"""
This script takes the txt split files produced by `split_data.py` and computes the medians for each patch in the dataset
and saves them to disk. The medians are computed for each band and for each month in the year.
It also computes the (per-split) pixel counts for each class in the dataset and saves them to a metadata file along
with other metadata to be used directly in training.
"""

import argparse
from functools import partial
from multiprocessing import Pool, Manager
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from numpy.lib.stride_tricks import as_strided
from pycocotools.coco import COCO
from tqdm import tqdm
import wandb
import hashlib

from utils.constants import BANDS, IMG_SIZE, REFERENCE_BAND, MEDIANS_DTYPE, LABEL_DTYPE
from utils.medians import get_medians_subpatch_path
from utils.medians_metadata import MediansMetadata


def parse_arguments():
    # Parse user arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--splits_artifact', type=str, required=False,
                        help='Wandb artifact of type \'split\' containing the txt files with train/val/test splits.')
    parser.add_argument('--coco_path', type=str, default=None, required=False,
                        help='Root path for coco file, such as "coco_files".')
    parser.add_argument('--coco_prefix', type=str, default=None, required=False,
                        help='Prefix for the coco files, which will be combined with the suffix "_coco_{mode}.json".')
    parser.add_argument('--netcdf_path', type=str, default='dataset/netcdf', required=False,
                        help='Path to the netCDF files. Default "dataset/netcdf".')

    parser.add_argument('--group_freq', type=str, default='1MS', required=False,
                        help='The frequency to aggregate medians with. Default "1MS".')
    parser.add_argument('--bands', nargs='+', default=BANDS.keys(), required=False,
                        help='The bands to use. Default all.')
    parser.add_argument('--output_size', nargs='+', type=int, default=[IMG_SIZE, IMG_SIZE], required=False,
                        help='The size of the medians (height, width). If none given, the output will be of the same size.')

    parser.add_argument('--num_workers', type=int, default=8, required=False,
                        help='The number of workers to use for parallel computation. Default 8.')
    parser.add_argument('--num_readers', type=int, default=8, required=False,
                        help='The max number of simultaneous readers. Default 8.')
    parser.add_argument('--num_writers', type=int, default=8, required=False,
                        help='The max number of simultaneous writers. Default 8.')

    return parser.parse_args()


def sliding_window_view(arr, window_shape, steps):
    """
    Code taken from:
        https://gist.github.com/meowklaski/4bda7c86c6168f3557657d5fb0b5395a

    Produce a view from a sliding, striding window over `arr`.
        The window is only placed in 'valid' positions - no overlapping
        over the boundary.
        Parameters
        ----------
        arr : numpy.ndarray, shape=(...,[x, (...), z])
            The array to slide the window over.
        window_shape : Sequence[int]
            The shape of the window to raster: [Wx, (...), Wz],
            determines the length of [x, (...), z]
        steps : Sequence[int]
            The step size used when applying the window
            along the [x, (...), z] directions: [Sx, (...), Sz]
        Returns
        -------
        view of `arr`, shape=([X, (...), Z], ..., [Wx, (...), Wz])
            Where X = (x - Wx) // Sx + 1
    """
    in_shape = np.array(arr.shape[-len(steps):])  # [x, (...), z]
    window_shape = np.array(window_shape)  # [Wx, (...), Wz]
    steps = np.array(steps)  # [Sx, (...), Sz]
    nbytes = arr.strides[-1]  # size (bytes) of an element in `arr`

    # number of per-byte steps to take to fill window
    window_strides = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)
    # number of per-byte steps to take to place window
    step_strides = tuple(window_strides[-len(steps):] * steps)
    # number of bytes to step to populate sliding window view
    strides = tuple(int(i) * nbytes for i in step_strides + window_strides)

    outshape = tuple((in_shape - window_shape) // steps + 1)
    # outshape: ([X, (...), Z], ..., [Wx, (...), Wz])
    outshape = outshape + arr.shape[:-len(steps)] + tuple(window_shape)
    return as_strided(arr, shape=outshape, strides=strides, writeable=False)


def process_patch(out_path, data_path, bands, group_freq, output_size, semaphore_read, semaphore_write, job):
    mode, num_patches, patch_id, patch_info = job

    patch_dir = out_path / mode / str(patch_id).rjust(len(str(num_patches)), "0")
    patch_dir.mkdir(exist_ok=True, parents=True)

    # Calculate medians
    with semaphore_read:
        netcdf = netCDF4.Dataset(data_path / patch_info['file_name'], 'r')

    year = netcdf.patch_year

    # Calculate the padding between two neighboring subpatches
    patch_size = tuple(netcdf["labels"].dimensions[d].size for d in ["y", "x"])
    sliding_window_step = [
        patch_size[i] // (patch_size[i] // output_size[i])
        for i in range(2)
    ]

    # output intervals
    date_range = pd.date_range(start=f'{year}-01-01', end=f'{int(year) + 1}-01-01', freq=group_freq)
    num_buckets = len(date_range) - 1

    # out, aggregated array
    medians = np.empty((len(bands), num_buckets, patch_size[0], patch_size[1]), dtype=MEDIANS_DTYPE)
    for band_id, band in enumerate(bands):
        # Load band data
        band_data = xr.open_dataset(xr.backends.NetCDF4DataStore(netcdf[band]))

        # Aggregate into time bins
        band_data = band_data.groupby_bins(
            'time',
            bins=date_range,
            right=True,
            include_lowest=False,
            labels=date_range[:-1]
        ).median(dim='time')

        # Upsample so months without data are initiated with NaN values
        # band_data = band_data.resample(time_bins=group_freq).median(dim='time_bins')

        # Fill:
        # NaN months with linear interpolation
        # NaN months outsize of range (e.g. month 12) using extrapolation
        band_data = band_data.interpolate_na(dim='time_bins', method='linear', fill_value='extrapolate')

        # Keep values within requested time window
        band_data = band_data.isel(time_bins=slice(0, 0 + num_buckets))

        # Convert to numpy array
        band_data = band_data[band].values

        # If expand ratio is 1, that means current band has the same resolution as reference band
        expand_ratio = int(BANDS[band] / BANDS[REFERENCE_BAND])

        # If resolution does not match reference band, stretch it
        if expand_ratio != 1:
            band_data = np.repeat(band_data, expand_ratio, axis=1)
            band_data = np.repeat(band_data, expand_ratio, axis=2)

        assert num_buckets == band_data.shape[0], f"Expected {num_buckets} time bins, got {band_data.shape[0]}"
        medians[band_id, :, :, :] = np.expand_dims(band_data, axis=0)

    # Reshape so window length is first
    medians = medians.transpose(1, 0, 2, 3)

    medians_subpatches = sliding_window_view(
        arr=medians,
        window_shape=[num_buckets, len(bands), output_size[0], output_size[1]],
        steps=[1, 1, sliding_window_step[0], sliding_window_step[1]]) \
        .astype(MEDIANS_DTYPE) \
        .squeeze(axis=(0, 1,))  # corresponding to dimensions num_bins and num_bands, since there is no sliding
    # shape: (subpatches_in_row, subpatches_in_col, bins, bands, height, width)

    # Labels: Load and Convert to numpy array
    labels = xr.open_dataset(xr.backends.NetCDF4DataStore(netcdf['labels']))['labels'].values
    labels = sliding_window_view(labels, output_size, sliding_window_step) \
        .astype(LABEL_DTYPE)
    # shape: (subpatches_in_row, subpatches_in_col, height, width)

    assert medians_subpatches.shape[:2] == labels.shape[:2]

    # Save medians and labels
    num_subpatches = medians_subpatches.shape[0] * medians_subpatches.shape[1]
    with semaphore_write:
        for i in range(medians_subpatches.shape[0]):
            for j in range(medians_subpatches.shape[1]):
                sub_idx = i * medians_subpatches.shape[1] + j
                np.save(get_medians_subpatch_path(patch_dir, sub_idx, num_subpatches), medians_subpatches[i, j, :, :, :, :])
                np.save(get_medians_subpatch_path(patch_dir, sub_idx, num_subpatches, labels=True), labels[i, j, :, :])

    # Compute pixel counts for each class
    class_pixel_counts = {
        int(c): int(np.sum(labels == c))
        for c in np.unique(labels)
    }

    return num_subpatches, mode, class_pixel_counts


def generate_jobs_and_metadata(args, bands, splits_artifact=None):
    # Accumulate metadata of each split
    splits_metadata = {}
    # Create a job for each patch
    jobs = []

    if splits_artifact:
        patch_counts: pd.DataFrame = splits_artifact.get("patch_counts").get_dataframe()
        splits_dir = Path(splits_artifact.download()) / "splits"
        for _, row in patch_counts.iterrows():
            split_file = splits_dir / f"{row['target']}.txt"
            split_df = pd.read_csv(split_file, header=None, names=['path'])

            # Store metadata
            splits_metadata[row['target']] = MediansMetadata(
                bands=bands,
                img_size=args.output_size,
                num_patches=row["count"],
            )

            # Create a job for each patch
            jobs.extend(
                (split_file.stem, row["count"], i + 1, {'file_name': patch_row['path']})
                for i, patch_row in split_df.iterrows()
            )
    else:
        for mode in ['train', 'val', 'test']:
            coco_path = Path(args.coco_path)
            coco_prefix = f"{args.coco_prefix}_" if args.coco_prefix else ""
            coco = COCO(coco_path / f"{coco_prefix}coco_{mode}.json")

            # We are assuming that each coco file has patches with ids from 1 to n
            num_patches = len(coco.imgs)
            # Store metadata
            splits_metadata[mode] = MediansMetadata(
                bands=bands,
                img_size=args.output_size,
                num_patches=0,
            )

            # Create a job for each patch
            jobs.extend((mode, num_patches, patch_id, patch_info) for patch_id, patch_info in coco.imgs.items())

    return jobs, splits_metadata


def main():
    args = parse_arguments()

    if not args.coco_path and not args.splits_artifact:
        raise ValueError("One of coco_path and splits_artifact must be set.")
    if args.coco_path and args.splits_artifact:
        raise ValueError("Both coco_path and splits_artifact cannot be set at the same time. Please choose one of them.")
    if args.output_size and len(args.output_size) != 2:
        raise ValueError(f"Output size must be a tuple of 2 integers. Got {args.output_size}")

    with wandb.init(
            project='agri-strat',
            job_type='compute-medians',
            config=vars(args),
    ) as run:
        netcdf_path = Path(args.netcdf_path)
        bands = sorted(args.bands)

        splits_artifact = None
        if args.splits_artifact:
            splits_artifact = run.use_artifact(args.splits_artifact, type='split')

        jobs, splits_metadata = generate_jobs_and_metadata(args, bands, splits_artifact)

        with Manager() as manager:
            semaphore_read = manager.Semaphore(args.num_readers)
            semaphore_write = manager.Semaphore(args.num_writers)

            splits_name = splits_artifact.metadata.get("split_rules_name") if splits_artifact else (args.coco_prefix or "")
            splits_prefix = f"{splits_name}_" if splits_name else ""
            splits_id = f"{splits_prefix}{args.group_freq}_{''.join(args.bands)}_{args.output_size[0]}x{args.output_size[1]}"
            out_path = Path('dataset') / 'medians' / splits_id
            out_path.mkdir(exist_ok=False, parents=True)

            # Process patches in parallel
            func = partial(process_patch,
                           out_path, netcdf_path, bands, args.group_freq, args.output_size,
                           semaphore_read, semaphore_write)

            with tqdm(total=len(jobs)) as pbar:
                pool = Pool(args.num_workers)
                for num_subpatches, mode, class_pixel_counts in pool.imap_unordered(func, jobs):

                    # Store the number of subpatches per patch (should be the same for all patches)
                    if not splits_metadata[mode].num_subpatches_per_patch:
                        splits_metadata[mode].num_subpatches_per_patch = num_subpatches
                    else:
                        assert splits_metadata[mode].num_subpatches_per_patch == num_subpatches, (
                            f"Expected the same number of subpatches in each patch but got "
                            f"{splits_metadata[mode].num_subpatches_per_patch} and {num_subpatches} for some patches")

                    # Store the pixel counts for each class
                    dictionary = splits_metadata[mode].class_pixel_counts
                    for k, v in class_pixel_counts.items():
                        dictionary[k] = dictionary.get(k, 0) + v

                    # Update the progress bar
                    pbar.update(1)

        # Save metadata of each split
        for mode, medians_metadata in splits_metadata.items():
            metadata_filename = out_path / f"{mode}_meta.json"
            with open(metadata_filename, 'w') as f:
                f.write(medians_metadata.to_json())
            print(f"Medians metadata written to {metadata_filename}.")

            short_hash = hashlib.sha1(splits_id.encode("utf-8")).hexdigest()[-4:]
            artifact = wandb.Artifact(
                name=f"{splits_prefix}medians_{short_hash}_{mode}",
                type="medians",
                metadata={
                    "split_rules_name": splits_name,
                    "splits_id": splits_id,
                    "split": mode,
                },
            )
            artifact.add_file(metadata_filename.as_posix(), name="meta.json")

            # The stub file is a workaround to allow referencing a directory.
            # Referencing the dir directly would make wandb reference all contained files instead, whereas the main
            # point is to preserve the path to the directory.
            stub_file = out_path / mode / f".stub"
            open(stub_file, 'a').close()
            artifact.add_reference(stub_file.absolute().as_uri(), name="medians_stub", checksum=False)

            run.log_artifact(artifact)


if __name__ == '__main__':
    main()
