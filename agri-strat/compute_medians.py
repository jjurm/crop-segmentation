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

from utils.constants import BANDS, IMG_SIZE, REFERENCE_BAND, MEDIANS_DTYPE, LABEL_DTYPE
from utils.medians import get_medians_subpatch_path
from utils.medians_metadata import MediansMetadata, MediansMetadataPerSplit
import utils


def parse_arguments():
    # Parse user arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_path', type=str, default='coco_files', required=False,
                        help='Root path for coco files. Default "coco_files".')
    parser.add_argument('--coco_prefix', type=str, default=None, required=False,
                        help='Prefix for the coco files, which will be combined with the suffix "_coco_{mode}.json". Default None.')
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

    parser.add_argument('--out_path', type=str, default='dataset/medians', required=False,
                        help='Path to export the medians into. Default "dataset/medians".')
    parser.add_argument("--auto_subdir", action='store_true', default=False, required=False,
                        help='Create an automatically named subdirectory for the medians under out_path. Default False.')

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
    patch_size = (int(patch_info['height']), int(patch_info['width']))
    sliding_window_step = [
        patch_size[i] // (patch_size[i] // output_size[i])
        for i in range(2)
    ]

    # output intervals
    date_range = pd.date_range(start=f'{year}-01-01', end=f'{int(year) + 1}-01-01', freq=group_freq)
    num_buckets = len(date_range) - 1

    # out, aggregated array
    medians = np.empty((len(bands), num_buckets, patch_info['height'], patch_info['width']), dtype=MEDIANS_DTYPE)
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

    return num_subpatches


def main():
    args = parse_arguments()

    coco_path = Path(args.coco_path)
    coco_prefix = f"{args.coco_prefix}_" if args.coco_prefix else ""
    netcdf_path = Path(args.netcdf_path)
    bands = sorted(args.bands)

    if args.output_size and len(args.output_size) != 2:
        raise ValueError(f"Output size must be a tuple of 2 integers. Got {args.output_size}")
    out_path = Path(args.out_path)
    if args.auto_subdir:
        out_path = out_path / f"{coco_prefix}{args.group_freq}_{''.join(args.bands)}_{args.output_size[0]}x{args.output_size[1]}"
    # Create medians folder if it doesn't exist
    out_path.mkdir(exist_ok=True, parents=True)

    medians_metadata = MediansMetadata(
        bands=bands,
        img_size=args.output_size,
    )

    with Manager() as manager:
        semaphore_read = manager.Semaphore(args.num_readers)
        semaphore_write = manager.Semaphore(args.num_writers)

        # Create a job for each patch
        jobs = []
        for mode in ['train', 'val', 'test']:
            coco = COCO(coco_path / f"{coco_prefix}coco_{mode}.json")

            # We are assuming that each coco file has patches with ids from 1 to n
            num_patches = len(coco.imgs)
            setattr(medians_metadata, mode, MediansMetadataPerSplit(size=num_patches))

            jobs.extend((mode, num_patches, patch_id, patch_info) for patch_id, patch_info in coco.imgs.items())

        # Process patches in parallel
        func = partial(process_patch, out_path, netcdf_path, bands, args.group_freq, args.output_size, semaphore_read, semaphore_write)

        with tqdm(total=len(jobs)) as pbar:
            pool = Pool(args.num_workers)
            for num_subpatches in pool.imap_unordered(func, jobs):
                if not medians_metadata.num_subpatches:
                    medians_metadata.num_subpatches = num_subpatches
                else:
                    assert medians_metadata.num_subpatches == num_subpatches, f"Expected the same number of subpatches in each patch but got {medians_metadata.num_subpatches} and {num_subpatches} for some patches"
                pbar.update(1)

    # Save metadata
    with open(out_path / utils.medians_metadata.FILENAME, 'w') as f:
        f.write(medians_metadata.to_json())


if __name__ == '__main__':
    main()
