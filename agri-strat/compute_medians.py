#!/usr/bin/env python
"""
This script computes the medians of the bands in the netCDF files and saves them to disk. The medians are computed
for each subpatch of the netCDF files, and the subpatches (and the labels) are saved to disk as numpy arrays.
"""

import argparse
import hashlib
import itertools
import os
from functools import partial
from multiprocessing import Pool, Manager
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm

import wandb
from utils.constants import BANDS, IMG_SIZE, REFERENCE_BAND, MEDIANS_DTYPE, LABEL_DTYPE
from utils.medians_metadata import MediansMetadata


def parse_arguments():
    # Parse user arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--netcdf_path', type=Path, default=None, required=False,
                        help='Path to the netCDF files. Default $NETCDF_PATH or "dataset/netcdf".')
    parser.add_argument('--medians_path', type=str, default=None, required=False,
                        help='Path to the directory with subdirectories of medians. Defaults to $MEDIANS_PATH or '
                             'dataset/medians.')

    parser.add_argument('--group_freq', type=str, default='1MS', required=False,
                        help='The frequency to aggregate medians with. Default "1MS".')
    parser.add_argument('--bands', nargs='+', default=BANDS.keys(), required=False,
                        help='The bands to use. Default all.')
    parser.add_argument('--output_size', nargs=2, type=int, default=[IMG_SIZE, IMG_SIZE], required=False,
                        help='The size of the medians (height, width). If none given, the output will be of the same '
                             'size.')

    parser.add_argument('--num_workers', type=int, default=8, required=False,
                        help='The number of workers to use for parallel computation. Default 8.')
    parser.add_argument('--num_readers', type=int, default=8, required=False,
                        help='The max number of simultaneous readers. Default 8.')
    parser.add_argument('--num_writers', type=int, default=8, required=False,
                        help='The max number of simultaneous writers. Default 8.')
    parser.add_argument('--limit_patches', type=int, default=None, required=False,
                        help='Limit the number of patches to process, for debugging. Default: None.')
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


def process_patch(out_path, data_path, bands, group_freq, output_size, semaphore_read, semaphore_write,
                  patch_relative_path):
    # Calculate medians
    with semaphore_read:
        netcdf = netCDF4.Dataset(data_path / patch_relative_path, 'r')
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
    target_filename = out_path / (str(patch_relative_path) + ".npz")
    with semaphore_write:
        target_filename.parent.mkdir(exist_ok=True, parents=True)
        np.savez(target_filename,
                 medians=medians_subpatches,
                 labels=labels)

    netcdf.close()

    return patch_size, num_subpatches


def main():
    with wandb.init(
            project='agri-strat',
            job_type='compute-medians',
            config=vars(parse_arguments()),
    ) as run:
        if run.config["limit_patches"] is not None:
            run.tags = run.tags + ("devtest",)

        print("Listing patches...")
        netcdf_path = run.config["netcdf_path"] or Path(os.environ.get("NETCDF_PATH", "dataset/netcdf"))
        medians_path = run.config["medians_path"] or os.environ.get("MEDIANS_PATH", "dataset/medians")
        jobs = []
        patches_generator = netcdf_path.glob('**/*.nc')
        if run.config["limit_patches"] is not None:
            patches_generator = itertools.islice(patches_generator, run.config["limit_patches"])
        for patch_path in patches_generator:
            jobs.append(patch_path.relative_to(netcdf_path))

        metadata = MediansMetadata(
            bands=sorted(run.config["bands"]),
            img_size=run.config["output_size"],
            num_patches=len(jobs),
        )

        print("Computing medians...")
        with Manager() as manager:
            semaphore_read = manager.Semaphore(run.config["num_readers"])
            semaphore_write = manager.Semaphore(run.config["num_writers"])

            run_name_short_hash = hashlib.sha1(run.id.encode("utf-8")).hexdigest()[-6:]
            medians_name = f"{run.config['group_freq']}_{''.join(run.config['bands'])}_{run.config['output_size'][0]}x{run.config['output_size'][1]}"
            medians_dir_name = f"{medians_name}-{run_name_short_hash}"
            out_path = Path(medians_path) / medians_dir_name
            out_path.mkdir(exist_ok=False, parents=True)

            # Process patches in parallel
            func = partial(process_patch,
                           out_path, netcdf_path, metadata.bands, run.config["group_freq"], run.config["output_size"],
                           semaphore_read, semaphore_write)

            with tqdm(total=len(jobs)) as pbar:
                pool = Pool(run.config["num_workers"])
                for patch_size, num_subpatches in pool.imap_unordered(func, jobs):

                    if not metadata.patch_size:
                        metadata.patch_size = patch_size
                        metadata.num_subpatches_per_patch = num_subpatches
                    else:
                        assert metadata.patch_size == patch_size, (
                            f"Expected the same patch size for all patches but got "
                            f"{metadata.patch_size} and {patch_size} for some patches")

                    # Update the progress bar
                    pbar.update(1)

        # Log an artifact with metadata
        metadata_filename = out_path / "meta.json"
        with open(metadata_filename, 'w') as f:
            f.write(metadata.to_json())
        print(f"Medians metadata written to {metadata_filename}.")

        artifact = wandb.Artifact(
            name=medians_name,
            type="medians",
            metadata={
                "medians_path": medians_path,
                # Consumer scripts should receive path as arg instead of using this
                "medians_dir_name": medians_dir_name,
                "num_patches": len(jobs),
            },
        )
        artifact.add_file(metadata_filename.as_posix(), name="meta.json")

        # The stub file is a workaround to allow referencing a directory.
        # Referencing the dir directly would make wandb reference all contained files instead, whereas the main
        # point is to preserve the path to the directory.
        stub_file = out_path / f".stub"
        open(stub_file, 'a').close()
        artifact.add_reference(stub_file.absolute().as_uri(), name="medians_stub", checksum=False)

        run.log_artifact(artifact)


if __name__ == '__main__':
    main()
