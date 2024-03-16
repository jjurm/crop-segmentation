from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset, get_worker_info

from utils.constants import MEDIANS_DTYPE
from utils.label_encoder import LabelEncoder
from utils.medians_metadata import MediansMetadata

# Divider for normalizing tiff data to [0-1] range
NORMALIZATION_DIV = 10000


class MediansDataset(IterableDataset):
    def __init__(
            self,
            split_file: Path,
            patch_count: int,
            # the parameters below do not differ between train/val/test datasets
            medians_subdir: Path,
            medians_metadata: MediansMetadata,
            bins_range: tuple[int, int],
            label_encoder: LabelEncoder,
            requires_norm: bool,
            shuffle: bool = False,
            batched: bool = True,  # When true, each sample is a batch of subpatches
            skip_zero_label_subpatches: bool = False,
    ) -> None:
        super().__init__()
        self.patch_count = patch_count
        self.medians_subdir = medians_subdir
        self.medians_metadata = medians_metadata
        self.bins_range = bins_range
        self.label_encoder = label_encoder
        self.requires_norm = requires_norm
        self.batched = batched
        self.skip_zero_label_subpatches = skip_zero_label_subpatches

        df = pd.read_csv(split_file, header=None, names=["path"])["path"]
        if shuffle:
            # Shuffling is done before workers are spawned so that it is the same for all workers
            df = df.sample(frac=1)
        self.split_df = df

    def load_medians(self, patch_relative_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads precomputed medians for requested path.
        Medians are already padded and aggregated, so no need for further processing.
        """
        filename = self.medians_subdir / (str(patch_relative_path) + ".npz")
        npz = np.load(filename)

        medians_raw = npz['medians']
        # shape: (subpatch_y, subpatch_x, bins, bands, height, width)
        medians = medians_raw[:, :, self.bins_range[0] - 1:self.bins_range[1]].astype(MEDIANS_DTYPE)

        # Read labels
        labels = npz['labels']
        # shape: (subpatch_y, subpatch_x, height, width)

        return medians, labels

    def get_worker_patches(self):
        worker_info = get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            paths = self.split_df
        else:  # in a worker process, split workload
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            paths = self.split_df.iloc[worker_id::num_workers]
        return paths

    def __iter__(self):
        paths = self.get_worker_patches()

        for patch_path in paths:
            medians, labels = self.load_medians(patch_path)

            # Normalize data to range [0-1]
            if self.requires_norm:
                medians = medians / NORMALIZATION_DIV

            # Map labels to 0-len(unique(crop_id)) see config
            labels_mapped = np.zeros_like(labels)
            for crop_id, linear_id in self.label_encoder.dataset_to_model.items():
                labels_mapped[labels == crop_id] = linear_id
            # All classes NOT in the linear encoder's values are already mapped to 0
            labels_mapped = labels_mapped.astype(np.int64)

            batch = []
            for subpatch_y in range(medians.shape[0]):
                for subpatch_x in range(medians.shape[1]):
                    if self.skip_zero_label_subpatches and not np.any(labels_mapped[subpatch_y, subpatch_x]):
                        continue
                    batch.append({
                        'medians': medians[subpatch_y, subpatch_x],
                        'labels': labels_mapped[subpatch_y, subpatch_x],
                        # 'parcels': (labels != 0),
                        'patch_path': patch_path,
                        'subpatch_yx': (subpatch_y, subpatch_x),
                    })

            if self.batched:
                yield batch
            else:
                for sample in batch:
                    yield sample

    def get_medians_shape(self):
        return (
            self.bins_range[1] - self.bins_range[0] + 1,
            len(self.medians_metadata.bands),
            *self.medians_metadata.img_size
        )

    def get_labels_shape(self):
        return *self.medians_metadata.img_size,
