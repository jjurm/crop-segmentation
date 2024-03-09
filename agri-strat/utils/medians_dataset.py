from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils.constants import MEDIANS_DTYPE
from utils.medians import get_medians_subpatch_path
from utils.medians_metadata import MediansMetadata

# Divider for normalizing tiff data to [0-1] range
NORMALIZATION_DIV = 10000


class MediansDataset(Dataset):
    def __init__(
            self,
            split_file: Path,
            patch_count: int,
            # the parameters below do not differ between train/val/test datasets
            medians_subdir: Path,
            medians_metadata: MediansMetadata,
            bins_range: tuple[int, int],
            linear_encoder: dict,
            requires_norm: bool,
    ) -> None:
        self.patch_count = patch_count
        self.medians_subdir = medians_subdir
        self.medians_metadata = medians_metadata
        self.bins_range = bins_range
        self.linear_encoder = linear_encoder
        self.requires_norm = requires_norm

        self.split_df = pd.read_csv(split_file, header=None, names=["path"])["path"]

    def load_medians(self, patch_dir: Path, subpatch_id: int, num_subpatches_per_patch: int) -> tuple[
        np.ndarray, np.ndarray]:
        """
        Loads precomputed medians for requested path.
        Medians are already padded and aggregated, so no need for further processing.
        """
        medians_raw = np.load(
            get_medians_subpatch_path(patch_dir, subpatch_id, num_subpatches_per_patch),
            mmap_mode='r',
        )
        # shape: (bins, bands, height, width)

        medians = medians_raw[self.bins_range[0] - 1:self.bins_range[1]].astype(MEDIANS_DTYPE)

        # Read labels
        labels = np.load(get_medians_subpatch_path(patch_dir, subpatch_id, num_subpatches_per_patch, labels=True))
        # shape: (height, width)

        return medians, labels

    def get_medians_shape(self):
        return (
            self.bins_range[1] - self.bins_range[0] + 1,
            len(self.medians_metadata.bands),
            *self.medians_metadata.img_size
        )

    def get_labels_shape(self):
        return *self.medians_metadata.img_size,

    def __getitem__(self, idx: int) -> dict:
        # The data item index (`idx`) corresponds to a single sequence.
        # In order to fetch the correct sequence, we must determine exactly which
        # patch, subpatch and bins it corresponds to.
        patch_id = idx // self.medians_metadata.num_subpatches_per_patch
        patch_path = self.split_df[patch_id]
        subpatch_id = idx % self.medians_metadata.num_subpatches_per_patch

        # Medians are already computed, therefore we just load them
        patch_dir = self.medians_subdir / patch_path
        medians, labels = self.load_medians(patch_dir, subpatch_id, self.medians_metadata.num_subpatches_per_patch)

        # Normalize data to range [0-1]
        if self.requires_norm:
            medians = medians / NORMALIZATION_DIV

        # Map labels to 0-len(unique(crop_id)) see config
        # labels = np.vectorize(self.linear_encoder.get)(labels)
        labels_mapped = np.zeros_like(labels)
        for crop_id, linear_id in self.linear_encoder.items():
            labels_mapped[labels == crop_id] = linear_id
        # All classes NOT in the linear encoder's values are already mapped to 0

        return {
            'medians': medians,
            'labels': labels_mapped.astype(np.int64),
            # 'parcels': (labels != 0),
            'idx': idx,
            'patch_path': patch_path,
            'subpatch_id': subpatch_id,
        }

    def __len__(self):
        """
        Computes the total number of produced sequences
        """
        return self.patch_count * self.medians_metadata.num_subpatches_per_patch
