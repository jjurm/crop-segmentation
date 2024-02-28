from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from utils.constants import MEDIANS_DTYPE, LABEL_DTYPE
from utils.medians import get_medians_subpatch_path
from utils.medians_metadata import MediansMetadata

# Divider for normalizing tiff data to [0-1] range
NORMALIZATION_DIV = 10000


class MediansDataset(Dataset):
    def __init__(
            self,
            split: str,
            saved_medians_path: Path,
            bins_range: tuple[int, int],
            linear_encoder: dict,
            metadata: MediansMetadata,
            requires_norm: bool = True,
    ) -> None:
        # number of total patches is given by number of patches in coco
        self.num_patches = metadata.get_split(split).size
        self.num_subpatches = metadata.num_subpatches
        self.requires_norm = requires_norm
        self.linear_encoder = linear_encoder
        self.medians_dir = saved_medians_path / split
        self.bins_range = bins_range
        self.metadata = metadata

    def load_medians(self, patch_dir: Path, subpatch_id: int, num_subpatches: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads precomputed medians for requested path.
        Medians are already padded and aggregated, so no need for further processing.
        """
        medians_raw = np.load(
            get_medians_subpatch_path(patch_dir, subpatch_id, num_subpatches),
            mmap_mode='r',
        )
        # shape: (bins, bands, height, width)

        medians = medians_raw[self.bins_range[0] - 1:self.bins_range[1]].astype(MEDIANS_DTYPE)

        # Read labels
        labels = np.load(get_medians_subpatch_path(patch_dir, subpatch_id, num_subpatches, labels=True))
        # shape: (height, width)

        return medians, labels

    def __getitem__(self, idx: int) -> dict:
        # The data item index (`idx`) corresponds to a single sequence.
        # In order to fetch the correct sequence, we must determine exactly which
        # patch, subpatch and bins it corresponds to.
        patch_id = (idx // self.num_subpatches) % self.num_patches + 1
        subpatch_id = idx % self.num_subpatches

        # They are already computed, therefore we just load them
        patch_dir = Path(self.medians_dir) / str(patch_id).rjust(len(str(self.num_patches)), "0")
        medians, labels = self.load_medians(patch_dir, subpatch_id, self.num_subpatches)

        # Normalize data to range [0-1]
        if self.requires_norm:
            medians = medians / NORMALIZATION_DIV

        # Map labels to 0-len(unique(crop_id)) see config
        # labels = np.vectorize(self.linear_encoder.get)(labels)
        labels_mapped = np.zeros_like(labels)
        for crop_id, linear_id in self.linear_encoder.items():
            labels_mapped[labels == crop_id] = linear_id
        # All classes NOT in linear encoder's values are already mapped to 0

        return {
            'medians': medians,
            'labels': labels_mapped.astype(np.int64),
            'parcels': (labels != 0),
            'idx': idx
        }

    def __len__(self):
        '''
        Computes the total number of produced sequences
        '''
        return self.num_patches * self.num_subpatches
