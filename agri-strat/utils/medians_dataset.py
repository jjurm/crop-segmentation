from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
import wandb
from urllib.parse import urlparse

from utils.constants import MEDIANS_DTYPE, LABEL_DTYPE
from utils.medians import get_medians_subpatch_path
from utils.medians_metadata import MediansMetadata

# Divider for normalizing tiff data to [0-1] range
NORMALIZATION_DIV = 10000


class MediansDataset(Dataset):
    def __init__(
            self,
            medians_artifact: str,
            bins_range: tuple[int, int],
            linear_encoder: dict,
            requires_norm: bool = True,
    ) -> None:
        # number of total patches is given by number of patches in coco
        self.requires_norm = requires_norm
        self.linear_encoder = linear_encoder
        self.bins_range = bins_range

        medians_artifact = wandb.run.use_artifact(medians_artifact, type='medians')

        # Load metadata
        metadata_path = medians_artifact.get_entry("meta.json").download()
        with open(metadata_path, 'r') as f:
            self.metadata = MediansMetadata.from_json(f.read())

        # Get the directory where computed medians are stored
        medians_url = urlparse(medians_artifact.manifest.entries["medians_stub"].ref)
        assert medians_url.scheme == "file"
        self.medians_dir = Path(medians_url.path).parent

    def load_medians(self, patch_dir: Path, subpatch_id: int, num_subpatches_per_patch: int) -> tuple[np.ndarray, np.ndarray]:
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

    def __getitem__(self, idx: int) -> dict:
        # The data item index (`idx`) corresponds to a single sequence.
        # In order to fetch the correct sequence, we must determine exactly which
        # patch, subpatch and bins it corresponds to.
        patch_id = (idx // self.metadata.num_subpatches_per_patch) % self.metadata.num_patches + 1
        subpatch_id = idx % self.metadata.num_subpatches_per_patch

        # Medians are already computed, therefore we just load them
        patch_dir = self.medians_dir / str(patch_id).rjust(len(str(self.metadata.num_patches)), "0")
        medians, labels = self.load_medians(patch_dir, subpatch_id, self.metadata.num_subpatches_per_patch)

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
            'parcels': (labels != 0),
            'idx': idx
        }

    def __len__(self):
        """
        Computes the total number of produced sequences
        """
        return self.metadata.num_patches * self.metadata.num_subpatches_per_patch
