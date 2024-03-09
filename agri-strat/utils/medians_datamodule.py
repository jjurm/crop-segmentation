from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import wandb
from torch.utils.data import DataLoader

from utils.cached_dataset import CachedMediansDataset
from utils.medians_dataset import MediansDataset
from utils.medians_metadata import MediansMetadata


class MediansDataModule(pl.LightningDataModule):
    def __init__(
            self,
            medians_artifact: str,
            medians_path: str,
            split_artifact: str,
            bins_range: tuple[int, int],
            linear_encoder: dict,
            requires_norm: bool,
            batch_size: int,
            num_workers: int,
            cache_dataset: bool,
    ):
        super().__init__()

        self.medians_artifact = medians_artifact
        self.medians_path = medians_path
        self.split_artifact = split_artifact
        self.bins_range = bins_range
        self.linear_encoder = linear_encoder
        self.requires_norm = requires_norm
        self.cache_dataset = cache_dataset

        self.dataloader_args = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': True,
            'persistent_workers': True,
        }

        # prepare_data() will set these
        self.metadata_path: Path | None = None
        self.medians_subdir: Path | None = None
        self.splits_dir: Path | None = None
        self.patch_counts: pd.DataFrame | None = {}
        self.pixel_counts: dict[str, dict[int, int]] | None = {}

        # setup() will set these
        self.metadata: MediansMetadata | None = None
        self.dataset_train: MediansDataset | None = None
        self.dataset_val: MediansDataset | None = None
        self.dataset_test: MediansDataset | None = None

    # noinspection PyUnresolvedReferences
    def prepare_data(self):
        # Medians metadata and directory
        medians_artifact = wandb.run.use_artifact(self.medians_artifact, type='medians')
        self.metadata_path = Path(medians_artifact.get_entry("meta.json").download())
        self.medians_subdir = Path(self.medians_path) / medians_artifact.metadata["medians_dir_name"]

        # Splits
        splits_artifact = wandb.run.use_artifact(self.split_artifact, type='split')
        self.splits_dir = Path(splits_artifact.download()) / "splits"
        for i, row in splits_artifact.get("patch_counts").get_dataframe().iterrows():
            split = row["target"]
            self.patch_counts[split] = int(row["count"])
            self.pixel_counts[split] = splits_artifact.get(f"pixel_counts_{split}").get_dataframe() \
                .set_index("class")["count"].to_dict()

    def setup(self, stage: str = 'fit'):
        with open(self.metadata_path, 'r') as f:
            self.metadata = MediansMetadata.from_json(f.read())

        common_config = {
            'use_cache': self.cache_dataset,
            'medians_subdir': self.medians_subdir,
            'medians_metadata': self.metadata,
            'bins_range': self.bins_range,
            'linear_encoder': self.linear_encoder,
            'requires_norm': self.requires_norm,
        }

        if stage == 'fit':
            self.dataset_train = CachedMediansDataset(split_file=(self.splits_dir / "train.txt"),
                                                      patch_count=self.patch_counts["train"],
                                                      **common_config)
            self.dataset_val = CachedMediansDataset(split_file=(self.splits_dir / "val.txt"),
                                                    patch_count=self.patch_counts["val"],
                                                    **common_config)
        elif stage == 'test':
            self.dataset_test = MediansDataset(split_file=(self.splits_dir / "test.txt"),
                                               patch_count=self.patch_counts["test"],
                                               **common_config)
        else:
            raise ValueError(f"stage = {stage}, expected: 'fit' or 'test'")

    def train_dataloader(self):
        return DataLoader(self.dataset_train, shuffle=True, **self.dataloader_args)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, shuffle=False, **self.dataloader_args)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, shuffle=False, **self.dataloader_args)

    def get_bands(self) -> list[str]:
        """
        Returns a list of bands that the model should be created for.
        """
        return self.metadata.bands
