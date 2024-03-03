from pathlib import Path

import lightning.pytorch as pl
from torch.utils.data import DataLoader

import utils.medians_metadata
from utils.medians_dataset import MediansDataset
from utils.medians_metadata import MediansMetadata


class MediansDataModule(pl.LightningDataModule):
    def __init__(
            self,
            medians_artifacts: dict[str, str],
            bins_range: tuple[int, int],
            linear_encoder: dict,
            requires_norm: bool,
            batch_size: int,
            num_workers: int,
    ):
        super().__init__()

        self.medians_artifacts = medians_artifacts
        self.bins_range = bins_range
        self.linear_encoder = linear_encoder
        self.requires_norm = requires_norm

        self.dataloader_args = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': True,
        }

        self.dataset_train: MediansDataset | None = None
        self.dataset_val: MediansDataset | None = None
        self.dataset_test: MediansDataset | None = None

    def prepare_data(self):
        pass

    def setup(self, stage: str = 'fit'):
        common_config = {
            'bins_range': self.bins_range,
            'linear_encoder': self.linear_encoder,
            'requires_norm': self.requires_norm,
        }

        if stage == 'fit':
            assert "train" in self.medians_artifacts and "val" in self.medians_artifacts, \
                "Both --train_medians_artifact and --val_medians_artifact are required for fitting."
            self.dataset_train = MediansDataset(medians_artifact=self.medians_artifacts["train"], **common_config)
            self.dataset_val = MediansDataset(medians_artifact=self.medians_artifacts["val"], **common_config)
        elif stage == 'test':
            assert "test" in self.medians_artifacts, "--test_medians_artifact is required for testing."
            self.dataset_test = MediansDataset(medians_artifact=self.medians_artifacts["test"], **common_config)
        else:
            raise ValueError(f"stage = {stage}, expected: 'fit' or 'test'")

    def train_dataloader(self):
        return DataLoader(self.dataset_train, shuffle=True, **self.dataloader_args)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, shuffle=False, **self.dataloader_args)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, shuffle=False, **self.dataloader_args)

    def get_num_bands(self):
        """
        Returns number of bands that the model should be created for.
        """
        assert self.dataset_train is not None or self.dataset_test is not None, "No dataset loaded."
        dataset = self.dataset_train if self.dataset_train is not None else self.dataset_test
        return len(dataset.metadata.bands)
