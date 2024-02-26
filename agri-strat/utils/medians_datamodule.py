from pathlib import Path

import lightning.pytorch as pl
from torch.utils.data import DataLoader

import utils.medians_metadata
from utils.medians_dataset import MediansDataset
from utils.medians_metadata import MediansMetadata


class MediansDataModule(pl.LightningDataModule):
    def __init__(
            self,
            saved_medians_path: Path,
            bins_range: tuple[int, int],
            linear_encoder: dict,
            requires_norm: bool,
            batch_size: int,
            num_workers: int,
    ):
        super().__init__()

        self.saved_medians_path = saved_medians_path
        self.bins_range = bins_range
        self.linear_encoder = linear_encoder
        self.requires_norm = requires_norm

        self.dataloader_args = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': True,
        }

        self.metadata: MediansMetadata | None = None
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

    def prepare_data(self):
        pass

    def setup(self, stage: str = 'fit'):
        # load mendians' metadata
        with open(self.saved_medians_path / utils.medians_metadata.FILENAME, 'r') as f:
            self.metadata = MediansMetadata.from_json(f.read())

        common_config = {
            'saved_medians_path': self.saved_medians_path,
            'bins_range': self.bins_range,
            'linear_encoder': self.linear_encoder,
            'metadata': self.metadata,
            'requires_norm': self.requires_norm,
        }

        if stage == 'fit':
            self.dataset_train = MediansDataset(split='train', **common_config)
            self.dataset_val = MediansDataset(split='val', **common_config)
        elif stage == 'test':
            self.dataset_test = MediansDataset(split='test', **common_config)
        else:
            raise ValueError(f"stage = {stage}, expected: 'fit' or 'test'")

    def train_dataloader(self):
        return DataLoader(self.dataset_train, shuffle=True, **self.dataloader_args)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, shuffle=False, **self.dataloader_args)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, shuffle=False, **self.dataloader_args)
