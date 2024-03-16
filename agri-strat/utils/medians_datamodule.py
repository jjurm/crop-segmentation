from math import ceil
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import wandb
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter.utils import IterableWrapperIterDataPipe

from utils.medians_dataset import MediansDataset
from utils.medians_metadata import MediansMetadata


# TODO consider using the following function to have
# - different seed for each epoch
# - reproducibility even if loading from a checkpoing (need to plug in the epoch number)
# def worker_init_fn(id, split_seed: int):
#     # Recommended by NumPy Rng Author: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
#     # Another good resource: https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
#     process_seed = torch.initial_seed()
#     # Back out the base_seed so we can use all the bits.
#     base_seed = process_seed - id
#     # TODO: split_seed seems to have no impact.
#     ss = np.random.SeedSequence(
#         [id, base_seed, split_seed]
#     )  # Rylan added split seed.
#     # More than 128 bits (4 32-bit words) would be overkill.
#     np_rng_seed = ss.generate_state(4)
#     np.random.seed(np_rng_seed)


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
            shuffle_buffer_num_patches: int,
            skip_zero_label_subpatches: bool,
    ):
        super().__init__()

        self.medians_artifact = medians_artifact
        self.medians_path = medians_path
        self.split_artifact = split_artifact
        self.bins_range = bins_range
        self.linear_encoder = linear_encoder
        self.requires_norm = requires_norm
        self.batch_size = batch_size
        self.cache_dataset = cache_dataset
        self.shuffle_buffer_num_patches = shuffle_buffer_num_patches
        self.skip_zero_label_subpatches = skip_zero_label_subpatches

        self.dataloader_args = {
            'batch_size': None,
            'num_workers': num_workers,
            'pin_memory': True,
            'persistent_workers': True,
            'drop_last': False,
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
            'medians_subdir': self.medians_subdir,
            'medians_metadata': self.metadata,
            'bins_range': self.bins_range,
            'linear_encoder': self.linear_encoder,
            'requires_norm': self.requires_norm,
        }

        if stage == 'fit':
            self.dataset_train = MediansDataset(split_file=(self.splits_dir / "train.txt"),
                                                patch_count=self.patch_counts["train"],
                                                shuffle=True, batched=False,
                                                skip_zero_label_subpatches=self.skip_zero_label_subpatches,
                                                **common_config)
            self.dataset_val = MediansDataset(split_file=(self.splits_dir / "val.txt"),
                                              patch_count=self.patch_counts["val"],
                                              batched=True,
                                              **common_config)
        elif stage == 'test':
            self.dataset_test = MediansDataset(split_file=(self.splits_dir / "test.txt"),
                                               patch_count=self.patch_counts["test"],
                                               batched=True,
                                               **common_config)
        else:
            raise ValueError(f"stage = {stage}, expected: 'fit' or 'test'")

    def train_dataloader(self):
        dataloader = DataLoader(self.dataset_train, prefetch_factor=2 * self.batch_size, **self.dataloader_args)
        pipe = IterableWrapperIterDataPipe(dataloader, deepcopy=False) \
            .shuffle(buffer_size=self.shuffle_buffer_num_patches * self.metadata.num_subpatches_per_patch) \
            .batch(batch_size=self.batch_size).collate()
        return pipe

    def val_dataloader(self):
        dataloader = DataLoader(self.dataset_val, **self.dataloader_args)
        pipe = IterableWrapperIterDataPipe(dataloader, deepcopy=False) \
            .unbatch() \
            .batch(batch_size=self.batch_size).collate()
        return pipe

    def test_dataloader(self):
        dataloader = DataLoader(self.dataset_test, **self.dataloader_args)
        pipe = IterableWrapperIterDataPipe(dataloader, deepcopy=False) \
            .unbatch() \
            .batch(batch_size=self.batch_size).collate()
        return pipe

    def get_bands(self) -> list[str]:
        """
        Returns a list of bands that the model should be created for.
        """
        return self.metadata.bands

    def get_approx_num_batches(self, split: str) -> int:
        """
        If skip_zero_label_subpatches is True, the number of batches is not known in advance, so we return an upper bound.
        Otherwise, returns the exact number of batches.
        :param split: train/val/test
        """
        approx_num_samples = self.patch_counts[split] * self.metadata.num_subpatches_per_patch
        if self.dataloader_args.get("drop_last", False):
            approx_num_batches = approx_num_samples // self.batch_size
        else:
            approx_num_batches = ceil(approx_num_samples / self.batch_size)
        return approx_num_batches
