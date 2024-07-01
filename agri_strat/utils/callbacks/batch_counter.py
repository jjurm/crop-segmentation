from typing import Any, Dict

import lightning as pl
import wandb

from agri_strat.utils.medians_datamodule import MediansDataModule


class BatchCounterCallback(pl.Callback):
    """
    This callback is used to set Trainer's max_batches for train, val and test loops, which fixes the progress bar when
    an IterableDataset without __len__ is used. The callback assumes that each iteration has the same number of batches
    and that only one dataloader is used for each set.

    In the first epoch it uses the datamodule's estimated number of batches for each set, which is expected to be
    an upper bound of the true number. In the following epochs it uses the number of batches counted previously.
    """

    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        checkpoint["trainer/true_batch_count"] = self.train_true_batch_count
        checkpoint["trainer/true_batch_count_val"] = self.val_true_batch_count
        checkpoint["trainer/true_batch_count_test"] = self.test_true_batch_count

    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                           checkpoint: Dict[str, Any]) -> None:
        if "trainer/true_batch_count" in checkpoint:
            self.train_true_batch_count = checkpoint["trainer/true_batch_count"]
        if "trainer/true_batch_count_val" in checkpoint:
            self.val_true_batch_count = checkpoint["trainer/true_batch_count_val"]
        if "trainer/true_batch_count_test" in checkpoint:
            self.test_true_batch_count = checkpoint["trainer/true_batch_count_test"]

    def __init__(self, datamodule: MediansDataModule, set_trainer_max_batches: bool):
        self.datamodule = datamodule
        self.set_trainer_max_batches = set_trainer_max_batches

        self.train_batch_counter = None
        self.train_true_batch_count = None
        self.val_batch_counter = None
        self.val_true_batch_count = None
        self.test_batch_counter = None
        self.test_true_batch_count = None

    def _train_set_guessed_num_batches(self, trainer: pl.Trainer):
        guessed_num_batches = self.train_true_batch_count or self.datamodule.get_approx_num_batches("train")
        trainer.fit_loop.max_batches = guessed_num_batches

    def _val_set_guessed_num_batches(self, trainer: pl.Trainer):
        guessed_num_batches = self.val_true_batch_count or self.datamodule.get_approx_num_batches("val")
        trainer.fit_loop.epoch_loop.val_loop._max_batches = [guessed_num_batches]

    def _test_set_guessed_num_batches(self, trainer: pl.Trainer):
        guessed_num_batches = self.test_true_batch_count or self.datamodule.get_approx_num_batches("test")
        trainer.test_loop._max_batches = [guessed_num_batches]

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_batch_counter = 0
        if self.set_trainer_max_batches:
            self._train_set_guessed_num_batches(trainer)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.sanity_checking:
            return
        self.val_batch_counter = 0
        if self.set_trainer_max_batches:
            self._val_set_guessed_num_batches(trainer)

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.test_batch_counter = 0
        if self.set_trainer_max_batches:
            self._test_set_guessed_num_batches(trainer)

    def on_train_batch_start(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        self.train_batch_counter += 1

    def on_validation_batch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if trainer.sanity_checking:
            return
        self.val_batch_counter += 1

    def on_test_batch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        self.test_batch_counter += 1

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_true_batch_count = self.train_batch_counter
        wandb.run.summary["trainer/true_batch_count"] = self.train_true_batch_count

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.sanity_checking:
            return
        self.val_true_batch_count = self.val_batch_counter
        wandb.run.summary["trainer/true_batch_count_val"] = self.val_true_batch_count

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.test_true_batch_count = self.test_batch_counter
        wandb.run.summary["trainer/true_batch_count_test"] = self.test_true_batch_count
