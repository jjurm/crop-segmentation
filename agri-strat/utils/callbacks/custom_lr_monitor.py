from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor


class CustomLearningRateMonitor(LearningRateMonitor):
    """
    Modifies the LearningRateMonitor in that the learning rate with the epoch frequency is logged at the end of each
    epoch, not at the beginning.
    """

    def on_train_epoch_start(self, trainer: "pl.Trainer", *args: Any, **kwargs: Any) -> None:
        # do nothing
        pass

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_start(trainer, pl_module)
