import lightning.pytorch as pl


class PrintStageCallback(pl.Callback):
    """
    Prints the current stage of the training loop.
    """

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("Training...")

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("Validating...")

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("Testing...")
