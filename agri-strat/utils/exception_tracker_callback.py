import lightning as pl
import wandb


class ExceptionTrackerCallback(pl.Callback):
    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        if isinstance(exception, KeyboardInterrupt):
            print("Caught KeyboardInterrupt, stopping training")
            wandb.run.tags = wandb.run.tags + ("interrupted",)
            trainer.should_stop = True
        else:
            wandb.run.tags = wandb.run.tags + ("exception",)
