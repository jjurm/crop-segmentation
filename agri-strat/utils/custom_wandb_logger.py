"""
Like WandbLogger, but also cleans up model artifacts without an alias.
"""

import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


class CustomWandbLogger(WandbLogger):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self._checkpoint_name is not None, "The checkpoint name must be set."

        self._custom_wandb_api = wandb.Api(overrides={"project": self.experiment.project})

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        super().after_save_checkpoint(checkpoint_callback)

        # Clean up previous artifacts.
        # Adapted from https://gitbook-docs.wandb.ai/guides/artifacts/api#cleaning-up-unused-versions
        for version in self._custom_wandb_api.artifacts("model", self._checkpoint_name):
            # Clean up all versions that don't have an alias such as 'latest'.
            if len(version.aliases) == 0:
                version.delete()
