"""
Like WandbLogger, but also cleans up model artifacts without an alias.
"""
from pathlib import Path

import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.utilities import _scan_checkpoints
from torch import Tensor


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

    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        """
        This function is copied from the superclass and modified
        """

        # get checkpoints to be saved with associated score
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

        # log iteratively all new checkpoints
        for t, p, s, tag in checkpoints:
            metadata = {
                "score": s.item() if isinstance(s, Tensor) else s,
                "original_filename": Path(p).name,
                checkpoint_callback.__class__.__name__: {
                    k: getattr(checkpoint_callback, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                    ]
                    # ensure it does not break if `ModelCheckpoint` args change
                    if hasattr(checkpoint_callback, k)
                },
            }
            if not self._checkpoint_name:
                self._checkpoint_name = f"model-{self.experiment.id}"
            artifact = wandb.Artifact(name=self._checkpoint_name, type="model", metadata=metadata)
            artifact.add_file(p, name="model.ckpt")
            aliases = ["latest", "best"] if p == checkpoint_callback.best_model_path else ["latest"]
            self.experiment.log_artifact(artifact, aliases=aliases)

            # The added line to wait for the artifact upload (before artifacts are accessed)
            artifact.wait()

            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t
