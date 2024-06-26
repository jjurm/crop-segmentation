"""
Like WandbLogger, but also cleans up model artifacts without an alias.
"""
from pathlib import Path
from typing import Mapping, Optional

import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.utilities import _scan_checkpoints
from lightning_fabric.utilities.logger import _add_prefix
from torch import Tensor
from wandb.sdk.wandb_summary import SummaryDict


class CustomWandbLogger(WandbLogger):
    """
    Modifies the WandbLogger in the following:
    - log_metrics() doesn't log metrics immediately, but aggregates them and waits for a call to log_now()
    - cleans up model artifacts without an alias.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self._checkpoint_name is not None, "The checkpoint name must be set."

        self._custom_wandb_api = wandb.Api(overrides={"project": self.experiment.project})
        self.aggregated_metrics = {}

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        if step is not None:
            metrics_dict = dict(metrics, **{"trainer/global_step": step})
        else:
            metrics_dict = metrics
        self.aggregated_metrics.update(metrics_dict)

    def log_now(self, dry_run: bool):
        if not dry_run:
            self.experiment.log(self.aggregated_metrics)
        self.aggregated_metrics = {}

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        super().after_save_checkpoint(checkpoint_callback)

        # Clean up previous artifacts.
        self._clean_up_unaliased_artifacts()

    def _clean_up_unaliased_artifacts(self):
        # Adapted from https://gitbook-docs.wandb.ai/guides/artifacts/api#cleaning-up-unused-versions
        for version in self._custom_wandb_api.artifacts("model", self._checkpoint_name):
            # Clean up all versions that don't have an alias such as 'latest'.
            if len({"latest", "best"}.intersection(version.aliases)) == 0:
                version.delete(delete_aliases=True)

    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        """
        This function is copied from the superclass and modified
        """

        # get checkpoints to be saved with associated score
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)
        assert len(checkpoints) <= 1, "Logging multiple checkpoints at once is unexpected"

        # log iteratively all new checkpoints
        for t, p, s, tag in checkpoints:
            metadata = {
                "score": s.item() if isinstance(s, Tensor) else s,
                "summary": {
                    k: dict(v) if isinstance(v, SummaryDict) else v
                    for k, v in dict(self.experiment.summary).items()
                    if (k.startswith("val/") and (not isinstance(v, SummaryDict) or "_type" not in dict(v)))
                       or k.startswith("trainer/")
                       or k in ["_step", "epoch", "val_epoch", "lr-Adam"]
                       or (k.startswith("train/") and k.endswith("_epoch"))
                },
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
            aliases = [
                "latest",
                "epoch={epoch:02d}".format(epoch=self.experiment.summary['epoch']),
            ]
            if p == checkpoint_callback.best_model_path:
                aliases.append("best")
            self.experiment.log_artifact(artifact, aliases=aliases)

            # The added line to wait for the artifact upload (before artifacts are accessed)
            artifact.wait()

            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t
