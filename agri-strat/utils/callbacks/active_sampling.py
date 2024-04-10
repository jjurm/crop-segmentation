from functools import partial

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import tqdm

from utils.class_weights import ClassWeights
from utils.medians_datamodule import MediansDataModule


class ScorePerGroup:
    def __init__(self, monitor_fields: list[str]):
        self.monitor_fields = monitor_fields
        self.scores = {}

    def get_key(self, batch, i):
        key = tuple(batch[field][i] for field in self.monitor_fields)
        return key

    def update(self, batch, i, score, weight=1.):
        key = self.get_key(batch, i)
        if key not in self.scores:
            self.scores[key] = {"loss": 0., "weight": 0.}
        self.scores[key]["loss"] += score
        self.scores[key]["weight"] += weight


# TODO implement entropy and margin metrics, + any val metric, loss, weighted by pixels or samples,
#  uncertainty estimation with MC-Dropout,
#  Reproducible Holdout loss (rhloss)
#  https://blog.dataiku.com/active-sampling-data-selection-for-efficient-model-training
class ActiveSamplingCallback(pl.Callback):
    def __init__(
            self,
            datamodule: MediansDataModule,
            class_weights: ClassWeights,
            metric: str,
            monitor_fields: list[str],
            sample_fraction: float = 1.0,
    ):
        self.datamodule = datamodule
        self.metric = metric
        self.monitor_fields = monitor_fields
        self.sample_fraction = sample_fraction

        self.loss_nll_parcel = nn.NLLLoss(weight=class_weights.class_weights_weighted, ignore_index=0)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        score_per_group = ScorePerGroup(self.monitor_fields)
        inference_step = partial(self.inference_step, trainer=trainer, score_per_group=score_per_group)
        # Evaluate the model on the training data

        with torch.inference_mode():
            for batch, batch_idx, dataloader_idx in tqdm.tqdm(trainer.train_dataloader, desc="AL inference"):
                inference_step(batch, batch_idx)

        self.datamodule.dataset_train.add_priority_to_patches(partial(self.priority_fn, scores=score_per_group.scores))

    def priority_fn(self, scores, row):
        key = tuple(row[field] for field in self.monitor_fields)
        if key in scores:
            return scores[key]["loss"] / scores[key]["weight"]
        else:
            return np.inf

    def inference_step(self, batch, batch_idx, trainer: "pl.Trainer", score_per_group: ScorePerGroup):
        inputs, labels = batch['medians'], batch['labels']  # (B, T, C, H, W), (B, H, W)
        output = trainer.model(inputs)

        for i, patch_path in enumerate(batch["patch_path"]):
            n_pixels = int((batch["labels"][i] != 0).sum().item())
            outputs_ = output[i].unsqueeze(0)
            labels_ = batch["labels"][i].unsqueeze(0)
            loss_nll_parcel = self.loss_nll_parcel(outputs_, labels_)
            score_per_group.update(batch, i, loss_nll_parcel, weight=n_pixels)
