from functools import partial
from typing import Callable

import torch

from utils.active_sampling.relevance_score.score_fn import OutputLabelsScoreFn


class LossScoreFn(OutputLabelsScoreFn):
    def __init__(self, loss_fn: Callable, ignore_index: int = None, **kwargs):
        self.ignore_index = ignore_index
        self.loss_fn = partial(loss_fn, ignore_index=ignore_index, **kwargs | dict(reduction="none"))

    def _score(self, output, labels) -> torch.Tensor:
        loss = self.loss_fn(output, labels)

        # Average over all dimensions except the batch dimension, to get a single score per sample.
        # Calculates the mean loss per sample across non-ignored pixels.
        # TODO test whether mean/sum is better
        loss_parcel = loss[labels != self.ignore_index]
        loss_per_sample = loss_parcel.mean(dim=tuple(range(1, loss_parcel.ndim)))

        return loss_per_sample
