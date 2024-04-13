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
        mask = labels != self.ignore_index
        reduce_dims = tuple(range(1, mask.ndim))
        denom = torch.sum(mask, dim=reduce_dims)
        loss_per_sample = loss.sum(dim=reduce_dims) / denom

        return loss_per_sample
