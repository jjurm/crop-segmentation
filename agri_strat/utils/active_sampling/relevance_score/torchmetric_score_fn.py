from functools import partial
from typing import Callable

import torch
from torch import nn

from agri_strat.score_fn import ScoreFn


class TorchMetricScoreFn(ScoreFn):
    """
    A ScoreFn that uses a torch metric to score the model's output on the labels.
    The metric should be a function corresponding to any subclass of MulticlassStatScores.
    """

    def __init__(
            self,
            metric: Callable,
            **kwargs,
    ):
        self.metric = partial(metric, **kwargs | dict(multidim_average="samplewise"))

    def score(self, batch, model: nn.Module) -> torch.Tensor:
        output = model(batch['medians'])
        return self.metric(output, batch['labels'])
