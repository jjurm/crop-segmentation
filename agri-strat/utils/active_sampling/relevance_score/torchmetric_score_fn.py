from functools import partial
from typing import Callable

import torch

from score_fn import OutputLabelsScoreFn


class TorchMetricScoreFn(OutputLabelsScoreFn):
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

    def _score(self, output, labels) -> torch.Tensor:
        return self.metric(output, labels)
