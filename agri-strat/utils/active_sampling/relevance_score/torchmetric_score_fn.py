from functools import partial
from typing import Callable

from score_fn import OutputLabelsScoreFn


class TorchMetricScoreFn(OutputLabelsScoreFn):
    def __init__(
            self,
            metric: Callable,
            **kwargs,
    ):
        self.metric = partial(metric, **kwargs)

    def _score(self, output, labels):
        return self.metric(output, labels)
