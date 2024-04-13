from abc import ABC, abstractmethod

import torch
from torch import nn


class ScoreFn(ABC):
    @abstractmethod
    def score(self, batch, model: nn.Module) -> torch.Tensor:
        """
        Scores the batch on the model.
        :param batch: The batch to score (collated).
        :param model: The model to score on.
        :return: An array of relevance scores. Higher means the sample is more relevant.
        """
        pass


class OutputLabelsScoreFn(ScoreFn):
    @abstractmethod
    def _score(self, output, labels) -> torch.Tensor:
        """
        Scores model's output on the labels.
        :return: A tensor of relevance scores, one for each sample in the batch. Higher means the sample is more
        relevant.
        """
        pass

    def score(self, batch, model: nn.Module) -> torch.Tensor:
        output = model(batch['medians'])
        return self._score(output, batch['labels'])
