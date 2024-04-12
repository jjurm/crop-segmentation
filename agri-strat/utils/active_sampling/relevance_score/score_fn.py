from abc import ABC, abstractmethod

from torch import nn


class ScoreFn(ABC):
    @abstractmethod
    def score(self, batch, model: nn.Module):
        """
        Scores the batch on the model.
        :return: A relevance score. Higher means more relevant.
        """
        pass


class OutputLabelsScoreFn(ScoreFn):
    @abstractmethod
    def _score(self, output, labels):
        pass

    def score(self, batch, model: nn.Module):
        inputs, labels = batch['medians'], batch['labels']
        batch_size = inputs.shape[0]
        output = model(inputs)
        return self._score(output, labels)
