from abc import abstractmethod

import torch
from torch import nn


class ScoreFn(nn.Module):
    @abstractmethod
    def score(self, batch, model: nn.Module) -> torch.Tensor:
        """
        Scores the batch on the model.
        :param batch: The batch to score (collated).
        :param model: The model to score on.
        :return: An array of relevance scores. Higher means the sample is more relevant.
        """
        pass
