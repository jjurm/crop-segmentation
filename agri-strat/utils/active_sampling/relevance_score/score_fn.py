from abc import abstractmethod, ABC

import torch
from torch import nn


class ScoreFn(nn.Module, ABC):
    @abstractmethod
    def score(self, batch, model: nn.Module) -> torch.Tensor:
        """
        Scores the batch on the model.
        :param batch: The batch to score (collated).
        :param model: The model to score on.
        :return: An array of relevance scores. Higher means the sample is more relevant.
        """
        pass


class WeightedScoreFn(ScoreFn, ABC):
    def __init__(self, score_fns: list[ScoreFn], weights: torch.Tensor) -> None:
        super().__init__()
        self.score_fns = nn.ModuleList(score_fns)
        self.register_buffer('weights', weights.unsqueeze(1))

    def score(self, batch, model: nn.Module) -> torch.Tensor:
        scores = torch.stack([fn.score(batch, model) for fn in self.score_fns], dim=0)
        return torch.sum(scores * self.get_buffer("weights"), dim=0)
