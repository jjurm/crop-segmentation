import torch
from torch import nn

from utils.active_sampling.relevance_score.score_fn import ScoreFn


class RandomScoreFn(ScoreFn):
    def __init__(self, seed) -> None:
        super().__init__()
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def score(self, batch, model: nn.Module) -> torch.Tensor:
        batch_size = batch['medians'].shape[0]
        return torch.rand(batch_size, generator=self.generator).to(device=batch['medians'].device)
