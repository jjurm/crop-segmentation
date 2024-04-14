from abc import abstractmethod
from typing import Type

import torch
import wandb
from torch import nn

from utils.active_sampling.relevance_score.score_fn import ScoreFn


class AbstractLossScoreFn(ScoreFn):
    def __init__(self, ignore_index: int = None):
        super().__init__()
        self.ignore_index = ignore_index

    @abstractmethod
    def _score(self, inputs, labels, model) -> torch.Tensor:
        pass

    def score(self, batch, model: nn.Module) -> torch.Tensor:
        loss = self._score(batch['medians'], batch['labels'], model)

        # Average over all dimensions except the batch dimension, to get a single score per sample.
        # Calculates the mean loss per sample across non-ignored pixels.
        # TODO test whether mean/sum is better
        mask = batch['labels'] != self.ignore_index
        reduce_dims = tuple(range(1, mask.ndim))
        denom = torch.sum(mask, dim=reduce_dims)
        loss_per_sample = loss.sum(dim=reduce_dims) / denom

        return loss_per_sample


class LossScoreFn(AbstractLossScoreFn):
    def __init__(
            self,
            loss_fn: nn.Module,
            ignore_index: int = None
    ):
        super().__init__(ignore_index)
        self.loss_fn = loss_fn

    def _score(self, inputs, labels, model) -> torch.Tensor:
        output = model(inputs)
        return self.loss_fn(output, labels)


class RHOLossScoreFn(LossScoreFn):
    def __init__(
            self,
            loss_fn: nn.Module,
            irreducible_loss_model_artifact: str | None,
            irreducible_loss_model_class: Type[nn.Module],
            irreducible_loss_model_kwargs: dict,
            ignore_index: int = None,
    ):
        super().__init__(loss_fn, ignore_index)

        self.il_model = None
        if irreducible_loss_model_artifact is not None:
            model_artifact = wandb.run.use_artifact(irreducible_loss_model_artifact, type="model")
            model_ckpt = torch.load(model_artifact.file())
            state_dict = {
                k.removeprefix("model."): v
                for k, v in model_ckpt["state_dict"].items()
                if k.startswith("model.")
            }
            il_model = irreducible_loss_model_class(**irreducible_loss_model_kwargs)
            il_model.load_state_dict(state_dict)
            print("loaded IL model")
            self.il_model = il_model

    def _score(self, inputs, labels, model) -> torch.Tensor:
        model_output = model(inputs)
        loss = self.loss_fn(model_output, labels)

        if self.il_model is not None:
            il_output = self.il_model(inputs)
            il_loss = self.loss_fn(il_output, labels)
            loss = loss - il_loss

        return loss
