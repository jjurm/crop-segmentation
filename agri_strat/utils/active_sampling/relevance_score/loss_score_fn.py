from abc import abstractmethod

import torch
import wandb
from torch import nn

from agri_strat.models.model_class import get_model_class
from agri_strat.utils.active_sampling.relevance_score.score_fn import ScoreFn


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
        model_output = model(inputs)
        return self.loss_fn(model_output, labels)


class IHOLossScoreFn(LossScoreFn):
    """
    Calculates the Irreducible Holdout Loss part of the RHO-Loss score function from https://arxiv.org/abs/2206.07137.
    """

    def __init__(
            self,
            loss_fn: nn.Module,
            irreducible_loss_model_artifact: str | None,
            ignore_index: int = None,
    ):
        super().__init__(loss_fn, ignore_index)

        assert irreducible_loss_model_artifact is not None
        model_artifact = wandb.run.use_artifact(irreducible_loss_model_artifact, type="model")
        model_ckpt = torch.load(model_artifact.file())
        state_dict = {
            k.removeprefix("model."): v
            for k, v in model_ckpt["state_dict"].items()
            if k.startswith("model.")
        }
        il_model_class = get_model_class(model_ckpt["hyper_parameters"]["model"])
        il_model = il_model_class(**model_ckpt["hyper_parameters"]["model_kwargs"])
        il_model.load_state_dict(state_dict)

        for param in il_model.parameters():
            param.requires_grad = False

        il_model.eval()

        print(f"loaded IL model from artifact: {irreducible_loss_model_artifact}")
        self.il_model = il_model

    def _score(self, inputs, labels, model) -> torch.Tensor:
        il_output = self.il_model(inputs)
        return self.loss_fn(il_output, labels)
