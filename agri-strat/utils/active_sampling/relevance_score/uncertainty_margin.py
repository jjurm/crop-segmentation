import torch
from torch import nn

from utils.active_sampling.relevance_score.score_fn import ScoreFn


class UncertaintyMarginScoreFn(ScoreFn):
    """
    Calculates the difference between the highest non-correct class probability and the correct class probability.
    """

    def score(self, batch, model: nn.Module) -> torch.Tensor:
        model_output = model(batch['medians'])  # shape (batch_size, num_classes, ...)
        labels = batch['labels']  # shape (batch_size, ...)

        # Get the correct class probabilities using torch.gather
        correct_class_probs = torch.gather(model_output, 1, labels.unsqueeze(1)).squeeze(1)

        # Set the correct class probabilities to -inf
        model_output.scatter_(1, labels.unsqueeze(1), float('-inf'))
        # Get the highest non-correct class probabilities
        highest_non_correct_class_probs = model_output.max(dim=1).values

        # Calculate the uncertainty margin
        return highest_non_correct_class_probs - correct_class_probs
