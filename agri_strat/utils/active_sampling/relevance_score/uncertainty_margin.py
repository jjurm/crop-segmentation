import torch

from agri_strat.utils.active_sampling.relevance_score.loss_score_fn import AbstractLossScoreFn


class UncertaintyMarginScoreFn(AbstractLossScoreFn):
    """
    Calculates the difference between the highest non-correct class probability and the correct class probability.
    """

    def _score(self, inputs, labels, model) -> torch.Tensor:
        model_output = model(inputs)  # shape (batch_size, num_classes, ...)
        # labels have shape (batch_size, ...)

        # Get the correct class probabilities using torch.gather
        correct_class_probs = torch.gather(model_output, 1, labels.unsqueeze(1)).squeeze(1)

        # Set the correct class probabilities to -inf
        model_output.scatter_(1, labels.unsqueeze(1), float('-inf'))
        # Get the highest non-correct class probabilities
        highest_non_correct_class_probs = model_output.max(dim=1).values

        # Calculate the uncertainty margin per pixel
        return highest_non_correct_class_probs - correct_class_probs
