from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import functional_datapipe, IterDataPipe, DataChunk
from torch.utils.data.datapipes.iter.grouping import UnBatcherIterDataPipe
from torchdata.datapipes.iter import IterableWrapper

from utils.active_sampling.relevance_score.loss_score_fn import RHOLossScoreFn, LossScoreFn
from utils.active_sampling.relevance_score.score_fn import ScoreFn, WeightedScoreFn
from utils.active_sampling.relevance_score.uncertainty_margin import UncertaintyMarginScoreFn
from utils.class_weights import ClassWeights


@functional_datapipe('unbatch_tensor')
class TensorUnBatcherIterDataPipe(UnBatcherIterDataPipe):
    r"""
    Allows unbatching also tensors and numpy arrays.
    """

    def __init__(self, datapipe: IterDataPipe):
        super().__init__(datapipe, unbatch_level=1)

    def _dive(self, element, unbatch_level):
        if unbatch_level < -1:
            raise ValueError("unbatch_level must be -1 or >= 0")
        if unbatch_level == -1:
            if isinstance(element, (list, DataChunk)):
                for item in element:
                    yield from self._dive(item, unbatch_level=-1)
            else:
                yield element
        elif unbatch_level == 0:
            yield element
        else:
            if isinstance(element, (list, DataChunk, torch.Tensor, np.ndarray)):
                for item in element:
                    yield from self._dive(item, unbatch_level=unbatch_level - 1)
            else:
                raise IndexError(f"unbatch_level {self.unbatch_level} exceeds the depth of the DataPipe")


def _get_single_active_sampling_relevancy_score_fn(
        active_sampling_relevancy_score: str,
        class_weights: ClassWeights,
        parcel_loss: bool,
) -> Tuple[float, ScoreFn]:
    if "*" in active_sampling_relevancy_score:
        weight, active_sampling_relevancy_score = active_sampling_relevancy_score.split("*")
        weight = float(weight)
    else:
        weight = 1.0

    if active_sampling_relevancy_score == "loss":
        fn = LossScoreFn(
            loss_fn=nn.NLLLoss(weight=class_weights.class_weights_weighted, reduction="none"),
            ignore_index=0 if parcel_loss else None,
        )
    elif active_sampling_relevancy_score.startswith("rho-loss-"):
        fn = RHOLossScoreFn(
            loss_fn=nn.NLLLoss(weight=class_weights.class_weights_weighted, reduction="none"),
            ignore_index=0 if parcel_loss else None,
            irreducible_loss_model_artifact=active_sampling_relevancy_score.removeprefix("rho-loss-"),
        )
    elif active_sampling_relevancy_score == "uncertainty-margin":
        fn = UncertaintyMarginScoreFn(
            ignore_index=0 if parcel_loss else None,
        )
    else:
        raise ValueError(f"Unsupported active_sampling_relevancy_score: {active_sampling_relevancy_score}")

    return weight, fn


def get_active_sampling_relevancy_score_fn(
        active_sampling_relevancy_score: list[str],
        class_weights: ClassWeights,
        parcel_loss: bool,
) -> Optional[ScoreFn]:
    # No active sampling
    if active_sampling_relevancy_score is None:
        return None
    filtered_fns = [fn for fn in active_sampling_relevancy_score if fn != "none"]
    if len(filtered_fns) == 0:
        return None

    # Single active sampling function
    # if len(filtered_fns) == 1:
    #     _, fn = _get_single_active_sampling_relevancy_score_fn(filtered_fns[0], class_weights, parcel_loss)
    #     print(f"Using active sampling relevancy score function: {fn}")
    #     return fn

    # Multiple active sampling functions
    weights_fns = [
        _get_single_active_sampling_relevancy_score_fn(fn, class_weights, parcel_loss)
        for fn in filtered_fns
    ]
    weights = torch.tensor([weight for weight, _ in weights_fns])
    fns = [fn for _, fn in weights_fns]
    print(f"Using active sampling relevancy score functions: {weights_fns}")
    return WeightedScoreFn(fns, weights=weights)


def _index_contained_in(indices, x):
    return x[0] in indices


def _get_1(x):
    return x[1]


# TODO implement entropy and margin metrics, + any val metric, loss, weighted by pixels or samples,
#  uncertainty estimation with MC-Dropout,
#  Reproducible Holdout loss (rhloss)
#  https://blog.dataiku.com/active-sampling-data-selection-for-efficient-model-training
class ActiveSampler(nn.Module):
    def __init__(
            self,
            batch_size: int,
            n_batches_per_block: int,
            accumulate_grad_batches: int,
            relevancy_score_fn: ScoreFn | None,
    ):
        """
        :param batch_size: The physical batch size
        :param n_batches_per_block: (At most) how many batches to sample in a block.
        :param relevancy_score_fn: A function that computes the relevancy score for each sample in a batch
        """
        super().__init__()
        self.batch_size = batch_size
        self.n_batches_per_block = n_batches_per_block
        self.accumulate_grad_batches = accumulate_grad_batches
        self.relevancy_score_fn = relevancy_score_fn

    def __call__(self, block, block_idx, model):
        """
        Receives an uncollated block and returns a block with at most (n_batches_per_block * accumulate_grad_batches
        * batch_size) samples with the highest relevancy score.

        If relevancy_score_fn is None, the block is returned as is.
        """
        want_samples = self.n_batches_per_block * self.accumulate_grad_batches * self.batch_size
        block_size = len(block)
        run_active_sampling = block_size > want_samples

        if run_active_sampling:
            if self.relevancy_score_fn is None:
                raise ValueError("relevancy_score_fn must be provided if block_size is larger than 1.")

            with torch.no_grad():
                score_fn = partial(self.relevancy_score_fn.score, model=model)
                scores = np.fromiter(
                    IterableWrapper(block, deepcopy=False)
                    .batch(batch_size=self.batch_size)
                    .collate()
                    .map(score_fn)
                    .unbatch_tensor(),
                    dtype=float,
                )
                assert len(scores) == block_size

                # Get want_samples samples with the highest relevancy score
                # ignore NaNs, which are considered large in python
                count_nans = np.isnan(scores).sum()
                n_ignore = min(count_nans, block_size - want_samples)
                unsorted_indices = np.argpartition(scores, -want_samples - n_ignore)[-want_samples - n_ignore:]
                if n_ignore > 0:
                    unsorted_indices = unsorted_indices[:-n_ignore]
                indices = set(unsorted_indices)

            # The following returns samples in their original order
            block = IterableWrapper(block, deepcopy=False) \
                .enumerate() \
                .filter(partial(_index_contained_in, indices)) \
                .map(_get_1)

        return block
