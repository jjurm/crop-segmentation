from functools import partial

import numpy as np
from torch.utils.data.datapipes.iter.utils import IterableWrapperIterDataPipe

from utils.active_sampling.relevance_score.score_fn import ScoreFn


class ScorePerGroup:
    def __init__(self, monitor_fields: list[str]):
        self.monitor_fields = monitor_fields
        self.scores = {}

    def get_key(self, batch, i):
        key = tuple(batch[field][i] for field in self.monitor_fields)
        return key

    def update(self, batch, i, score, weight=1.):
        key = self.get_key(batch, i)
        if key not in self.scores:
            self.scores[key] = {"loss": 0., "weight": 0.}
        self.scores[key]["loss"] += score
        self.scores[key]["weight"] += weight


# TODO implement entropy and margin metrics, + any val metric, loss, weighted by pixels or samples,
#  uncertainty estimation with MC-Dropout,
#  Reproducible Holdout loss (rhloss)
#  https://blog.dataiku.com/active-sampling-data-selection-for-efficient-model-training
class ActiveSampler:
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
        block_size = block.shape[0]
        run_active_sampling = (block_size > want_samples) and (self.relevancy_score_fn is not None)

        if run_active_sampling:
            score_fn = partial(self.relevancy_score_fn.score, model=model)
            scores = np.array(IterableWrapperIterDataPipe(block, deepcopy=False) \
                              .batch(self.batch_size) \
                              .collate() \
                              .map(score_fn) \
                              .unbatch())
            assert len(scores) == block_size

            # Get want_samples samples with the highest relevancy score
            # ignore NaNs, which are considered large in python
            count_nans = np.isnan(scores).sum()
            n_ignore_nans = min(count_nans, block_size - want_samples)
            unsorted_indices = np.argpartition(scores, -want_samples - n_ignore_nans) \
                [-want_samples - n_ignore_nans:-n_ignore_nans]
            indices = np.sort(unsorted_indices)  # return samples in their original order
            block = [block[i] for i in indices]

        return block
