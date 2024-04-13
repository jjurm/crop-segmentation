from functools import partial

import numpy as np
import torch
from torch.utils.data import functional_datapipe, IterDataPipe, DataChunk
from torch.utils.data.datapipes.iter.grouping import UnBatcherIterDataPipe
from torchdata.datapipes.iter import IterableWrapper

from utils.active_sampling.relevance_score.score_fn import ScoreFn


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
        block_size = len(block)
        run_active_sampling = (block_size > want_samples) and (self.relevancy_score_fn is not None)

        if run_active_sampling:
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
                .filter(lambda x: x[0] in indices) \
                .map(lambda x: x[1])

        return block
