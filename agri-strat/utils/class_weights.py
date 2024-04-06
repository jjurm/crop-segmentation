import numpy as np
import torch

from utils.label_encoder import LabelEncoder


def _calculate_class_weights(relative_class_frequencies) -> torch.Tensor:
    """
    Calculate class weights for the loss function.
    """
    n_classes = len(relative_class_frequencies)
    class_weights = [
        1 / (n_classes * freq) if freq != 0 else 0
        for freq in relative_class_frequencies
    ]
    return torch.tensor(class_weights)


def _calculate_class_counts_frequencies(class_counts: dict[int, int], label_encoder: LabelEncoder, parcel_loss: bool):
    mapped_counts = np.array([
        0 if i == 0 and parcel_loss else
        sum(class_counts[dataset_label] for dataset_label in dataset_labels)
        for i, dataset_labels in label_encoder.entries
    ])
    total_count = np.sum(mapped_counts)
    relative_class_frequencies = mapped_counts / total_count
    return mapped_counts, relative_class_frequencies


class ClassWeights:
    def __init__(
            self,
            class_counts: dict[int, int],
            label_encoder: LabelEncoder,
            parcel_loss: bool,
            weighted_loss: bool,
            class_weights_weight: float,
    ):
        self._class_counts = class_counts
        self._label_encoder = label_encoder
        self._parcel_loss = parcel_loss

        self._mapped_counts, self.relative_class_frequencies = _calculate_class_counts_frequencies(
            class_counts, label_encoder, parcel_loss)
        self._class_weights = _calculate_class_weights(self.relative_class_frequencies)

        # interpolate between class weights and uniform weights
        if weighted_loss:
            class_weights_weighted = class_weights_weight * self._class_weights + \
                                     (1 - class_weights_weight) * torch.ones(label_encoder.num_classes)
            class_weights_weighted = class_weights_weighted.float().cuda()
        else:
            class_weights_weighted = None
        self.class_weights_weighted = class_weights_weighted

    def get_wandb_table(self, wandb=None):
        table = wandb.Table(
            columns=["IDs", "Class Name", "Pixel count", "Weight"],
            data=[
                [",".join(map(str, dataset_labels)), name, self._mapped_counts[i], self._class_weights[i]]
                for i, dataset_labels, name in self._label_encoder.entries_with_name
                if (i != 0 or not self._parcel_loss)  # if parcel_loss, skip the zero class
            ],
        )
        return table
