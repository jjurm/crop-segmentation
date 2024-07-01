import torch
import wandb
from agri_strat.utils.class_weights import _calculate_class_weights, _calculate_class_counts_frequencies
from agri_strat.utils.label_encoder import LabelEncoder


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
        self._class_weights = torch.tensor(_calculate_class_weights(self.relative_class_frequencies))

        # interpolate between class weights and uniform weights
        if weighted_loss:
            class_weights_weighted = class_weights_weight * self._class_weights + \
                                     (1 - class_weights_weight) * torch.ones(label_encoder.num_classes)
            class_weights_weighted = class_weights_weighted.float().cuda()
        else:
            class_weights_weighted = None
        self.class_weights_weighted = class_weights_weighted

    def get_wandb_table(self):
        table = wandb.Table(
            columns=["IDs", "Class Name", "Pixel count", "Weight"],
            data=[
                [",".join(map(str, dataset_labels)), name, self._mapped_counts[i], self._class_weights[i]]
                for i, dataset_labels, name in self._label_encoder.entries_with_name
                if (i != 0 or not self._parcel_loss)  # if parcel_loss, skip the zero class
            ],
        )
        return table
