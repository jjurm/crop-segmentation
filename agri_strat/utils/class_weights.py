import numpy as np
from agri_strat.utils.label_encoder import LabelEncoder


def _calculate_class_weights(relative_class_frequencies):
    """
    Calculate class weights for the loss function.
    """
    n_classes = len(relative_class_frequencies)
    class_weights = [
        1 / (n_classes * freq) if freq != 0 else 0
        for freq in relative_class_frequencies
    ]
    return class_weights


def _calculate_class_counts_frequencies(class_counts: dict[int, int], label_encoder: LabelEncoder, parcel_loss: bool):
    mapped_counts = np.array([
        0 if i == 0 and parcel_loss else
        sum(class_counts[dataset_label] for dataset_label in dataset_labels)
        for i, dataset_labels in label_encoder.entries
    ])
    total_count = np.sum(mapped_counts)
    relative_class_frequencies = mapped_counts / total_count
    return mapped_counts, relative_class_frequencies
