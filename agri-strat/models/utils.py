import numpy as np


def initialize_last_layer_bias(last_layer, relative_class_frequencies):
    """
    Initialize the bias of the last layer to reflect the class priors
    """
    if relative_class_frequencies is not None:
        num_classes = last_layer.out_channels
        assert len(relative_class_frequencies) == num_classes, (
            f"Length of relative_class_frequencies={len(relative_class_frequencies)} must be equal to "
            f"num_classes={num_classes}")
        min_freq = min(f for f in relative_class_frequencies if f > 0)
        for i, freq in enumerate(relative_class_frequencies):
            freq_to_use = max(freq, min_freq)
            last_layer.bias.data[i].fill_(-np.log((1 - freq_to_use) / freq_to_use))
