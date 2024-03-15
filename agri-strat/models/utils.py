import numpy as np


def initialize_last_layer_bias(last_layer, relative_class_frequencies):
    """
    Initialize the bias of the last layer to reflect the class priors
    """
    if relative_class_frequencies is not None:
        num_classes = last_layer.out_channels
        assert len(relative_class_frequencies) == num_classes, ("Length of relative_class_frequencies must be "
                                                                "equal to num_classes")
        for i, freq in enumerate(relative_class_frequencies):
            last_layer.bias.data[i].fill_(-np.log((1 - freq) / freq))


def calculate_class_frequencies(class_counts, parcel_loss):
    """
    Calculate the relative class frequencies
    :param class_counts:
    :param parcel_loss: if True, the frequency of the zero class is set to zero
    :return:
    """
    total_count = sum(class_counts.values())
    relative_class_frequencies = {
        cls: 0 if cls == 0 and parcel_loss else (count / total_count)
        for cls, count in class_counts.items()}
    return relative_class_frequencies
