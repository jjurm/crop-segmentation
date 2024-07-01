from typing import Type

from torch import nn

from agri_strat.models.convstar import ConvSTAR
from agri_strat.models.unet import UNet


def get_model_class(model) -> Type[nn.Module]:
    if model == "unet":
        model_class = UNet
    elif model == "convstar":
        model_class = ConvSTAR
    else:
        raise ValueError(f"model = {model}, expected: 'unet' or 'convstar'")
    return model_class
