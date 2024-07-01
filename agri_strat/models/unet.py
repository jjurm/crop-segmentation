"""\
Implementation of the model proposed in:
- Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." 2015.

Code adopted from:
https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/vision/unet.py

Adopted from:
https://github.com/Orion-AI-Lab/S4A-Models/blob/master/model/PAD_unet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from agri_strat.models.utils import initialize_last_layer_bias


class DoubleConv(nn.Module):
    """[ Conv2d => BatchNorm (optional) => ReLU ] x 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # Apply kaiming_uniform_ initialization
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Downscale with MaxPool => DoubleConvolution block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upsampling (by either bilinear interpolation or transpose convolutions) followed by concatenation of feature
    map from contracting path, followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, num_classes, num_bands, num_time_steps, num_layers=3, relative_class_frequencies=None):
        """
        Parameters:
        -----------
        num_layers: int, default 3
            The number of layers to use in each path.
        """
        super(UNet, self).__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")
        self.num_layers = num_layers

        input_channels = num_bands * num_time_steps  # bands * time steps

        # Encoder
        # -------
        layers = [DoubleConv(input_channels, 64)]

        feats = 64
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        # Decoder
        # --------
        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, False))
            feats //= 2

        last_layer = nn.Conv2d(feats, num_classes, kernel_size=1)
        initialize_last_layer_bias(last_layer, relative_class_frequencies)

        layers.append(last_layer)
        layers.append(nn.LogSoftmax(dim=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # inputs have the shape (B, T, C, H, W)

        # Concatenate time series along channels dimension
        b, t, c, h, w = x.size()
        x = x.view(b, -1, h, w)  # (B, T * C, H, W)

        xi = [self.layers[0](x)]

        # Down path
        for layer in self.layers[1:self.num_layers]:
            xi.append(layer(xi[-1]))

        # Up path
        for i, layer in enumerate(self.layers[self.num_layers:-2]):
            xi[-1] = layer(xi[-1], xi[-2 - i])

        xi[-1] = self.layers[-2](xi[-1])

        # Softmax
        return self.layers[-1](xi[-1])
