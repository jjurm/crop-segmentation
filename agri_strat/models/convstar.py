"""
Implementation of the convSTAR model.
- Turkoglu, Mehmet Ozgur, et al. "Crop mapping from image time series: Deep learning with multi-scale label hierarchies." Remote Sensing of Environment 264 (2021): 112603.

Code taken from:
https://github.com/0zgur0/ms-convSTAR

Adapted from:
https://github.com/Orion-AI-Lab/S4A-Models/blob/master/model/PAD_convSTAR.py
"""

import torch
import torch.nn as nn
from torch.nn import init

from agri_strat.models.utils import initialize_last_layer_bias


class ConvSTARCell(nn.Module):
    """
    Generate a convolutional STAR cell
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal_(self.update.weight)
        init.orthogonal_(self.gate.weight)
        init.constant_(self.update.bias, 0.)
        init.constant_(self.gate.bias, 1.)

    def forward(self, input_, prev_state):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = torch.zeros(state_size).cuda()
            else:
                prev_state = torch.zeros(state_size)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        gain = torch.sigmoid(self.gate(stacked_inputs))
        update = torch.tanh(self.update(input_))
        new_state = gain * prev_state + (1 - gain) * update

        return new_state


class ConvSTAR(nn.Module):
    def __init__(self, num_classes, input_size=4, hidden_sizes=64, kernel_sizes=3, num_layers=3,
                 relative_class_frequencies=None, **kwargs):
        """
        Parameters:
        -----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        num_layers : integer. number of chained `ConvSTARCell`.
        """
        super(ConvSTAR, self).__init__()

        self.input_size = input_size

        if not isinstance(hidden_sizes, list):
            self.hidden_sizes = [hidden_sizes] * num_layers
        else:
            assert len(hidden_sizes) == num_layers, '`hidden_sizes` must have the same length as num_layers'
            self.hidden_sizes = hidden_sizes

        if not isinstance(kernel_sizes, list):
            self.kernel_sizes = [kernel_sizes] * num_layers
        else:
            assert len(kernel_sizes) == num_layers, '`kernel_sizes` must have the same length as num_layers'
            self.kernel_sizes = kernel_sizes

        self.num_layers = num_layers

        cells = []
        for i in range(self.num_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i - 1]

            cell = ConvSTARCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            name = 'ConvSTARCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

        self.conv2d = nn.Conv2d(
            in_channels=self.hidden_sizes[-1],
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0
        )
        initialize_last_layer_bias(self.conv2d, relative_class_frequencies)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden=None):
        """
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).

        Returns
        -------
        The prediction of the last hidden layer.
        """
        if not hidden:
            hidden = [None] * self.num_layers

        x = x.transpose(1, 2)  # (B, T, C, H, W) -> (B, C, T, H, W)

        batch_size, channels, timesteps, height, width = x.size()

        # retain tensors in list to allow different hidden sizes
        upd_hidden = []

        for timestep in range(timesteps):
            input_ = x[:, :, timestep, :, :]

            for layer_idx in range(self.num_layers):
                cell = self.cells[layer_idx]
                cell_hidden = hidden[layer_idx]

                if layer_idx == 0:
                    upd_cell_hidden = cell(input_, cell_hidden)
                else:
                    upd_cell_hidden = cell(upd_hidden[-1], cell_hidden)
                upd_hidden.append(upd_cell_hidden)
                # update input_ to the last updated hidden layer for next pass
                hidden[layer_idx] = upd_cell_hidden

        # Keep only the last output for an N-to-1 scheme
        x = hidden[-1]  # (L, B, C, H, W) -> (B, C, H, W)
        x = self.softmax(self.conv2d(x))  # (B, K, H, W)

        return x
