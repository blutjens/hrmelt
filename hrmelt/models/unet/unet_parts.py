""" Parts of the U-Net model 
Src: https://github.com/milesial/Pytorch-UNet/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, List, Any
from hrmelt.utils.utils import lookup_torch_activation

class ResConv(nn.Module):
    """(convolution => [BN] => activation)+residual"""

    def __init__(self, in_channels, out_channels, 
        mid_channels=None, activation='relu'):
        super().__init__()

        act_fn = lookup_torch_activation(activation)

        if not mid_channels:
            mid_channels = out_channels

        self.res_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            act_fn
        )

    def forward(self, x):
        return self.res_conv(x) + x

class DoubleConv(nn.Module):
    """(convolution => [BN] => activation) * 2"""

    def __init__(self, in_channels, out_channels, 
        mid_channels=None, activation='relu'):
        super().__init__()

        # Define activation function
        act_fn = lookup_torch_activation(activation)

        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            act_fn,
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            act_fn
        )

    def forward(self, x):
        return self.double_conv(x)

class MaskAttentionLayer(nn.Module):
    """
    Use attention to mask out areas of the input. 
    # todo: verify if this works. 
    Source: ChatGPT
    """
    def __init__(self, input_size):
        super(MaskAttentionLayer, self).__init__()
        # Attention weights for each pixel in the image.
        self.w = nn.Parameter(torch.randn(input_size[0], input_size[1], 1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x, mask):
        """
        Inputs:
            x torch.Tensor(batch, channel, height, width)
        Returns:
            attended_x torch.Tensor(batch, channel)
        """
        scores = torch.matmul(x, self.w) + self.b
        mask = mask.float()
        masked_scores = scores * mask - (1 - mask) * 1e9
        weights = F.softmax(masked_scores, dim=(1,2))
        attended_x = torch.sum(x * weights.unsqueeze(-1), dim=(1,2))
        return attended_x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, 
        activation='relu'):
        
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, 
                activation=activation)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, 
        bilinear=True, activation='relu'):

        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, activation=activation)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, activation=activation)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation=None):
        super(OutConv, self).__init__()
        
        act_fn = lookup_torch_activation(activation)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            act_fn
        )

    def forward(self, x):
        return self.conv(x)