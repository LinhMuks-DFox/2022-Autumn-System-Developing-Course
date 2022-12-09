from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

IntOrIntTuple = Union[int, Tuple[int, int]]


class ImgEncoder(nn.Module):
    def __init__(self, output_size, linear_input_size: Tuple[int, int]):
        super().__init__()
        self.convolution_layers = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7),  # 28 * 28 -> 22 * 22?
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=3),  # 22 * 22 -> 20 * 20
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        # [(W−K+2P)/S]+1
        self.linear_layers = nn.Sequential(
            nn.Linear(linear_input_size[0] * linear_input_size[1], output_size),
            nn.ReLU()
        )

    def forward(self, sample):
        ret = sample
        for conv in self.convolution_layers:
            ret = conv(ret)
        ret = self.flatten(ret)
        for lin in self.linear_layers:
            ret = lin(ret)
        return ret


class ImgDecoder(nn.Module):
    def __init__(self, intput_size, linear_input_size: Tuple[int, int]):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(intput_size, linear_input_size[0] * linear_input_size[1]),
            nn.ReLU()
        )
        self.unflatten = nn.Unflatten(1, (1, *linear_input_size))
        self.conv2dTrans = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=3),  # 28 * 28 -> 22 * 22?
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, kernel_size=7),  # 22 * 22 -> 20 * 20
            nn.ReLU()
        )

    def forward(self, sample):
        ret = sample
        for lin in self.linear_layers:
            ret = lin(ret)
        ret = self.unflatten(ret)
        for conv in self.conv2dTrans:
            ret = conv(ret)
        return ret


