import numpy as np
from torch import nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 depthwise=False, relu6=False):
        if depthwise and (in_channels != out_channels):
            raise ValueError("The number of input and output channels in the depthwise "
                             "convolutional layer should be equal.")
        layers = [
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=self._auto_pad(kernel_size),
                      groups=in_channels if depthwise else 1,
                      bias=False),
            nn.BatchNorm2d(out_channels)
        ]

        if relu6:
            layers.append(nn.ReLU6(inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))

        super().__init__(*layers)

    @staticmethod
    def _auto_pad(k):
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        if np.sum(p) == 0:
            return 0
        return p
