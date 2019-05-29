##
# sumpool2d.py -- Torch implementation of SumPooling2D layer (CNN architectures)
##

import torch.nn as nn
import numpy as np
import pandas as pd
from operator import mul
from functools import reduce
from typing import Optional, Union, List, Tuple

ArrayLike = Union[np.ndarray, List, Tuple]


class TorchSumPooling2dLayer(nn.Module):
    """
    Torch implementation of SumPooling2d for spiking neurons
    """

    def __init__(
        self,
        kernel_size: ArrayLike = (1, 1),
        strides: Optional[ArrayLike] = None,
        padding: ArrayLike = (0, 0, 0, 0),
        name: str = "pooling2d",
    ):
        """
        Torch implementation of SumPooling using the LPPool2d module
        """
        nn.Module.__init__(self)  # Init nn.Module
        self.padding = padding
        self.kernel_size = kernel_size
        self.name = name
        if strides is None:
            strides = kernel_size
        self.strides = strides
        if padding == (0, 0, 0, 0):
            self.pad = None
        else:
            self.pad = nn.ZeroPad2d(padding)
        self.pool = nn.LPPool2d(1, kernel_size=kernel_size, stride=strides)

    def forward(self, tsrBinaryInput):
        _, self.nInChannels, h, w = list(tsrBinaryInput.shape)
        if self.pad is None:
            tsrPoolOut = self.pool(tsrBinaryInput)
        else:
            tsrPoolOut = self.pool(self.pad(tsrBinaryInput))
        self.outShape = tsrPoolOut.shape[1:]
        self.tsrNumSpikes = tsrPoolOut
        return tsrPoolOut

    @property
    def nOutChannels(self):
        return self.nInChannels

    def summary(self):
        """
        Returns the summary of this layer as a pandas Series
        """
        summary = pd.Series(
            {
                "Layer": self.name,
                "Output Shape": (tuple(self.outShape)),
                "Padding": tuple(self.padding),
                "Kernel": tuple(self.kernel_size),
                "Stride": tuple(self.strides),
                "FanOutPrev": reduce(
                    mul, np.array(self.kernel_size) / np.array(self.strides), 1
                ),
                "Neurons": 0,
                "KernelMem": 0,
                "BiasMem": 0,
            }
        )
        return summary
