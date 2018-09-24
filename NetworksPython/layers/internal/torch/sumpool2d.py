import torch.nn as nn
import numpy as np
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
    ):
        """
        Torch implementation of SumPooling using the LPPool2d module
        """
        super(TorchSumPooling2dLayer, self).__init__()  # Init nn.Module
        self.pad = nn.ZeroPad2d(padding)
        self.pool = nn.LPPool2d(1, kernel_size=kernel_size, stride=strides)

    def forward(self, tsrBinaryInput):
        tsrPoolOut = self.pool(self.pad(tsrBinaryInput))
        return tsrPoolOut
