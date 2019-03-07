import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple

ArrayLike = Union[np.ndarray, List, Tuple]


class TorchCropping2dLayer(nn.Module):
    """
    Torch implementation of SumPooling2d for spiking neurons
    """

    def __init__(self, cropping: ArrayLike = ((0, 0), (0, 0)), strName="crop2d"):
        """
        Torch implementation of SumPooling using the LPPool2d module
        """
        super(TorchCropping2dLayer, self).__init__()  # Init nn.Module
        self.padding = ()
        self.kernel_size = ()
        self.strides = ()
        self.top_crop, self.bottom_crop = cropping[0]
        self.left_crop, self.right_crop = cropping[1]
        self.strName = strName

    def forward(self, tsrBinaryInput):
        _, self.nInChannels, h, w = list(tsrBinaryInput.shape)
        tsrCropOut = tsrBinaryInput[
            :,
            :,
            self.top_crop : h - self.bottom_crop,
            self.left_crop : w - self.right_crop,
        ]
        self.outShape = tsrCropOut.shape[1:]
        self.tsrNumSpikes = tsrCropOut
        return tsrCropOut

    @property
    def nOutChannels(self):
        return self.nInChannels

    def summary(self):
        """
        Returns a summary of this layer as a pandas Series
        """
        summary = pd.Series(
            {
                "Layer": self.strName,
                "Output Shape": tuple(self.outShape),
                "Cropping": (
                    self.top_crop,
                    self.bottom_crop,
                    self.left_crop,
                    self.right_crop,
                ),
            }
        )
        return summary
