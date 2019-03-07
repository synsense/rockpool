import numpy as np
import pandas as pd
import torch.nn as nn
from typing import Union, List, Tuple

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]


class TorchZeroPad2dLayer(nn.Module):
    def __init__(self, padding: ArrayLike = (0, 0, 0, 0), strName: str = "zeropad2d"):
        """
        Pytorch implementation of a spiking neuron with convolutional inputs
        SUBTRACT superseeds Reset value
        """
        nn.Module.__init__(self)  # Init nn.Module
        self.padding = padding
        self.pad = nn.ZeroPad2d(padding)
        self.strName = strName

    def reset_states(self):
        """
        Reset the state of all neurons in this layer
        """
        pass

    def forward(self, tsrBinaryInput):
        _, self.nInChannels, h, w = list(tsrBinaryInput.shape)
        # Convolve all inputs at once
        tsrPadOut = self.pad(tsrBinaryInput)

        self.tsrNumSpikes = tsrPadOut
        self.outShape = tsrPadOut.shape[1:]
        return tsrPadOut.float()  # Float is just to keep things compatible

    @property
    def nOutChannels(self):
        return self.nInChannels

    def summary(self):
        """
        Returns the summary of this layer as a pandas Series
        """
        summary = pd.Series(
            {
                "Layer": self.strName,
                "Output Shape": str(list(self.outShape)),
                "Padding": str(self.padding),
                # "Kernel": str(None),
                # "Stride": str(None),
                "Neurons": 0,
                "KernelMem": 0,
                "BiasMem": 0,
            },
            index=[0],
        )
        return summary
