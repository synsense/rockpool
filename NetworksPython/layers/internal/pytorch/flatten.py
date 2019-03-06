import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple

ArrayLike = Union[np.ndarray, List, Tuple]


class TorchFlatten2dLayer(nn.Module):
    """
    Equivalent to keras flatten
    """

    def __init__(self, strName="crop2d"):
        """
        Torch implementation of SumPooling using the LPPool2d module
        """
        nn.Module.__init__(self)  # Init nn.Module
        self.strName = strName

    def forward(self, tsrBinaryInput):
        nBatch = len(tsrBinaryInput)
        # Temporary modify LQ, due to keras weights generation change
        # tsrBinaryInput = tsrBinaryInput.permute(0, 2, 3, 1)
        tsrFlattenOut = tsrBinaryInput.contiguous().view(nBatch, -1)
        self.outShape = tsrFlattenOut.shape[1:]
        self.tsrNumSpikes = tsrFlattenOut
        return tsrFlattenOut

    @property
    def nOutChannels(self):
        return 1

    def summary(self):
        """
        Returns a summary of this layer as a pandas Series
        """
        summary = pd.Series(
            {
                "Layer": self.strName,
                "Output Shape": str(list(self.outShape)),
                # "Padding": str(None),
                # "Kernel": str(None),
                # "Stride": str(None),
                "Neurons": 0,
                "KernelMem": 0,
                "BiasMem": 0,
            }
        )
        return summary
