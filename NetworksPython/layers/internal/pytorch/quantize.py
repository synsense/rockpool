import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple

ArrayLike = Union[np.ndarray, List, Tuple]


class TorchQuantizeLayer(nn.Module):
    """
    Equivalent to keras flatten
    """

    def __init__(self, name="quantize"):
        """
        Torch implementation of Quantizing the output spike count.
        """
        nn.Module.__init__(self)  # Init nn.Module
        self.name = name

    def forward(self, tsrInput):
        nBatch = len(tsrInput)
        tsrOut = tsrInput.int().float()
        self.outShape = tsrOut.shape[1:]
        self.tsrNumSpikes = tsrOut
        return tsrOut

    @property
    def nOutChannels(self):
        return 1

    def summary(self):
        """
        Returns a summary of this layer as a pandas Series
        """
        summary = pd.DataFrame(
            {
                "Layer": self.name,
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
