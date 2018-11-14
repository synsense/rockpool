import numpy as np
import pandas as pd
import torch.nn as nn
from typing import Optional, Union, List, Tuple
from operator import mul
from functools import reduce

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]


class TorchSpikingConv2dLayer(nn.Module):
    def __init__(
        self,
        nInChannels: int = 1,
        nOutChannels: int = 1,
        kernel_size: ArrayLike = (1, 1),
        strides: ArrayLike = (1, 1),
        padding: ArrayLike = (0, 0, 0, 0),
        bias: bool = True,
        fVThresh: float = 8,
        fVThreshLow: Optional[float] = None,
        fVSubtract: Optional[float] = None,
        fVReset: float = 0,
        strName: str = "conv2d",
    ):
        """
        Pytorch implementation of a spiking neuron with convolutional inputs
        SUBTRACT superseeds Reset value
        """
        super(TorchSpikingConv2dLayer, self).__init__()  # Init nn.Module
        if padding != (0, 0, 0, 0):
            self.pad = nn.ZeroPad2d(padding)
        else:
            self.pad = None
        self.conv = nn.Conv2d(
            nInChannels,
            nOutChannels,
            kernel_size=kernel_size,
            stride=strides,
            bias=bias,
        )
        # Initialize neuron states
        self.fVSubtract = fVSubtract
        self.fVReset = fVReset
        self.fVThresh = fVThresh
        self.fVThreshLow = fVThreshLow
        self.strName = strName

        # Layer convolutional properties
        self.nInChannels = nInChannels
        self.nOutChannels = nOutChannels
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.bias = bias

        # Blank parameter place holders
        self.tsrNumSpikes = None
        self.tsrState = None

    @property
    def fVThreshLow(self):
        return self._fVThreshLow

    @fVThreshLow.setter
    def fVThreshLow(self, fVNewThreshLow):
        self._fVThreshLow = fVNewThreshLow
        if fVNewThreshLow is None:
            try:
                del (self.threshLower)
            except AttributeError:
                pass
        else:
            # Relu on the layer
            self.threshLower = nn.Threshold(fVNewThreshLow, fVNewThreshLow)

    def resetStates(self):
        """
        Reset the state of all neurons in this layer
        """
        if self.tsrState is None:
            return
        else:
            self.tsrState.zero_()

    def forward(self, tsrBinaryInput):
        # Determine no. of time steps from input
        nNumTimeSteps = len(tsrBinaryInput)

        # Convolve all inputs at once
        if self.pad is None:
            tsrConvOut = self.conv(tsrBinaryInput)
        else:
            tsrConvOut = self.conv(self.pad(tsrBinaryInput))

        ## - Count number of spikes for each neuron in each time step
        # vnNumSpikes = np.zeros(tsrConvOut.shape[1:], int)

        # Local variables
        tsrState = self.tsrState
        fVSubtract = self.fVSubtract
        fVThresh = self.fVThresh
        fVThreshLow = self.fVThreshLow
        fVReset = self.fVReset

        # Initialize state as required
        # Create a vector to hold all output spikes
        if self.tsrNumSpikes is None or len(self.tsrNumSpikes) != len(tsrBinaryInput):
            del (self.tsrNumSpikes)  # Free memory just to be sure
            self.tsrNumSpikes = tsrConvOut.new_zeros(
                nNumTimeSteps, *tsrConvOut.shape[1:]
            ).int()

        self.tsrNumSpikes.zero_()
        tsrNumSpikes = self.tsrNumSpikes

        if self.tsrState is None:
            self.tsrState = tsrConvOut.new_zeros(tsrConvOut.shape[1:])

        tsrState = self.tsrState

        # Loop over time steps
        for iCurrentTimeStep in range(nNumTimeSteps):
            tsrState = tsrState + tsrConvOut[iCurrentTimeStep]

            # - Reset or subtract from membrane state after spikes
            if fVSubtract is not None:
                # Calculate number of spikes to be generated
                tsrNumSpikes[iCurrentTimeStep] = (tsrState >= fVThresh).int() + (
                    tsrState - fVThresh > 0
                ).int() * (tsrState / fVSubtract).int()
                ## - Subtract from states
                tsrState = tsrState - (
                    fVSubtract * tsrNumSpikes[iCurrentTimeStep].float()
                )
            else:
                # - Check threshold crossings for spikes
                vbRecSpikeRaster = tsrState >= fVThresh
                # - Add to spike counter
                tsrNumSpikes[iCurrentTimeStep] = vbRecSpikeRaster
                # - Reset neuron states
                tsrState = (
                    vbRecSpikeRaster.float() * fVReset
                    + tsrState * (vbRecSpikeRaster ^ 1).float()
                )

            if fVThreshLow is not None:
                tsrState = self.threshLower(tsrState)  # Lower bound on the activation

        self.tsrState = tsrState
        self.tsrNumSpikes = tsrNumSpikes
        self.outShape = tsrNumSpikes.shape[1:]
        return tsrNumSpikes.float()  # Float is just to keep things compatible

    def summary(self):
        summary = pd.DataFrame(
            {
                "Layer": self.strName,
                "Output Shape": str(list(self.outShape)),
                "Padding": str(self.padding),
                "Kernel": str(self.kernel_size),
                "Stride": str(self.strides),
                "FanOutPrev": reduce(
                    mul, np.array(self.kernel_size) / np.array(self.strides), 1
                )
                * self.nOutChannels,
                "Neurons": reduce(mul, list(self.outShape[-3:]), 1),
                "KernelMem": self.nInChannels
                * self.nOutChannels
                * reduce(mul, self.kernel_size, 1),
                "BiasMem": self.bias * self.nOutChannels,
            },
            index=[0],
        )
        return summary
