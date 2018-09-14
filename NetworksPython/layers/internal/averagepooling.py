###
# averagepooling.py - Class implementing a pooling layer.
###

from typing import Optional
import numpy as np
import skimage.measure
from warnings import warn
from typing import Optional, Union, List, Tuple, Generator
import torch.nn as nn

# from typing import Optional, Union, List, Tuple
from ...timeseries import TSEvent

# from ..layer import Layer
from .iaf_cl import CLIAF
from .cnnweights import CNNWeight

# - Absolute tolerance, e.g. for comparing float values
fTolAbs = 1e-9

# - Type alias for array-like objects

ArrayLike = Union[np.ndarray, List, Tuple]


class AveragePooling2D(CLIAF):
    """
    AveragePooling: Implements average pooling by simply merging inputs. So this is more of sum than average pooling.
    """

    def __init__(
        self,
        inShape: tuple,
        pool_size: tuple = (1, 1),
        img_data_format="channels_last",
        tDt: float = 1,
        strName: str = "unnamed",
    ):
        """
        :param inShape:     tuple Input shape
        :param pool_size:   tuple Pooling width along each dimension
        :param img_data_format: str 'channels_first' or 'channels_last'
        :param strName:     str  Name of this layer.
        :param tDt:         float  Time step for simulations
        """

        if img_data_format == "channels_last":
            nKernels = inShape[-1]
        elif img_data_format == "channels_first":
            nKernels = inShape[0]
        self.img_data_format = img_data_format
        mfW = CNNWeight(
            inShape=inShape,
            nKernels=nKernels,
            kernel_size=pool_size,
            strides=pool_size,
            img_data_format=img_data_format,
        )  # Simple hack

        # Call parent constructor
        super().__init__(mfWIn=mfW, tDt=tDt, strName=strName)
        self.pool_size = pool_size
        self.reset_state()

    def _prepare_input(
        self, tsInput: Optional[TSEvent] = None, nNumTimeSteps: int = 1
    ) -> np.ndarray:
        """
        Prepare input stream and return a binarized vector of spikes
        """
        # - End time of evolution
        tFinal = self.t + nNumTimeSteps * self.tDt
        # - Extract spike timings and channels
        if tsInput is not None:
            if tsInput.isempty():
                # Return an empty list with all zeros
                vbSpikeRaster = np.zeros((self.nSizeIn), bool)
            else:
                # Ensure number of channels is atleast as many as required
                try:
                    assert tsInput.nNumChannels >= self.nSizeIn
                except AssertionError as err:
                    warn(
                        self.strName
                        + ": Expanding input dimensions to match layer size."
                    )
                    tsInput.nNumChannels = self.nSizeIn

                # Extract spike data from the input variable
                mfSpikeRaster = tsInput.xraster(
                    tDt=self.tDt, tStart=self.t, tStop=tFinal
                )

                ## - Make sure size is correct
                # mfSpikeRaster = mfSpikeRaster[:nNumTimeSteps, :]
                # assert mfSpikeRaster.shape == (nNumTimeSteps, self.nSizeIn)
                yield from mfSpikeRaster  # Yield a single time step
                return
        else:
            # Return an empty list with all zeros
            vbSpikeRaster = np.zeros((self.nSizeIn), bool)

        while True:
            yield vbSpikeRaster

    def evolve(
        self,
        tsInput: Optional[TSEvent] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
        bVerbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:  TSEvent  Input spike trian
        :param tDuration:   float    Simulation/Evolution time
        :param bVerbose:    bool Currently no effect, just for conformity
        :return:          TSEvent  output spike series

        """

        # Compute number of simulation time steps
        if nNumTimeSteps is None:
            nNumTimeSteps = int((tDuration + fTolAbs) // self.tDt)

        # - Generate input in rasterized form, get actual evolution duration
        mfInptSpikeRaster = np.array(list(self._prepare_input(tsInput, nNumTimeSteps)))

        # Reshape input data
        mfInptSpikeRaster = mfInptSpikeRaster.reshape((-1, *self.mfW.inShape))

        # Do average pooling here
        if self.img_data_format == "channels_last":
            mbOutRaster = skimage.measure.block_reduce(
                mfInptSpikeRaster, (1, *self.pool_size, 1), np.sum
            )
        elif self.img_data_format == "channels_first":
            mbOutRaster = skimage.measure.block_reduce(
                mfInptSpikeRaster, (1, 1, *self.pool_size, 1), np.sum
            )

        # Reshape the output to flat indices
        mbOutRaster = mbOutRaster.reshape((-1, self.nSize))

        # Convert raster to indices
        ltSpikeTimes, liSpikeIDs = np.nonzero(mbOutRaster)

        # Convert arrays to TimeSeries objects
        tseOut = TSEvent(
            vtTimeTrace=ltSpikeTimes, vnChannels=liSpikeIDs, nNumChannels=self.nSize
        )

        # Update time
        self._t += self.tDt * nNumTimeSteps

        return tseOut

    @property
    def mfW(self):
        return self._mfWIn

    @mfW.setter
    def mfW(self, mfNewW):
        self.mfWIn = mfNewW


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
