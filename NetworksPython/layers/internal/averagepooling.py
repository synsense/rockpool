###
# averagepooling.py - Class implementing a pooling layer.
###

from typing import Optional
import numpy as np
import skimage.measure

# from typing import Optional, Union, List, Tuple
from ...timeseries import TSEvent

# from ..layer import Layer
from ..iaf_cl import CLIAF
from ..cnnweights import CNNWeight


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

        # - Generate input in rasterized form, get actual evolution duration
        mfInptSpikeRaster, nNumTimeSteps = self._prepare_input(
            tsInput, tDuration, nNumTimeSteps
        )
        print(np.sum(mfInptSpikeRaster))

        # Do average pooling here
        print(mfInptSpikeRaster.shape)

        # Reshape input data
        mfInptSpikeRaster = mfInptSpikeRaster.reshape((-1, *self.mfW.inShape))
        print(mfInptSpikeRaster.shape)

        if self.img_data_format == "channels_last":
            mbOutRaster = skimage.measure.block_reduce(
                mfInptSpikeRaster, (1, *self.pool_size, 1), np.sum
            )
        elif self.img_data_format == "channels_first":
            mbOutRaster = skimage.measure.block_reduce(
                mfInptSpikeRaster, (1, 1, *self.pool_size, 1), np.sum
            )
        print(mbOutRaster.shape)

        # Reshape the output to flat indices
        mbOutRaster = mbOutRaster.reshape((-1, self.nSize))
        print(mbOutRaster.shape)

        # Convert raster to indices
        ltSpikeTimes, liSpikeIDs = np.nonzero(mbOutRaster)

        # Convert arrays to TimeSeries objects
        tseOut = TSEvent(
            vtTimeTrace=ltSpikeTimes, vnChannels=liSpikeIDs, nNumChannels=self.nSize
        )

        print(np.sum(mfInptSpikeRaster), np.sum(mbOutRaster))

        # Update time
        self._t += self.tDt * nNumTimeSteps

        return tseOut

    @property
    def mfW(self):
        return self._mfWIn

    @mfW.setter
    def mfW(self, mfNewW):
        self.mfWIn = mfNewW


#    @property
#    def cOutput(self):
#        return TSEvent
#
#    @property
#    def cInput(self):
#        return TSEvent
