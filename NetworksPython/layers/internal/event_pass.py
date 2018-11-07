import numpy as np
from ...timeseries import TimeSeries
from ..layer import Layer
from typing import Optional, Union, Tuple, List, Callable
from warnings import warn
from NetworksPython import TSEvent

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Relative tolerance for float comparions
fTolerance = 1e-5


class PassThroughEvents(Layer):
    def __init__(
        self,
        mnW: np.ndarray,
        tDt: float=0.001,
        fNoiseStd: Optional[float]=None,
        strName: str="unnamed",
    ):
        """
        PassThroughEvents class - route events to different channels

        :param mnW:         np.ndarray Positive integer weight matrix for this layer
        :param tDt:         float Time step duration. Only used for determining evolution period and internal clock.
        :param fNoiseStd:   float Not actually used
        :param strName:     str Name of this layer. Default: 'unnamed'
        """

        # - Weights should be of integer type
        mnW = np.asarray(mnW, int)

        if fNoiseStd is not None:
            warn("Layer `{}`: fNoiseStd is not used in this layer.".format(strName))
        
        # - Initialize parent class
        super().__init__(
            mfW=mnW,
            tDt=tDt,
            fNoiseStd=fNoiseStd,
            strName=strName,
        )

        self.reset_all()

    def evolve(
        self,
        tsInput: Optional[TSEvent] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
        bVerbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSEvent  Input spike trian
        :param tDuration:       float    Simulation/Evolution time
        :param nNumTimeSteps    int      Number of evolution time steps
        :param bVerbose:        bool     Currently no effect, just for conformity
        :return:                TSEvent  output spike series

        """

        nNumTimeSteps = self._determine_timesteps(tsInput, tDuration, nNumTimeSteps)
        tEnd = self.t + self.tDt * nNumTimeSteps

        # - Handle empty inputs
        if tsInput is None or tsInput.vtTimeTrace.size == 0:
            return TSEvent(None, None, nNumChannels=self.nSize)#, tStart=self.t, tStop=tEnd)
        
        nNumInputEvents = tsInput.vtTimeTrace.size
        # - Boolean raster of input events - each row corresponds to one event (not timepoint)
        mbInputChannelRaster = np.zeros((nNumInputEvents, self.nSizeIn), bool)
        mbInputChannelRaster[np.arange(nNumInputEvents), tsInput.vnChannels] = True
        # - Integer raster of output events with number of occurences
        #   Each row corresponds to one input event (not timepoint)
        mnOutputChannelRaster = mbInputChannelRaster @ self.mnW
        ## -- Extract channels from raster
        # - Number of repetitions for each output event in temporal order
        #   (every self.nSize events occur simultaneously)
        vnRepetitions = mnOutputChannelRaster.flatten()
        # - Output channels corresponding to vnRepetitions
        vnChannelMask = np.tile(np.arange(self.nSize), nNumInputEvents)
        # - Output channel train
        vnChannelsOut = np.repeat(vnChannelMask, vnRepetitions)
        # - Output time trace consits of elements from input time trace
        #   repeated by the number of output events they result in
        vnNumOutputEventsPerInputEvent = np.sum(mnOutputChannelRaster, axis=1)
        vtTimeTraceOut = np.repeat(tsInput.vtTimeTrace, vnNumOutputEventsPerInputEvent)

        # - Output time series
        tseOut = TSEvent(
            vtTimeTrace=vtTimeTraceOut,
            vnChannels=vnChannelsOut,
            nNumChannels=self.nSize,
            # tStart=self.t,
            # tStop=tEnd,
            strName="transformed event raster"
        )

        # - Update clock
        self._nTimeStep += nNumTimeSteps

        return tseOut

    @property
    def cInput(self):
        return self.TSEvent

    @property
    def cOutput(self):
        return self.TSEvent    

    @property
    def mnW(self):
        return self._mfW

    @mnW.setter
    def mnW(self, mnNewW):
        self.mfW = np.asarray(mnNewW, int)

