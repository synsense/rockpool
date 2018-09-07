"""
updwon.py - Feedforward layer that converts each analogue input channel to one spiking up and one down channel
"""

import numpy as np
from typing import Optional, Union, Tuple, List

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    bUseTqdm = False
else:
    bUseTqdm = True

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Local imports
from ...timeseries import TSContinuous, TSEvent
from ..layer import Layer

__all__ = ["FFUpDown"]

## - FFUpDown - Class: Define a spiking feedforward layer to convert analogue inputs to up and down channels
class FFUpDown(Layer):
    """
    FFUpDown - Class: Define a spiking feedforward layer to convert analogue inputs to up and down channels
    """

    ## - Constructor
    def __init__(
        self,
        mfW: Union[int, np.ndarray],
        tDt: float = 0.001,
        fNoiseStd: float = 0,
        vfThrUp: Union[ArrayLike, float] = 0.001,
        vfThrDown: Union[ArrayLike, float] = 0.001,
        strName: str = "unnamed",
    ):
        """
        FFUpDown - Construct a spiking feedforward layer to convert analogue inputs to up and down channels
        
        :param mfW:         np.array MxN weight matrix.
            Unlike other Layer classes, only important thing about mfW is mfW.shape[0].
            It determines the number of input channels (self.nSizeIn). The output size
            (corresponding to self.nSize) is always 2*self.nSizeIn (one up and one down
            channel for each input). The values of the weight matrix do not have any effect.
            Therefore also an integer (corresponding to nSizeIn) can be passed.
        :param tDt:         float Time-step. Default: 0.1 ms
        :param fNoiseStd:   float Noise std. dev. per second. Default: 0

        :param vfThrUp:     array-like Thresholds for creating up-spikes
        :param vfThrDown:   array-like Thresholds for creating down-spikes

        :param strName:     str Name for the layer. Default: 'unnamed'
        """

        if np.size(mfW) == 1:
            nSizeIn = mfW
        else:
            nSizeIn = mfW.shape[0]

        # - Call super constructor
        super().__init__(
            mfW=np.zeros((nSizeIn, 2*nSizeIn)),
            tDt=tDt,
            fNoiseStd=fNoiseStd,
            strName=strName
        )

        # - Store layer parameters
        self.vfThrUp = vfThrUp
        self.vfThrDown = vfThrDown

        self.reset_all()

    def evolve(
        self,
        tsInput: Optional[TSContinuous] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
        bVerbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSContinuous  Input spike trian
        :param tDuration:       float    Simulation/Evolution time
        :param nNumTimeSteps    int      Number of evolution time steps
        :param bVerbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        # - Prepare time base
        __, mfInputStep, nNumTimeSteps = self._prepare_input(
            tsInput, tDuration, nNumTimeSteps
        )

        # - Add noise to input
        mfInputStep += np.random.randn(*mfInputStep.shape) * self.fNoiseStd

        # - Prepare local variables
        vfThrUp = self.vfThrUp
        vfThrDown = self.vfThrDown

        # - Lists for storing spikes
        lnTSSpike = list()
        liSpikeIDs = list()

        rangeIterator = range(nNumTimeSteps)
        if bVerbose and bUseTqdm:
            # - Add tqdm output
            rangeIterator = tqdm(rangeIterator)
        
        # - Initialize state for comparing values: If self.vState exists, assume input continues from
        #   previous evolution. Otherwise start with initial input data
        vState = mfInputStep[0] if self.vState is None else self.vState
        
        for iCurrentTS in rangeIterator:
            # - Indices of inputs where upper threshold is passed
            viUp, = np.where(mfInputStep[iCurrentTS] > vState + vfThrUp)
            # - Indices of inputs where lower threshold is passed
            viDown, = np.where(mfInputStep[iCurrentTS] < vState - vfThrDown)
            # - Update state
            vState[viUp] += vfThrUp[viUp]
            vState[viDown] -= vfThrDown[viDown]
            # - Append spikes to lists
            lnTSSpike += (viUp.size + viDown.size) * [iCurrentTS]
            # - Up channels have even, down channels odd IDs
            liSpikeIDs += list(2*viUp) + list(2*viDown + 1)

        # - Store state for future evolutions
        self.vState = vState

        # - Output time series
        vtSpikeTimes = (np.array(lnTSSpike) + 1 + self._nTimeStep) * self.tDt
        tseOut = TSEvent(
            vtTimeTrace=vtSpikeTimes, vnChannels=liSpikeIDs, nNumChannels=2*self.nSizeIn
        )

        # - Update time
        self._nTimeStep += nNumTimeSteps

        return tseOut


    def reset_state(self):
        # - Store None as state to indicate that future evolutions do not continue from previous input
        self.vState = None


    @property
    def cOutput(self):
        return TSEvent
    
    @property
    def vState(self):
        return self._vState

    @vState.setter
    # Note that vState here is of size self.nSizeIn and not self.nSize
    def vState(self, vNewState):
        if vNewState is None:
            self._vState = None
        else:
            self._vState = self._expand_to_size(vNewState, self.nSizeIn, "vState")

    @property
    def vfThrUp(self):
        return self._vfThrUp

    @vfThrUp.setter
    def vfThrUp(self, vfNewThr):
        assert (np.array(vfNewThr) >= 0).all(), "vfThrUp must not be negative."

        self._vfThrUp = self._expand_to_size(vfNewThr, self.nSizeIn, "vfThrUp", bAllowNone=False)
    
    @property
    def vfThrDown(self):
        return self._vfThrDown

    @vfThrDown.setter
    def vfThrDown(self, vfNewThr):
        assert (np.array(vfNewThr) >= 0).all(), "vfThrDown must not be negative."
        self._vfThrDown = self._expand_to_size(vfNewThr, self.nSizeIn, "vfThrDown", bAllowNone=False)
    