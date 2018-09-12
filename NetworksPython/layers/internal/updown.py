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
        This layer is exceptional in that self.vState has the same size as self.nSizeIn, not self.nSize. 
        It corresponds to the input, inferred from the output spikes by inverting the up-/down-algorithm.
        
        :param mfW:         np.array MxN weight matrix.
            Unlike other Layer classes, only important thing about mfW its shape. The first
            dimension determines the number of input channels (self.nSizeIn). The second
            dimension corresponds to nSize and has to be n*2*nSizeIn, n up and n down
            channels for each input). If n>1 the up-/and down-spikes are distributed over
            multipple channels. The values of the weight matrix do not have any effect.
            It is also possible to pass only an integer, which will correspond to nSizeIn.
            nSize is then set to 2*nSizeIn, i.e. n=1. Alternatively a tuple of two values,
            corresponding to nSizeIn and n can be passed.
        :param tDt:         float Time-step. Default: 0.1 ms
        :param fNoiseStd:   float Noise std. dev. per second. Default: 0

        :param vfThrUp:     array-like Thresholds for creating up-spikes
        :param vfThrDown:   array-like Thresholds for creating down-spikes

        :param strName:     str Name for the layer. Default: 'unnamed'
        """

        if np.size(mfW) == 1:
            nSizeIn = mfW
            nSize = 2 * nSizeIn
            # - On how many output channels is the are the up-/down-spikes from each input distributed
            self._nMultiChannel = 1
        elif np.size(mfW) == 2:
            # - Tuple determining shape
            (nSizeIn, self._nMultiChannel) = mfW
            nSize = 2 * self._nMultiChannel * nSizeIn
        else:
            (nSizeIn, nSize) = mfW.shape
            assert (
                nSize % (2 * nSizeIn) == 0
            ), "Layer `{}`: nSize (here {}) must be a multiple of 2*nSizeIn (here {}).".format(
                strName, nSize, nSizeIn
            )
            # - On how many output channels is the are the up-/down-spikes from each input distributed
            self._nMultiChannel = nSize / (2 * nSizeIn)

        # - Call super constructor
        super().__init__(
            mfW=np.zeros((nSizeIn, nSize)),
            tDt=tDt,
            fNoiseStd=fNoiseStd,
            strName=strName,
        )

        # - Store layer parameters
        self.vfThrUp = vfThrUp
        self.vfThrDown = vfThrDown

        self.reset_all()

    # @profile
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
            liSpikeIDs += list(2 * viUp) + list(2 * viDown + 1)

        # - Store state for future evolutions
        self.vState = vState

        ## -- Distribute output spikes over output channels
        vnSpikeIDs = np.array(liSpikeIDs)
        # - Array to hold distributed channel IDs
        vnChannels = np.zeros(vnSpikeIDs.size, int)
        for nSpikeID in range(2 * self.nSizeIn):
            viSpikeIndices, = np.where(vnSpikeIDs == nSpikeID)
            nNumEvents = viSpikeIndices.size
            vnChannels[viSpikeIndices] = np.tile(
                np.arange(self._nMultiChannel) + self._nMultiChannel * nSpikeID,
                int(np.ceil(nNumEvents / self._nMultiChannel)),
            )[:nNumEvents]

        # - Output time series
        vtSpikeTimes = (np.array(lnTSSpike) + 1 + self._nTimeStep) * self.tDt
        tseOut = TSEvent(
            vtTimeTrace=vtSpikeTimes,
            vnChannels=vnChannels,
            nNumChannels=2 * self.nSizeIn * self._nMultiChannel,
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

        self._vfThrUp = self._expand_to_size(
            vfNewThr, self.nSizeIn, "vfThrUp", bAllowNone=False
        )

    @property
    def vfThrDown(self):
        return self._vfThrDown

    @vfThrDown.setter
    def vfThrDown(self, vfNewThr):
        assert (np.array(vfNewThr) >= 0).all(), "vfThrDown must not be negative."
        self._vfThrDown = self._expand_to_size(
            vfNewThr, self.nSizeIn, "vfThrDown", bAllowNone=False
        )
