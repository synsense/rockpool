"""
updwon.py - Feedforward layer that converts each analogue input channel to one spiking up and one down channel
            Run in batch mode like FFUpDownTorch to save memory, but do not use pytorch. FFUpDownTorch seems
            to be slower..
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
# - Default maximum numbers of time steps for a single evolution batch
nDefaultMaxNumTimeSteps = 5000

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
        nRepeatOutput: int = 1,
        tDt: float = 0.001,
        vtTauDecay: Union[ArrayLike, float, None] = None,
        fNoiseStd: float = 0,
        vfThrUp: Union[ArrayLike, float] = 0.001,
        vfThrDown: Union[ArrayLike, float] = 0.001,
        strName: str = "unnamed",
        nMaxNumTimeSteps: int = nDefaultMaxNumTimeSteps,
    ):
        """
        FFUpDownBatch - Construct a spiking feedforward layer to convert analogue inputs to up and down channels
        This layer is exceptional in that self.vState has the same size as self.nSizeIn, not self.nSize.
        It corresponds to the input, inferred from the output spikes by inverting the up-/down-algorithm.

        :param mfW:         np.array MxN weight matrix.
            Unlike other Layer classes, only important thing about mfW its shape. The first
            dimension determines the number of input channels (self.nSizeIn). The second
            dimension corresponds to nSize and has to be n*2*nSizeIn, n up and n down
            channels for each input). If n>1 the up-/and down-spikes are distributed over
            multiple channels. The values of the weight matrix do not have any effect.
            It is also possible to pass only an integer, which will correspond to nSizeIn.
            nSize is then set to 2*nSizeIn, i.e. n=1. Alternatively a tuple of two values,
            corresponding to nSizeIn and n can be passed.
        :param tDt:         float Time-step. Default: 0.1 ms
        :param vtTauDecay:  array-like  States that tracks input signal for threshold comparison
                                        decay with this time constant unless it is None

        :param fNoiseStd:   float Noise std. dev. per second. Default: 0

        :param vfThrUp:     array-like Thresholds for creating up-spikes
        :param vfThrDown:   array-like Thresholds for creating down-spikes

        :param strName:     str Name for the layer. Default: 'unnamed'

        :nMaxNumTimeSteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        if np.size(mfW) == 1:
            nSizeIn = mfW
            nSize = 2 * nSizeIn * nRepeatOutput
            # - On how many output channels is the are the up-/down-spikes from each input distributed
            self._nMultiChannel = 1
        elif np.size(mfW) == 2:
            # - Tuple determining shape
            (nSizeIn, self._nMultiChannel) = mfW
            nSize = 2 * self._nMultiChannel * nSizeIn * nRepeatOutput
        else:
            (nSizeIn, nSize) = mfW.shape
            assert (
                nSize % (2 * nSizeIn) == 0
            ), "Layer `{}`: nSize (here {}) must be a multiple of 2*nSizeIn (here {}).".format(
                strName, nSize, nSizeIn
            )
            # - On how many output channels is the are the up-/down-spikes from each input distributed
            self._nMultiChannel = nSize / (2 * nSizeIn)
            nSize *= nRepeatOutput
        # - Make sure self._nMultiChannel is an integer
        self._nMultiChannel = int(self._nMultiChannel)

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
        self.vtTauDecay = vtTauDecay
        self.nMaxNumTimeSteps = nMaxNumTimeSteps
        self.nRepeatOutput = nRepeatOutput

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
        __, mfInput, nNumTimeSteps = self._prepare_input(
            tsInput, tDuration, nNumTimeSteps
        )

        if self.fNoiseStd > 0:
            # - Add noise to input
            mfInputStep += np.random.randn(*mfInput.shape) * self.fNoiseStd

        # - Tensor for collecting output spike raster
        mbOutputSpikes = np.zeros((nNumTimeSteps, 2*self.nSizeIn))

        # - Iterate over batches and run evolution
        iCurrentIndex = 0
        for mfCurrentInput, nCurrNumTS in self._batch_data(
                mfInput, nNumTimeSteps, self.nMaxNumTimeSteps
            ):
            mbOutputSpikes[iCurrentIndex : iCurrentIndex+nCurrNumTS] = self._single_batch_evolution(
                mfCurrentInput,
                nCurrNumTS,
                bVerbose,
            )
            iCurrentIndex += nCurrNumTS

        ## -- Distribute output spikes over output channels by assigning to each channel
        ##    an interval of length self._nMultiChannel.
        # - Set each event to the first element of its corresponding interval
        vnTSSpike, vnSpikeIDs = np.where(mbOutputSpikes)
        vnSpikeIDs *= self._nMultiChannel
        # - Repeat output spikes
        vnSpikeIDs = vnSpikeIDs.repeat(self.nRepeatOutput)
        # - Add a repeating series of (0,1,2,..,self._nMultiChannel) to distribute the
        #   events over the interval
        vnDistribute = np.tile(
            np.arange(self._nMultiChannel),
            int(np.ceil(vnSpikeIDs.size / self._nMultiChannel))
        )[:vnSpikeIDs.size]
        vnSpikeIDs += vnDistribute

        # - Output time series
        vtSpikeTimes = (vnTSSpike.repeat(self.nRepeatOutput) + 1 + self._nTimeStep) * self.tDt
        tseOut = TSEvent(
            vtTimeTrace=vtSpikeTimes,
            vnChannels=vnSpikeIDs,
            nNumChannels=2 * self.nSizeIn * self._nMultiChannel,
            strName="Spikes from analogue",
        )

        # - Update time
        self._nTimeStep += nNumTimeSteps

        return tseOut

    @profile
    def _batch_data(
        self, mfInput: np.ndarray, nNumTimeSteps: int, nMaxNumTimeSteps: int = None,
    ) -> (np.ndarray, int):
        """_batch_data: Generator that returns the data in batches"""
        # - Handle None for nMaxNumTimeSteps
        nMaxNumTimeSteps = nNumTimeSteps if nMaxNumTimeSteps is None else nMaxNumTimeSteps
        nStart = 0
        while nStart < nNumTimeSteps:
            # - Endpoint of current batch
            nEnd = min(nStart + nMaxNumTimeSteps, nNumTimeSteps)
            # - Data for current batch
            mfCurrentInput = mfInput[nStart:nEnd]
            yield mfCurrentInput, nEnd-nStart
            # - Update nStart
            nStart = nEnd

    @profile
    def _single_batch_evolution(
        self,
        mfInput: np.ndarray,
        nNumTimeSteps: int,
        bVerbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input for a single batch

        :param mfInput:     np.ndarray   Input
        :param nNumTimeSteps:   int      Number of evolution time steps
        :param bVerbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        # - Prepare local variables
        vfThrUp = self.vfThrUp
        vfThrDown = self.vfThrDown
        vfDecayFactor = self._vfDecayFactor

        # - Arrays for collecting spikes
        mbSpikeRaster = np.zeros((nNumTimeSteps, 2*self.nSizeIn))

        # - Initialize state for comparing values: If self.vState exists, assume input continues from
        #   previous evolution. Otherwise start with initial input data
        vState = mfInput[0] if self._vState is None else self._vState.copy()

        for iCurrentTS in range(nNumTimeSteps):
            # - Decay mechanism
            vState *= vfDecayFactor
            # - Indices of inputs where upper threshold is passed
            vbUp = mfInput[iCurrentTS] > vState + vfThrUp
            # - Indices of inputs where lower threshold is passed
            vbDown = mfInput[iCurrentTS] < vState - vfThrDown
            # - Update state
            vState += vfThrUp * vbUp
            vState -= vfThrDown * vbDown
            # - Append spikes to array
            mbSpikeRaster[iCurrentTS, ::2] = vbUp
            mbSpikeRaster[iCurrentTS, 1::2] = vbDown

        # - Store state for future evolutions
        self._vState = vState.copy()

        return mbSpikeRaster

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

    @property
    def vtTauDecay(self):
        vtTau = np.repeat(None, self.nSizeIn)
        # - Treat decay factors of 1 as not decaying (i.e. set them None)
        vbDecay = self._vfDecayFactor != 1
        vtTau[vbDecay] = self.tDt / (1 - self._vfDecayFactor[vbDecay])
        return vtTau

    @vtTauDecay.setter
    def vtTauDecay(self, vtNewTau):
        vtNewTau = self._expand_to_size(vtNewTau, self.nSizeIn, "vtTauDecay", bAllowNone=True)
        # - Find entries which are not None, indicating decay
        vbDecay = np.array([tTau is not None for tTau in vtNewTau])
        # - Check for too small entries
        assert (vtNewTau[vbDecay] >= self.tDt).all(), (
            "Layer `{}`: Entries of vtTauDecay must be greater or equal to tDt ({}).".format(self.strName, self.tDt)
        )
        self._vfDecayFactor = np.ones(self.nSizeIn)  # No decay corresponds to decay factor 1
        self._vfDecayFactor[vbDecay] = 1 - self.tDt / vtNewTau[vbDecay]
