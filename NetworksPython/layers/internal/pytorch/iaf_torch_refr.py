###
# iaf_torch_refr.py - Like iaf_torch.py but classes support refractory period.
###


# - Imports
import numpy as np
import torch

from ....timeseries import TSContinuous, TSEvent
from ...layer import Layer
from ..timedarray_shift import TimedArray as TAShift

from typing import Optional, Union, Tuple, List

from time import time

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Configure exports
__all__ = ["FFIAFTorch", "FFIAFSpkInTorch", "RecIAFTorch", "RecIAFSpkInTorch"]

# - Absolute tolerance, e.g. for comparing float values
fTolAbs = 1e-9
# - Default maximum numbers of time steps for a single evolution batch
nDefaultMaxNumTimeSteps = 400

## - FFIAFTorch - Class: define a spiking feedforward layer with spiking outputs
class FFIAFTorch(Layer):
    """ FFIAFTorch - Class: define a spiking feedforward layer with spiking outputs
    """

    ## - Constructor
    def __init__(
        self,
        mfW: np.ndarray,
        vfBias: Union[float, np.ndarray] = 0.015,
        tDt: float = 0.0001,
        fNoiseStd: float = 0,
        vtTauN: Union[float, np.ndarray] = 0.02,
        vfVThresh: Union[float, np.ndarray] = -0.055,
        vfVReset: Union[float, np.ndarray] = -0.065,
        vfVRest: Union[float, np.ndarray] = -0.065,
        tRefractoryTime = 0,
        strName: str = "unnamed",
        bRecord: bool = False,
        nMaxNumTimeSteps: int = nDefaultMaxNumTimeSteps,
    ):
        """
        FFIAFTorch - Construct a spiking feedforward layer with IAF neurons, running on GPU, using torch
                     Inputs are continuous currents; outputs are spiking events

        :param mfW:             np.array MxN weight matrix.
        :param vfBias:          np.array Nx1 bias vector. Default: 10mA

        :param tDt:             float Time-step. Default: 0.1 ms
        :param fNoiseStd:       float Noise std. dev. per second. Default: 0

        :param vtTauN:          np.array Nx1 vector of neuron time constants. Default: 20ms

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param vfVReset:        np.array Nx1 vector of neuron thresholds. Default: -65mV
        :param vfVRest:         np.array Nx1 vector of neuron thresholds. Default: -65mV

        :param tRefractoryTime: float Refractory period after each spike. Default: 0ms

        :param strName:         str Name for the layer. Default: 'unnamed'

        :param bRecord:         bool Record membrane potential during evolutions

        :nMaxNumTimeSteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        # - Call super constructor
        super().__init__(
            mfW=np.asarray(mfW), tDt=tDt, fNoiseStd=fNoiseStd, strName=strName
        )

        # - Set device to cuda if available and determine how tensors should be instantiated
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.tensors = torch.cuda
        else:
            self.device = torch.device("cpu")
            print("Layer `{}`: Using CPU as CUDA is not available.".format(strName))
            self.tensors = torch

        # - Record neuron parameters
        self.vfVThresh = vfVThresh
        self.vfVReset = vfVReset
        self.vfVRest = vfVRest
        self.vtTauN = vtTauN
        self.vfBias = vfBias
        self.mfW = mfW
        self.bRecord = bRecord
        self.nMaxNumTimeSteps = nMaxNumTimeSteps
        self.tRefractoryTime = tRefractoryTime

        # - Store "reset" state
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
        evolve : Function to evolve the states of this layer given an input. Automatically splits evolution in batches,

        :param tsSpkInput:      TSContinuous  Input spike trian
        :param tDuration:       float    Simulation/Evolution time
        :param nNumTimeSteps:   int      Number of evolution time steps
        :param bVerbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        # - Layer time step before evolution
        nTimeStepStart = self._nTimeStep

        # - Prepare input signal
        mfInput, nNumTimeSteps = self._prepare_input(tsInput, tDuration, nNumTimeSteps)

        # - Tensor for collecting output spike raster
        mbSpiking = torch.ByteTensor(nNumTimeSteps, self.nSize).fill_(0)

        # - Tensor for recording states
        if self.bRecord:
            self.mfRecordStates = (
                self.tensors.FloatTensor(2 * nNumTimeSteps + 1, self.nSize)
                .fill_(0)
                .cpu()
            )
            self.mfRecordSynapses = (
                self.tensors.FloatTensor(nNumTimeSteps + 1, self.nSize).fill_(0).cpu()
            )
            self.mfRecordStates[0] = self._vState
            self.mfRecordSynapses[0] = self._vSynapseState

        # - Iterate over batches and run evolution
        iCurrentIndex = 0
        for mfCurrentInput, nCurrNumTS in self._batch_data(
            mfInput, nNumTimeSteps, self.nMaxNumTimeSteps
        ):
            mbSpiking[
                iCurrentIndex : iCurrentIndex + nCurrNumTS
            ] = self._single_batch_evolution(
                mfCurrentInput, iCurrentIndex, nCurrNumTS, bVerbose
            )
            iCurrentIndex += nCurrNumTS

        # - Store recorded states in timeseries
        if self.bRecord:
            vtRecTimesStates = np.repeat(
                (nTimeStepStart + np.arange(nNumTimeSteps + 1)) * self.tDt, 2
            )[1:]
            vtRecTimesSynapses = (
                nTimeStepStart + np.arange(nNumTimeSteps + 1)
            ) * self.tDt
            self.tscRecStates = TSContinuous(
                vtRecTimesStates, self.mfRecordStates.numpy()
            )
            self.tscRecSynapses = TSContinuous(
                vtRecTimesSynapses, self.mfRecordSynapses.numpy()
            )

        # - Start and stop times for output time series
        tStart = nTimeStepStart * self.tDt
        tStop = (nTimeStepStart + nNumTimeSteps) * self.tDt

        # - Output timeseries
        if (mbSpiking == 0).all():
            tseOut = TSEvent(
                None, tStart=tStart, tStop=tStop, nNumChannels=self.nSize,
            )
        else:
            vnSpikeTimeIndices, vnChannels = torch.nonzero(mbSpiking).t()
            vtSpikeTimings = (
                nTimeStepStart + vnSpikeTimeIndices + 1
            ).float() * self.tDt

            tseOut = TSEvent(
                vtTimeTrace=np.clip(vtSpikeTimings.numpy(), tStart, tStop-fTolAbs*10**6),  # Clip due to possible numerical errors
                vnChannels=vnChannels.numpy(),
                nNumChannels=self.nSize,
                strName="Layer `{}` spikes".format(self.strName),
                tStart=tStart,
                tStop=tStop,
            )

        return tseOut

    # @profile
    def _batch_data(
        self, mfInput: np.ndarray, nNumTimeSteps: int, nMaxNumTimeSteps: int = None
    ) -> (np.ndarray, int):
        """_batch_data: Generator that returns the data in batches"""
        # - Handle None for nMaxNumTimeSteps
        nMaxNumTimeSteps = (
            nNumTimeSteps if nMaxNumTimeSteps is None else nMaxNumTimeSteps
        )
        nStart = 0
        while nStart < nNumTimeSteps:
            # - Endpoint of current batch
            nEnd = min(nStart + nMaxNumTimeSteps, nNumTimeSteps)
            # - Data for current batch
            mfCurrentInput = mfInput[nStart:nEnd]
            yield mfCurrentInput, nEnd - nStart
            # - Update nStart
            nStart = nEnd

    # @profile
    def _single_batch_evolution(
        self,
        mfInput: np.ndarray,
        nEvolutionTimeStep: int,
        nNumTimeSteps: Optional[int] = None,
        bVerbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input for a single batch

        :param mfInput:     np.ndarray   Input to layer as matrix
        :param nEvolutionTimeStep int    Time step within current evolution at beginning of current batch
        :param nNumTimeSteps:   int      Number of evolution time steps
        :param bVerbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        # - Get synapse input to neurons
        mfNeuralInput, nNumTimeSteps = self._prepare_neural_input(
            mfInput, nNumTimeSteps
        )

        # - Update synapse state to end of evolution before rest potential and bias are added to input
        self._vSynapseState = mfNeuralInput[-1].clone()

        if self.bRecord:
            # - Tensor for recording synapse and neuron states
            mfRecordStates = self.tensors.FloatTensor(
                2 * nNumTimeSteps, self.nSize
            ).fill_(0)
            # - Store synapse states
            self.mfRecordSynapses[
                nEvolutionTimeStep + 1 : nEvolutionTimeStep + nNumTimeSteps + 1
            ] = mfNeuralInput.cpu()

        # - Tensor for collecting spike data
        mbSpiking = self.tensors.ByteTensor(nNumTimeSteps, self.nSize).fill_(0)

        # - Get local variables
        vState = self._vState.clone()
        vfAlpha = self._vfAlpha
        vfVThresh = self._vfVThresh
        vfVReset = self._vfVReset
        bRecord = self.bRecord
        nRefractorySteps = self._nRefractorySteps
        vnRefractoryCountdownSteps = self._vnRefractoryCountdownSteps.clone()


        # - Include resting potential and bias in input for fewer computations
        mfNeuralInput += self._vfVRest + self._vfBias

        # - Evolve neuron states
        for nStep in range(nNumTimeSteps):
            # - Determine refractory neurons
            vbNotRefractory = (vnRefractoryCountdownSteps == 0).float()
            # - Decrement refractory countdown
            vnRefractoryCountdownSteps -= 1
            vnRefractoryCountdownSteps.clamp_(min=0)
            # - Incremental state update from input
            vState += vfAlpha * (mfNeuralInput[nStep] - vState) * vbNotRefractory
            # - Store updated state before spike
            if bRecord:
                mfRecordStates[2 * nStep] = vState
            # - Spiking
            vbSpiking = (vState > vfVThresh).float()
            # - State reset
            vState += (vfVReset - vState) * vbSpiking
            # - Store spikes
            mbSpiking[nStep] = vbSpiking
            # - Update refractory countdown
            vnRefractoryCountdownSteps += nRefractorySteps * vbSpiking
            # - Store updated state after spike
            if bRecord:
                mfRecordStates[2 * nStep + 1] = vState
            del vbSpiking

        # - Store recorded neuron states
        if bRecord:
            self.mfRecordStates[
                2 * nEvolutionTimeStep
                + 1 : 2 * (nEvolutionTimeStep + nNumTimeSteps)
                + 1
            ] = mfRecordStates.cpu()

        # - Store updated state and update clock
        self._vState = vState
        self._vnRefractoryCountdownSteps = vnRefractoryCountdownSteps
        self._nTimeStep += nNumTimeSteps

        return mbSpiking.cpu()

    # @profile
    def _prepare_neural_input(
        self, mfInput: np.array, nNumTimeSteps: Optional[int] = None
    ) -> (np.ndarray, int):
        """
        _prepare_neural_input : Prepare the weighted, noisy synaptic input to the neurons
                                and return it together with number of evolution time steps

        :param mfInput          np.ndarray  External input spike raster
        :param nNumTimeSteps    int      Number of evolution time steps
        :return:
                mfNeuralInput   np.ndarray  Input to neurons
                nNumTimeSteps   int         Number of evolution time steps

        """
        # - Prepare mfInput
        mfInput = torch.from_numpy(mfInput).float().to(self.device)
        # - Weight inputs
        mfNeuralInput = torch.mm(mfInput, self._mfW)

        # - Add noise trace
        if self.fNoiseStd > 0:
            mfNeuralInput += (
                torch.randn(nNumTimeSteps, self.nSize).float().to(self.device)
                # - Standard deviation slightly smaller than expected (due to brian??),
                #   therefore correct with empirically found factor 1.63
                * self.fNoiseStd
                * torch.sqrt(2. * self._vtTauN / self.tDt)
                * 1.63
            )

        return mfNeuralInput, nNumTimeSteps

    # @profile
    def _prepare_input(
        self,
        tsInput: TSContinuous = None,
        tDuration: float = None,
        nNumTimeSteps: int = None,
    ) -> (torch.Tensor, int):
        """
        _prepare_input - Sample input, set up time base

        :param tsInput:       TimeSeries TxM or Tx1 Input signals for this layer
        :param tDuration:     float Duration of the desired evolution, in seconds
        :param nNumTimeSteps: int Number of evolution time steps

        :return: (vtTimeBase, mfInputStep, tDuration)
            mfInputStep:    ndarray (T1xN) Discretised input signal for layer
            nNumTimeSteps:  int Actual number of evolution time steps
        """
        if nNumTimeSteps is None:
            # - Determine nNumTimeSteps
            if tDuration is None:
                # - Determine tDuration
                assert (
                    tsInput is not None
                ), "Layer `{}`: One of `nNumTimeSteps`, `tsInput` or `tDuration` must be supplied".format(
                    self.strName
                )

                if tsInput.bPeriodic:
                    # - Use duration of periodic TimeSeries, if possible
                    tDuration = tsInput.tDuration

                else:
                    # - Evolve until the end of the input TImeSeries
                    tDuration = tsInput.tStop - self.t
                    assert tDuration > 0, (
                        "Layer `{}`: Cannot determine an appropriate evolution duration.".format(
                            self.strName
                        )
                        + " `tsInput` finishes before the current evolution time."
                    )
            nNumTimeSteps = int(np.floor((tDuration + fTolAbs) / self.tDt))
        else:
            assert isinstance(
                nNumTimeSteps, int
            ), "Layer `{}`: nNumTimeSteps must be of type int.".format(self.strName)

        # - Generate discrete time base
        vtTimeBase = self._gen_time_trace(self.t, nNumTimeSteps)

        if tsInput is not None:
            # - Make sure vtTimeBase matches tsInput
            if not isinstance(tsInput, TSEvent):
                if not tsInput.bPeriodic:
                    # - If time base limits are very slightly beyond tsInput.tStart and tsInput.tStop, match them
                    if (
                        tsInput.tStart - 1e-3 * self.tDt
                        <= vtTimeBase[0]
                        <= tsInput.tStart
                    ):
                        vtTimeBase[0] = tsInput.tStart
                    if (
                        tsInput.tStop
                        <= vtTimeBase[-1]
                        <= tsInput.tStop + 1e-3 * self.tDt
                    ):
                        vtTimeBase[-1] = tsInput.tStop

                # - Warn if evolution period is not fully contained in tsInput
                if not (tsInput.contains(vtTimeBase) or tsInput.bPeriodic):
                    print(
                        "Layer `{}`: Evolution period (t = {} to {}) ".format(
                            self.strName, vtTimeBase[0], vtTimeBase[-1]
                        )
                        + "not fully contained in input signal (t = {} to {})".format(
                            tsInput.tStart, tsInput.tStop
                        )
                    )

            # - Sample input trace and check for correct dimensions
            mfInputStep = self._check_input_dims(tsInput(vtTimeBase))

            # - Treat "NaN" as zero inputs
            mfInputStep[np.isnan(mfInputStep)] = 0

        else:
            # - Assume zero inputs
            mfInputStep = np.zeros((nNumTimeSteps, self.nSizeIn))

        return (mfInputStep, nNumTimeSteps)

    def reset_state(self):
        """ .reset_state() - Method: reset the internal state of the layer
            Usage: .reset_state()
        """
        self.vState = self.vfVReset
        self.vSynapseState = 0
        self._vnRefractoryCountdownSteps = torch.zeros(self.nSize).to(self.device)

    ### --- Properties

    @property
    def cOutput(self):
        return TSEvent

    @property
    def tRefractoryTime(self):
        return self._nRefractorySteps * self.tDt

    @tRefractoryTime.setter
    def tRefractoryTime(self, tNewTime):
        self._nRefractorySteps = int(np.round(tNewTime / self.tDt))

    @property
    def vtRefractoryCountdown(self):
        return self._vnRefractoryCountdownSteps.cpu().numpy() * self.tDt

    @property
    def vState(self):
        return self._vState.cpu().numpy()

    @vState.setter
    def vState(self, vNewState):
        vNewState = np.asarray(self._expand_to_net_size(vNewState, "vState"))
        self._vState = torch.from_numpy(vNewState).to(self.device).float()

    @property
    def vtTauN(self):
        return self._vtTauN.cpu().numpy()

    @vtTauN.setter
    def vtTauN(self, vtNewTauN):
        vtNewTauN = np.asarray(self._expand_to_net_size(vtNewTauN, "vtTauN"))
        self._vtTauN = torch.from_numpy(vtNewTauN).to(self.device).float()
        if (self.tDt >= self._vtTauN).any():
            print(
                "Layer `{}`: tDt is larger than some of the vtTauN. This can cause numerical instabilities.".format(
                    self.strName
                )
            )

    @property
    def vfAlpha(self):
        return self._vfAlpha.cpu().numpy()

    @property
    def _vfAlpha(self):
        return self.tDt / self._vtTauN

    @property
    def vfBias(self):
        return self._vfBias.cpu().numpy()

    @vfBias.setter
    def vfBias(self, vfNewBias):
        vfNewBias = np.asarray(self._expand_to_net_size(vfNewBias, "vfBias"))
        self._vfBias = torch.from_numpy(vfNewBias).to(self.device).float()

    @property
    def vfVThresh(self):
        return self._vfVThresh.cpu().numpy()

    @vfVThresh.setter
    def vfVThresh(self, vfNewVThresh):
        vfNewVThresh = np.asarray(self._expand_to_net_size(vfNewVThresh, "vfVThresh"))
        self._vfVThresh = torch.from_numpy(vfNewVThresh).to(self.device).float()

    @property
    def vfVRest(self):
        return self._vfVRest.cpu().numpy()

    @vfVRest.setter
    def vfVRest(self, vfNewVRest):
        vfNewVRest = np.asarray(self._expand_to_net_size(vfNewVRest, "vfVRest"))
        self._vfVRest = torch.from_numpy(vfNewVRest).to(self.device).float()

    @property
    def vfVReset(self):
        return self._vfVReset.cpu().numpy()

    @vfVReset.setter
    def vfVReset(self, vfNewVReset):
        vfNewVReset = np.asarray(self._expand_to_net_size(vfNewVReset, "vfVReset"))
        self._vfVReset = torch.from_numpy(vfNewVReset).to(self.device).float()

    @property
    def vSynapseState(self):
        return self._vSynapseState.cpu().numpy()

    @vSynapseState.setter
    def vSynapseState(self, vfNewState):
        vfNewState = np.asarray(self._expand_to_net_size(vfNewState, "vSynapseState"))
        self._vSynapseState = torch.from_numpy(vfNewState).to(self.device).float()

    @property
    def t(self):
        return self._nTimeStep * self.tDt

    @property
    def mfW(self):
        return self._mfW.cpu().numpy()

    @mfW.setter
    def mfW(self, mfNewW):
        mfNewW = self._expand_to_shape(
            mfNewW, (self.nSizeIn, self.nSize), "mfW", bAllowNone=False
        )
        self._mfW = torch.from_numpy(mfNewW).to(self.device).float()


# - FFIAFSpkInTorch - Class: Spiking feedforward layer with spiking in- and outputs
class FFIAFSpkInTorch(FFIAFTorch):
    """ FFIAFSpkInTorch - Class: Spiking feedforward layer with spiking in- and outputs
    """

    ## - Constructor
    def __init__(
        self,
        mfW: np.ndarray,
        vfBias: np.ndarray = 0.01,
        tDt: float = 0.0001,
        fNoiseStd: float = 0,
        vtTauN: np.ndarray = 0.02,
        vtTauS: np.ndarray = 0.02,
        vfVThresh: np.ndarray = -0.055,
        vfVReset: np.ndarray = -0.065,
        vfVRest: np.ndarray = -0.065,
        tRefractoryTime=0,
        strName: str = "unnamed",
        bRecord: bool = False,
        nMaxNumTimeSteps: int = nDefaultMaxNumTimeSteps,
    ):
        """
        FFIAFSpkInTorch - Construct a spiking feedforward layer with IAF neurons, running on GPU, using torch
                          in- and outputs are spiking events

        :param mfW:             np.array MxN weight matrix.
        :param vfBias:          np.array Nx1 bias vector. Default: 10mA

        :param tDt:             float Time-step. Default: 0.1 ms
        :param fNoiseStd:       float Noise std. dev. per second. Default: 0

        :param vtTauN:          np.array Nx1 vector of neuron time constants. Default: 20ms
        :param vtTauS:          np.array Nx1 vector of synapse time constants. Default: 20ms

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param vfVReset:        np.array Nx1 vector of neuron thresholds. Default: -65mV
        :param vfVRest:         np.array Nx1 vector of neuron thresholds. Default: -65mV

        :param tRefractoryTime: float Refractory period after each spike. Default: 0ms

        :param strName:         str Name for the layer. Default: 'unnamed'

        :param bRecord:         bool Record membrane potential during evolutions

        :nMaxNumTimeSteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        # - Call super constructor
        super().__init__(
            mfW=mfW,
            vfBias=vfBias,
            tDt=tDt,
            fNoiseStd=fNoiseStd,
            vtTauN=vtTauN,
            vfVThresh=vfVThresh,
            vfVReset=vfVReset,
            vfVRest=vfVRest,
            tRefractoryTime=tRefractoryTime,
            strName=strName,
            bRecord=bRecord,
            nMaxNumTimeSteps=nMaxNumTimeSteps,
        )

        # - Record neuron parameters
        self.vtTauS = vtTauS

    # @profile
    def _prepare_neural_input(
        self, mfInput: np.array, nNumTimeSteps: Optional[int] = None
    ) -> (np.ndarray, int):
        """
        _prepare_neural_input : Prepare the weighted, noisy synaptic input to the neurons
                                and return it together with number of evolution time steps

        :param mfInput:     np.ndarray   Input to layer as matrix
        :param nNumTimeSteps    int      Number of evolution time steps
        :return:
                mfNeuralInput   np.ndarray  Input to neurons
                nNumTimeSteps   int         Number of evolution time steps

        """
        # - Prepare mfInput
        mfInput = torch.from_numpy(mfInput).float().to(self.device)
        # - Weight inputs
        mfWeightedInput = torch.mm(mfInput, self._mfW)

        # - Add noise trace
        if self.fNoiseStd > 0:
            mfWeightedInput += (
                torch.randn(nNumTimeSteps, self.nSize).float().to(self.device)
                # - Standard deviation slightly smaller than expected (due to brian??),
                #   therefore correct with empirically found factor 1.63
                * self.fNoiseStd
                * torch.sqrt(2. * self._vtTauN / self.tDt)
                * 1.63
            )

        # - Include previous synaptic states
        mfWeightedInput[0] = self._vSynapseState * torch.exp(-self.tDt / self._vtTauS)

        # - Reshape input for convolution
        mfWeightedInput = mfWeightedInput.t().reshape(1, self.nSize, -1)

        # - Apply exponential filter to input
        vtTimes = (
            torch.arange(nNumTimeSteps).to(self.device).reshape(1, -1).float()
            * self.tDt
        )
        mfKernels = torch.exp(-vtTimes / self._vtTauS.reshape(-1, 1))
        # - Reverse on time axis and reshape to match convention of pytorch
        mfKernels = mfKernels.flip(1).reshape(self.nSize, 1, nNumTimeSteps)
        # - Object for applying convolution
        convSynapses = torch.nn.Conv1d(
            self.nSize,
            self.nSize,
            nNumTimeSteps,
            padding=nNumTimeSteps - 1,
            groups=self.nSize,
            bias=False,
        ).to(self.device)
        convSynapses.weight.data = mfKernels
        # - Filtered synaptic currents
        mfNeuralInput = convSynapses(mfWeightedInput)[0].detach().t()[:nNumTimeSteps]

        return mfNeuralInput, nNumTimeSteps

    # @profile
    def _prepare_input(
        self,
        tsInput: Optional[TSEvent] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
    ) -> (np.ndarray, int):
        """
        _prepare_input - Sample input, set up time base

        :param tsInput:      TimeSeries TxM or Tx1 Input signals for this layer
        :param tDuration:    float Duration of the desired evolution, in seconds
        :param nNumTimeSteps int Number of evolution time steps

        :return:
            mfSpikeRaster:    ndarray Boolean raster containing spike info
            nNumTimeSteps:    ndarray Number of evlution time steps
        """
        if nNumTimeSteps is None:
            # - Determine nNumTimeSteps
            if tDuration is None:
                # - Determine tDuration
                assert (
                    tsInput is not None
                ), "Layer {}: One of `nNumTimeSteps`, `tsInput` or `tDuration` must be supplied".format(
                    self.strName
                )

                if tsInput.bPeriodic:
                    # - Use duration of periodic TimeSeries, if possible
                    tDuration = tsInput.tDuration

                else:
                    # - Evolve until the end of the input TImeSeries
                    tDuration = tsInput.tStop - self.t
                    assert tDuration > 0, (
                        "Layer {}: Cannot determine an appropriate evolution duration.".format(
                            self.strName
                        )
                        + "`tsInput` finishes before the current "
                        "evolution time."
                    )
            # - Discretize tDuration wrt self.tDt
            nNumTimeSteps = int(np.floor((tDuration + fTolAbs) / self.tDt))
        else:
            assert isinstance(
                nNumTimeSteps, int
            ), "Layer `{}`: nNumTimeSteps must be of type int.".format(self.strName)

        # - Extract spike timings and channels
        if tsInput is not None:
            # Extract spike data from the input variable
            __, __, mfSpikeRaster, __ = tsInput.raster(
                tDt=self.tDt,
                tStart=self.t,
                tStop=(self._nTimeStep + nNumTimeSteps) * self._tDt,
                vnSelectChannels=np.arange(self.nSizeIn),
            )
            # - Convert to supported format
            mfSpikeRaster = mfSpikeRaster.astype(int)
            # - Make sure size is correct
            mfSpikeRaster = mfSpikeRaster[:nNumTimeSteps, :]

        else:
            mfSpikeRaster = np.zeros((nNumTimeSteps, self.nSizeIn))

        return mfSpikeRaster, nNumTimeSteps

    @property
    def cInput(self):
        return TSEvent

    @property
    def vtTauS(self):
        return self._vtTauS.cpu().numpy()

    @vtTauS.setter
    def vtTauS(self, vtNewTauS):
        vtNewTauS = np.asarray(self._expand_to_net_size(vtNewTauS, "vtTauS"))
        self._vtTauS = torch.from_numpy(vtNewTauS).to(self.device).float()


## - RecIAFTorch - Class: define a spiking recurrent layer with spiking outputs
class RecIAFTorch(FFIAFTorch):
    """ FFIAFTorch - Class: define a spiking recurrent layer with spiking outputs
    """

    ## - Constructor
    def __init__(
        self,
        mfW: np.ndarray,
        vfBias: Union[float, np.ndarray] = 0.015,
        tDt: float = 0.0001,
        fNoiseStd: float = 0,
        vtTauN: Union[float, np.ndarray] = 0.02,
        vtTauSynR: Union[float, np.ndarray] = 0.05,
        vfVThresh: Union[float, np.ndarray] = -0.055,
        vfVReset: Union[float, np.ndarray] = -0.065,
        vfVRest: Union[float, np.ndarray] = -0.065,
        tRefractoryTime=0,
        strName: str = "unnamed",
        bRecord: bool = False,
        nMaxNumTimeSteps: int = nDefaultMaxNumTimeSteps,
    ):
        """
        FFIAFTorch - Construct a spiking recurrent layer with IAF neurons, running on GPU, using torch
                     Inputs are continuous currents; outputs are spiking events

        :param mfW:             np.array MxN weight matrix.
        :param vfBias:          np.array Nx1 bias vector. Default: 0.015

        :param tDt:             float Time-step. Default: 0.0001
        :param fNoiseStd:       float Noise std. dev. per second. Default: 0

        :param vtTauN:          np.array Nx1 vector of neuron time constants. Default: 0.02
        :param vtTauSynR:       np.array NxN vector of recurrent synaptic time constants. Default: 0.005

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds. Default: -0.055
        :param vfVReset:        np.array Nx1 vector of neuron thresholds. Default: -0.065
        :param vfVRest:         np.array Nx1 vector of neuron thresholds. Default: -0.065

        :param tRefractoryTime: float Refractory period after each spike. Default: 0

        :param strName:         str Name for the layer. Default: 'unnamed'

        :param bRecord:         bool Record membrane potential during evolutions. Default: False

        :nMaxNumTimeSteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        assert (
            np.atleast_2d(mfW).shape[0] == np.atleast_2d(mfW).shape[1]
        ), "Layer `{}`: mfW must be a square matrix.".format(strName)

        # - Call super constructor
        super().__init__(
            mfW=mfW,
            vfBias=vfBias,
            tDt=tDt,
            fNoiseStd=fNoiseStd,
            vtTauN=vtTauN,
            vfVThresh=vfVThresh,
            vfVReset=vfVReset,
            vfVRest=vfVRest,
            tRefractoryTime=tRefractoryTime,
            strName=strName,
            bRecord=bRecord,
            nMaxNumTimeSteps=nMaxNumTimeSteps,
        )

        # - Record neuron parameters
        self.vtTauSynR = vtTauSynR

    # @profile
    def _single_batch_evolution(
        self,
        mfInput: np.ndarray,
        nEvolutionTimeStep: int,
        nNumTimeSteps: Optional[int] = None,
        bVerbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input for a single batch

        :param mfInput:     np.ndarray   Input to layer as matrix
        :param nEvolutionTimeStep int    Time step within current evolution at beginning of current batch
        :param nNumTimeSteps:   int      Number of evolution time steps
        :param bVerbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        mfNeuralInput, nNumTimeSteps = self._prepare_neural_input(
            mfInput, nNumTimeSteps
        )

        if self.bRecord:
            # - Tensor for recording synapse and neuron states
            mfRecordStates = self.tensors.FloatTensor(
                2 * nNumTimeSteps, self.nSize
            ).fill_(0)

        # - Tensor for collecting spike data
        mbSpiking = self.tensors.ByteTensor(nNumTimeSteps, self.nSize).fill_(0)

        # - Get local variables
        vState = self._vState.clone()
        vfAlpha = self._vfAlpha
        vfVThresh = self._vfVThresh
        vfVReset = self._vfVReset
        bRecord = self.bRecord
        mfKernels = self._mfKernelsRec
        nNumTSKernel = mfKernels.shape[0]
        mfWRec = self._mfW
        nRefractorySteps = self._nRefractorySteps
        vnRefractoryCountdownSteps = self._vnRefractoryCountdownSteps.clone()

        # - Include resting potential and bias in input for fewer computations
        # - Omit latest time point, which is only used for carrying over synapse state to new batch
        mfNeuralInput[:-1] += self._vfVRest + self._vfBias

        # - Evolve neuron states
        for nStep in range(nNumTimeSteps):
            # - Determine refractory neurons
            vbNotRefractory = (vnRefractoryCountdownSteps == 0).float()
            # - Decrement refractory countdown
            vnRefractoryCountdownSteps -= 1
            vnRefractoryCountdownSteps.clamp_(min=0)
            # - Incremental state update from input
            vState += vfAlpha * (mfNeuralInput[nStep] - vState) * vbNotRefractory
            # - Store updated state before spike
            if bRecord:
                mfRecordStates[2 * nStep] = vState
            # - Spiking
            vbSpiking = (vState > vfVThresh).float()
            # - State reset
            vState += (vfVReset - vState) * vbSpiking
            # - Store spikes
            mbSpiking[nStep] = vbSpiking
            # - Update refractory countdown
            vnRefractoryCountdownSteps += nRefractorySteps * vbSpiking
            # - Store updated state after spike
            if bRecord:
                mfRecordStates[2 * nStep + 1] = vState
            # - Add filtered recurrent spikes to input
            nTSRecurrent = min(nNumTSKernel, nNumTimeSteps - nStep)
            mfNeuralInput[nStep + 1 : nStep + 1 + nTSRecurrent] += mfKernels[
                :nTSRecurrent
            ] * torch.mm(vbSpiking.reshape(1, -1), mfWRec)

            del vbSpiking

        # - Store recorded neuron and synapse states
        if bRecord:
            self.mfRecordStates[
                2 * nEvolutionTimeStep
                + 1 : 2 * (nEvolutionTimeStep + nNumTimeSteps)
                + 1
            ] = mfRecordStates.cpu()
            self.mfRecordSynapses[
                nEvolutionTimeStep + 1 : nEvolutionTimeStep + nNumTimeSteps + 1
            ] = (
                mfNeuralInput[:nNumTimeSteps]
                - self._vfVRest
                - self._vfBias  # Introduces slight numerical error in stored synapses of about 1e-9
            ).cpu()

        # - Store updated neuron and synapse states and update clock
        self._vState = vState
        self._vSynapseState = mfNeuralInput[-1].clone()
        self._nTimeStep += nNumTimeSteps

        return mbSpiking.cpu()

    # @profile
    def _prepare_neural_input(
        self, mfInput: np.array, nNumTimeSteps: Optional[int] = None
    ) -> (np.ndarray, int):
        """
        _prepare_neural_input : Prepare the noisy synaptic input to the neurons
                                and return it together with number of evolution time steps

        :param tsSpkInput:      TSContinuous  Input spike trian
        :param tDuration:       float    Simulation/Evolution time
        :param nNumTimeSteps    int      Number of evolution time steps
        :return:
                mfNeuralInput   np.ndarray  Input to neurons
                nNumTimeSteps   int         Number of evolution time steps

        """

        nNumTimeSteps = mfInput.shape[0] if nNumTimeSteps is None else nNumTimeSteps

        # - Prepare mfInput, with additional time step for carrying over recurrent spikes between batches
        mfNeuralInput = self.tensors.FloatTensor(nNumTimeSteps + 1, self.nSize).fill_(0)
        mfNeuralInput[:-1] = torch.from_numpy(mfInput).float()
        # - Carry over filtered recurrent spikes from previous batch
        nTSRecurrent = min(mfNeuralInput.shape[0], self._mfKernelsRec.shape[0])
        mfNeuralInput[:nTSRecurrent] += (
            self._mfKernelsRec[:nTSRecurrent] * self._vSynapseState
        )

        # - Add noise trace
        if self.fNoiseStd > 0:
            mfNeuralInput += (
                torch.randn(nNumTimeSteps + 1, self.nSize).float().to(self.device)
                # - Standard deviation slightly smaller than expected (due to brian??),
                #   therefore correct with empirically found factor 1.63
                * self.fNoiseStd
                * torch.sqrt(2. * self._vtTauN / self.tDt)
                * 1.63
            )

        return mfNeuralInput, nNumTimeSteps

    @property
    def vtTauSynR(self):
        return self._vtTauSynR.cpu().numpy()

    @vtTauSynR.setter
    def vtTauSynR(self, vtNewTauSynR):
        vtNewTauSynR = np.asarray(self._expand_to_net_size(vtNewTauSynR, "vtTauSynR"))
        if (vtNewTauSynR < self.tDt).any():
            print(
                "Layer `{}`: tDt is larger than some of the vtTauSynR. This can cause numerical instabilities.".format(
                    self.strName
                )
            )

        self._vtTauSynR = torch.from_numpy(vtNewTauSynR).to(self.device).float()

        # - Kernel for filtering recurrent spikes
        nKernelSize = 50 * int(
            np.amax(vtNewTauSynR) / self.tDt
        )  # - Values smaller than ca. 1e-21 are neglected
        vtTimes = (
            torch.arange(nKernelSize).to(self.device).reshape(-1, 1).float() * self.tDt
        )
        self._mfKernelsRec = torch.exp(-vtTimes / self._vtTauSynR.reshape(1, -1))

    @property
    def tDt(self):
        return self._tDt

    @tDt.setter
    def tDt(self, tNewDt):
        assert tNewDt > 0, "Layer `{}`: tDt must be greater than 0.".format(
            self.strName
        )
        self._tDt = tNewDt
        if hasattr(self, "vtTauSynR"):
            # - Update filter for recurrent spikes if already exists
            self.vtTauSynR = self.vtTauSynR


## - RecIAFSpkInTorch - Class: define a spiking recurrent layer with spiking in- and outputs
class RecIAFSpkInTorch(RecIAFTorch):
    """ RecIAFSpkInTorch - Class: define a spiking recurrent layer with spiking in- and outputs
    """

    ## - Constructor
    def __init__(
        self,
        mfWIn: np.ndarray,
        mfWRec: np.ndarray,
        vfBias: Union[float, np.ndarray] = 0.0105,
        tDt: float = 0.0001,
        fNoiseStd: float = 0,
        vtTauN: Union[float, np.ndarray] = 0.02,
        vtTauSInp: Union[float, np.ndarray] = 0.05,
        vtTauSRec: Union[float, np.ndarray] = 0.05,
        vfVThresh: Union[float, np.ndarray] = -0.055,
        vfVReset: Union[float, np.ndarray] = -0.065,
        vfVRest: Union[float, np.ndarray] = -0.065,
        tRefractoryTime=0,
        strName: str = "unnamed",
        bRecord: bool = False,
        bAddEvents: bool = True,
        nMaxNumTimeSteps: int = nDefaultMaxNumTimeSteps,
    ):
        """
        RecIAFSpkInTorch - Construct a spiking recurrent layer with IAF neurons, running on GPU, using torch
                           Inputs and outputs are spiking events

        :param mfWIn:           np.array MxN input weight matrix.
        :param mfWRec:          np.array NxN recurrent weight matrix.
        :param vfBias:          np.array Nx1 bias vector. Default: 0.0105

        :param tDt:             float Time-step. Default: 0.0001
        :param fNoiseStd:       float Noise std. dev. per second. Default: 0

        :param vtTauN:          np.array Nx1 vector of neuron time constants. Default: 0.02
        :param vtTauSInp:       np.array Nx1 vector of synapse time constants. Default: 0.05
        :param vtTauSRec:       np.array Nx1 vector of synapse time constants. Default: 0.05

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds. Default: -0.055
        :param vfVReset:        np.array Nx1 vector of neuron thresholds. Default: -0.065
        :param vfVRest:         np.array Nx1 vector of neuron thresholds. Default: -0.065

        :param tRefractoryTime: float Refractory period after each spike. Default: 0

        :param strName:         str Name for the layer. Default: 'unnamed'

        :param bRecord:         bool Record membrane potential during evolutions. Default: False

        :bAddEvents:            bool     If during evolution multiple input events arrive during one
                                         time step for a channel, count their actual number instead of
                                         just counting them as one (This might make less sense for
                                         refractory neurons).

        :nMaxNumTimeSteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        # - Call super constructor
        Layer.__init__(self, mfW=mfWIn, tDt=tDt, fNoiseStd=fNoiseStd, strName=strName)

        # - Set device to cuda if available and determine how tensors should be instantiated
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.tensors = torch.cuda
        else:
            self.device = torch.device("cpu")
            print("Layer `{}`: Using CPU as CUDA is not available.".format(strName))
            self.tensors = torch

        # - Bypass property setter to avoid unnecessary convolution kernel update
        assert (
            type(nMaxNumTimeSteps) == int and nMaxNumTimeSteps > 0.
        ), "Layer `{}`: nMaxNumTimeSteps must be an integer greater than 0.".format(
            self.strName
        )
        self._nMaxNumTimeSteps = nMaxNumTimeSteps

        # - Record neuron parameters
        self.vfVThresh = vfVThresh
        self.vfVReset = vfVReset
        self.vfVRest = vfVRest
        self.vtTauN = vtTauN
        self.vtTauSRec = vtTauSRec
        self.vtTauSInp = vtTauSInp
        self.vfBias = vfBias
        self.mfWIn = mfWIn
        self.mfWRec = mfWRec
        self.bRecord = bRecord
        self.bAddEvents = bAddEvents
        self.tRefractoryTime = tRefractoryTime

        # - Store "reset" state
        self.reset_all()
        
    # @profile
    def _prepare_neural_input(
        self, mfInput: np.array, nNumTimeSteps: Optional[int] = None
    ) -> (np.ndarray, int):
        """
        _prepare_neural_input : Prepare the noisy synaptic input to the neurons
                                and return it together with number of evolution time steps

        :param mfInput          np.ndarray  External input spike raster
        :param nNumTimeSteps    int         Number of evolution time steps
        :return:
                mfNeuralInput   np.ndarray  Input to neurons
                nNumTimeSteps   int         Number of evolution time steps

        """

        nNumTimeSteps = mfInput.shape[0] if nNumTimeSteps is None else nNumTimeSteps

        # - Prepare external input
        mfInput = torch.from_numpy(mfInput).float().to(self.device)
        # - Weigh inputs
        mfWeightedInput = torch.mm(mfInput, self._mfWIn)
        # - Carry over external inputs from last batch
        mfWeightedInput[0] = self._vSynapseStateInp.clone() * torch.exp(
            -self.tDt / self._vtTauSInp
        )
        # - Reshape input for convolution
        mfWeightedInput = mfWeightedInput.t().reshape(1, self.nSize, -1)
        # - Apply exponential filter to external input
        vtTimes = (
            torch.arange(nNumTimeSteps).to(self.device).reshape(1, -1).float()
            * self.tDt
        )
        mfInputKernels = torch.exp(-vtTimes / self._vtTauSInp.reshape(-1, 1))
        # - Reverse on time axis and reshape to match convention of pytorch
        mfInputKernels = mfInputKernels.flip(1).reshape(self.nSize, 1, nNumTimeSteps)
        # - Object for applying convolution
        convSynapses = torch.nn.Conv1d(
            self.nSize,
            self.nSize,
            nNumTimeSteps,
            padding=nNumTimeSteps - 1,
            groups=self.nSize,
            bias=False,
        ).to(self.device)
        convSynapses.weight.data = mfInputKernels
        # - Filtered synaptic currents
        mfFilteredExternalInput = (
            convSynapses(mfWeightedInput)[0].detach().t()[:nNumTimeSteps]
        )
        # - Store filtered input from last time step for carry-over to next batch
        self._vSynapseStateInp = mfFilteredExternalInput[-1].clone()

        # - Prepare input to neurons, with additional time step for carrying over recurrent spikes between batches
        mfNeuralInput = self.tensors.FloatTensor(nNumTimeSteps + 1, self.nSize).fill_(0)
        # - Filtered external input
        mfNeuralInput[:-1] = mfFilteredExternalInput
        # - Carry over filtered recurrent spikes from previous batch
        nTSRecurrent = min(mfNeuralInput.shape[0], self._mfKernelsRec.shape[0])
        mfNeuralInput[:nTSRecurrent] += (
            self._mfKernelsRec[:nTSRecurrent] * self._vSynapseState
        )

        # - Add noise trace
        if self.fNoiseStd > 0:
            mfNeuralInput += (
                torch.randn(nNumTimeSteps + 1, self.nSize).float().to(self.device)
                # - Standard deviation slightly smaller than expected (due to brian??),
                #   therefore correct with empirically found factor 1.63
                * self.fNoiseStd
                * torch.sqrt(2. * self._vtTauN / self.tDt)
                * 1.63
            )

        return mfNeuralInput, nNumTimeSteps

    # @profile
    def _prepare_input(
        self,
        tsInput: Optional[TSEvent] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
    ) -> (np.ndarray, int):
        """
        _prepare_input - Sample input, set up time base

        :param tsInput:      TimeSeries TxM or Tx1 Input signals for this layer
        :param tDuration:    float Duration of the desired evolution, in seconds
        :param nNumTimeSteps int Number of evolution time steps

        :return:
            mnSpikeRaster:    Tensor Boolean raster containing spike info
            nNumTimeSteps:    ndarray Number of evlution time steps
        """
        if nNumTimeSteps is None:
            # - Determine nNumTimeSteps
            if tDuration is None:
                # - Determine tDuration
                assert (
                    tsInput is not None
                ), "Layer {}: One of `nNumTimeSteps`, `tsInput` or `tDuration` must be supplied".format(
                    self.strName
                )

                if tsInput.bPeriodic:
                    # - Use duration of periodic TimeSeries, if possible
                    tDuration = tsInput.tDuration

                else:
                    # - Evolve until the end of the input TImeSeries
                    tDuration = tsInput.tStop - self.t
                    assert tDuration > 0, (
                        "Layer {}: Cannot determine an appropriate evolution duration.".format(
                            self.strName
                        )
                        + "`tsInput` finishes before the current "
                        "evolution time."
                    )
            # - Discretize tDuration wrt self.tDt
            nNumTimeSteps = int(np.floor((tDuration + fTolAbs) / self.tDt))
        else:
            assert isinstance(
                nNumTimeSteps, int
            ), "Layer `{}`: nNumTimeSteps must be of type int.".format(self.strName)

        # - Extract spike timings and channels
        if tsInput is not None:
            # Extract spike data from the input variable
            __, __, mnSpikeRaster, __ = tsInput.raster(
                tDt=self.tDt,
                tStart=self.t,
                tStop=(self._nTimeStep + nNumTimeSteps) * self._tDt,
                vnSelectChannels=np.arange(
                    self.nSizeIn
                ),  ## This causes problems when tsInput has no events in some channels
                bAddEvents=self.bAddEvents,  # Allow for multiple input spikes per time step
            )
            # - Convert to supportedformat
            mnSpikeRaster = mnSpikeRaster.astype(int)
            # - Make sure size is correct
            mnSpikeRaster = mnSpikeRaster[:nNumTimeSteps, :]

        else:
            mnSpikeRaster = np.zeros((nNumTimeSteps, self.nSizeIn))

        return mnSpikeRaster, nNumTimeSteps

    def reset_all(self):
        super().reset_all()
        self.vSynapseStateInp = 0

    def _update_rec_kernel(self):
        # - Kernel for filtering recurrent spikes
        nKernelSize = min(
            50
            * int(
                torch.max(self._vtTauSRec) / self.tDt
            ),  # - Values smaller than ca. 1e-21 are neglected
            self._nMaxNumTimeSteps
            + 1,  # Kernel does not need to be larger than batch duration
        )
        vtTimes = (
            torch.arange(nKernelSize).to(self.device).reshape(-1, 1).float() * self.tDt
        )
        self._mfKernelsRec = torch.exp(-vtTimes / self._vtTauSRec.reshape(1, -1))
        print(
            "Layer `{}`: Recurrent filter kernels have been updated.".format(
                self.strName
            )
        )

    @property
    def cInput(self):
        return TSEvent

    @property
    def tDt(self):
        return self._tDt

    @tDt.setter
    def tDt(self, tNewDt):
        assert tNewDt > 0, "Layer `{}`: tDt must be greater than 0.".format(
            self.strName
        )
        self._tDt = tNewDt
        if hasattr(self, "vtTauSRec"):
            # - Update filter for recurrent spikes if already exists
            self.vtTauSRec = self.vtTauSRec

    @property
    def vtTauSRec(self):
        return self._vtTauSRec.cpu().numpy()

    @vtTauSRec.setter
    def vtTauSRec(self, vtNewTauSRec):
        vtNewTauSRec = np.asarray(self._expand_to_net_size(vtNewTauSRec, "vtTauSRec"))
        if (vtNewTauSRec < self.tDt).any():
            print(
                "Layer `{}`: tDt is larger than some of the vtTauSRec. This can cause numerical instabilities.".format(
                    self.strName
                )
            )

        self._vtTauSRec = torch.from_numpy(vtNewTauSRec).to(self.device).float()
        self._update_rec_kernel()

    @property
    def vtTauSInp(self):
        return self._vtTauSInp.cpu().numpy()

    @vtTauSInp.setter
    def vtTauSInp(self, vtNewTauSInp):
        vtNewTauSInp = np.asarray(self._expand_to_net_size(vtNewTauSInp, "vtTauSInp"))
        self._vtTauSInp = torch.from_numpy(vtNewTauSInp).to(self.device).float()

    @property
    def mfWIn(self):
        return self._mfWIn.cpu().numpy()

    @mfWIn.setter
    def mfWIn(self, mfNewW):
        mfNewW = self._expand_to_shape(
            mfNewW, (self.nSizeIn, self.nSize), "mfWIn", bAllowNone=False
        )
        self._mfWIn = torch.from_numpy(mfNewW).to(self.device).float()

    @property
    def mfWRec(self):
        return self._mfWRec.cpu().numpy()

    @mfWRec.setter
    def mfWRec(self, mfNewW):
        mfNewW = self._expand_to_shape(
            mfNewW, (self.nSize, self.nSize), "mfWRec", bAllowNone=False
        )
        self._mfWRec = torch.from_numpy(mfNewW).to(self.device).float()

    # mfW as alias for mfWRec
    @property
    def mfW(self):
        return self.mfWRec

    @mfW.setter
    def mfW(self, mfNewW):
        self.mfWRec = mfNewW

    # _mfW as alias for _mfWRec
    @property
    def _mfW(self):
        return self._mfWRec

    @_mfW.setter
    def _mfW(self, mfNewW):
        self._mfWRec = mfNewW

    @property
    def vSynapseStateInp(self):
        return self._vSynapseStateInp.cpu().numpy()

    @vSynapseStateInp.setter
    def vSynapseStateInp(self, vfNewState):
        vfNewState = np.asarray(
            self._expand_to_net_size(vfNewState, "vSynapseStateInp")
        )
        self._vSynapseStateInp = torch.from_numpy(vfNewState).to(self.device).float()

    @property
    def nMaxNumTimeSteps(self):
        return self._nMaxNumTimeSteps

    @nMaxNumTimeSteps.setter
    def nMaxNumTimeSteps(self, nNewMax):
        assert (
            type(nNewMax) == int and nNewMax > 0.
        ), "Layer `{}`: nMaxNumTimeSteps must be an integer greater than 0.".format(
            self.strName
        )
        self._nMaxNumTimeSteps = nNewMax
        self._update_rec_kernel()
