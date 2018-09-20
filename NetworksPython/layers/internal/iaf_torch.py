###
# iaf_torch.py - Classes implementing recurrent and feedforward layers consisting of standard IAF neurons in in torch (GPU)
###


# - Imports
import numpy as np
import torch

from ...timeseries import TSContinuous, TSEvent

from ..layer import Layer
from .timedarray_shift import TimedArray as TAShift

from typing import Optional, Union, Tuple, List

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Configure exports
__all__ = [
    "FFIAFTorch",
]

# - Absolute tolerance, e.g. for comparing float values
fTolAbs = 1e-9

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
        tRefractoryTime=0,
        strName: str = "unnamed",
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
        """

        # - Call super constructor (`asarray` is used to strip units)
        super().__init__(
            mfW=np.asarray(mfW),
            tDt=np.asarray(tDt),
            fNoiseStd=np.asarray(fNoiseStd),
            strName=strName,
        )

        # - Set cuda as backend
        assert torch.cuda.is_available(), "Layer `{}`: Cuda not available.".format(self.strName)
        self.gpu = torch.device("cuda")

        # - Record neuron parameters
        self.vfVThresh = vfVThresh
        self.vfVReset = vfVReset
        self.vfVRest = vfVRest
        self.vtTauN = vtTauN
        self.vfBias = vfBias
        self.mfW = mfW

        # - Store "reset" state
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
        vtTimeBase, mfInput, nNumTimeSteps = self._prepare_input(
            tsInput, tDuration, nNumTimeSteps
        )

        # - Weight inputs
        mfNeuralInput = torch.mm(mfInput,self._mfW)

        # - Add noise trace
        mfNeuralInput += (
            torch.randn(vtTimeBase.flatten().shape[0], self.nSize).to(self.gpu).double()
            # - Standard deviation slightly smaller than expected (due to brian??),
            #   therefore correct with empirically found factor 1.63
            * self.fNoiseStd
            * torch.sqrt(2. * self._vtTauN / torch.from_numpy(self.tDt).to(self.gpu))
            * 1.63
        )

        # - Include resting potential and bias in input for fewer computations
        mfNeuralInput += self._vfVRest + self._vfBias

        # - Get local variables
        vState = self._vState.clone()
        vfAlpha = self._vfAlpha
        vfVThresh = self._vfVThresh
        vfVRest = self._vfVRest

        # - lists for collecting spike data
        lnSpikeTimeIndices = list()
        lnSpikeChannels = list()

        # - Useful for obtaining spike indices from boolean
        vfArange = torch.arange(self.nSize)
        
        for nStep in range(nNumTimeSteps):
            # - Incremental state update from input
            vState += vfAlpha * (mfNeuralInput[nStep] - vState)
            # - Spiking
            vbSpiking = vState > vfVThresh
            vState[vbSpiking] = vfVRest[vbSpiking]
            lnSpikeTimeIndices += torch.sum(vbSpiking).item() * [nStep]
            lnSpikeChannels += list(vfArange[vbSpiking].numpy())
            

        # - Store updated state
        self._vState = vState

        # - Build response TimeSeries
        vtSpikeTimings = (np.array(lnSpikeTimeIndices) + self._nTimeStep) * self.tDt

        # - Update clock
        self._nTimeStep += nNumTimeSteps

        return TSEvent(vtSpikeTimings, lnSpikeChannels, strName="Layer `{}` spikes".format(self.strName))

    def _prepare_input(
        self,
        tsInput: TSContinuous = None,
        tDuration: float = None,
        nNumTimeSteps: int = None,
    ) -> (torch.Tensor, torch.Tensor, int):
        """
        _prepare_input - Sample input, set up time base

        :param tsInput:       TimeSeries TxM or Tx1 Input signals for this layer
        :param tDuration:     float Duration of the desired evolution, in seconds
        :param nNumTimeSteps: int Number of evolution time steps

        :return: (vtTimeBase, mfInputStep, tDuration)
            vtTimeBase:     Tensor T1 Discretised time base for evolution
            mfInputStep:    Tensor (T1xN) Discretised input signal for layer
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
            mfInputStep = torch.from_numpy(self._check_input_dims(tsInput(vtTimeBase))).to(self.gpu)

            # - Treat "NaN" as zero inputs
            mfInputStep[torch.isnan(mfInputStep)] = 0

        else:
            # - Assume zero inputs
            mfInputStep = torch.zeros((vtTimeBase.flatten().shape[0], self.nSizeIn))

        return (
            torch.from_numpy(vtTimeBase).to(self.gpu), mfInputStep, nNumTimeSteps
        )

    def reset_state(self):
        """ .reset_state() - Method: reset the internal state of the layer
            Usage: .reset_state()
        """
        self.vState = self.vfVReset

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
    def vState(self):
        return self._vState.cpu().numpy()

    @vState.setter
    def vState(self, vNewState):
        vNewState =         (
            np.asarray(self._expand_to_net_size(vNewState, "vState"))
        )
        self._vState = torch.from_numpy(vNewState).to(self.gpu)

    @property
    def vtTauN(self):
        return self._vtTauN.cpu().numpy()

    @vtTauN.setter
    def vtTauN(self, vtNewTauN):
        vtNewTauN = (
            np.asarray(self._expand_to_net_size(vtNewTauN, "vtTauN"))
        )
        self._vtTauN = torch.from_numpy(vtNewTauN).to(self.gpu)

    @property
    def vfAlpha(self):
        return self._vfAlpha.cpu().numpy()

    @property
    def _vfAlpha(self):
        return torch.from_numpy(self.tDt).to(self.gpu) / self._vtTauN 

    @property
    def vfBias(self):
        return self._vfBias.cpu().numpy()

    @vfBias.setter
    def vfBias(self, vfNewBias):
        vfNewBias = (
            np.asarray(self._expand_to_net_size(vfNewBias, "vfBias"))
        )
        self._vfBias = torch.from_numpy(vfNewBias).to(self.gpu)

    @property
    def vfVThresh(self):
        return self._vfVThresh.cpu().numpy()

    @vfVThresh.setter
    def vfVThresh(self, vfNewVThresh):
        vfNewVThresh = (
            np.asarray(self._expand_to_net_size(vfNewVThresh, "vfVThresh"))
        )
        self._vfVThresh = torch.from_numpy(vfNewVThresh).to(self.gpu)

    @property
    def vfVRest(self):
        return self._vfVRest.cpu().numpy()

    @vfVRest.setter
    def vfVRest(self, vfNewVRest):
        vfNewVRest = (
            np.asarray(self._expand_to_net_size(vfNewVRest, "vfVRest"))
        )
        self._vfVRest = torch.from_numpy(vfNewVRest).to(self.gpu)

    @property
    def vfVReset(self):
        return self._vfVReset.cpu().numpy()

    @vfVReset.setter
    def vfVReset(self, vfNewVReset):
        vfNewVReset = (
            np.asarray(self._expand_to_net_size(vfNewVReset, "vfVReset"))
        )
        self._vfVReset = torch.from_numpy(vfNewVReset).to(self.gpu)

    @property
    def t(self):
        return self._nTimeStep * self.tDt

    @property
    def mfW(self):
        return self._mfW.cpu().numpy()

    @mfW.setter
    def mfW(self, mfNewW):
        mfNewW  = self._expand_to_shape(mfNewW, (self.nSizeIn, self.nSize), "mfW", bAllowNone=False)
        self._mfW = torch.from_numpy(mfNewW).to(self.gpu)
    