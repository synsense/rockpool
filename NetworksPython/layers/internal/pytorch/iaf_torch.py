###
# iaf_torch.py - Classes implementing recurrent and feedforward layers consisting of standard IAF neurons in in torch (GPU)
#                For each layer there is one (slightly more efficient) version without refractoriness and one with.
###


# - Imports
import json
from typing import Optional, Union
from warnings import warn
import numpy as np
import torch

from ....timeseries import TSContinuous, TSEvent
from ...layer import Layer, RefProperty

# - Configure exports
__all__ = ["FFIAFTorch", "FFIAFSpkInTorch", "RecIAFTorch", "RecIAFSpkInTorch"]

# - Absolute tolerance, e.g. for comparing float values
tol_abs = 1e-9
# - Default maximum numbers of time steps for a single evolution batch
nDefaultMaxNumTimeSteps = 400


## - _RefractoryBase - Class: Base class for providing refractoriness-related properties
##                            and methods so that refractory layers can inherit them
class _RefractoryBase:
    def _single_batch_evolution(
        self,
        mfInput: np.ndarray,
        nEvolutionTimeStep: int,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input for a single batch

        :param mfInput:     np.ndarray   Input to layer as matrix
        :param nEvolutionTimeStep int    Time step within current evolution at beginning of current batch
        :param num_timesteps:   int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """
        # - Get synapse input to neurons
        mfNeuralInput, num_timesteps = self._prepare_neural_input(
            mfInput, num_timesteps
        )

        # - Update synapse state to end of evolution before rest potential and bias are added to input
        self._vSynapseState = mfNeuralInput[-1].clone()

        if self.bRecord:
            # - Tensor for recording synapse and neuron states
            mfRecordStates = self.tensors.FloatTensor(
                2 * num_timesteps, self.size
            ).fill_(0)
            # - Store synapse states
            self.mfRecordSynapses[
                nEvolutionTimeStep + 1 : nEvolutionTimeStep + num_timesteps + 1
            ] = mfNeuralInput.cpu()

        # - Tensor for collecting spike data
        mbSpiking = self.tensors.ByteTensor(num_timesteps, self.size).fill_(0)

        # - Get local variables
        state = self._state.clone()
        vfAlpha = self._vfAlpha
        vfVThresh = self._vfVThresh
        vfVReset = self._vfVReset
        bRecord = self.bRecord
        nRefractorySteps = self._nRefractorySteps
        vnRefractoryCountdownSteps = self._vnRefractoryCountdownSteps.clone()

        # - Include resting potential and bias in input for fewer computations
        mfNeuralInput += self._vfVRest + self._vfBias

        # - Evolve neuron states
        for nStep in range(num_timesteps):
            # - Determine refractory neurons
            vbNotRefractory = (vnRefractoryCountdownSteps == 0).float()
            # - Decrement refractory countdown
            vnRefractoryCountdownSteps -= 1
            vnRefractoryCountdownSteps.clamp_(min=0)
            # - Incremental state update from input
            state += vfAlpha * (mfNeuralInput[nStep] - state) * vbNotRefractory
            # - Store updated state before spike
            if bRecord:
                mfRecordStates[2 * nStep] = state
            # - Spiking
            vbSpiking = (state > vfVThresh).float()
            # - State reset
            state += (vfVReset - state) * vbSpiking
            # - Store spikes
            mbSpiking[nStep] = vbSpiking
            # - Update refractory countdown
            vnRefractoryCountdownSteps += nRefractorySteps * vbSpiking
            # - Store updated state after spike
            if bRecord:
                mfRecordStates[2 * nStep + 1] = state
            del vbSpiking

        # - Store recorded neuron states
        if bRecord:
            self.mfRecordStates[
                2 * nEvolutionTimeStep
                + 1 : 2 * (nEvolutionTimeStep + num_timesteps)
                + 1
            ] = mfRecordStates.cpu()

        # - Store updated state and update clock
        self._state = state
        self._vnRefractoryCountdownSteps = vnRefractoryCountdownSteps
        self._timestep += num_timesteps

        return mbSpiking.cpu()

    def reset_state(self):
        """ .reset_state() - Method: reset the internal state of the layer
            Usage: .reset_state()
        """
        self.state = self.vfVReset
        self.vSynapseState = 0
        self._vnRefractoryCountdownSteps = torch.zeros(self.size).to(self.device)

    ### --- Properties

    @property
    def tRefractoryTime(self):
        return self._nRefractorySteps * self.dt

    @tRefractoryTime.setter
    def tRefractoryTime(self, tNewTime):
        self._nRefractorySteps = int(np.round(tNewTime / self.dt))

    @property
    def vtRefractoryCountdown(self):
        return self._vnRefractoryCountdownSteps.cpu().numpy() * self.dt


## - FFIAFTorch - Class: define a spiking feedforward layer with spiking outputs
class FFIAFTorch(Layer):
    """ FFIAFTorch - Class: define a spiking feedforward layer with spiking outputs
    """

    ## - Constructor
    def __init__(
        self,
        weights: np.ndarray,
        vfBias: Union[float, np.ndarray] = 0.015,
        dt: float = 0.0001,
        noise_std: float = 0,
        vtTauN: Union[float, np.ndarray] = 0.02,
        vfVThresh: Union[float, np.ndarray] = -0.055,
        vfVReset: Union[float, np.ndarray] = -0.065,
        vfVRest: Union[float, np.ndarray] = -0.065,
        name: str = "unnamed",
        bRecord: bool = False,
        nMaxNumTimeSteps: int = nDefaultMaxNumTimeSteps,
    ):
        """
        FFIAFTorch - Construct a spiking feedforward layer with IAF neurons, running on GPU, using torch
                     Inputs are continuous currents; outputs are spiking events

        :param weights:             np.array MxN weight matrix.
        :param vfBias:          np.array Nx1 bias vector. Default: 10mA

        :param dt:             float Time-step. Default: 0.1 ms
        :param noise_std:       float Noise std. dev. per second. Default: 0

        :param vtTauN:          np.array Nx1 vector of neuron time constants. Default: 20ms

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param vfVReset:        np.array Nx1 vector of neuron thresholds. Default: -65mV
        :param vfVRest:         np.array Nx1 vector of neuron thresholds. Default: -65mV

        :param name:         str Name for the layer. Default: 'unnamed'

        :param bRecord:         bool Record membrane potential during evolutions

        :nMaxNumTimeSteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        # - Call super constructor
        super().__init__(
            weights=np.asarray(weights), dt=dt, noise_std=noise_std, name=name
        )

        # - Set device to cuda if available and determine how tensors should be instantiated
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.tensors = torch.cuda
        else:
            self.device = torch.device("cpu")
            print("Layer `{}`: Using CPU as CUDA is not available.".format(name))
            self.tensors = torch

        # - Record neuron parameters
        self.vfVThresh = vfVThresh
        self.vfVReset = vfVReset
        self.vfVRest = vfVRest
        self.vtTauN = vtTauN
        self.vfBias = vfBias
        self.weights = weights
        self.bRecord = bRecord
        self.nMaxNumTimeSteps = nMaxNumTimeSteps

        # - Store "reset" state
        self.reset_all()

    # @profile
    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input. Automatically splits evolution in batches,

        :param tsSpkInput:      TSContinuous  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps:   int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        # - Layer time step before evolution
        nTimeStepStart = self._timestep

        # - Prepare input signal
        mfInput, num_timesteps = self._prepare_input(ts_input, duration, num_timesteps)

        # - Tensor for collecting output spike raster
        mbSpiking = torch.ByteTensor(num_timesteps, self.size).fill_(0)

        # - Tensor for recording states
        if self.bRecord:
            self.mfRecordStates = (
                self.tensors.FloatTensor(2 * num_timesteps + 1, self.size)
                .fill_(0)
                .cpu()
            )
            self.mfRecordSynapses = (
                self.tensors.FloatTensor(num_timesteps + 1, self.size).fill_(0).cpu()
            )
            self.mfRecordStates[0] = self._state
            self.mfRecordSynapses[0] = self._vSynapseState

        # - Iterate over batches and run evolution
        iCurrentIndex = 0
        for mfCurrentInput, nCurrNumTS in self._batch_data(
            mfInput, num_timesteps, self.nMaxNumTimeSteps
        ):
            mbSpiking[
                iCurrentIndex : iCurrentIndex + nCurrNumTS
            ] = self._single_batch_evolution(
                mfCurrentInput, iCurrentIndex, nCurrNumTS, verbose
            )
            iCurrentIndex += nCurrNumTS

        # - Store recorded states in timeseries
        if self.bRecord:
            vtRecTimesStates = np.repeat(
                (nTimeStepStart + np.arange(num_timesteps + 1)) * self.dt, 2
            )[1:]
            vtRecTimesSynapses = (
                nTimeStepStart + np.arange(num_timesteps + 1)
            ) * self.dt
            self.tscRecStates = TSContinuous(
                vtRecTimesStates, self.mfRecordStates.numpy()
            )
            self.tscRecSynapses = TSContinuous(
                vtRecTimesSynapses, self.mfRecordSynapses.numpy()
            )

        # - Start and stop times for output time series
        t_start = nTimeStepStart * self.dt
        t_stop = (nTimeStepStart + num_timesteps) * self.dt

        # - Output timeseries
        if (mbSpiking == 0).all():
            tseOut = TSEvent(None, t_start=t_start, t_stop=t_stop, num_channels=self.size)
        else:
            vnSpikeTimeIndices, vnChannels = torch.nonzero(mbSpiking).t()
            vtSpikeTimings = (
                nTimeStepStart + vnSpikeTimeIndices + 1
            ).float() * self.dt

            tseOut = TSEvent(
                times=np.clip(
                    vtSpikeTimings.numpy(), t_start, t_stop - tol_abs * 10 ** 6
                ),  # Clip due to possible numerical errors
                channels=vnChannels.numpy(),
                num_channels=self.size,
                name="Layer `{}` spikes".format(self.name),
                t_start=t_start,
                t_stop=t_stop,
            )

        return tseOut

    # @profile
    def _batch_data(
        self, mfInput: np.ndarray, num_timesteps: int, nMaxNumTimeSteps: int = None
    ) -> (np.ndarray, int):
        """_batch_data: Generator that returns the data in batches"""
        # - Handle None for nMaxNumTimeSteps
        nMaxNumTimeSteps = (
            num_timesteps if nMaxNumTimeSteps is None else nMaxNumTimeSteps
        )
        nStart = 0
        while nStart < num_timesteps:
            # - Endpoint of current batch
            nEnd = min(nStart + nMaxNumTimeSteps, num_timesteps)
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
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input for a single batch

        :param mfInput:     np.ndarray   Input to layer as matrix
        :param nEvolutionTimeStep int    Time step within current evolution at beginning of current batch
        :param num_timesteps:   int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        # - Get synapse input to neurons
        mfNeuralInput, num_timesteps = self._prepare_neural_input(
            mfInput, num_timesteps
        )

        # - Update synapse state to end of evolution before rest potential and bias are added to input
        self._vSynapseState = mfNeuralInput[-1].clone()

        if self.bRecord:
            # - Tensor for recording synapse and neuron states
            mfRecordStates = self.tensors.FloatTensor(
                2 * num_timesteps, self.size
            ).fill_(0)
            # - Store synapse states
            self.mfRecordSynapses[
                nEvolutionTimeStep + 1 : nEvolutionTimeStep + num_timesteps + 1
            ] = mfNeuralInput.cpu()

        # - Tensor for collecting spike data
        mbSpiking = self.tensors.ByteTensor(num_timesteps, self.size).fill_(0)

        # - Get local variables
        state = self._state.clone()
        vfAlpha = self._vfAlpha
        vfVThresh = self._vfVThresh
        vfVReset = self._vfVReset
        bRecord = self.bRecord

        # - Include resting potential and bias in input for fewer computations
        mfNeuralInput += self._vfVRest + self._vfBias

        # - Evolve neuron states
        for nStep in range(num_timesteps):
            # - Incremental state update from input
            state += vfAlpha * (mfNeuralInput[nStep] - state)
            # - Store updated state before spike
            if bRecord:
                mfRecordStates[2 * nStep] = state
            # - Spiking
            vbSpiking = (state > vfVThresh).float()
            # - State reset
            state += (vfVReset - state) * vbSpiking
            # - Store spikes
            mbSpiking[nStep] = vbSpiking
            # - Store updated state after spike
            if bRecord:
                mfRecordStates[2 * nStep + 1] = state
            del vbSpiking

        # - Store recorded neuron states
        if bRecord:
            self.mfRecordStates[
                2 * nEvolutionTimeStep
                + 1 : 2 * (nEvolutionTimeStep + num_timesteps)
                + 1
            ] = mfRecordStates.cpu()

        # - Store updated state and update clock
        self._state = state
        self._timestep += num_timesteps

        return mbSpiking.cpu()

    # @profile
    def _prepare_neural_input(
        self, mfInput: np.array, num_timesteps: Optional[int] = None
    ) -> (np.ndarray, int):
        """
        _prepare_neural_input : Prepare the weighted, noisy synaptic input to the neurons
                                and return it together with number of evolution time steps

        :param mfInput:     np.ndarray   Input to layer as matrix
        :param num_timesteps    int      Number of evolution time steps
        :return:
                mfNeuralInput   np.ndarray  Input to neurons
                num_timesteps   int         Number of evolution time steps

        """
        # - Prepare mfInput
        mfInput = torch.from_numpy(mfInput).float().to(self.device)
        # - Weight inputs
        mfNeuralInput = torch.mm(mfInput, self._mfW)

        # - Add noise trace
        if self.noise_std > 0:
            mfNeuralInput += (
                torch.randn(num_timesteps, self.size).float().to(self.device)
                # - Standard deviation slightly smaller than expected (due to brian??),
                #   therefore correct with empirically found factor 1.63
                * self.noise_std
                * torch.sqrt(2.0 * self._vtTauN / self.dt)
                * 1.63
            )

        return mfNeuralInput, num_timesteps

    # @profile
    def _prepare_input(
        self,
        ts_input: TSContinuous = None,
        duration: float = None,
        num_timesteps: int = None,
    ) -> (torch.Tensor, int):
        """
        _prepare_input - Sample input, set up time base

        :param ts_input:       TimeSeries TxM or Tx1 Input signals for this layer
        :param duration:     float Duration of the desired evolution, in seconds
        :param num_timesteps: int Number of evolution time steps

        :return: (vtTimeBase, mfInputStep, duration)
            mfInputStep:    ndarray (T1xN) Discretised input signal for layer
            num_timesteps:  int Actual number of evolution time steps
        """
        if num_timesteps is None:
            # - Determine num_timesteps
            if duration is None:
                # - Determine duration
                assert (
                    ts_input is not None
                ), "Layer `{}`: One of `num_timesteps`, `ts_input` or `duration` must be supplied".format(
                    self.name
                )

                if ts_input.periodic:
                    # - Use duration of periodic TimeSeries, if possible
                    duration = ts_input.duration

                else:
                    # - Evolve until the end of the input TImeSeries
                    duration = ts_input.t_stop - self.t
                    assert duration > 0, (
                        "Layer `{}`: Cannot determine an appropriate evolution duration.".format(
                            self.name
                        )
                        + " `ts_input` finishes before the current evolution time."
                    )
            num_timesteps = int(np.floor((duration + tol_abs) / self.dt))
        else:
            assert isinstance(
                num_timesteps, int
            ), "Layer `{}`: num_timesteps must be of type int.".format(self.name)

        # - Generate discrete time base
        vtTimeBase = self._gen_time_trace(self.t, num_timesteps)

        if ts_input is not None:
            # - Make sure vtTimeBase matches ts_input
            if not isinstance(ts_input, TSEvent):
                if not ts_input.periodic:
                    # - If time base limits are very slightly beyond ts_input.t_start and ts_input.t_stop, match them
                    if (
                        ts_input.t_start - 1e-3 * self.dt
                        <= vtTimeBase[0]
                        <= ts_input.t_start
                    ):
                        vtTimeBase[0] = ts_input.t_start
                    if (
                        ts_input.t_stop
                        <= vtTimeBase[-1]
                        <= ts_input.t_stop + 1e-3 * self.dt
                    ):
                        vtTimeBase[-1] = ts_input.t_stop

                # - Warn if evolution period is not fully contained in ts_input
                if not (ts_input.contains(vtTimeBase) or ts_input.periodic):
                    print(
                        "Layer `{}`: Evolution period (t = {} to {}) ".format(
                            self.name, vtTimeBase[0], vtTimeBase[-1]
                        )
                        + "not fully contained in input signal (t = {} to {})".format(
                            ts_input.t_start, ts_input.t_stop
                        )
                    )

            # - Sample input trace and check for correct dimensions
            mfInputStep = self._check_input_dims(ts_input(vtTimeBase))

            # - Treat "NaN" as zero inputs
            mfInputStep[np.isnan(mfInputStep)] = 0

        else:
            # - Assume zero inputs
            mfInputStep = np.zeros((num_timesteps, self.size_in))

        return (mfInputStep, num_timesteps)

    def reset_state(self):
        """ .reset_state() - Method: reset the internal state of the layer
            Usage: .reset_state()
        """
        self.state = self.vfVReset
        self.vSynapseState = 0


    def to_dict(self):

        essentialDict = {}
        essentialDict["name"] = self.name
        essentialDict["weights"] = self._mfW.cpu().tolist()
        essentialDict["dt"] = self.dt
        essentialDict["noise_std"] = self.noise_std
        essentialDict["nMaxNumTimeSteps"] = self.nMaxNumTimeSteps
        essentialDict["vfVThresh"] = self._vfVThresh.cpu().tolist()
        essentialDict["vfVReset"] = self._vfVReset.cpu().tolist()
        essentialDict["vfVRest"] = self._vfVReset.cpu().tolist()
        essentialDict["vtTauN"] = self._vtTauN.cpu().tolist()
        essentialDict["vfBias"] = self._vfBias.cpu().tolist()
        essentialDict["bRecord"] = self.bRecord
        essentialDict["ClassName"] = "FFIAFTorch"

        return essentialDict

    def save(self, essentialDict, filename):
        with open(filename, "w") as f:
            json.dump(essentialDict, f)

    @staticmethod
    def load_from_file(filename):
        """
        load the layer from a file
        :param filename: str with the filename that includes the dict that initializes the layer
        :return: FFIAFTorch layer
        """
        with open(filename, "r") as f:
            config = json.load(f)
        return FFIAFTorch(
            weights=config["weights"],
            vfBias=config["vfBias"],
            dt=config["dt"],
            noise_std=config["noise_std"],
            vtTauN=config["vtTauN"],
            vfVThresh=config["vfVThresh"],
            vfVReset=config["vfVReset"],
            vfVRest=config["vfVRest"],
            name=config["name"],
            bRecord=config["bRecord"],
            nMaxNumTimeSteps=config["nMaxNumTimeSteps"],
        )

    @staticmethod
    def load_from_dict(config):
        """
        load the layer from a dict
        :param config: dict information for the initialization
        :return: FFIAFTorch layer
        """
        return FFIAFTorch(
            weights=config["weights"],
            vfBias=config["vfBias"],
            dt=config["dt"],
            noise_std=config["noise_std"],
            vtTauN=config["vtTauN"],
            vfVThresh=config["vfVThresh"],
            vfVReset=config["vfVReset"],
            vfVRest=config["vfVRest"],
            name=config["name"],
            bRecord=config["bRecord"],
            nMaxNumTimeSteps=config["nMaxNumTimeSteps"],
        )

    ### --- Properties

    @property
    def output_type(self):
        return TSEvent

    @RefProperty
    def state(self):
        return self._state

    @state.setter
    def state(self, vNewState):
        vNewState = np.asarray(self._expand_to_net_size(vNewState, "state"))
        self._state = torch.from_numpy(vNewState).to(self.device).float()

    @RefProperty
    def vtTauN(self):
        return self._vtTauN

    @vtTauN.setter
    def vtTauN(self, vtNewTauN):
        vtNewTauN = np.asarray(self._expand_to_net_size(vtNewTauN, "vtTauN"))
        self._vtTauN = torch.from_numpy(vtNewTauN).to(self.device).float()
        if (self.dt >= self._vtTauN).any():
            print(
                "Layer `{}`: dt is larger than some of the vtTauN. This can cause numerical instabilities.".format(
                    self.name
                )
            )

    @property
    def vfAlpha(self):
        warn(
            "Layer `{}`: Changing values of returned object by item assignment will not have effect on layer's vfAlpha".format(
                self.name
            )
        )
        return self._vfAlpha.cpu().numpy()

    @property
    def _vfAlpha(self):
        return self.dt / self._vtTauN

    @RefProperty
    def vfBias(self):
        return self._vfBias

    @vfBias.setter
    def vfBias(self, vfNewBias):
        vfNewBias = np.asarray(self._expand_to_net_size(vfNewBias, "vfBias"))
        self._vfBias = torch.from_numpy(vfNewBias).to(self.device).float()

    @RefProperty
    def vfVThresh(self):
        return self._vfVThresh

    @vfVThresh.setter
    def vfVThresh(self, vfNewVThresh):
        vfNewVThresh = np.asarray(self._expand_to_net_size(vfNewVThresh, "vfVThresh"))
        self._vfVThresh = torch.from_numpy(vfNewVThresh).to(self.device).float()

    @RefProperty
    def vfVRest(self):
        return self._vfVRest

    @vfVRest.setter
    def vfVRest(self, vfNewVRest):
        vfNewVRest = np.asarray(self._expand_to_net_size(vfNewVRest, "vfVRest"))
        self._vfVRest = torch.from_numpy(vfNewVRest).to(self.device).float()

    @RefProperty
    def vfVReset(self):
        return self._vfVReset

    @vfVReset.setter
    def vfVReset(self, vfNewVReset):
        vfNewVReset = np.asarray(self._expand_to_net_size(vfNewVReset, "vfVReset"))
        self._vfVReset = torch.from_numpy(vfNewVReset).to(self.device).float()

    @RefProperty
    def vSynapseState(self):
        return self._vSynapseState

    @vSynapseState.setter
    def vSynapseState(self, vfNewState):
        vfNewState = np.asarray(self._expand_to_net_size(vfNewState, "vSynapseState"))
        self._vSynapseState = torch.from_numpy(vfNewState).to(self.device).float()

    @property
    def t(self):
        return self._timestep * self.dt

    @RefProperty
    def weights(self):
        return self._mfW

    @weights.setter
    def weights(self, mfNewW):
        mfNewW = self._expand_to_shape(
            mfNewW, (self.size_in, self.size), "weights", bAllowNone=False
        )
        self._mfW = torch.from_numpy(mfNewW).to(self.device).float()


## - FFIAFRefrTorch - Class: define a spiking feedforward layer with spiking outputs and refractoriness
class FFIAFRefrTorch(_RefractoryBase, FFIAFTorch):
    """ FFIAFRefrTorch - Class: define a spiking feedforward layer with spiking outputs and refractoriness
    """

    ## - Constructor
    def __init__(
        self,
        weights: np.ndarray,
        vfBias: Union[float, np.ndarray] = 0.015,
        dt: float = 0.0001,
        noise_std: float = 0,
        vtTauN: Union[float, np.ndarray] = 0.02,
        vfVThresh: Union[float, np.ndarray] = -0.055,
        vfVReset: Union[float, np.ndarray] = -0.065,
        vfVRest: Union[float, np.ndarray] = -0.065,
        tRefractoryTime=0,
        name: str = "unnamed",
        bRecord: bool = False,
        nMaxNumTimeSteps: int = nDefaultMaxNumTimeSteps,
    ):
        """
        FFIAFRefrTorch - Construct a spiking feedforward layer with IAF neurons, running on GPU, using torch
                         Inputs are continuous currents; outputs are spiking events. Support Refractoriness.

        :param weights:             np.array MxN weight matrix.
        :param vfBias:          np.array Nx1 bias vector. Default: 10mA

        :param dt:             float Time-step. Default: 0.1 ms
        :param noise_std:       float Noise std. dev. per second. Default: 0

        :param vtTauN:          np.array Nx1 vector of neuron time constants. Default: 20ms

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param vfVReset:        np.array Nx1 vector of neuron thresholds. Default: -65mV
        :param vfVRest:         np.array Nx1 vector of neuron thresholds. Default: -65mV

        :param tRefractoryTime: float Refractory period after each spike. Default: 0ms

        :param name:         str Name for the layer. Default: 'unnamed'

        :param bRecord:         bool Record membrane potential during evolutions

        :nMaxNumTimeSteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        # - Call super constructor
        super().__init__(
            weights=np.asarray(weights),
            vfBias=vfBias,
            dt=dt,
            noise_std=noise_std,
            vtTauN=vtTauN,
            vfVThresh=vfVThresh,
            vfVReset=vfVReset,
            vfVRest=vfVRest,
            name=name,
            bRecord=bRecord,
            nMaxNumTimeSteps=nMaxNumTimeSteps,
        )

        self.tRefractoryTime = tRefractoryTime


# - FFIAFSpkInTorch - Class: Spiking feedforward layer with spiking in- and outputs
class FFIAFSpkInTorch(FFIAFTorch):
    """ FFIAFSpkInTorch - Class: Spiking feedforward layer with spiking in- and outputs
    """

    ## - Constructor
    def __init__(
        self,
        weights: np.ndarray,
        vfBias: np.ndarray = 0.01,
        dt: float = 0.0001,
        noise_std: float = 0,
        vtTauN: np.ndarray = 0.02,
        vtTauS: np.ndarray = 0.02,
        vfVThresh: np.ndarray = -0.055,
        vfVReset: np.ndarray = -0.065,
        vfVRest: np.ndarray = -0.065,
        name: str = "unnamed",
        bRecord: bool = False,
        nMaxNumTimeSteps: int = nDefaultMaxNumTimeSteps,
    ):
        """
        FFIAFSpkInTorch - Construct a spiking feedforward layer with IAF neurons, running on GPU, using torch
                          in- and outputs are spiking events

        :param weights:             np.array MxN weight matrix.
        :param vfBias:          np.array Nx1 bias vector. Default: 10mA

        :param dt:             float Time-step. Default: 0.1 ms
        :param noise_std:       float Noise std. dev. per second. Default: 0

        :param vtTauN:          np.array Nx1 vector of neuron time constants. Default: 20ms
        :param vtTauS:          np.array Nx1 vector of synapse time constants. Default: 20ms

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param vfVReset:        np.array Nx1 vector of neuron thresholds. Default: -65mV
        :param vfVRest:         np.array Nx1 vector of neuron thresholds. Default: -65mV

        :param name:         str Name for the layer. Default: 'unnamed'

        :param bRecord:         bool Record membrane potential during evolutions

        :nMaxNumTimeSteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        # - Call super constructor
        super().__init__(
            weights=weights,
            vfBias=vfBias,
            dt=dt,
            noise_std=noise_std,
            vtTauN=vtTauN,
            vfVThresh=vfVThresh,
            vfVReset=vfVReset,
            vfVRest=vfVRest,
            name=name,
            bRecord=bRecord,
            nMaxNumTimeSteps=nMaxNumTimeSteps,
        )

        # - Record neuron parameters
        self.vtTauS = vtTauS

    # @profile
    def _prepare_neural_input(
        self, mfInput: np.array, num_timesteps: Optional[int] = None
    ) -> (np.ndarray, int):
        """
        _prepare_neural_input : Prepare the weighted, noisy synaptic input to the neurons
                                and return it together with number of evolution time steps

        :param mfInput:         np.ndarray    Input data
        :param num_timesteps    int      Number of evolution time steps
        :return:
                mfNeuralInput   np.ndarray  Input to neurons
                num_timesteps   int         Number of evolution time steps

        """
        # - Prepare mfInput
        mfInput = torch.from_numpy(mfInput).float().to(self.device)

        # - Weight inputs
        mfWeightedInput = torch.mm(mfInput, self._mfW)

        # - Add noise trace
        if self.noise_std > 0:
            mfWeightedInput += (
                torch.randn(num_timesteps, self.size).float().to(self.device)
                # - Standard deviation slightly smaller than expected (due to brian??),
                #   therefore correct with empirically found factor 1.63
                * self.noise_std
                * torch.sqrt(2.0 * self._vtTauN / self.dt)
                * 1.63
            )

        # - Include previous synaptic states
        mfWeightedInput[0] = self._vSynapseState * torch.exp(-self.dt / self._vtTauS)

        # - Reshape input for convolution
        mfWeightedInput = mfWeightedInput.t().reshape(1, self.size, -1)

        # - Apply exponential filter to input
        vtTimes = (
            torch.arange(num_timesteps).to(self.device).reshape(1, -1).float()
            * self.dt
        )
        mfKernels = torch.exp(-vtTimes / self._vtTauS.reshape(-1, 1))
        # - Reverse on time axis and reshape to match convention of pytorch
        mfKernels = mfKernels.flip(1).reshape(self.size, 1, num_timesteps)
        # - Object for applying convolution
        convSynapses = torch.nn.Conv1d(
            self.size,
            self.size,
            num_timesteps,
            padding=num_timesteps - 1,
            groups=self.size,
            bias=False,
        ).to(self.device)
        convSynapses.weight.data = mfKernels
        # - Filtered synaptic currents
        mfNeuralInput = convSynapses(mfWeightedInput)[0].detach().t()[:num_timesteps]

        return mfNeuralInput, num_timesteps

    # @profile
    def _prepare_input(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> (np.ndarray, int):
        """
        _prepare_input - Sample input, set up time base

        :param ts_input:      TimeSeries TxM or Tx1 Input signals for this layer
        :param duration:    float Duration of the desired evolution, in seconds
        :param num_timesteps int Number of evolution time steps

        :return:
            mfSpikeRaster:    ndarray Boolean raster containing spike info
            num_timesteps:    ndarray Number of evlution time steps
        """
        if num_timesteps is None:
            # - Determine num_timesteps
            if duration is None:
                # - Determine duration
                assert (
                    ts_input is not None
                ), "Layer {}: One of `num_timesteps`, `ts_input` or `duration` must be supplied".format(
                    self.name
                )

                if ts_input.periodic:
                    # - Use duration of periodic TimeSeries, if possible
                    duration = ts_input.duration

                else:
                    # - Evolve until the end of the input TImeSeries
                    duration = ts_input.t_stop - self.t
                    assert duration > 0, (
                        "Layer {}: Cannot determine an appropriate evolution duration.".format(
                            self.name
                        )
                        + "`ts_input` finishes before the current "
                        "evolution time."
                    )
            # - Discretize duration wrt self.dt
            num_timesteps = int(np.floor((duration + tol_abs) / self.dt))
        else:
            assert isinstance(
                num_timesteps, int
            ), "Layer `{}`: num_timesteps must be of type int.".format(self.name)

        # - Extract spike timings and channels
        if ts_input is not None:
            # Extract spike data from the input variable
            mfSpikeRaster = ts_input.raster(
                dt=self.dt,
                t_start=self.t,
                t_stop=(self._timestep + num_timesteps) * self._dt,
                channels=np.arange(self.size_in),
            )
            # - Convert to supported format
            mfSpikeRaster = mfSpikeRaster.astype(int)
            # - Make sure size is correct
            mfSpikeRaster = mfSpikeRaster[:num_timesteps, :]

        else:
            mfSpikeRaster = np.zeros((num_timesteps, self.size_in))

        return mfSpikeRaster, num_timesteps

    @property
    def input_type(self):
        return TSEvent

    @RefProperty
    def vtTauS(self):
        return self._vtTauS

    @vtTauS.setter
    def vtTauS(self, vtNewTauS):
        vtNewTauS = np.asarray(self._expand_to_net_size(vtNewTauS, "vtTauS"))
        self._vtTauS = torch.from_numpy(vtNewTauS).to(self.device).float()


# - FFIAFSpkInRefrTorch - Class: Spiking feedforward layer with spiking in- and outputs and refractoriness
class FFIAFSpkInRefrTorch(_RefractoryBase, FFIAFSpkInTorch):
    """ FFIAFSpkInTorch - Class: Spiking feedforward layer with spiking in- and outputs and refractoriness
    """

    ## - Constructor
    def __init__(
        self,
        weights: np.ndarray,
        vfBias: np.ndarray = 0.01,
        dt: float = 0.0001,
        noise_std: float = 0,
        vtTauN: np.ndarray = 0.02,
        vtTauS: np.ndarray = 0.02,
        vfVThresh: np.ndarray = -0.055,
        vfVReset: np.ndarray = -0.065,
        vfVRest: np.ndarray = -0.065,
        tRefractoryTime=0,
        name: str = "unnamed",
        bRecord: bool = False,
        nMaxNumTimeSteps: int = nDefaultMaxNumTimeSteps,
    ):
        """
        FFIAFSpkInTorch - Construct a spiking feedforward layer with IAF neurons, running on GPU, using torch
                          in- and outputs are spiking events. Support refractoriness.

        :param weights:             np.array MxN weight matrix.
        :param vfBias:          np.array Nx1 bias vector. Default: 10mA

        :param dt:             float Time-step. Default: 0.1 ms
        :param noise_std:       float Noise std. dev. per second. Default: 0

        :param vtTauN:          np.array Nx1 vector of neuron time constants. Default: 20ms
        :param vtTauS:          np.array Nx1 vector of synapse time constants. Default: 20ms

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param vfVReset:        np.array Nx1 vector of neuron thresholds. Default: -65mV
        :param vfVRest:         np.array Nx1 vector of neuron thresholds. Default: -65mV

        :param tRefractoryTime: float Refractory period after each spike. Default: 0ms

        :param name:         str Name for the layer. Default: 'unnamed'

        :param bRecord:         bool Record membrane potential during evolutions

        :nMaxNumTimeSteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        # - Call super constructor
        super().__init__(
            weights=weights,
            vfBias=vfBias,
            dt=dt,
            noise_std=noise_std,
            vtTauN=vtTauN,
            vtTauS=vtTauS,
            vfVThresh=vfVThresh,
            vfVReset=vfVReset,
            vfVRest=vfVRest,
            name=name,
            bRecord=bRecord,
            nMaxNumTimeSteps=nMaxNumTimeSteps,
        )

        self.tRefractoryTime = tRefractoryTime


## - RecIAFTorch - Class: define a spiking recurrent layer with spiking outputs
class RecIAFTorch(FFIAFTorch):
    """ FFIAFTorch - Class: define a spiking recurrent layer with spiking outputs
    """

    ## - Constructor
    def __init__(
        self,
        weights: np.ndarray,
        vfBias: Union[float, np.ndarray] = 0.015,
        dt: float = 0.0001,
        noise_std: float = 0,
        vtTauN: Union[float, np.ndarray] = 0.02,
        vtTauSynR: Union[float, np.ndarray] = 0.05,
        vfVThresh: Union[float, np.ndarray] = -0.055,
        vfVReset: Union[float, np.ndarray] = -0.065,
        vfVRest: Union[float, np.ndarray] = -0.065,
        name: str = "unnamed",
        bRecord: bool = False,
        nMaxNumTimeSteps: int = nDefaultMaxNumTimeSteps,
    ):
        """
        FFIAFTorch - Construct a spiking recurrent layer with IAF neurons, running on GPU, using torch
                     Inputs are continuous currents; outputs are spiking events

        :param weights:             np.array MxN weight matrix.
        :param vfBias:          np.array Nx1 bias vector. Default: 0.015

        :param dt:             float Time-step. Default: 0.0001
        :param noise_std:       float Noise std. dev. per second. Default: 0

        :param vtTauN:          np.array Nx1 vector of neuron time constants. Default: 0.02
        :param vtTauSynR:       np.array NxN vector of recurrent synaptic time constants. Default: 0.005

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds. Default: -0.055
        :param vfVReset:        np.array Nx1 vector of neuron thresholds. Default: -0.065
        :param vfVRest:         np.array Nx1 vector of neuron thresholds. Default: -0.065

        :param name:         str Name for the layer. Default: 'unnamed'

        :param bRecord:         bool Record membrane potential during evolutions. Default: False

        :nMaxNumTimeSteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        assert (
            np.atleast_2d(weights).shape[0] == np.atleast_2d(weights).shape[1]
        ), "Layer `{}`: weights must be a square matrix.".format(name)

        # - Call super constructor
        super().__init__(
            weights=weights,
            vfBias=vfBias,
            dt=dt,
            noise_std=noise_std,
            vtTauN=vtTauN,
            vfVThresh=vfVThresh,
            vfVReset=vfVReset,
            vfVRest=vfVRest,
            name=name,
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
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input for a single batch

        :param mfInput:     np.ndarray   Input to layer as matrix
        :param nEvolutionTimeStep int    Time step within current evolution at beginning of current batch
        :param num_timesteps:   int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        mfNeuralInput, num_timesteps = self._prepare_neural_input(
            mfInput, num_timesteps
        )

        if self.bRecord:
            # - Tensor for recording synapse and neuron states
            mfRecordStates = self.tensors.FloatTensor(
                2 * num_timesteps, self.size
            ).fill_(0)

        # - Tensor for collecting spike data
        mbSpiking = self.tensors.ByteTensor(num_timesteps, self.size).fill_(0)

        # - Get local variables
        state = self._state.clone()
        vfAlpha = self._vfAlpha
        vfVThresh = self._vfVThresh
        vfVReset = self._vfVReset
        bRecord = self.bRecord
        mfKernels = self._mfKernelsRec
        nNumTSKernel = mfKernels.shape[0]
        weights_rec = self._mfW

        # - Include resting potential and bias in input for fewer computations
        # - Omit latest time point, which is only used for carrying over synapse state to new batch
        mfNeuralInput[:-1] += self._vfVRest + self._vfBias

        # - Evolve neuron states
        for nStep in range(num_timesteps):
            # - Incremental state update from input
            state += vfAlpha * (mfNeuralInput[nStep] - state)
            # - Store updated state before spike
            if bRecord:
                mfRecordStates[2 * nStep] = state
            # - Spiking
            vbSpiking = (state > vfVThresh).float()
            # - State reset
            state += (vfVReset - state) * vbSpiking
            # - Store spikes
            mbSpiking[nStep] = vbSpiking
            # - Store updated state after spike
            if bRecord:
                mfRecordStates[2 * nStep + 1] = state
            # - Add filtered recurrent spikes to input
            nTSRecurrent = min(nNumTSKernel, num_timesteps - nStep)
            mfNeuralInput[nStep + 1 : nStep + 1 + nTSRecurrent] += mfKernels[
                :nTSRecurrent
            ] * torch.mm(vbSpiking.reshape(1, -1), weights_rec)

            del vbSpiking

        # - Store recorded neuron and synapse states
        if bRecord:
            self.mfRecordStates[
                2 * nEvolutionTimeStep
                + 1 : 2 * (nEvolutionTimeStep + num_timesteps)
                + 1
            ] = mfRecordStates.cpu()
            self.mfRecordSynapses[
                nEvolutionTimeStep + 1 : nEvolutionTimeStep + num_timesteps + 1
            ] = (
                mfNeuralInput[:num_timesteps]
                - self._vfVRest
                - self._vfBias  # Introduces slight numerical error in stored synapses of about 1e-9
            ).cpu()

        # - Store updated neuron and synapse states and update clock
        self._state = state
        self._vSynapseState = mfNeuralInput[-1].clone()
        self._timestep += num_timesteps

        return mbSpiking.cpu()

    # @profile
    def _prepare_neural_input(
        self, mfInput: np.array, num_timesteps: Optional[int] = None
    ) -> (np.ndarray, int):
        """
        _prepare_neural_input : Prepare the noisy synaptic input to the neurons
                                and return it together with number of evolution time steps

        :param tsSpkInput:      TSContinuous  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :return:
                mfNeuralInput   np.ndarray  Input to neurons
                num_timesteps   int         Number of evolution time steps

        """

        num_timesteps = mfInput.shape[0] if num_timesteps is None else num_timesteps

        # - Prepare mfInput, with additional time step for carrying over recurrent spikes between batches
        mfNeuralInput = self.tensors.FloatTensor(num_timesteps + 1, self.size).fill_(0)
        mfNeuralInput[:-1] = torch.from_numpy(mfInput).float()
        # - Carry over filtered recurrent spikes from previous batch
        nTSRecurrent = min(mfNeuralInput.shape[0], self._mfKernelsRec.shape[0])
        mfNeuralInput[:nTSRecurrent] += (
            self._mfKernelsRec[:nTSRecurrent] * self._vSynapseState
        )

        # - Add noise trace
        if self.noise_std > 0:
            mfNeuralInput += (
                torch.randn(num_timesteps + 1, self.size).float().to(self.device)
                # - Standard deviation slightly smaller than expected (due to brian??),
                #   therefore correct with empirically found factor 1.63
                * self.noise_std
                * torch.sqrt(2.0 * self._vtTauN / self.dt)
                * 1.63
            )

        return mfNeuralInput, num_timesteps

    @property
    def vtTauSynR(self):
        return self._vtTauSynR.cpu().numpy()

    @vtTauSynR.setter
    def vtTauSynR(self, vtNewTauSynR):
        vtNewTauSynR = np.asarray(self._expand_to_net_size(vtNewTauSynR, "vtTauSynR"))
        if (vtNewTauSynR < self.dt).any():
            print(
                "Layer `{}`: dt is larger than some of the vtTauSynR. This can cause numerical instabilities.".format(
                    self.name
                )
            )

        self._vtTauSynR = torch.from_numpy(vtNewTauSynR).to(self.device).float()

        # - Kernel for filtering recurrent spikes
        nKernelSize = 50 * int(
            np.amax(vtNewTauSynR) / self.dt
        )  # - Values smaller than ca. 1e-21 are neglected
        vtTimes = (
            torch.arange(nKernelSize).to(self.device).reshape(-1, 1).float() * self.dt
        )
        self._mfKernelsRec = torch.exp(-vtTimes / self._vtTauSynR.reshape(1, -1))

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, tNewDt):
        assert tNewDt > 0, "Layer `{}`: dt must be greater than 0.".format(
            self.name
        )
        self._dt = tNewDt
        if hasattr(self, "vtTauSynR"):
            # - Update filter for recurrent spikes if already exists
            self.vtTauSynR = self.vtTauSynR


## - RecIAFRefrTorch - Class: define a spiking recurrent layer with spiking outputs and refractoriness
class RecIAFRefrTorch(_RefractoryBase, RecIAFTorch):
    """ FFIAFRefrTorch - Class: define a spiking recurrent layer with spiking outputs and refractoriness
    """

    ## - Constructor
    def __init__(
        self,
        weights: np.ndarray,
        vfBias: Union[float, np.ndarray] = 0.015,
        dt: float = 0.0001,
        noise_std: float = 0,
        vtTauN: Union[float, np.ndarray] = 0.02,
        vtTauSynR: Union[float, np.ndarray] = 0.05,
        vfVThresh: Union[float, np.ndarray] = -0.055,
        vfVReset: Union[float, np.ndarray] = -0.065,
        vfVRest: Union[float, np.ndarray] = -0.065,
        tRefractoryTime=0,
        name: str = "unnamed",
        bRecord: bool = False,
        nMaxNumTimeSteps: int = nDefaultMaxNumTimeSteps,
    ):
        """
        FFIAFRefrTorch - Construct a spiking recurrent layer with IAF neurons, running on GPU, using torch
                         Inputs are continuous currents; outputs are spiking events. Support refractoriness

        :param weights:             np.array MxN weight matrix.
        :param vfBias:          np.array Nx1 bias vector. Default: 0.015

        :param dt:             float Time-step. Default: 0.0001
        :param noise_std:       float Noise std. dev. per second. Default: 0

        :param vtTauN:          np.array Nx1 vector of neuron time constants. Default: 0.02
        :param vtTauSynR:       np.array NxN vector of recurrent synaptic time constants. Default: 0.005

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds. Default: -0.055
        :param vfVReset:        np.array Nx1 vector of neuron thresholds. Default: -0.065
        :param vfVRest:         np.array Nx1 vector of neuron thresholds. Default: -0.065

        :param tRefractoryTime: float Refractory period after each spike. Default: 0

        :param name:         str Name for the layer. Default: 'unnamed'

        :param bRecord:         bool Record membrane potential during evolutions. Default: False

        :nMaxNumTimeSteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        # - Call super constructor
        super().__init__(
            weights=weights,
            vfBias=vfBias,
            dt=dt,
            noise_std=noise_std,
            vtTauN=vtTauN,
            vtTauSynR=vtTauSynR,
            vfVThresh=vfVThresh,
            vfVReset=vfVReset,
            vfVRest=vfVRest,
            name=name,
            bRecord=bRecord,
            nMaxNumTimeSteps=nMaxNumTimeSteps,
        )

        self.tRefractoryTime = tRefractoryTime

    def _single_batch_evolution(
        self,
        mfInput: np.ndarray,
        nEvolutionTimeStep: int,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input for a single batch

        :param mfInput:     np.ndarray   Input to layer as matrix
        :param nEvolutionTimeStep int    Time step within current evolution at beginning of current batch
        :param num_timesteps:   int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """
        mfNeuralInput, num_timesteps = self._prepare_neural_input(
            mfInput, num_timesteps
        )

        if self.bRecord:
            # - Tensor for recording synapse and neuron states
            mfRecordStates = self.tensors.FloatTensor(
                2 * num_timesteps, self.size
            ).fill_(0)

        # - Tensor for collecting spike data
        mbSpiking = self.tensors.ByteTensor(num_timesteps, self.size).fill_(0)

        # - Get local variables
        state = self._state.clone()
        vfAlpha = self._vfAlpha
        vfVThresh = self._vfVThresh
        vfVReset = self._vfVReset
        bRecord = self.bRecord
        mfKernels = self._mfKernelsRec
        nNumTSKernel = mfKernels.shape[0]
        weights_rec = self._mfW
        nRefractorySteps = self._nRefractorySteps
        vnRefractoryCountdownSteps = self._vnRefractoryCountdownSteps.clone()

        # - Include resting potential and bias in input for fewer computations
        # - Omit latest time point, which is only used for carrying over synapse state to new batch
        mfNeuralInput[:-1] += self._vfVRest + self._vfBias

        # - Evolve neuron states
        for nStep in range(num_timesteps):
            # - Determine refractory neurons
            vbNotRefractory = (vnRefractoryCountdownSteps == 0).float()
            # - Decrement refractory countdown
            vnRefractoryCountdownSteps -= 1
            vnRefractoryCountdownSteps.clamp_(min=0)
            # - Incremental state update from input
            state += vfAlpha * (mfNeuralInput[nStep] - state) * vbNotRefractory
            # - Store updated state before spike
            if bRecord:
                mfRecordStates[2 * nStep] = state
            # - Spiking
            vbSpiking = (state > vfVThresh).float()
            # - State reset
            state += (vfVReset - state) * vbSpiking
            # - Store spikes
            mbSpiking[nStep] = vbSpiking
            # - Update refractory countdown
            vnRefractoryCountdownSteps += nRefractorySteps * vbSpiking
            # - Store updated state after spike
            if bRecord:
                mfRecordStates[2 * nStep + 1] = state
            # - Add filtered recurrent spikes to input
            nTSRecurrent = min(nNumTSKernel, num_timesteps - nStep)
            mfNeuralInput[nStep + 1 : nStep + 1 + nTSRecurrent] += mfKernels[
                :nTSRecurrent
            ] * torch.mm(vbSpiking.reshape(1, -1), weights_rec)

            del vbSpiking

        # - Store recorded neuron and synapse states
        if bRecord:
            self.mfRecordStates[
                2 * nEvolutionTimeStep
                + 1 : 2 * (nEvolutionTimeStep + num_timesteps)
                + 1
            ] = mfRecordStates.cpu()
            self.mfRecordSynapses[
                nEvolutionTimeStep + 1 : nEvolutionTimeStep + num_timesteps + 1
            ] = (
                mfNeuralInput[:num_timesteps]
                - self._vfVRest
                - self._vfBias  # Introduces slight numerical error in stored synapses of about 1e-9
            ).cpu()

        # - Store updated neuron and synapse states and update clock
        self._state = state
        self._vSynapseState = mfNeuralInput[-1].clone()
        self._timestep += num_timesteps

        return mbSpiking.cpu()


## - RecIAFSpkInTorch - Class: define a spiking recurrent layer with spiking in- and outputs
class RecIAFSpkInTorch(RecIAFTorch):
    """ RecIAFSpkInTorch - Class: define a spiking recurrent layer with spiking in- and outputs
    """

    ## - Constructor
    def __init__(
        self,
        weights_in: np.ndarray,
        weights_rec: np.ndarray,
        vfBias: Union[float, np.ndarray] = 0.0105,
        dt: float = 0.0001,
        noise_std: float = 0,
        vtTauN: Union[float, np.ndarray] = 0.02,
        vtTauSInp: Union[float, np.ndarray] = 0.05,
        vtTauSRec: Union[float, np.ndarray] = 0.05,
        vfVThresh: Union[float, np.ndarray] = -0.055,
        vfVReset: Union[float, np.ndarray] = -0.065,
        vfVRest: Union[float, np.ndarray] = -0.065,
        name: str = "unnamed",
        bRecord: bool = False,
        bAddEvents: bool = True,
        nMaxNumTimeSteps: int = nDefaultMaxNumTimeSteps,
    ):
        """
        RecIAFSpkInTorch - Construct a spiking recurrent layer with IAF neurons, running on GPU, using torch
                           Inputs and outputs are spiking events

        :param weights_in:           np.array MxN input weight matrix.
        :param weights_rec:          np.array NxN recurrent weight matrix.
        :param vfBias:          np.array Nx1 bias vector. Default: 0.0105

        :param dt:             float Time-step. Default: 0.0001
        :param noise_std:       float Noise std. dev. per second. Default: 0

        :param vtTauN:          np.array Nx1 vector of neuron time constants. Default: 0.02
        :param vtTauSInp:       np.array Nx1 vector of synapse time constants. Default: 0.05
        :param vtTauSRec:       np.array Nx1 vector of synapse time constants. Default: 0.05

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds. Default: -0.055
        :param vfVReset:        np.array Nx1 vector of neuron thresholds. Default: -0.065
        :param vfVRest:         np.array Nx1 vector of neuron thresholds. Default: -0.065

        :param name:         str Name for the layer. Default: 'unnamed'

        :param bRecord:         bool Record membrane potential during evolutions. Default: False

        :bAddEvents:            bool     If during evolution multiple input events arrive during one
                                         time step for a channel, count their actual number instead of
                                         just counting them as one (This might make less sense for
                                         refractory neurons).

        :nMaxNumTimeSteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        # - Call super constructor
        Layer.__init__(self, weights=weights_in, dt=dt, noise_std=noise_std, name=name)

        # - Set device to cuda if available and determine how tensors should be instantiated
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.tensors = torch.cuda
        else:
            self.device = torch.device("cpu")
            print("Layer `{}`: Using CPU as CUDA is not available.".format(name))
            self.tensors = torch

        # - Bypass property setter to avoid unnecessary convolution kernel update
        assert (
            type(nMaxNumTimeSteps) == int and nMaxNumTimeSteps > 0.0
        ), "Layer `{}`: nMaxNumTimeSteps must be an integer greater than 0.".format(
            self.name
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
        self.weights_in = weights_in
        self.weights_rec = weights_rec
        self.bRecord = bRecord
        self.bAddEvents = bAddEvents

        # - Store "reset" state
        self.reset_all()

    # @profile
    def _prepare_neural_input(
        self, mfInput: np.array, num_timesteps: Optional[int] = None
    ) -> (np.ndarray, int):
        """
        _prepare_neural_input : Prepare the noisy synaptic input to the neurons
                                and return it together with number of evolution time steps

        :param mfInput          np.ndarray  External input spike raster
        :param num_timesteps    int         Number of evolution time steps
        :return:
                mfNeuralInput   np.ndarray  Input to neurons
                num_timesteps   int         Number of evolution time steps

        """

        num_timesteps = mfInput.shape[0] if num_timesteps is None else num_timesteps

        # - Prepare external input
        mfInput = torch.from_numpy(mfInput).float().to(self.device)
        # - Weigh inputs
        mfWeightedInput = torch.mm(mfInput, self._weights_in)
        # - Carry over external inputs from last batch
        mfWeightedInput[0] = self._vSynapseStateInp.clone() * torch.exp(
            -self.dt / self._vtTauSInp
        )
        # - Reshape input for convolution
        mfWeightedInput = mfWeightedInput.t().reshape(1, self.size, -1)
        # - Apply exponential filter to external input
        vtTimes = (
            torch.arange(num_timesteps).to(self.device).reshape(1, -1).float()
            * self.dt
        )
        mfInputKernels = torch.exp(-vtTimes / self._vtTauSInp.reshape(-1, 1))
        # - Reverse on time axis and reshape to match convention of pytorch
        mfInputKernels = mfInputKernels.flip(1).reshape(self.size, 1, num_timesteps)
        # - Object for applying convolution
        convSynapses = torch.nn.Conv1d(
            self.size,
            self.size,
            num_timesteps,
            padding=num_timesteps - 1,
            groups=self.size,
            bias=False,
        ).to(self.device)
        convSynapses.weight.data = mfInputKernels
        # - Filtered synaptic currents
        mfFilteredExternalInput = (
            convSynapses(mfWeightedInput)[0].detach().t()[:num_timesteps]
        )
        # - Store filtered input from last time step for carry-over to next batch
        self._vSynapseStateInp = mfFilteredExternalInput[-1].clone()

        # - Prepare input to neurons, with additional time step for carrying over recurrent spikes between batches
        mfNeuralInput = self.tensors.FloatTensor(num_timesteps + 1, self.size).fill_(0)
        # - Filtered external input
        mfNeuralInput[:-1] = mfFilteredExternalInput
        # - Carry over filtered recurrent spikes from previous batch
        nTSRecurrent = min(mfNeuralInput.shape[0], self._mfKernelsRec.shape[0])
        mfNeuralInput[:nTSRecurrent] += (
            self._mfKernelsRec[:nTSRecurrent] * self._vSynapseState
        )

        # - Add noise trace
        if self.noise_std > 0:
            mfNeuralInput += (
                torch.randn(num_timesteps + 1, self.size).float().to(self.device)
                # - Standard deviation slightly smaller than expected (due to brian??),
                #   therefore correct with empirically found factor 1.63
                * self.noise_std
                * torch.sqrt(2.0 * self._vtTauN / self.dt)
                * 1.63
            )

        return mfNeuralInput, num_timesteps

    # @profile
    def _prepare_input(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> (np.ndarray, int):
        """
        _prepare_input - Sample input, set up time base

        :param ts_input:      TimeSeries TxM or Tx1 Input signals for this layer
        :param duration:    float Duration of the desired evolution, in seconds
        :param num_timesteps int Number of evolution time steps

        :return:
            mnSpikeRaster:    Tensor Boolean raster containing spike info
            num_timesteps:    ndarray Number of evlution time steps
        """
        if num_timesteps is None:
            # - Determine num_timesteps
            if duration is None:
                # - Determine duration
                assert (
                    ts_input is not None
                ), "Layer {}: One of `num_timesteps`, `ts_input` or `duration` must be supplied".format(
                    self.name
                )

                if ts_input.periodic:
                    # - Use duration of periodic TimeSeries, if possible
                    duration = ts_input.duration

                else:
                    # - Evolve until the end of the input TImeSeries
                    duration = ts_input.t_stop - self.t
                    assert duration > 0, (
                        "Layer {}: Cannot determine an appropriate evolution duration.".format(
                            self.name
                        )
                        + "`ts_input` finishes before the current "
                        "evolution time."
                    )
            # - Discretize duration wrt self.dt
            num_timesteps = int(np.floor((duration + tol_abs) / self.dt))
        else:
            assert isinstance(
                num_timesteps, int
            ), "Layer `{}`: num_timesteps must be of type int.".format(self.name)

        # - Extract spike timings and channels
        if ts_input is not None:
            # Extract spike data from the input variable
            mnSpikeRaster = ts_input.raster(
                dt=self.dt,
                t_start=self.t,
                t_stop=(self._timestep + num_timesteps) * self._dt,
                channels=np.arange(
                    self.size_in
                ),  # This causes problems when ts_input has no events in some channels
                add_events=self.bAddEvents,  # Allow for multiple input spikes per time step
            )
            # - Convert to supportedformat
            mnSpikeRaster = mnSpikeRaster.astype(int)
            # - Make sure size is correct
            mnSpikeRaster = mnSpikeRaster[:num_timesteps, :]

        else:
            mnSpikeRaster = np.zeros((num_timesteps, self.size_in))

        return mnSpikeRaster, num_timesteps

    def reset_all(self):
        super().reset_all()
        self.vSynapseStateInp = 0

    def to_dict(self):

        essentialDict = {}
        essentialDict["name"] = self.name
        essentialDict["weights_rec"] = self._weights_rec.cpu().tolist()
        essentialDict["dt"] = self.dt
        essentialDict["noise_std"] = self.noise_std
        essentialDict["nMaxNumTimeSteps"] = self.nMaxNumTimeSteps
        essentialDict["vfVThresh"] = self._vfVThresh.cpu().tolist()
        essentialDict["vfVReset"] = self._vfVReset.cpu().tolist()
        essentialDict["vfVRest"] = self._vfVReset.cpu().tolist()
        essentialDict["vtTauN"] = self._vtTauN.cpu().tolist()
        essentialDict["vtTauSRec"] = self._vtTauSRec.cpu().tolist()
        essentialDict["vtTauSInp"] = self._vtTauSInp.cpu().tolist()
        essentialDict["vfBias"] = self._vfBias.cpu().tolist()
        essentialDict["weights_in"] = self._weights_in.cpu().tolist()
        essentialDict["bRecord"] = self.bRecord
        essentialDict["bAddEvents"] = self.bAddEvents
        essentialDict["ClassName"] = "RecIAFSpkInTorch"

        return essentialDict

    def save(self, essentialDict, filename):
        with open(filename, "w") as f:
            json.dump(essentialDict, f)

    @staticmethod
    def load(filename):
        with open(filename, "r") as f:
            config = json.load(f)
        return RecIAFSpkInTorch(
            weights_in=config["weights_in"],
            weights_rec=config["weights_rec"],
            vfBias=config["vfBias"],
            dt=config["dt"],
            noise_std=config["noise_std"],
            vtTauN=config["vtTauN"],
            vtTauSInp=config["vtTauSInp"],
            vtTauSRec=config["vtTauSRec"],
            vfVThresh=config["vfVThresh"],
            vfVReset=config["vfVReset"],
            vfVRest=config["vfVRest"],
            name=config["name"],
            bRecord=config["bRecord"],
            bAddEvents=config["bAddEvents"],
            nMaxNumTimeSteps=config["nMaxNumTimeSteps"],
        )
    @staticmethod
    def load_from_dict(config):

        return RecIAFSpkInTorch(
            weights_in=config["weights_in"],
            weights_rec=config["weights_rec"],
            vfBias=config["vfBias"],
            dt=config["dt"],
            noise_std=config["noise_std"],
            vtTauN=config["vtTauN"],
            vtTauSInp=config["vtTauSInp"],
            vtTauSRec=config["vtTauSRec"],
            vfVThresh=config["vfVThresh"],
            vfVReset=config["vfVReset"],
            vfVRest=config["vfVRest"],
            name=config["name"],
            bRecord=config["bRecord"],
            bAddEvents=config["bAddEvents"],
            nMaxNumTimeSteps=config["nMaxNumTimeSteps"],
        )

    def _update_rec_kernel(self):
        # - Kernel for filtering recurrent spikes
        nKernelSize = min(
            50
            * int(
                torch.max(self._vtTauSRec) / self.dt
            ),  # - Values smaller than ca. 1e-21 are neglected
            self._nMaxNumTimeSteps
            + 1,  # Kernel does not need to be larger than batch duration
        )
        vtTimes = (
            torch.arange(nKernelSize).to(self.device).reshape(-1, 1).float() * self.dt
        )
        self._mfKernelsRec = torch.exp(-vtTimes / self._vtTauSRec.reshape(1, -1))
        print(
            "Layer `{}`: Recurrent filter kernels have been updated.".format(
                self.name
            )
        )

    @property
    def input_type(self):
        return TSEvent

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, tNewDt):
        assert tNewDt > 0, "Layer `{}`: dt must be greater than 0.".format(
            self.name
        )
        self._dt = tNewDt
        if hasattr(self, "vtTauSRec"):
            # - Update filter for recurrent spikes if already exists
            self.vtTauSRec = self.vtTauSRec

    @RefProperty
    def vtTauSRec(self):
        return self._vtTauSRec

    @vtTauSRec.setter
    def vtTauSRec(self, vtNewTauSRec):
        vtNewTauSRec = np.asarray(self._expand_to_net_size(vtNewTauSRec, "vtTauSRec"))
        if (vtNewTauSRec < self.dt).any():
            print(
                "Layer `{}`: dt is larger than some of the vtTauSRec. This can cause numerical instabilities.".format(
                    self.name
                )
            )

        self._vtTauSRec = torch.from_numpy(vtNewTauSRec).to(self.device).float()
        self._update_rec_kernel()

    @RefProperty
    def vtTauSInp(self):
        return self._vtTauSInp

    @vtTauSInp.setter
    def vtTauSInp(self, vtNewTauSInp):
        vtNewTauSInp = np.asarray(self._expand_to_net_size(vtNewTauSInp, "vtTauSInp"))
        self._vtTauSInp = torch.from_numpy(vtNewTauSInp).to(self.device).float()

    @RefProperty
    def weights_in(self):
        return self._weights_in

    @weights_in.setter
    def weights_in(self, mfNewW):
        mfNewW = self._expand_to_shape(
            mfNewW, (self.size_in, self.size), "weights_in", bAllowNone=False
        )
        self._weights_in = torch.from_numpy(mfNewW).to(self.device).float()

    @RefProperty
    def weights_rec(self):
        return self._weights_rec

    @weights_rec.setter
    def weights_rec(self, mfNewW):
        mfNewW = self._expand_to_shape(
            mfNewW, (self.size, self.size), "weights_rec", bAllowNone=False
        )
        self._weights_rec = torch.from_numpy(mfNewW).to(self.device).float()

    # weights as alias for weights_rec
    @property
    def weights(self):
        return self.weights_rec

    @weights.setter
    def weights(self, mfNewW):
        self.weights_rec = mfNewW

    # _mfW as alias for _weights_rec
    @property
    def _mfW(self):
        return self._weights_rec

    @_mfW.setter
    def _mfW(self, mfNewW):
        self._weights_rec = mfNewW

    @RefProperty
    def vSynapseStateInp(self):
        return self._vSynapseStateInp

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
            type(nNewMax) == int and nNewMax > 0.0
        ), "Layer `{}`: nMaxNumTimeSteps must be an integer greater than 0.".format(
            self.name
        )
        self._nMaxNumTimeSteps = nNewMax
        self._update_rec_kernel()


## - RecIAFSpkInRefrTorch - Class: define a spiking recurrent layer with spiking in- and outputs and refractoriness
class RecIAFSpkInRefrTorch(_RefractoryBase, RecIAFSpkInTorch):
    """ RecIAFSpkInRefrTorch - Class: define a spiking recurrent layer with spiking in- and outputs and refractoriness
    """

    ## - Constructor
    def __init__(
        self,
        weights_in: np.ndarray,
        weights_rec: np.ndarray,
        vfBias: Union[float, np.ndarray] = 0.0105,
        dt: float = 0.0001,
        noise_std: float = 0,
        vtTauN: Union[float, np.ndarray] = 0.02,
        vtTauSInp: Union[float, np.ndarray] = 0.05,
        vtTauSRec: Union[float, np.ndarray] = 0.05,
        vfVThresh: Union[float, np.ndarray] = -0.055,
        vfVReset: Union[float, np.ndarray] = -0.065,
        vfVRest: Union[float, np.ndarray] = -0.065,
        tRefractoryTime=0,
        name: str = "unnamed",
        bRecord: bool = False,
        bAddEvents: bool = True,
        nMaxNumTimeSteps: int = nDefaultMaxNumTimeSteps,
    ):
        """
        RecIAFSpkInRefrTorch - Construct a spiking recurrent layer with IAF neurons, running on GPU, using torch
                               Inputs and outputs are spiking events. Support refractoriness

        :param weights_in:           np.array MxN input weight matrix.
        :param weights_rec:          np.array NxN recurrent weight matrix.
        :param vfBias:          np.array Nx1 bias vector. Default: 0.0105

        :param dt:             float Time-step. Default: 0.0001
        :param noise_std:       float Noise std. dev. per second. Default: 0

        :param vtTauN:          np.array Nx1 vector of neuron time constants. Default: 0.02
        :param vtTauSInp:       np.array Nx1 vector of synapse time constants. Default: 0.05
        :param vtTauSRec:       np.array Nx1 vector of synapse time constants. Default: 0.05

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds. Default: -0.055
        :param vfVReset:        np.array Nx1 vector of neuron thresholds. Default: -0.065
        :param vfVRest:         np.array Nx1 vector of neuron thresholds. Default: -0.065

        :param tRefractoryTime: float Refractory period after each spike. Default: 0

        :param name:         str Name for the layer. Default: 'unnamed'

        :param bRecord:         bool Record membrane potential during evolutions. Default: False

        :bAddEvents:            bool     If during evolution multiple input events arrive during one
                                         time step for a channel, count their actual number instead of
                                         just counting them as one (This might make less sense for
                                         refractory neurons).

        :nMaxNumTimeSteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        # - Call super constructor
        super().__init__(
            weights_in=weights_in,
            weights_rec=weights_rec,
            vfBias=vfBias,
            dt=dt,
            noise_std=noise_std,
            vtTauN=vtTauN,
            vtTauSInp=vtTauSInp,
            vtTauSRec=vtTauSRec,
            vfVThresh=vfVThresh,
            vfVReset=vfVReset,
            vfVRest=vfVRest,
            name=name,
            bRecord=bRecord,
            nMaxNumTimeSteps=nMaxNumTimeSteps,
        )

        self.tRefractoryTime = tRefractoryTime

    def _single_batch_evolution(
        self,
        mfInput: np.ndarray,
        nEvolutionTimeStep: int,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input for a single batch

        :param mfInput:            Input to layer as matrix
        :param nEvolutionTimeStep  Time step within current evolution at beginning of current batch
        :param num_timesteps:      Number of evolution time steps
        :param verbose:           Currently no effect, just for conformity
        :return:                   output spike series

        """
        mfNeuralInput, num_timesteps = self._prepare_neural_input(
            mfInput, num_timesteps
        )

        if self.bRecord:
            # - Tensor for recording synapse and neuron states
            mfRecordStates = self.tensors.FloatTensor(
                2 * num_timesteps, self.size
            ).fill_(0)

        # - Tensor for collecting spike data
        mbSpiking = self.tensors.ByteTensor(num_timesteps, self.size).fill_(0)

        # - Get local variables
        state = self._state.clone()
        vfAlpha = self._vfAlpha
        vfVThresh = self._vfVThresh
        vfVReset = self._vfVReset
        bRecord = self.bRecord
        mfKernels = self._mfKernelsRec
        nNumTSKernel = mfKernels.shape[0]
        weights_rec = self._mfW
        nRefractorySteps = self._nRefractorySteps
        vnRefractoryCountdownSteps = self._vnRefractoryCountdownSteps.clone()

        # - Include resting potential and bias in input for fewer computations
        # - Omit latest time point, which is only used for carrying over synapse state to new batch
        mfNeuralInput[:-1] += self._vfVRest + self._vfBias

        # - Evolve neuron states
        for nStep in range(num_timesteps):
            # - Determine refractory neurons
            vbNotRefractory = (vnRefractoryCountdownSteps == 0).float()
            # - Decrement refractory countdown
            vnRefractoryCountdownSteps -= 1
            vnRefractoryCountdownSteps.clamp_(min=0)
            # - Incremental state update from input
            state += vfAlpha * (mfNeuralInput[nStep] - state) * vbNotRefractory
            # - Store updated state before spike
            if bRecord:
                mfRecordStates[2 * nStep] = state
            # - Spiking
            vbSpiking = (state > vfVThresh).float()
            # - State reset
            state += (vfVReset - state) * vbSpiking
            # - Store spikes
            mbSpiking[nStep] = vbSpiking
            # - Update refractory countdown
            vnRefractoryCountdownSteps += nRefractorySteps * vbSpiking
            # - Store updated state after spike
            if bRecord:
                mfRecordStates[2 * nStep + 1] = state
            # - Add filtered recurrent spikes to input
            nTSRecurrent = min(nNumTSKernel, num_timesteps - nStep)
            mfNeuralInput[nStep + 1 : nStep + 1 + nTSRecurrent] += mfKernels[
                :nTSRecurrent
            ] * torch.mm(vbSpiking.reshape(1, -1), weights_rec)

            del vbSpiking

        # - Store recorded neuron and synapse states
        if bRecord:
            self.mfRecordStates[
                2 * nEvolutionTimeStep
                + 1 : 2 * (nEvolutionTimeStep + num_timesteps)
                + 1
            ] = mfRecordStates.cpu()
            self.mfRecordSynapses[
                nEvolutionTimeStep + 1 : nEvolutionTimeStep + num_timesteps + 1
            ] = (
                mfNeuralInput[:num_timesteps]
                - self._vfVRest
                - self._vfBias  # Introduces slight numerical error in stored synapses of about 1e-9
            ).cpu()

        # - Store updated neuron and synapse states and update clock
        self._state = state
        self._vSynapseState = mfNeuralInput[-1].clone()
        self._timestep += num_timesteps

        return mbSpiking.cpu()


## - RecIAFSpkInRefrCLTorch - Class: like RecIAFSpkInTorch but with leak that is constant over time.
class RecIAFSpkInRefrCLTorch(RecIAFSpkInRefrTorch):
    """ RecIAFSpkInRefrCLTorch - Class: like RecIAFSpkInTorch but with leak that
                                        is constant over time.
    """

    ## - Constructor
    def __init__(
        self,
        weights_in: np.ndarray,
        weights_rec: np.ndarray,
        vfBias: Union[float, np.ndarray] = 0.0105,
        dt: float = 0.0001,
        vfLeakRate: Union[float, np.ndarray] = 0.02,
        vtTauN: Union[float, np.ndarray] = 0.02,
        vtTauSInp: Union[float, np.ndarray] = 0.05,
        vtTauSRec: Union[float, np.ndarray] = 0.05,
        vfVThresh: Union[float, np.ndarray] = -0.055,
        vfVReset: Union[float, np.ndarray] = -0.065,
        vfVRest: Union[float, np.ndarray, None] = -0.065,
        vfStateMin: Union[float, np.ndarray, None] = -0.085,
        tRefractoryTime=0,
        name: str = "unnamed",
        bRecord: bool = False,
        bAddEvents: bool = True,
        nMaxNumTimeSteps: int = nDefaultMaxNumTimeSteps,
    ):
        """
        RecIAFSpkInRefrCLTorch - Construct a spiking recurrent layer with IAF neurons, running on GPU, using torch
                                 Inputs and outputs are spiking events. Support refractoriness. Constant leak

        :param weights_in:           np.array MxN input weight matrix.
        :param weights_rec:          np.array NxN recurrent weight matrix.
        :param vfBias:          np.array Nx1 bias vector. Default: 0.0105

        :param dt:             float Time-step. Default: 0.0001
        :param vtTauN:          np.array Nx1 vector of neuron time constants. Default: 0.02

        :param vfLeakRate:      np.array Nx1 vector of constant neuron leakage in V/s. Default: 0.02
        :param vtTauSInp:       np.array Nx1 vector of synapse time constants. Default: 0.05
        :param vtTauSRec:       np.array Nx1 vector of synapse time constants. Default: 0.05

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds. Default: -0.055
        :param vfVReset:        np.array Nx1 vector of neuron reset potential. Default: -0.065
        :param vfVRest:         np.array Nx1 vector of neuron resting potential. Default: -0.065
                                If None, leak will always be negative (for positive entries of vfLeakRate)
        :param vfStateMin:      np.array Nx1 vector of lower limits for neuron states. Default: -0.85
                                If None, there are no lower limits

        :param tRefractoryTime: float Refractory period after each spike. Default: 0

        :param name:         str Name for the layer. Default: 'unnamed'

        :param bRecord:         bool Record membrane potential during evolutions. Default: False

        :bAddEvents:            bool     If during evolution multiple input events arrive during one
                                         time step for a channel, count their actual number instead of
                                         just counting them as one (This might make less sense for
                                         refractory neurons).

        :nMaxNumTimeSteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        # - Call super constructor
        super().__init__(
            weights_in=weights_in,
            weights_rec=weights_rec,
            vfBias=vfBias,
            dt=dt,
            noise_std=0,
            vtTauN=vtTauN,
            vtTauSInp=vtTauSInp,
            vtTauSRec=vtTauSRec,
            vfVThresh=vfVThresh,
            vfVReset=vfVReset,
            vfVRest=vfVRest,
            tRefractoryTime=tRefractoryTime,
            name=name,
            bRecord=bRecord,
            nMaxNumTimeSteps=nMaxNumTimeSteps,
        )
        self.vfLeakRate = vfLeakRate
        self.vfStateMin = vfStateMin

    def _single_batch_evolution(
        self,
        mfInput: np.ndarray,
        nEvolutionTimeStep: int,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input for a single batch

        :param mfInput:     np.ndarray   Input to layer as matrix
        :param nEvolutionTimeStep int    Time step within current evolution at beginning of current batch
        :param num_timesteps:   int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """
        mfNeuralInput, num_timesteps = self._prepare_neural_input(
            mfInput, num_timesteps
        )

        if self.bRecord:
            # - Tensor for recording synapse and neuron states
            mfRecordStates = self.tensors.FloatTensor(
                2 * num_timesteps, self.size
            ).fill_(0)

        # - Tensor for collecting spike data
        mbSpiking = self.tensors.ByteTensor(num_timesteps, self.size).fill_(0)

        # - Get local variables
        state = self._state.clone()
        vfAlpha = self._vfAlpha
        vLeak = self._vfLeakRate * self.dt
        vfVThresh = self._vfVThresh
        vfVReset = self._vfVReset
        vfVRest = self._vfVRest
        vfStateMin = self._vfStateMin
        bRecord = self.bRecord
        mfKernels = self._mfKernelsRec
        nNumTSKernel = mfKernels.shape[0]
        weights_rec = self._mfW
        nRefractorySteps = self._nRefractorySteps
        vnRefractoryCountdownSteps = self._vnRefractoryCountdownSteps.clone()

        if vfVRest is None:
            vLeakUpdate = vLeak

        # - Evolve neuron states
        for nStep in range(num_timesteps):
            # - Determine refractory neurons
            vbNotRefractory = (vnRefractoryCountdownSteps == 0).float()
            # - Decrement refractory countdown
            vnRefractoryCountdownSteps -= 1
            vnRefractoryCountdownSteps.clamp_(min=0)
            # - Incremental state update from input
            if vfVRest is not None:
                # - Leak moves state towards `vfVRest`
                vLeakUpdate = vLeak * (
                    (state < vfVRest).float() - (state > vfVRest).float()
                )
            state += vbNotRefractory * vfAlpha * (mfNeuralInput[nStep] + vLeakUpdate)
            if vfStateMin is not None:
                # - Keep states above lower limits
                state = torch.max(state, vfStateMin)
            # - Store updated state before spike
            if bRecord:
                mfRecordStates[2 * nStep] = state
            # - Spiking
            vbSpiking = (state > vfVThresh).float()
            # - State reset
            state += (vfVReset - state) * vbSpiking
            # - Store spikes
            mbSpiking[nStep] = vbSpiking
            # - Update refractory countdown
            vnRefractoryCountdownSteps += nRefractorySteps * vbSpiking
            # - Store updated state after spike
            if bRecord:
                mfRecordStates[2 * nStep + 1] = state
            # - Add filtered recurrent spikes to input
            nTSRecurrent = min(nNumTSKernel, num_timesteps - nStep)
            mfNeuralInput[nStep + 1 : nStep + 1 + nTSRecurrent] += mfKernels[
                :nTSRecurrent
            ] * torch.mm(vbSpiking.reshape(1, -1), weights_rec)

            del vbSpiking

        # - Store recorded neuron and synapse states
        if bRecord:
            self.mfRecordStates[
                2 * nEvolutionTimeStep
                + 1 : 2 * (nEvolutionTimeStep + num_timesteps)
                + 1
            ] = mfRecordStates.cpu()
            self.mfRecordSynapses[
                nEvolutionTimeStep + 1 : nEvolutionTimeStep + num_timesteps + 1
            ] = (mfNeuralInput[:num_timesteps]).cpu()

        # - Store updated neuron and synapse states and update clock
        self._state = state
        self._vSynapseState = mfNeuralInput[-1].clone()
        self._timestep += num_timesteps

        return mbSpiking.cpu()

    def reset_state(self):
        super().reset_state()
        # - Set previous synaptic input to 0
        self._last_synaptic = self.tensors.FloatTensor(self._state.size()).fill_(0)

    @RefProperty
    def vfLeakRate(self):
        return self._vfLeakRate

    @vfLeakRate.setter
    def vfLeakRate(self, vfNewRate):
        vfNewRate = np.asarray(self._expand_to_net_size(vfNewRate, "vfLeakRate"))
        self._vfLeakRate = torch.from_numpy(vfNewRate).to(self.device).float()

    @RefProperty
    def vfStateMin(self):
        return self._vfStateMin

    @vfStateMin.setter
    def vfStateMin(self, vfNewMin):
        if vfNewMin is None:
            self._vfStateMin = None
        else:
            vfNewMin = np.asarray(self._expand_to_net_size(vfNewMin, "vfStateMin"))
            self._vfStateMin = torch.from_numpy(vfNewMin).to(self.device).float()

    @RefProperty
    def vfVRest(self):
        return self._vfVRest

    @vfVRest.setter
    def vfVRest(self, vfNewVRest):
        if vfNewVRest is None:
            self._vfVRest = None
        else:
            vfNewVRest = np.asarray(self._expand_to_net_size(vfNewVRest, "vfVRest"))
            self._vfVRest = torch.from_numpy(vfNewVRest).to(self.device).float()
