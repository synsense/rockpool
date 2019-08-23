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
from ....utilities import RefProperty
from ...layer import Layer

# - Configure exports
__all__ = ["FFIAFTorch", "FFIAFSpkInTorch", "RecIAFTorch", "RecIAFSpkInTorch"]

# - Absolute tolerance, e.g. for comparing float values
tol_abs = 1e-9
# - Default maximum numbers of time steps for a single evolution batch
MAX_NUM_TIMESTEPS_DEFAULT = 400


## - _RefractoryBase - Class: Base class for providing refractoriness-related properties
##                            and methods so that refractory layers can inherit them
class _RefractoryBase:
    def _single_batch_evolution(
        self,
        inp: np.ndarray,
        evolution_timestep: int,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input for a single batch

        :param inp:     np.ndarray   Input to layer as matrix
        :param evolution_timestep int    Time step within current evolution at beginning of current batch
        :param num_timesteps:   int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """
        # - Get synapse input to neurons
        neural_input, num_timesteps = self._prepare_neural_input(inp, num_timesteps)

        # - Update synapse state to end of evolution before rest potential and bias are added to input
        self._synapse_state = neural_input[-1].clone()

        if self.record:
            # - Tensor for recording synapse and neuron states
            record_states = self.tensors.FloatTensor(
                2 * num_timesteps, self.size
            ).fill_(0)
            # - Store synapse states
            self.synapse_recording[
                evolution_timestep + 1 : evolution_timestep + num_timesteps + 1
            ] = neural_input.cpu()

        # - Tensor for collecting spike data
        matr_is_spiking = self.tensors.ByteTensor(num_timesteps, self.size).fill_(0)

        # - Get local variables
        state = self._state.clone()
        alpha = self._alpha
        v_thresh = self._v_thresh
        v_reset = self._v_reset
        record = self.record
        num_refractory_steps = self._num_refractory_steps
        nums_refr_ctdwn_steps = self._nums_refr_ctdwn_steps.clone()

        # - Include resting potential and bias in input for fewer computations
        neural_input += self._v_rest + self._bias

        # - Evolve neuron states
        for step in range(num_timesteps):
            # - Determine refractory neurons
            is_not_refractory = (nums_refr_ctdwn_steps == 0).float()
            # - Decrement refractory countdown
            nums_refr_ctdwn_steps -= 1
            nums_refr_ctdwn_steps.clamp_(min=0)
            # - Incremental state update from input
            state += alpha * (neural_input[step] - state) * is_not_refractory
            # - Store updated state before spike
            if record:
                record_states[2 * step] = state
            # - Spiking
            is_spiking = (state > v_thresh).float()
            # - State reset
            state += (v_reset - state) * is_spiking
            # - Store spikes
            matr_is_spiking[step] = is_spiking
            # - Update refractory countdown
            nums_refr_ctdwn_steps += num_refractory_steps * is_spiking
            # - Store updated state after spike
            if record:
                record_states[2 * step + 1] = state
            del is_spiking

        # - Store recorded neuron states
        if record:
            self.record_states[
                2 * evolution_timestep
                + 1 : 2 * (evolution_timestep + num_timesteps)
                + 1
            ] = record_states.cpu()

        # - Store updated state and update clock
        self._state = state
        self._nums_refr_ctdwn_steps = nums_refr_ctdwn_steps
        self._timestep += num_timesteps

        return matr_is_spiking.cpu()

    def reset_state(self):
        """ .reset_state() - arguments:: reset the internal state of the layer
            Usage: .reset_state()
        """
        self.state = self.v_reset
        self.synapse_state = 0
        self._nums_refr_ctdwn_steps = torch.zeros(self.size).to(self.device)

    ### --- Properties

    @property
    def refractory(self):
        return self._num_refractory_steps * self.dt

    @refractory.setter
    def refractory(self, tNewTime):
        self._num_refractory_steps = int(np.round(tNewTime / self.dt))

    @property
    def t_refr_countdown(self):
        return self._nums_refr_ctdwn_steps.cpu().numpy() * self.dt


## - FFIAFTorch - Class: define a spiking feedforward layer with spiking outputs
class FFIAFTorch(Layer):
    """ FFIAFTorch - Class: define a spiking feedforward layer with spiking outputs
    """

    ## - Constructor
    def __init__(
        self,
        weights: np.ndarray,
        bias: Union[float, np.ndarray] = 0.015,
        dt: float = 0.0001,
        noise_std: float = 0,
        tau_mem: Union[float, np.ndarray] = 0.02,
        v_thresh: Union[float, np.ndarray] = -0.055,
        v_reset: Union[float, np.ndarray] = -0.065,
        v_rest: Union[float, np.ndarray] = -0.065,
        name: str = "unnamed",
        record: bool = False,
        max_num_timesteps: int = MAX_NUM_TIMESTEPS_DEFAULT,
    ):
        """
        FFIAFTorch - Construct a spiking feedforward layer with IAF neurons, running on GPU, using torch
                     Inputs are continuous currents; outputs are spiking events

        :param weights:             np.array MxN weight matrix.
        :param bias:          np.array Nx1 bias vector. Default: 10mA

        :param dt:             float Time-step. Default: 0.1 ms
        :param noise_std:       float Noise std. dev. per second. Default: 0

        :param tau_mem:          np.array Nx1 vector of neuron time constants. Default: 20ms

        :param v_thresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param v_reset:        np.array Nx1 vector of neuron thresholds. Default: -65mV
        :param v_rest:         np.array Nx1 vector of neuron thresholds. Default: -65mV

        :param name:         str Name for the layer. Default: 'unnamed'

        :param record:         bool Record membrane potential during evolutions

        :max_num_timesteps:      int   Maximum number of timesteps during single evolution batch. Longer
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
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.tau_mem = tau_mem
        self.bias = bias
        self.weights = weights
        self.record = record
        self.max_num_timesteps = max_num_timesteps

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
        timestep_start = self._timestep

        # - Prepare input signal
        inp, num_timesteps = self._prepare_input(ts_input, duration, num_timesteps)

        # - Tensor for collecting output spike raster
        matr_is_spiking = torch.ByteTensor(num_timesteps, self.size).fill_(0)

        # - Tensor for recording states
        if self.record:
            self.record_states = (
                self.tensors.FloatTensor(2 * num_timesteps + 1, self.size)
                .fill_(0)
                .cpu()
            )
            self.synapse_recording = (
                self.tensors.FloatTensor(num_timesteps + 1, self.size).fill_(0).cpu()
            )
            self.record_states[0] = self._state
            self.synapse_recording[0] = self._synapse_state

        # - Iterate over batches and run evolution
        idx_curr = 0
        for matr_input_curr, num_ts_curr in self._batch_data(
            inp, num_timesteps, self.max_num_timesteps
        ):
            matr_is_spiking[
                idx_curr : idx_curr + num_ts_curr
            ] = self._single_batch_evolution(
                matr_input_curr, idx_curr, num_ts_curr, verbose
            )
            idx_curr += num_ts_curr

        # - Store recorded states in timeseries
        if self.record:
            rec_times_states = np.repeat(
                (timestep_start + np.arange(num_timesteps + 1)) * self.dt, 2
            )[1:]
            rec_times_synapses = (
                timestep_start + np.arange(num_timesteps + 1)
            ) * self.dt
            self.ts_rec_states = TSContinuous(
                rec_times_states, self.record_states.numpy()
            )
            self.ts_rec_synapses = TSContinuous(
                rec_times_synapses, self.synapse_recording.numpy()
            )

        # - Start and stop times for output time series
        t_start = timestep_start * self.dt
        t_stop = (timestep_start + num_timesteps) * self.dt

        # - Output timeseries
        if (matr_is_spiking == 0).all():
            event_out = TSEvent(
                None, t_start=t_start, t_stop=t_stop, num_channels=self.size
            )
        else:
            spiketime_indices, channels = torch.nonzero(matr_is_spiking).t()
            spike_times = (timestep_start + spiketime_indices + 1).float() * self.dt

            event_out = TSEvent(
                times=np.clip(
                    spike_times.numpy(), t_start, t_stop - tol_abs * 10 ** 6
                ),  # Clip due to possible numerical errors
                channels=channels.numpy(),
                num_channels=self.size,
                name="Layer `{}` spikes".format(self.name),
                t_start=t_start,
                t_stop=t_stop,
            )

        return event_out

    # @profile
    def _batch_data(
        self, inp: np.ndarray, num_timesteps: int, max_num_timesteps: int = None
    ) -> (np.ndarray, int):
        """_batch_data: Generator that returns the data in batches"""
        # - Handle None for max_num_timesteps
        max_num_timesteps = (
            num_timesteps if max_num_timesteps is None else max_num_timesteps
        )
        n_start = 0
        while n_start < num_timesteps:
            # - Endpoint of current batch
            n_end = min(n_start + max_num_timesteps, num_timesteps)
            # - Data for current batch
            matr_input_curr = inp[n_start:n_end]
            yield matr_input_curr, n_end - n_start
            # - Update n_start
            n_start = n_end

    # @profile
    def _single_batch_evolution(
        self,
        inp: np.ndarray,
        evolution_timestep: int,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input for a single batch

        :param inp:     np.ndarray   Input to layer as matrix
        :param evolution_timestep int    Time step within current evolution at beginning of current batch
        :param num_timesteps:   int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        # - Get synapse input to neurons
        neural_input, num_timesteps = self._prepare_neural_input(inp, num_timesteps)

        # - Update synapse state to end of evolution before rest potential and bias are added to input
        self._synapse_state = neural_input[-1].clone()

        if self.record:
            # - Tensor for recording synapse and neuron states
            record_states = self.tensors.FloatTensor(
                2 * num_timesteps, self.size
            ).fill_(0)
            # - Store synapse states
            self.synapse_recording[
                evolution_timestep + 1 : evolution_timestep + num_timesteps + 1
            ] = neural_input.cpu()

        # - Tensor for collecting spike data
        matr_is_spiking = self.tensors.ByteTensor(num_timesteps, self.size).fill_(0)

        # - Get local variables
        state = self._state.clone()
        alpha = self._alpha
        v_thresh = self._v_thresh
        v_reset = self._v_reset
        record = self.record

        # - Include resting potential and bias in input for fewer computations
        neural_input += self._v_rest + self._bias

        # - Evolve neuron states
        for step in range(num_timesteps):
            # - Incremental state update from input
            state += alpha * (neural_input[step] - state)
            # - Store updated state before spike
            if record:
                record_states[2 * step] = state
            # - Spiking
            is_spiking = (state > v_thresh).float()
            # - State reset
            state += (v_reset - state) * is_spiking
            # - Store spikes
            matr_is_spiking[step] = is_spiking
            # - Store updated state after spike
            if record:
                record_states[2 * step + 1] = state
            del is_spiking

        # - Store recorded neuron states
        if record:
            self.record_states[
                2 * evolution_timestep
                + 1 : 2 * (evolution_timestep + num_timesteps)
                + 1
            ] = record_states.cpu()

        # - Store updated state and update clock
        self._state = state
        self._timestep += num_timesteps

        return matr_is_spiking.cpu()

    # @profile
    def _prepare_neural_input(
        self, inp: np.array, num_timesteps: Optional[int] = None
    ) -> (np.ndarray, int):
        """
        _prepare_neural_input : Prepare the weighted, noisy synaptic input to the neurons
                                and return it together with number of evolution time steps

        :param inp:     np.ndarray   Input to layer as matrix
        :param num_timesteps    int      Number of evolution time steps
        :return:
                neural_input   np.ndarray  Input to neurons
                num_timesteps   int         Number of evolution time steps

        """
        # - Prepare inp
        inp = torch.from_numpy(inp).float().to(self.device)
        # - Weight inputs
        neural_input = torch.mm(inp, self._weights)

        # - Add noise trace
        if self.noise_std > 0:
            neural_input += (
                torch.randn(num_timesteps, self.size).float().to(self.device)
                # - Standard deviation slightly smaller than expected (due to brian??),
                #   therefore correct with empirically found factor 1.63
                * self.noise_std
                * torch.sqrt(2.0 * self._tau_mem / self.dt)
                * 1.63
            )

        return neural_input, num_timesteps

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

        :return: (time_base, input_steps, duration)
            input_steps:    ndarray (T1xN) Discretised input signal for layer
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
        time_base = self._gen_time_trace(self.t, num_timesteps)

        if ts_input is not None:
            # - Make sure time_base matches ts_input
            if not isinstance(ts_input, TSEvent):
                if not ts_input.periodic:
                    # - If time base limits are very slightly beyond ts_input.t_start and ts_input.t_stop, match them
                    if (
                        ts_input.t_start - 1e-3 * self.dt
                        <= time_base[0]
                        <= ts_input.t_start
                    ):
                        time_base[0] = ts_input.t_start
                    if (
                        ts_input.t_stop
                        <= time_base[-1]
                        <= ts_input.t_stop + 1e-3 * self.dt
                    ):
                        time_base[-1] = ts_input.t_stop

                # - Warn if evolution period is not fully contained in ts_input
                if not (ts_input.contains(time_base) or ts_input.periodic):
                    print(
                        "Layer `{}`: Evolution period (t = {} to {}) ".format(
                            self.name, time_base[0], time_base[-1]
                        )
                        + "not fully contained in input signal (t = {} to {})".format(
                            ts_input.t_start, ts_input.t_stop
                        )
                    )

            # - Sample input trace and check for correct dimensions
            input_steps = self._check_input_dims(ts_input(time_base))

            # - Treat "NaN" as zero inputs
            input_steps[np.isnan(input_steps)] = 0

        else:
            # - Assume zero inputs
            input_steps = np.zeros((num_timesteps, self.size_in))

        return (input_steps, num_timesteps)

    def reset_state(self):
        """ .reset_state() - arguments:: reset the internal state of the layer
            Usage: .reset_state()
        """
        self.state = self.v_reset
        self.synapse_state = 0

    def to_dict(self):

        essential_dict = {}
        essential_dict["name"] = self.name
        essential_dict["weights"] = self._weights.cpu().tolist()
        essential_dict["dt"] = self.dt
        essential_dict["noise_std"] = self.noise_std
        essential_dict["max_num_timesteps"] = self.max_num_timesteps
        essential_dict["v_thresh"] = self._v_thresh.cpu().tolist()
        essential_dict["v_reset"] = self._v_reset.cpu().tolist()
        essential_dict["v_rest"] = self._v_reset.cpu().tolist()
        essential_dict["tau_mem"] = self._tau_mem.cpu().tolist()
        essential_dict["bias"] = self._bias.cpu().tolist()
        essential_dict["record"] = self.record
        essential_dict["class_name"] = "FFIAFTorch"

        return essential_dict

    def save(self, essential_dict, filename):
        with open(filename, "w") as f:
            json.dump(essential_dict, f)

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
            bias=config["bias"],
            dt=config["dt"],
            noise_std=config["noise_std"],
            tau_mem=config["tau_mem"],
            v_thresh=config["v_thresh"],
            v_reset=config["v_reset"],
            v_rest=config["v_rest"],
            name=config["name"],
            record=config["record"],
            max_num_timesteps=config["max_num_timesteps"],
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
            bias=config["bias"],
            dt=config["dt"],
            noise_std=config["noise_std"],
            tau_mem=config["tau_mem"],
            v_thresh=config["v_thresh"],
            v_reset=config["v_reset"],
            v_rest=config["v_rest"],
            name=config["name"],
            record=config["record"],
            max_num_timesteps=config["max_num_timesteps"],
        )

    ### --- Properties

    @property
    def output_type(self):
        return TSEvent

    @RefProperty
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        new_state = np.asarray(self._expand_to_net_size(new_state, "state"))
        self._state = torch.from_numpy(new_state).to(self.device).float()

    @RefProperty
    def tau_mem(self):
        return self._tau_mem

    @tau_mem.setter
    def tau_mem(self, new_tau_mem):
        new_tau_mem = np.asarray(self._expand_to_net_size(new_tau_mem, "tau_mem"))
        self._tau_mem = torch.from_numpy(new_tau_mem).to(self.device).float()
        if (self.dt >= self._tau_mem).any():
            print(
                "Layer `{}`: dt is larger than some of the tau_mem. This can cause numerical instabilities.".format(
                    self.name
                )
            )

    @property
    def alpha(self):
        warn(
            "Layer `{}`: Changing values of returned object by item assignment will not have effect on layer's alpha".format(
                self.name
            )
        )
        return self._alpha.cpu().numpy()

    @property
    def _alpha(self):
        return self.dt / self._tau_mem

    @RefProperty
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, new_bias):
        new_bias = np.asarray(self._expand_to_net_size(new_bias, "bias"))
        self._bias = torch.from_numpy(new_bias).to(self.device).float()

    @RefProperty
    def v_thresh(self):
        return self._v_thresh

    @v_thresh.setter
    def v_thresh(self, new_v_thresh):
        new_v_thresh = np.asarray(self._expand_to_net_size(new_v_thresh, "v_thresh"))
        self._v_thresh = torch.from_numpy(new_v_thresh).to(self.device).float()

    @RefProperty
    def v_rest(self):
        return self._v_rest

    @v_rest.setter
    def v_rest(self, new_v_rest):
        new_v_rest = np.asarray(self._expand_to_net_size(new_v_rest, "v_rest"))
        self._v_rest = torch.from_numpy(new_v_rest).to(self.device).float()

    @RefProperty
    def v_reset(self):
        return self._v_reset

    @v_reset.setter
    def v_reset(self, new_v_reset):
        new_v_reset = np.asarray(self._expand_to_net_size(new_v_reset, "v_reset"))
        self._v_reset = torch.from_numpy(new_v_reset).to(self.device).float()

    @RefProperty
    def synapse_state(self):
        return self._synapse_state

    @synapse_state.setter
    def synapse_state(self, new_state):
        new_state = np.asarray(self._expand_to_net_size(new_state, "synapse_state"))
        self._synapse_state = torch.from_numpy(new_state).to(self.device).float()

    @property
    def t(self):
        return self._timestep * self.dt

    @RefProperty
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, new_w):
        new_w = self._expand_to_shape(
            new_w, (self.size_in, self.size), "weights", allow_none=False
        )
        self._weights = torch.from_numpy(new_w).to(self.device).float()


## - FFIAFRefrTorch - Class: define a spiking feedforward layer with spiking outputs and refractoriness
class FFIAFRefrTorch(_RefractoryBase, FFIAFTorch):
    """ FFIAFRefrTorch - Class: define a spiking feedforward layer with spiking outputs and refractoriness
    """

    ## - Constructor
    def __init__(
        self,
        weights: np.ndarray,
        bias: Union[float, np.ndarray] = 0.015,
        dt: float = 0.0001,
        noise_std: float = 0,
        tau_mem: Union[float, np.ndarray] = 0.02,
        v_thresh: Union[float, np.ndarray] = -0.055,
        v_reset: Union[float, np.ndarray] = -0.065,
        v_rest: Union[float, np.ndarray] = -0.065,
        refractory=0,
        name: str = "unnamed",
        record: bool = False,
        max_num_timesteps: int = MAX_NUM_TIMESTEPS_DEFAULT,
    ):
        """
        FFIAFRefrTorch - Construct a spiking feedforward layer with IAF neurons, running on GPU, using torch
                         Inputs are continuous currents; outputs are spiking events. Support Refractoriness.

        :param weights:             np.array MxN weight matrix.
        :param bias:          np.array Nx1 bias vector. Default: 10mA

        :param dt:             float Time-step. Default: 0.1 ms
        :param noise_std:       float Noise std. dev. per second. Default: 0

        :param tau_mem:          np.array Nx1 vector of neuron time constants. Default: 20ms

        :param v_thresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param v_reset:        np.array Nx1 vector of neuron thresholds. Default: -65mV
        :param v_rest:         np.array Nx1 vector of neuron thresholds. Default: -65mV

        :param refractory: float Refractory period after each spike. Default: 0ms

        :param name:         str Name for the layer. Default: 'unnamed'

        :param record:         bool Record membrane potential during evolutions

        :max_num_timesteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        # - Call super constructor
        super().__init__(
            weights=np.asarray(weights),
            bias=bias,
            dt=dt,
            noise_std=noise_std,
            tau_mem=tau_mem,
            v_thresh=v_thresh,
            v_reset=v_reset,
            v_rest=v_rest,
            name=name,
            record=record,
            max_num_timesteps=max_num_timesteps,
        )

        self.refractory = refractory


# - FFIAFSpkInTorch - Class: Spiking feedforward layer with spiking in- and outputs
class FFIAFSpkInTorch(FFIAFTorch):
    """ FFIAFSpkInTorch - Class: Spiking feedforward layer with spiking in- and outputs
    """

    ## - Constructor
    def __init__(
        self,
        weights: np.ndarray,
        bias: np.ndarray = 0.01,
        dt: float = 0.0001,
        noise_std: float = 0,
        tau_mem: np.ndarray = 0.02,
        tau_syn: np.ndarray = 0.02,
        v_thresh: np.ndarray = -0.055,
        v_reset: np.ndarray = -0.065,
        v_rest: np.ndarray = -0.065,
        name: str = "unnamed",
        record: bool = False,
        max_num_timesteps: int = MAX_NUM_TIMESTEPS_DEFAULT,
    ):
        """
        FFIAFSpkInTorch - Construct a spiking feedforward layer with IAF neurons, running on GPU, using torch
                          in- and outputs are spiking events

        :param weights:             np.array MxN weight matrix.
        :param bias:          np.array Nx1 bias vector. Default: 10mA

        :param dt:             float Time-step. Default: 0.1 ms
        :param noise_std:       float Noise std. dev. per second. Default: 0

        :param tau_mem:          np.array Nx1 vector of neuron time constants. Default: 20ms
        :param tau_syn:          np.array Nx1 vector of synapse time constants. Default: 20ms

        :param v_thresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param v_reset:        np.array Nx1 vector of neuron thresholds. Default: -65mV
        :param v_rest:         np.array Nx1 vector of neuron thresholds. Default: -65mV

        :param name:         str Name for the layer. Default: 'unnamed'

        :param record:         bool Record membrane potential during evolutions

        :max_num_timesteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        # - Call super constructor
        super().__init__(
            weights=weights,
            bias=bias,
            dt=dt,
            noise_std=noise_std,
            tau_mem=tau_mem,
            v_thresh=v_thresh,
            v_reset=v_reset,
            v_rest=v_rest,
            name=name,
            record=record,
            max_num_timesteps=max_num_timesteps,
        )

        # - Record neuron parameters
        self.tau_syn = tau_syn

    # @profile
    def _prepare_neural_input(
        self, inp: np.array, num_timesteps: Optional[int] = None
    ) -> (np.ndarray, int):
        """
        _prepare_neural_input : Prepare the weighted, noisy synaptic input to the neurons
                                and return it together with number of evolution time steps

        :param inp:         np.ndarray    Input data
        :param num_timesteps    int      Number of evolution time steps
        :return:
                neural_input   np.ndarray  Input to neurons
                num_timesteps   int         Number of evolution time steps

        """
        # - Prepare inp
        inp = torch.from_numpy(inp).float().to(self.device)

        # - Weight inputs
        weighted_input = torch.mm(inp, self._weights)

        # - Add noise trace
        if self.noise_std > 0:
            weighted_input += (
                torch.randn(num_timesteps, self.size).float().to(self.device)
                # - Standard deviation slightly smaller than expected (due to brian??),
                #   therefore correct with empirically found factor 1.63
                * self.noise_std
                * torch.sqrt(2.0 * self._tau_mem / self.dt)
                * 1.63
            )

        # - Include previous synaptic states
        weighted_input[0] += self._synapse_state * torch.exp(-self.dt / self._tau_syn)

        # - Reshape input for convolution
        weighted_input = weighted_input.t().reshape(1, self.size, -1)

        # - Apply exponential filter to input
        times = (
            torch.arange(num_timesteps).to(self.device).reshape(1, -1).float() * self.dt
        )
        matr_kernels = torch.exp(-times / self._tau_syn.reshape(-1, 1))
        # - Reverse on time axis and reshape to match convention of pytorch
        matr_kernels = matr_kernels.flip(1).reshape(self.size, 1, num_timesteps)
        # - Object for applying convolution
        convSynapses = torch.nn.Conv1d(
            self.size,
            self.size,
            num_timesteps,
            padding=num_timesteps - 1,
            groups=self.size,
            bias=False,
        ).to(self.device)
        convSynapses.weight.data = matr_kernels
        # - Filtered synaptic currents
        neural_input = convSynapses(weighted_input)[0].detach().t()[:num_timesteps]

        return neural_input, num_timesteps

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
            spike_raster:    ndarray Boolean raster containing spike info
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
            spike_raster = ts_input.raster(
                dt=self.dt,
                t_start=self.t,
                t_stop=(self._timestep + num_timesteps) * self._dt,
                channels=np.arange(self.size_in),
            )
            # - Convert to supported format
            spike_raster = spike_raster.astype(int)
            # - Make sure size is correct
            spike_raster = spike_raster[:num_timesteps, :]

        else:
            spike_raster = np.zeros((num_timesteps, self.size_in))

        return spike_raster, num_timesteps

    @property
    def input_type(self):
        return TSEvent

    @RefProperty
    def tau_syn(self):
        return self._tau_syn

    @tau_syn.setter
    def tau_syn(self, new_tau_syn):
        new_tau_syn = np.asarray(self._expand_to_net_size(new_tau_syn, "tau_syn"))
        self._tau_syn = torch.from_numpy(new_tau_syn).to(self.device).float()


# - FFIAFSpkInRefrTorch - Class: Spiking feedforward layer with spiking in- and outputs and refractoriness
class FFIAFSpkInRefrTorch(_RefractoryBase, FFIAFSpkInTorch):
    """ FFIAFSpkInTorch - Class: Spiking feedforward layer with spiking in- and outputs and refractoriness
    """

    ## - Constructor
    def __init__(
        self,
        weights: np.ndarray,
        bias: np.ndarray = 0.01,
        dt: float = 0.0001,
        noise_std: float = 0,
        tau_mem: np.ndarray = 0.02,
        tau_syn: np.ndarray = 0.02,
        v_thresh: np.ndarray = -0.055,
        v_reset: np.ndarray = -0.065,
        v_rest: np.ndarray = -0.065,
        refractory=0,
        name: str = "unnamed",
        record: bool = False,
        max_num_timesteps: int = MAX_NUM_TIMESTEPS_DEFAULT,
    ):
        """
        FFIAFSpkInTorch - Construct a spiking feedforward layer with IAF neurons, running on GPU, using torch
                          in- and outputs are spiking events. Support refractoriness.

        :param weights:             np.array MxN weight matrix.
        :param bias:          np.array Nx1 bias vector. Default: 10mA

        :param dt:             float Time-step. Default: 0.1 ms
        :param noise_std:       float Noise std. dev. per second. Default: 0

        :param tau_mem:          np.array Nx1 vector of neuron time constants. Default: 20ms
        :param tau_syn:          np.array Nx1 vector of synapse time constants. Default: 20ms

        :param v_thresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param v_reset:        np.array Nx1 vector of neuron thresholds. Default: -65mV
        :param v_rest:         np.array Nx1 vector of neuron thresholds. Default: -65mV

        :param refractory: float Refractory period after each spike. Default: 0ms

        :param name:         str Name for the layer. Default: 'unnamed'

        :param record:         bool Record membrane potential during evolutions

        :max_num_timesteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        # - Call super constructor
        super().__init__(
            weights=weights,
            bias=bias,
            dt=dt,
            noise_std=noise_std,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            v_thresh=v_thresh,
            v_reset=v_reset,
            v_rest=v_rest,
            name=name,
            record=record,
            max_num_timesteps=max_num_timesteps,
        )

        self.refractory = refractory


## - RecIAFTorch - Class: define a spiking recurrent layer with spiking outputs
class RecIAFTorch(FFIAFTorch):
    """ FFIAFTorch - Class: define a spiking recurrent layer with spiking outputs
    """

    ## - Constructor
    def __init__(
        self,
        weights: np.ndarray,
        bias: Union[float, np.ndarray] = 0.015,
        dt: float = 0.0001,
        noise_std: float = 0,
        tau_mem: Union[float, np.ndarray] = 0.02,
        tau_syn_r: Union[float, np.ndarray] = 0.05,
        v_thresh: Union[float, np.ndarray] = -0.055,
        v_reset: Union[float, np.ndarray] = -0.065,
        v_rest: Union[float, np.ndarray] = -0.065,
        name: str = "unnamed",
        record: bool = False,
        max_num_timesteps: int = MAX_NUM_TIMESTEPS_DEFAULT,
    ):
        """
        FFIAFTorch - Construct a spiking recurrent layer with IAF neurons, running on GPU, using torch
                     Inputs are continuous currents; outputs are spiking events

        :param weights:             np.array MxN weight matrix.
        :param bias:          np.array Nx1 bias vector. Default: 0.015

        :param dt:             float Time-step. Default: 0.0001
        :param noise_std:       float Noise std. dev. per second. Default: 0

        :param tau_mem:          np.array Nx1 vector of neuron time constants. Default: 0.02
        :param tau_syn_r:       np.array NxN vector of recurrent synaptic time constants. Default: 0.005

        :param v_thresh:       np.array Nx1 vector of neuron thresholds. Default: -0.055
        :param v_reset:        np.array Nx1 vector of neuron thresholds. Default: -0.065
        :param v_rest:         np.array Nx1 vector of neuron thresholds. Default: -0.065

        :param name:         str Name for the layer. Default: 'unnamed'

        :param record:         bool Record membrane potential during evolutions. Default: False

        :max_num_timesteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        assert (
            np.atleast_2d(weights).shape[0] == np.atleast_2d(weights).shape[1]
        ), "Layer `{}`: weights must be a square matrix.".format(name)

        # - Call super constructor
        super().__init__(
            weights=weights,
            bias=bias,
            dt=dt,
            noise_std=noise_std,
            tau_mem=tau_mem,
            v_thresh=v_thresh,
            v_reset=v_reset,
            v_rest=v_rest,
            name=name,
            record=record,
            max_num_timesteps=max_num_timesteps,
        )

        # - Record neuron parameters
        self.tau_syn_r = tau_syn_r

    # @profile
    def _single_batch_evolution(
        self,
        inp: np.ndarray,
        evolution_timestep: int,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input for a single batch

        :param inp:     np.ndarray   Input to layer as matrix
        :param evolution_timestep int    Time step within current evolution at beginning of current batch
        :param num_timesteps:   int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        neural_input, num_timesteps = self._prepare_neural_input(inp, num_timesteps)

        if self.record:
            # - Tensor for recording synapse and neuron states
            record_states = self.tensors.FloatTensor(
                2 * num_timesteps, self.size
            ).fill_(0)

        # - Tensor for collecting spike data
        matr_is_spiking = self.tensors.ByteTensor(num_timesteps, self.size).fill_(0)

        # - Get local variables
        state = self._state.clone()
        alpha = self._alpha
        v_thresh = self._v_thresh
        v_reset = self._v_reset
        record = self.record
        matr_kernels = self._mfKernelsRec
        num_ts_kernel = matr_kernels.shape[0]
        weights_rec = self._weights

        # - Include resting potential and bias in input for fewer computations
        # - Omit latest time point, which is only used for carrying over synapse state to new batch
        neural_input[:-1] += self._v_rest + self._bias

        # - Evolve neuron states
        for step in range(num_timesteps):
            # - Incremental state update from input
            state += alpha * (neural_input[step] - state)
            # - Store updated state before spike
            if record:
                record_states[2 * step] = state
            # - Spiking
            is_spiking = (state > v_thresh).float()
            # - State reset
            state += (v_reset - state) * is_spiking
            # - Store spikes
            matr_is_spiking[step] = is_spiking
            # - Store updated state after spike
            if record:
                record_states[2 * step + 1] = state
            # - Add filtered recurrent spikes to input
            ts_recurrent = min(num_ts_kernel, num_timesteps - step)
            neural_input[step + 1 : step + 1 + ts_recurrent] += matr_kernels[
                :ts_recurrent
            ] * torch.mm(is_spiking.reshape(1, -1), weights_rec)

            del is_spiking

        # - Store recorded neuron and synapse states
        if record:
            self.record_states[
                2 * evolution_timestep
                + 1 : 2 * (evolution_timestep + num_timesteps)
                + 1
            ] = record_states.cpu()
            self.synapse_recording[
                evolution_timestep + 1 : evolution_timestep + num_timesteps + 1
            ] = (
                neural_input[:num_timesteps]
                - self._v_rest
                - self._bias  # Introduces slight numerical error in stored synapses of about 1e-9
            ).cpu()

        # - Store updated neuron and synapse states and update clock
        self._state = state
        self._synapse_state = neural_input[-1].clone()
        self._timestep += num_timesteps

        return matr_is_spiking.cpu()

    # @profile
    def _prepare_neural_input(
        self, inp: np.array, num_timesteps: Optional[int] = None
    ) -> (np.ndarray, int):
        """
        _prepare_neural_input : Prepare the noisy synaptic input to the neurons
                                and return it together with number of evolution time steps

        :param tsSpkInput:      TSContinuous  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :return:
                neural_input   np.ndarray  Input to neurons
                num_timesteps   int         Number of evolution time steps

        """

        num_timesteps = inp.shape[0] if num_timesteps is None else num_timesteps

        # - Prepare inp, with additional time step for carrying over recurrent spikes between batches
        neural_input = self.tensors.FloatTensor(num_timesteps + 1, self.size).fill_(0)
        neural_input[:-1] = torch.from_numpy(inp).float()
        # - Carry over filtered recurrent spikes from previous batch
        ts_recurrent = min(neural_input.shape[0], self._mfKernelsRec.shape[0])
        neural_input[:ts_recurrent] += (
            self._mfKernelsRec[:ts_recurrent] * self._synapse_state
        )

        # - Add noise trace
        if self.noise_std > 0:
            neural_input += (
                torch.randn(num_timesteps + 1, self.size).float().to(self.device)
                # - Standard deviation slightly smaller than expected (due to brian??),
                #   therefore correct with empirically found factor 1.63
                * self.noise_std
                * torch.sqrt(2.0 * self._tau_mem / self.dt)
                * 1.63
            )

        return neural_input, num_timesteps

    @property
    def tau_syn_r(self):
        return self._vtTauSynR.cpu().numpy()

    @tau_syn_r.setter
    def tau_syn_r(self, vtNewTauSynR):
        vtNewTauSynR = np.asarray(self._expand_to_net_size(vtNewTauSynR, "tau_syn_r"))
        if (vtNewTauSynR < self.dt).any():
            print(
                "Layer `{}`: dt is larger than some of the tau_syn_r. This can cause numerical instabilities.".format(
                    self.name
                )
            )

        self._vtTauSynR = torch.from_numpy(vtNewTauSynR).to(self.device).float()

        # - Kernel for filtering recurrent spikes
        kernel_size = 50 * int(
            np.amax(vtNewTauSynR) / self.dt
        )  # - Values smaller than ca. 1e-21 are neglected
        times = (
            torch.arange(kernel_size).to(self.device).reshape(-1, 1).float() * self.dt
        )
        self._mfKernelsRec = torch.exp(-times / self._vtTauSynR.reshape(1, -1))

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, new_dt):
        assert new_dt > 0, "Layer `{}`: dt must be greater than 0.".format(self.name)
        self._dt = new_dt
        if hasattr(self, "tau_syn_r"):
            # - Update filter for recurrent spikes if already exists
            self.tau_syn_r = self.tau_syn_r


## - RecIAFRefrTorch - Class: define a spiking recurrent layer with spiking outputs and refractoriness
class RecIAFRefrTorch(_RefractoryBase, RecIAFTorch):
    """ FFIAFRefrTorch - Class: define a spiking recurrent layer with spiking outputs and refractoriness
    """

    ## - Constructor
    def __init__(
        self,
        weights: np.ndarray,
        bias: Union[float, np.ndarray] = 0.015,
        dt: float = 0.0001,
        noise_std: float = 0,
        tau_mem: Union[float, np.ndarray] = 0.02,
        tau_syn_r: Union[float, np.ndarray] = 0.05,
        v_thresh: Union[float, np.ndarray] = -0.055,
        v_reset: Union[float, np.ndarray] = -0.065,
        v_rest: Union[float, np.ndarray] = -0.065,
        refractory=0,
        name: str = "unnamed",
        record: bool = False,
        max_num_timesteps: int = MAX_NUM_TIMESTEPS_DEFAULT,
    ):
        """
        FFIAFRefrTorch - Construct a spiking recurrent layer with IAF neurons, running on GPU, using torch
                         Inputs are continuous currents; outputs are spiking events. Support refractoriness

        :param weights:             np.array MxN weight matrix.
        :param bias:          np.array Nx1 bias vector. Default: 0.015

        :param dt:             float Time-step. Default: 0.0001
        :param noise_std:       float Noise std. dev. per second. Default: 0

        :param tau_mem:          np.array Nx1 vector of neuron time constants. Default: 0.02
        :param tau_syn_r:       np.array NxN vector of recurrent synaptic time constants. Default: 0.005

        :param v_thresh:       np.array Nx1 vector of neuron thresholds. Default: -0.055
        :param v_reset:        np.array Nx1 vector of neuron thresholds. Default: -0.065
        :param v_rest:         np.array Nx1 vector of neuron thresholds. Default: -0.065

        :param refractory: float Refractory period after each spike. Default: 0

        :param name:         str Name for the layer. Default: 'unnamed'

        :param record:         bool Record membrane potential during evolutions. Default: False

        :max_num_timesteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        # - Call super constructor
        super().__init__(
            weights=weights,
            bias=bias,
            dt=dt,
            noise_std=noise_std,
            tau_mem=tau_mem,
            tau_syn_r=tau_syn_r,
            v_thresh=v_thresh,
            v_reset=v_reset,
            v_rest=v_rest,
            name=name,
            record=record,
            max_num_timesteps=max_num_timesteps,
        )

        self.refractory = refractory

    def _single_batch_evolution(
        self,
        inp: np.ndarray,
        evolution_timestep: int,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input for a single batch

        :param inp:     np.ndarray   Input to layer as matrix
        :param evolution_timestep int    Time step within current evolution at beginning of current batch
        :param num_timesteps:   int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """
        neural_input, num_timesteps = self._prepare_neural_input(inp, num_timesteps)

        if self.record:
            # - Tensor for recording synapse and neuron states
            record_states = self.tensors.FloatTensor(
                2 * num_timesteps, self.size
            ).fill_(0)

        # - Tensor for collecting spike data
        matr_is_spiking = self.tensors.ByteTensor(num_timesteps, self.size).fill_(0)

        # - Get local variables
        state = self._state.clone()
        alpha = self._alpha
        v_thresh = self._v_thresh
        v_reset = self._v_reset
        record = self.record
        matr_kernels = self._mfKernelsRec
        num_ts_kernel = matr_kernels.shape[0]
        weights_rec = self._weights
        num_refractory_steps = self._num_refractory_steps
        nums_refr_ctdwn_steps = self._nums_refr_ctdwn_steps.clone()

        # - Include resting potential and bias in input for fewer computations
        # - Omit latest time point, which is only used for carrying over synapse state to new batch
        neural_input[:-1] += self._v_rest + self._bias

        # - Evolve neuron states
        for step in range(num_timesteps):
            # - Determine refractory neurons
            is_not_refractory = (nums_refr_ctdwn_steps == 0).float()
            # - Decrement refractory countdown
            nums_refr_ctdwn_steps -= 1
            nums_refr_ctdwn_steps.clamp_(min=0)
            # - Incremental state update from input
            state += alpha * (neural_input[step] - state) * is_not_refractory
            # - Store updated state before spike
            if record:
                record_states[2 * step] = state
            # - Spiking
            is_spiking = (state > v_thresh).float()
            # - State reset
            state += (v_reset - state) * is_spiking
            # - Store spikes
            matr_is_spiking[step] = is_spiking
            # - Update refractory countdown
            nums_refr_ctdwn_steps += num_refractory_steps * is_spiking
            # - Store updated state after spike
            if record:
                record_states[2 * step + 1] = state
            # - Add filtered recurrent spikes to input
            ts_recurrent = min(num_ts_kernel, num_timesteps - step)
            neural_input[step + 1 : step + 1 + ts_recurrent] += matr_kernels[
                :ts_recurrent
            ] * torch.mm(is_spiking.reshape(1, -1), weights_rec)

            del is_spiking

        # - Store recorded neuron and synapse states
        if record:
            self.record_states[
                2 * evolution_timestep
                + 1 : 2 * (evolution_timestep + num_timesteps)
                + 1
            ] = record_states.cpu()
            self.synapse_recording[
                evolution_timestep + 1 : evolution_timestep + num_timesteps + 1
            ] = (
                neural_input[:num_timesteps]
                - self._v_rest
                - self._bias  # Introduces slight numerical error in stored synapses of about 1e-9
            ).cpu()

        # - Store updated neuron and synapse states and update clock
        self._state = state
        self._synapse_state = neural_input[-1].clone()
        self._timestep += num_timesteps

        return matr_is_spiking.cpu()


## - RecIAFSpkInTorch - Class: define a spiking recurrent layer with spiking in- and outputs
class RecIAFSpkInTorch(RecIAFTorch):
    """ RecIAFSpkInTorch - Class: define a spiking recurrent layer with spiking in- and outputs
    """

    ## - Constructor
    def __init__(
        self,
        weights_in: np.ndarray,
        weights_rec: np.ndarray,
        bias: Union[float, np.ndarray] = 0.0105,
        dt: float = 0.0001,
        noise_std: float = 0,
        tau_mem: Union[float, np.ndarray] = 0.02,
        tau_syn_inp: Union[float, np.ndarray] = 0.05,
        tau_syn_rec: Union[float, np.ndarray] = 0.05,
        v_thresh: Union[float, np.ndarray] = -0.055,
        v_reset: Union[float, np.ndarray] = -0.065,
        v_rest: Union[float, np.ndarray] = -0.065,
        name: str = "unnamed",
        record: bool = False,
        add_events: bool = True,
        max_num_timesteps: int = MAX_NUM_TIMESTEPS_DEFAULT,
    ):
        """
        RecIAFSpkInTorch - Construct a spiking recurrent layer with IAF neurons, running on GPU, using torch
                           Inputs and outputs are spiking events

        :param weights_in:           np.array MxN input weight matrix.
        :param weights_rec:          np.array NxN recurrent weight matrix.
        :param bias:          np.array Nx1 bias vector. Default: 0.0105

        :param dt:             float Time-step. Default: 0.0001
        :param noise_std:       float Noise std. dev. per second. Default: 0

        :param tau_mem:          np.array Nx1 vector of neuron time constants. Default: 0.02
        :param tau_syn_inp:       np.array Nx1 vector of synapse time constants. Default: 0.05
        :param tau_syn_rec:       np.array Nx1 vector of synapse time constants. Default: 0.05

        :param v_thresh:       np.array Nx1 vector of neuron thresholds. Default: -0.055
        :param v_reset:        np.array Nx1 vector of neuron thresholds. Default: -0.065
        :param v_rest:         np.array Nx1 vector of neuron thresholds. Default: -0.065

        :param name:         str Name for the layer. Default: 'unnamed'

        :param record:         bool Record membrane potential during evolutions. Default: False

        :add_events:            bool     If during evolution multiple input events arrive during one
                                         time step for a channel, count their actual number instead of
                                         just counting them as one (This might make less sense for
                                         refractory neurons).

        :max_num_timesteps:      int   Maximum number of timesteps during single evolution batch. Longer
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
            type(max_num_timesteps) == int and max_num_timesteps > 0.0
        ), "Layer `{}`: max_num_timesteps must be an integer greater than 0.".format(
            self.name
        )
        self._max_num_timesteps = max_num_timesteps

        # - Record neuron parameters
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.tau_mem = tau_mem
        self.tau_syn_rec = tau_syn_rec
        self.tau_syn_inp = tau_syn_inp
        self.bias = bias
        self.weights_in = weights_in
        self.weights_rec = weights_rec
        self.record = record
        self.add_events = add_events

        # - Store "reset" state
        self.reset_all()

    # @profile
    def _prepare_neural_input(
        self, inp: np.array, num_timesteps: Optional[int] = None
    ) -> (np.ndarray, int):
        """
        _prepare_neural_input : Prepare the noisy synaptic input to the neurons
                                and return it together with number of evolution time steps

        :param inp          np.ndarray  External input spike raster
        :param num_timesteps    int         Number of evolution time steps
        :return:
                neural_input   np.ndarray  Input to neurons
                num_timesteps   int         Number of evolution time steps

        """

        num_timesteps = inp.shape[0] if num_timesteps is None else num_timesteps

        # - Prepare external input
        inp = torch.from_numpy(inp).float().to(self.device)
        # - Weigh inputs
        weighted_input = torch.mm(inp, self._weights_in)
        # - Carry over external inputs from last batch
        weighted_input[0] += self._synapse_state_inp.clone() * torch.exp(
            -self.dt / self._vtTauSInp
        )
        # - Reshape input for convolution
        weighted_input = weighted_input.t().reshape(1, self.size, -1)
        # - Apply exponential filter to external input
        times = (
            torch.arange(num_timesteps).to(self.device).reshape(1, -1).float() * self.dt
        )
        matr_input_kernels = torch.exp(-times / self._vtTauSInp.reshape(-1, 1))
        # - Reverse on time axis and reshape to match convention of pytorch
        matr_input_kernels = matr_input_kernels.flip(1).reshape(
            self.size, 1, num_timesteps
        )
        # - Object for applying convolution
        convSynapses = torch.nn.Conv1d(
            self.size,
            self.size,
            num_timesteps,
            padding=num_timesteps - 1,
            groups=self.size,
            bias=False,
        ).to(self.device)
        convSynapses.weight.data = matr_input_kernels
        # - Filtered synaptic currents
        mfFilteredExternalInput = (
            convSynapses(weighted_input)[0].detach().t()[:num_timesteps]
        )
        # - Store filtered input from last time step for carry-over to next batch
        self._synapse_state_inp = mfFilteredExternalInput[-1].clone()

        # - Prepare input to neurons, with additional time step for carrying over recurrent spikes between batches
        neural_input = self.tensors.FloatTensor(num_timesteps + 1, self.size).fill_(0)
        # - Filtered external input
        neural_input[:-1] = mfFilteredExternalInput
        # - Carry over filtered recurrent spikes from previous batch
        ts_recurrent = min(neural_input.shape[0], self._mfKernelsRec.shape[0])
        neural_input[:ts_recurrent] += (
            self._mfKernelsRec[:ts_recurrent] * self._synapse_state
        )

        # - Add noise trace
        if self.noise_std > 0:
            neural_input += (
                torch.randn(num_timesteps + 1, self.size).float().to(self.device)
                # - Standard deviation slightly smaller than expected (due to brian??),
                #   therefore correct with empirically found factor 1.63
                * self.noise_std
                * torch.sqrt(2.0 * self._tau_mem / self.dt)
                * 1.63
            )

        return neural_input, num_timesteps

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
            spike_raster:    Tensor Boolean raster containing spike info
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
            spike_raster = ts_input.raster(
                dt=self.dt,
                t_start=self.t,
                t_stop=(self._timestep + num_timesteps) * self._dt,
                channels=np.arange(
                    self.size_in
                ),  # This causes problems when ts_input has no events in some channels
                add_events=self.add_events,  # Allow for multiple input spikes per time step
            )
            # - Convert to supportedformat
            spike_raster = spike_raster.astype(int)
            # - Make sure size is correct
            spike_raster = spike_raster[:num_timesteps, :]

        else:
            spike_raster = np.zeros((num_timesteps, self.size_in))

        return spike_raster, num_timesteps

    def reset_all(self):
        super().reset_all()
        self.synapse_state_inp = 0

    def to_dict(self):

        essential_dict = {}
        essential_dict["name"] = self.name
        essential_dict["weights_rec"] = self._weights_rec.cpu().tolist()
        essential_dict["dt"] = self.dt
        essential_dict["noise_std"] = self.noise_std
        essential_dict["max_num_timesteps"] = self.max_num_timesteps
        essential_dict["v_thresh"] = self._v_thresh.cpu().tolist()
        essential_dict["v_reset"] = self._v_reset.cpu().tolist()
        essential_dict["v_rest"] = self._v_reset.cpu().tolist()
        essential_dict["tau_mem"] = self._tau_mem.cpu().tolist()
        essential_dict["tau_syn_rec"] = self._vtTauSRec.cpu().tolist()
        essential_dict["tau_syn_inp"] = self._vtTauSInp.cpu().tolist()
        essential_dict["bias"] = self._bias.cpu().tolist()
        essential_dict["weights_in"] = self._weights_in.cpu().tolist()
        essential_dict["record"] = self.record
        essential_dict["add_events"] = self.add_events
        essential_dict["class_name"] = "RecIAFSpkInTorch"

        return essential_dict

    def save(self, essential_dict, filename):
        with open(filename, "w") as f:
            json.dump(essential_dict, f)

    @staticmethod
    def load(filename):
        with open(filename, "r") as f:
            config = json.load(f)
        return RecIAFSpkInTorch(
            weights_in=config["weights_in"],
            weights_rec=config["weights_rec"],
            bias=config["bias"],
            dt=config["dt"],
            noise_std=config["noise_std"],
            tau_mem=config["tau_mem"],
            tau_syn_inp=config["tau_syn_inp"],
            tau_syn_rec=config["tau_syn_rec"],
            v_thresh=config["v_thresh"],
            v_reset=config["v_reset"],
            v_rest=config["v_rest"],
            name=config["name"],
            record=config["record"],
            add_events=config["add_events"],
            max_num_timesteps=config["max_num_timesteps"],
        )

    @staticmethod
    def load_from_dict(config):

        return RecIAFSpkInTorch(
            weights_in=config["weights_in"],
            weights_rec=config["weights_rec"],
            bias=config["bias"],
            dt=config["dt"],
            noise_std=config["noise_std"],
            tau_mem=config["tau_mem"],
            tau_syn_inp=config["tau_syn_inp"],
            tau_syn_rec=config["tau_syn_rec"],
            v_thresh=config["v_thresh"],
            v_reset=config["v_reset"],
            v_rest=config["v_rest"],
            name=config["name"],
            record=config["record"],
            add_events=config["add_events"],
            max_num_timesteps=config["max_num_timesteps"],
        )

    def _update_rec_kernel(self):
        # - Kernel for filtering recurrent spikes
        kernel_size = min(
            50
            * int(
                torch.max(self._vtTauSRec) / self.dt
            ),  # - Values smaller than ca. 1e-21 are neglected
            self._max_num_timesteps
            + 1,  # Kernel does not need to be larger than batch duration
        )
        times = (
            torch.arange(kernel_size).to(self.device).reshape(-1, 1).float() * self.dt
        )
        self._mfKernelsRec = torch.exp(-times / self._vtTauSRec.reshape(1, -1))
        print(
            "Layer `{}`: Recurrent filter kernels have been updated.".format(self.name)
        )

    @property
    def input_type(self):
        return TSEvent

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, new_dt):
        assert new_dt > 0, "Layer `{}`: dt must be greater than 0.".format(self.name)
        self._dt = new_dt
        if hasattr(self, "tau_syn_rec"):
            # - Update filter for recurrent spikes if already exists
            self.tau_syn_rec = self.tau_syn_rec

    @RefProperty
    def tau_syn_rec(self):
        return self._vtTauSRec

    @tau_syn_rec.setter
    def tau_syn_rec(self, vtNewTauSRec):
        vtNewTauSRec = np.asarray(self._expand_to_net_size(vtNewTauSRec, "tau_syn_rec"))
        if (vtNewTauSRec < self.dt).any():
            print(
                "Layer `{}`: dt is larger than some of the tau_syn_rec. This can cause numerical instabilities.".format(
                    self.name
                )
            )

        self._vtTauSRec = torch.from_numpy(vtNewTauSRec).to(self.device).float()
        self._update_rec_kernel()

    @RefProperty
    def tau_syn_inp(self):
        return self._vtTauSInp

    @tau_syn_inp.setter
    def tau_syn_inp(self, vtNewTauSInp):
        vtNewTauSInp = np.asarray(self._expand_to_net_size(vtNewTauSInp, "tau_syn_inp"))
        self._vtTauSInp = torch.from_numpy(vtNewTauSInp).to(self.device).float()

    @RefProperty
    def weights_in(self):
        return self._weights_in

    @weights_in.setter
    def weights_in(self, new_w):
        new_w = self._expand_to_shape(
            new_w, (self.size_in, self.size), "weights_in", allow_none=False
        )
        self._weights_in = torch.from_numpy(new_w).to(self.device).float()

    @RefProperty
    def weights_rec(self):
        return self._weights_rec

    @weights_rec.setter
    def weights_rec(self, new_w):
        new_w = self._expand_to_shape(
            new_w, (self.size, self.size), "weights_rec", allow_none=False
        )
        self._weights_rec = torch.from_numpy(new_w).to(self.device).float()

    # weights as alias for weights_rec
    @property
    def weights(self):
        return self.weights_rec

    @weights.setter
    def weights(self, new_w):
        self.weights_rec = new_w

    # _weights as alias for _weights_rec
    @property
    def _weights(self):
        return self._weights_rec

    @_weights.setter
    def _weights(self, new_w):
        self._weights_rec = new_w

    @RefProperty
    def synapse_state_inp(self):
        return self._synapse_state_inp

    @synapse_state_inp.setter
    def synapse_state_inp(self, new_state):
        new_state = np.asarray(self._expand_to_net_size(new_state, "synapse_state_inp"))
        self._synapse_state_inp = torch.from_numpy(new_state).to(self.device).float()

    @property
    def max_num_timesteps(self):
        return self._max_num_timesteps

    @max_num_timesteps.setter
    def max_num_timesteps(self, new_max):
        assert (
            type(new_max) == int and new_max > 0.0
        ), "Layer `{}`: max_num_timesteps must be an integer greater than 0.".format(
            self.name
        )
        self._max_num_timesteps = new_max
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
        bias: Union[float, np.ndarray] = 0.0105,
        dt: float = 0.0001,
        noise_std: float = 0,
        tau_mem: Union[float, np.ndarray] = 0.02,
        tau_syn_inp: Union[float, np.ndarray] = 0.05,
        tau_syn_rec: Union[float, np.ndarray] = 0.05,
        v_thresh: Union[float, np.ndarray] = -0.055,
        v_reset: Union[float, np.ndarray] = -0.065,
        v_rest: Union[float, np.ndarray] = -0.065,
        refractory: float = 0,
        name: str = "unnamed",
        record: bool = False,
        add_events: bool = True,
        max_num_timesteps: int = MAX_NUM_TIMESTEPS_DEFAULT,
    ):
        """
        RecIAFSpkInRefrTorch - Construct a spiking recurrent layer with IAF neurons, running on GPU, using torch
                               Inputs and outputs are spiking events. Support refractoriness

        :param weights_in:           np.array MxN input weight matrix.
        :param weights_rec:          np.array NxN recurrent weight matrix.
        :param bias:          np.array Nx1 bias vector. Default: 0.0105

        :param dt:             float Time-step. Default: 0.0001
        :param noise_std:       float Noise std. dev. per second. Default: 0

        :param tau_mem:          np.array Nx1 vector of neuron time constants. Default: 0.02
        :param tau_syn_inp:       np.array Nx1 vector of synapse time constants. Default: 0.05
        :param tau_syn_rec:       np.array Nx1 vector of synapse time constants. Default: 0.05

        :param v_thresh:       np.array Nx1 vector of neuron thresholds. Default: -0.055
        :param v_reset:        np.array Nx1 vector of neuron thresholds. Default: -0.065
        :param v_rest:         np.array Nx1 vector of neuron thresholds. Default: -0.065

        :param refractory: float Refractory period after each spike. Default: 0

        :param name:         str Name for the layer. Default: 'unnamed'

        :param record:         bool Record membrane potential during evolutions. Default: False

        :add_events:            bool     If during evolution multiple input events arrive during one
                                         time step for a channel, count their actual number instead of
                                         just counting them as one (This might make less sense for
                                         refractory neurons).

        :max_num_timesteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        # - Call super constructor
        super().__init__(
            weights_in=weights_in,
            weights_rec=weights_rec,
            bias=bias,
            dt=dt,
            noise_std=noise_std,
            tau_mem=tau_mem,
            tau_syn_inp=tau_syn_inp,
            tau_syn_rec=tau_syn_rec,
            v_thresh=v_thresh,
            v_reset=v_reset,
            v_rest=v_rest,
            name=name,
            record=record,
            max_num_timesteps=max_num_timesteps,
        )

        self.refractory = refractory

    def _single_batch_evolution(
        self,
        inp: np.ndarray,
        evolution_timestep: int,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input for a single batch

        :param inp:            Input to layer as matrix
        :param evolution_timestep  Time step within current evolution at beginning of current batch
        :param num_timesteps:      Number of evolution time steps
        :param verbose:           Currently no effect, just for conformity
        :return:                   output spike series

        """
        neural_input, num_timesteps = self._prepare_neural_input(inp, num_timesteps)

        if self.record:
            # - Tensor for recording synapse and neuron states
            record_states = self.tensors.FloatTensor(
                2 * num_timesteps, self.size
            ).fill_(0)

        # - Tensor for collecting spike data
        matr_is_spiking = self.tensors.ByteTensor(num_timesteps, self.size).fill_(0)

        # - Get local variables
        state = self._state.clone()
        alpha = self._alpha
        v_thresh = self._v_thresh
        v_reset = self._v_reset
        record = self.record
        matr_kernels = self._mfKernelsRec
        num_ts_kernel = matr_kernels.shape[0]
        weights_rec = self._weights
        num_refractory_steps = self._num_refractory_steps
        nums_refr_ctdwn_steps = self._nums_refr_ctdwn_steps.clone()

        # - Include resting potential and bias in input for fewer computations
        # - Omit latest time point, which is only used for carrying over synapse state to new batch
        neural_input[:-1] += self._v_rest + self._bias

        # - Evolve neuron states
        for step in range(num_timesteps):
            # - Determine refractory neurons
            is_not_refractory = (nums_refr_ctdwn_steps == 0).float()
            # - Decrement refractory countdown
            nums_refr_ctdwn_steps -= 1
            nums_refr_ctdwn_steps.clamp_(min=0)
            # - Incremental state update from input
            state += alpha * (neural_input[step] - state) * is_not_refractory
            # - Store updated state before spike
            if record:
                record_states[2 * step] = state
            # - Spiking
            is_spiking = (state > v_thresh).float()
            # - State reset
            state += (v_reset - state) * is_spiking
            # - Store spikes
            matr_is_spiking[step] = is_spiking
            # - Update refractory countdown
            nums_refr_ctdwn_steps += num_refractory_steps * is_spiking
            # - Store updated state after spike
            if record:
                record_states[2 * step + 1] = state
            # - Add filtered recurrent spikes to input
            ts_recurrent = min(num_ts_kernel, num_timesteps - step)
            neural_input[step + 1 : step + 1 + ts_recurrent] += matr_kernels[
                :ts_recurrent
            ] * torch.mm(is_spiking.reshape(1, -1), weights_rec)

            del is_spiking

        # - Store recorded neuron and synapse states
        if record:
            self.record_states[
                2 * evolution_timestep
                + 1 : 2 * (evolution_timestep + num_timesteps)
                + 1
            ] = record_states.cpu()
            self.synapse_recording[
                evolution_timestep + 1 : evolution_timestep + num_timesteps + 1
            ] = (
                neural_input[:num_timesteps]
                - self._v_rest
                - self._bias  # Introduces slight numerical error in stored synapses of about 1e-9
            ).cpu()

        # - Store updated neuron and synapse states and update clock
        self._state = state
        self._synapse_state = neural_input[-1].clone()
        self._timestep += num_timesteps

        return matr_is_spiking.cpu()


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
        bias: Union[float, np.ndarray] = 0.0105,
        dt: float = 0.0001,
        leak_rate: Union[float, np.ndarray] = 0.02,
        tau_mem: Union[float, np.ndarray] = 0.02,
        tau_syn_inp: Union[float, np.ndarray] = 0.05,
        tau_syn_rec: Union[float, np.ndarray] = 0.05,
        v_thresh: Union[float, np.ndarray] = -0.055,
        v_reset: Union[float, np.ndarray] = -0.065,
        v_rest: Union[float, np.ndarray, None] = -0.065,
        state_min: Union[float, np.ndarray, None] = -0.085,
        refractory=0,
        name: str = "unnamed",
        record: bool = False,
        add_events: bool = True,
        max_num_timesteps: int = MAX_NUM_TIMESTEPS_DEFAULT,
    ):
        """
        RecIAFSpkInRefrCLTorch - Construct a spiking recurrent layer with IAF neurons, running on GPU, using torch
                                 Inputs and outputs are spiking events. Support refractoriness. Constant leak

        :param weights_in:           np.array MxN input weight matrix.
        :param weights_rec:          np.array NxN recurrent weight matrix.
        :param bias:          np.array Nx1 bias vector. Default: 0.0105

        :param dt:             float Time-step. Default: 0.0001
        :param tau_mem:          np.array Nx1 vector of neuron time constants. Default: 0.02

        :param leak_rate:      np.array Nx1 vector of constant neuron leakage in V/s. Default: 0.02
        :param tau_syn_inp:       np.array Nx1 vector of synapse time constants. Default: 0.05
        :param tau_syn_rec:       np.array Nx1 vector of synapse time constants. Default: 0.05

        :param v_thresh:       np.array Nx1 vector of neuron thresholds. Default: -0.055
        :param v_reset:        np.array Nx1 vector of neuron reset potential. Default: -0.065
        :param v_rest:         np.array Nx1 vector of neuron resting potential. Default: -0.065
                                If None, leak will always be negative (for positive entries of leak_rate)
        :param state_min:      np.array Nx1 vector of lower limits for neuron states. Default: -0.85
                                If None, there are no lower limits

        :param refractory: float Refractory period after each spike. Default: 0

        :param name:         str Name for the layer. Default: 'unnamed'

        :param record:         bool Record membrane potential during evolutions. Default: False

        :add_events:            bool     If during evolution multiple input events arrive during one
                                         time step for a channel, count their actual number instead of
                                         just counting them as one (This might make less sense for
                                         refractory neurons).

        :max_num_timesteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        # - Call super constructor
        super().__init__(
            weights_in=weights_in,
            weights_rec=weights_rec,
            bias=bias,
            dt=dt,
            noise_std=0,
            tau_mem=tau_mem,
            tau_syn_inp=tau_syn_inp,
            tau_syn_rec=tau_syn_rec,
            v_thresh=v_thresh,
            v_reset=v_reset,
            v_rest=v_rest,
            refractory=refractory,
            name=name,
            record=record,
            max_num_timesteps=max_num_timesteps,
        )
        self.leak_rate = leak_rate
        self.state_min = state_min

    def _single_batch_evolution(
        self,
        inp: np.ndarray,
        evolution_timestep: int,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input for a single batch

        :param inp:     np.ndarray   Input to layer as matrix
        :param evolution_timestep int    Time step within current evolution at beginning of current batch
        :param num_timesteps:   int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """
        neural_input, num_timesteps = self._prepare_neural_input(inp, num_timesteps)

        if self.record:
            # - Tensor for recording synapse and neuron states
            record_states = self.tensors.FloatTensor(
                2 * num_timesteps, self.size
            ).fill_(0)

        # - Tensor for collecting spike data
        matr_is_spiking = self.tensors.ByteTensor(num_timesteps, self.size).fill_(0)

        # - Get local variables
        state = self._state.clone()
        alpha = self._alpha
        v_leak = self._leak_rate * self.dt
        v_thresh = self._v_thresh
        v_reset = self._v_reset
        v_rest = self._v_rest
        state_min = self._state_min
        record = self.record
        matr_kernels = self._mfKernelsRec
        num_ts_kernel = matr_kernels.shape[0]
        weights_rec = self._weights
        num_refractory_steps = self._num_refractory_steps
        nums_refr_ctdwn_steps = self._nums_refr_ctdwn_steps.clone()

        if v_rest is None:
            v_leak_update = v_leak

        # - Evolve neuron states
        for step in range(num_timesteps):
            # - Determine refractory neurons
            is_not_refractory = (nums_refr_ctdwn_steps == 0).float()
            # - Decrement refractory countdown
            nums_refr_ctdwn_steps -= 1
            nums_refr_ctdwn_steps.clamp_(min=0)
            # - Incremental state update from input
            if v_rest is not None:
                # - Leak moves state towards `v_rest`
                v_leak_update = v_leak * (
                    (state < v_rest).float() - (state > v_rest).float()
                )
            state += is_not_refractory * alpha * (neural_input[step] + v_leak_update)
            if state_min is not None:
                # - Keep states above lower limits
                state = torch.max(state, state_min)
            # - Store updated state before spike
            if record:
                record_states[2 * step] = state
            # - Spiking
            is_spiking = (state > v_thresh).float()
            # - State reset
            state += (v_reset - state) * is_spiking
            # - Store spikes
            matr_is_spiking[step] = is_spiking
            # - Update refractory countdown
            nums_refr_ctdwn_steps += num_refractory_steps * is_spiking
            # - Store updated state after spike
            if record:
                record_states[2 * step + 1] = state
            # - Add filtered recurrent spikes to input
            ts_recurrent = min(num_ts_kernel, num_timesteps - step)
            neural_input[step + 1 : step + 1 + ts_recurrent] += matr_kernels[
                :ts_recurrent
            ] * torch.mm(is_spiking.reshape(1, -1), weights_rec)

            del is_spiking

        # - Store recorded neuron and synapse states
        if record:
            self.record_states[
                2 * evolution_timestep
                + 1 : 2 * (evolution_timestep + num_timesteps)
                + 1
            ] = record_states.cpu()
            self.synapse_recording[
                evolution_timestep + 1 : evolution_timestep + num_timesteps + 1
            ] = (neural_input[:num_timesteps]).cpu()

        # - Store updated neuron and synapse states and update clock
        self._state = state
        self._synapse_state = neural_input[-1].clone()
        self._timestep += num_timesteps

        return matr_is_spiking.cpu()

    def reset_state(self):
        super().reset_state()
        # - Set previous synaptic input to 0
        self._last_synaptic = self.tensors.FloatTensor(self._state.size()).fill_(0)

    @RefProperty
    def leak_rate(self):
        return self._leak_rate

    @leak_rate.setter
    def leak_rate(self, new_rate):
        new_rate = np.asarray(self._expand_to_net_size(new_rate, "leak_rate"))
        self._leak_rate = torch.from_numpy(new_rate).to(self.device).float()

    @RefProperty
    def state_min(self):
        return self._state_min

    @state_min.setter
    def state_min(self, new_min):
        if new_min is None:
            self._state_min = None
        else:
            new_min = np.asarray(self._expand_to_net_size(new_min, "state_min"))
            self._state_min = torch.from_numpy(new_min).to(self.device).float()

    @RefProperty
    def v_rest(self):
        return self._v_rest

    @v_rest.setter
    def v_rest(self, new_v_rest):
        if new_v_rest is None:
            self._v_rest = None
        else:
            new_v_rest = np.asarray(self._expand_to_net_size(new_v_rest, "v_rest"))
            self._v_rest = torch.from_numpy(new_v_rest).to(self.device).float()
