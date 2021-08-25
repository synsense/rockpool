###
# Classes implementing feedforward and recurrent layers consisting of I&F-neurons with constant leak. Clock based.
###

import numpy as np
from typing import Optional, Union
from collections import deque
from rockpool.timeseries import TSEvent, TSContinuous
from rockpool.utilities.property_arrays import ArrayLike
from rockpool.nn.layers.layer import Layer

from rockpool.nn.modules.timed_module import astimedmodule

FloatVector = Union[ArrayLike, float]

# - Try to import tqdm
try:
    from tqdm.autonotebook import tqdm

    __use_tqdm = True

except ModuleNotFoundError:
    __use_tqdm = False

    # - Define a fake tqdm shim
    def tqdm(iter, *args, **kwargs):
        return iter


# - Absolute tolerance, e.g. for comparing float values
tol_abs = 1e-9

__all__ = ["FFCLIAF", "RecCLIAF"]


class CLIAF_Base(Layer):
    """
    Abstract layer class of integrate and fire neurons with constant leak
    """

    def __init__(
        self,
        weights_in: np.ndarray,
        bias: FloatVector = 0.0,
        v_thresh: FloatVector = 8.0,
        v_reset: FloatVector = 0.0,
        v_subtract: Optional[FloatVector] = 8.0,
        dt: float = 1.0,
        monitor_id: Optional[Union[bool, int, ArrayLike]] = [],
        name: str = "unnamed",
    ):
        """
        Feedforward layer of integrate and fire neurons with constant leak

        :param FloatVector weights_in:              Input weight matrix
        :param FloatVector bias:                    Constant bias to be added to state at each time step. Default: ``0.0``
        :param FloatVector v_thresh:                Spiking threshold. Default: ``8.0``
        :param FloatVector v_reset:                 Reset potential after spike (also see param ``v_subtract``). Default: ``8.0``
        :param Optional[FloatVector] v_subtract:    If not ``None``, subtract provided values from neuron state after spike. Otherwise neurons will reset on each spike
        :param float dt:                            Time step for simulation of this layer, in s. Default: ``1.0``
        :param Optional[ArrayLike[int]] monitor_id: IDs of neurons to be recorded. Default: ``[]``, do not monitor any neurons
        :param Optional[str] name:                  Name of this layer. Default: ``'unnamed'``
        """

        # Call parent constructor
        super().__init__(weights=weights_in, dt=dt, name=name)

        # - Set neuron parameters
        self.weights_in = weights_in
        self.bias = bias
        self.v_thresh = v_thresh
        self.v_subtract = v_subtract
        self.v_reset = v_reset

        # - IDs of neurons to be recorded
        self.monitor_id = monitor_id

    def _add_to_record(
        self,
        state_time_series: list,
        t_now: float,
        id_out: Union[ArrayLike, bool] = True,
        state: Optional[np.ndarray] = None,
        debug: bool = False,
    ):
        """
        Convenience function to record current state of the layer or individual neuron

        :param list state_time_series:      A simple python list object to which the state needs to be appended
        :param float t_now:                 Current simulation time
        :param Optional[np.ndarray] id_out: Neuron IDs to record the state of. If ``True`` all the neuron's states will be added to the record. Default: ``True``, record all neurons
        :param Optional[np.ndarray] state:  If not ``None``, record this as state, otherwise record ``self.state``
        :param Optional[bool] debug:        If ``True``, print debug info. Default: ``False``, do not print debug info
        """

        state = self.state if state is None else state

        if id_out is True:
            id_out = np.arange(self.size)
        elif id_out is False:
            # - Do nothing
            return

        # Update record of state changes
        for id_out_iter in np.asarray(id_out):
            state_time_series.append([t_now, id_out_iter, state[id_out_iter]])
            if debug:
                print([t_now, id_out_iter, state[id_out_iter, 0]])

    def reset_time(self):
        """ Reset the internal clock of this layer """
        # - Set internal clock to 0
        self._timestep = 0

    def reset_state(self):
        """ Reset the internal state of this layer """
        # - Reset neuron state to 0
        self._state = self.v_reset

    def to_dict(self) -> dict:
        """
        Convert parameters of `self` to a dict if they are relevant for reconstructing an identical layer.

        :return dict:   Dictionary of layer parameters
        """
        config = super().to_dict()
        config.pop("weights")
        config.pop("noise_std")
        config["weights_in"] = self.weights_in.tolist()
        config["bias"] = self.bias.tolist()
        config["v_thresh"] = self.v_thresh.tolist()
        config["v_reset"] = self.v_reset.tolist()
        config["v_subtract"] = self.v_subtract.tolist()
        config["monitor_id"] = self.monitor_id.tolist()

        return config

    ### --- Properties

    @property
    def output_type(self):
        """
        (`.TSEvent`) Output subclass emitted by this layer (`.TSEvent`).
        """
        return TSEvent

    @property
    def input_type(self):
        """
        (`.TSEvent`) Input subclass accepted by this layer (`.TSEvent`).
        """
        return TSEvent

    @property
    def weights_in(self):
        """ (np.ndarray) Input weights for this layer [N_in, N]"""
        return self._weights_in

    @weights_in.setter
    def weights_in(self, new_w):
        assert (
            np.size(new_w) == self.size_in * self.size
        ), "`weights_in` must have [{}] elements.".format(self.size_in * self.size)
        self._weights_in = np.array(new_w).reshape(self.size_in, self.size)

    @property
    def state(self):
        """ (np.ndarray) Internal state of this layer """
        return self._state

    @state.setter
    def state(self, new_state):
        self._state = self._expand_to_net_size(new_state, "state", allow_none=False)

    @property
    def v_thresh(self):
        """ (np.ndarray) Threshold potental for the neurons in this layer [N,] """
        return self._v_thresh

    @v_thresh.setter
    def v_thresh(self, new_v_thresh):
        self._v_thresh = self._expand_to_net_size(
            new_v_thresh, "v_thresh", allow_none=False
        )

    @property
    def v_reset(self):
        """ (np.ndarray) Reset potential for the neurons in this layer [N,] """
        return self._v_reset

    @v_reset.setter
    def v_reset(self, new_v_reset):
        self._v_reset = self._expand_to_net_size(
            new_v_reset, "v_reset", allow_none=False
        )

    @property
    def v_subtract(self):
        """ (np.ndarray) Subtractive reset values for the neurons in this layer [N,] """
        return self._v_subtract

    @v_subtract.setter
    def v_subtract(self, new_v_state):
        if new_v_state is None:
            self._v_subtract = None
        else:
            self._v_subtract = self._expand_to_net_size(new_v_state, "v_subtract")

    @property
    def bias(self):
        """ (np.ndarray) Bias values for the neurons in this layer [N,] """
        return self._bias

    @bias.setter
    def bias(self, new_bias):
        self._bias = self._expand_to_net_size(new_bias, "bias", allow_none=False)

    @Layer.dt.setter
    def dt(self, new_dt):
        assert new_dt > 0, "`dt` must be greater than 0."
        self._dt = new_dt

    @property
    def monitor_id(self):
        """ (list) List of neurons that should be monitored during evolution """
        return self._id_monitor

    @monitor_id.setter
    def monitor_id(self, new_ids):
        if new_ids is True:
            self._id_monitor = np.arange(self.size)
        elif new_ids is None or new_ids is False or np.size(new_ids) == 0:
            self._id_monitor = np.array([])
        else:
            self._id_monitor = np.array(new_ids)


class FFCLIAF_Base(CLIAF_Base):
    """
    Feedforward layer of integrate and fire neurons with constant leak
    """

    def __init__(
        self,
        weights: np.ndarray,
        bias: FloatVector = 0.0,
        v_thresh: FloatVector = 8.0,
        v_reset: FloatVector = 0.0,
        v_subtract: Optional[FloatVector] = 8.0,
        dt: float = 1.0,
        monitor_id: Optional[Union[bool, int, ArrayLike]] = [],
        name: str = "unnamed",
    ):
        """
        Feedforward layer of integrate and fire neurons with constant leak

        :param np.ndarray weights:                  Input weight matrix [N_in, N]
        :param FloatVector bias:                    Constant bias to be added to state at each time step [N,]. Default: ``0.``
        :param FloatVector v_thresh:                Spiking threshold [N,]. Default: ``8.``
        :param FloatVector v_reset:                 Reset potential after spike [N,]. Default: ``0.``
        :param Optional[FloatVector] v_subtract:    If not ``None``, subtract provided values from neuron state after spike. Otherwise will reset. Default: ``8.``
        :param float dt:                            Time step for simulation of this layer, in s. Default: ``1.0``
        :param Optional[ArrayLike] monitor_id:      IDs of neurons to be recorded. Default: ``[]``, do not record neuron state
        :param Optional[str] name:                  Name of this layer. Default: ``'unnamed'``
        """

        # Call parent constructor
        super().__init__(
            weights_in=weights,
            bias=bias,
            v_thresh=v_thresh,
            v_reset=v_reset,
            v_subtract=v_subtract,
            dt=dt,
            monitor_id=monitor_id,
            name=name,
        )

        self.reset_state()

    def evolve(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        Function to evolve the states of this layer given an input

        :param Optional[TSEvent] ts_input:  Input spike train
        :param Optional[float] duration:    Simulation/Evolution time
        :param Optional[int] num_timesteps: Number of evolution time steps
        :param Optional[bool] verbose:      Currently no effect, just for conformity

        :return TSEvent:                    Output spike series
        """

        # - Generate input in rasterized form, get actual evolution duration
        __, inp_spike_raster, num_timesteps = self._prepare_input_events(
            ts_input, duration, num_timesteps
        )

        # Hold the sate of network at any time step when updated
        state_time_series = []
        spike_times = []
        spike_ids = []

        # Local variables
        state = self.state.astype(np.float32)
        v_thresh = self.v_thresh
        weights_in = self.weights_in
        bias = self.bias
        dt = self.dt
        size = self.size
        v_subtract = self.v_subtract
        v_reset = self.v_reset

        # - Indices of neurons to be monitored
        monitor_id = None if self.monitor_id.size == 0 else self.monitor_id
        # - Count number of spikes for each neuron in each time step
        vnNumSpikes = np.zeros(size, int)
        # - Time before first time step
        t_now = self.t

        if monitor_id is not None:
            # Record initial state of the network
            self._add_to_record(state_time_series, t_now)

        # Iterate over all time steps
        for cur_time_step in tqdm(range(inp_spike_raster.shape[0])):

            # - Spikes from input synapses
            is_inp_spike_raster = inp_spike_raster[cur_time_step]

            # Update neuron states
            update = is_inp_spike_raster @ weights_in

            # State update (write this way to avoid that type casting fails)
            state = state + update + bias

            # - Update current time
            t_now += dt

            if monitor_id is not None:
                # - Record state before reset
                self._add_to_record(
                    state_time_series, t_now, id_out=monitor_id, state=state
                )

            # - Reset spike counter
            vnNumSpikes[:] = 0

            # - Check threshold crossings for spikes
            rec_spike_raster = state >= v_thresh

            # - Reset or subtract from membrane state after spikes
            if v_subtract is not None:
                while rec_spike_raster.any():
                    # - Subtract from states
                    state[rec_spike_raster] -= v_subtract[rec_spike_raster]
                    # - Add to spike counter
                    vnNumSpikes[rec_spike_raster] += 1
                    # - Neurons that are still above threshold will emit another spike
                    rec_spike_raster = state >= v_thresh
            else:
                # - Add to spike counter
                vnNumSpikes = rec_spike_raster.astype(int)
                # - Reset neuron states
                state[rec_spike_raster] = v_reset[rec_spike_raster]

            # - Record spikes
            spike_times += [t_now] * np.sum(vnNumSpikes)
            spike_ids += list(np.repeat(np.arange(size), vnNumSpikes))

            if monitor_id is not None:
                # - Record state after reset
                self._add_to_record(
                    state_time_series, t_now, id_out=monitor_id, state=state
                )
            np.set_printoptions(precision=4, suppress=True)

        # - Update state
        self._state = state

        # - Start and stop times for output time series
        t_start = self._timestep * self.dt
        t_stop = (self._timestep + num_timesteps) * self.dt

        # Shift event times to middle of time bins
        spike_times = np.asarray(spike_times) - 0.5 * self.dt

        # Convert arrays to TimeSeries objects
        event_out = TSEvent(
            times=np.clip(
                spike_times, t_start, t_stop
            ),  # Clip due to possible numerical errors,
            channels=spike_ids,
            num_channels=self.size,
            t_start=t_start,
            t_stop=t_stop,
        )

        # Update time
        self._timestep += num_timesteps

        # TODO: Is there a time series object for this too?
        ts_state = np.array(state_time_series)

        # This is only for debugging purposes. Should ideally not be saved
        self._ts_state = ts_state

        return event_out

    def to_dict(self) -> dict:
        """
        Convert parameters of ``self`` to a dict if they are relevant for reconstructing an identical layer.
        """
        config = super().to_dict()
        config.pop("weights_in")
        config["weights"] = self.weights.tolist()
        return config

    # - weights as synonym for weights_in
    @property
    def weights(self):
        return self._weights_in

    @weights.setter
    def weights(self, new_w):
        self.weights_in = new_w


@astimedmodule(
    parameters=["weights", "bias"],
    states=["state", "_timestep"],
    simulation_parameters=["dt", "name", "v_thresh", "v_reset"],
)
class FFCLIAF(FFCLIAF_Base):
    pass


@astimedmodule(
    parameters=[
        "weights_in",
        "weights_rec",
        "bias",
    ],
    states=["state", "_num_rec_spikes_q", "_ts_until_refr_ends", "_timestep"],
    simulation_parameters=[
        "dt",
        "name",
        "v_thresh",
        "v_reset",
        "refractory",
    ],
)
class RecCLIAF(CLIAF_Base):
    """
    Recurrent layer of integrate and fire neurons with constant leak
    """

    def __init__(
        self,
        weights_in: np.ndarray,
        weights_rec: np.ndarray,
        bias: FloatVector = 0.0,
        v_thresh: FloatVector = 8.0,
        v_reset: FloatVector = 0.0,
        v_subtract: Optional[FloatVector] = 8.0,
        refractory: FloatVector = 0.0,
        dt: float = 1e-4,
        delay: Optional[float] = None,
        tTauBias: Optional[float] = None,
        monitor_id: Optional[Union[bool, int, ArrayLike]] = [],
        state_type: Union[type, str] = float,
        name: str = "unnamed",
    ):
        """
        Recurrent layer of integrate and fire neurons with constant leak

        :param np.ndarray weights_in:               Input weight matrix [N_in, N]
        :param np.ndarray weights_rec:              Recurrent weight matrix [N, N]

        :param FloatVector bias:                    Constant bias to be added to state at each time step [N,]. Default: ``0.``
        :param FloatVector v_thresh:                Spiking threshold [N,]. Default: ``8.``
        :param FloatVector v_reset:                 Reset potential after spike (also see param ``v_subtract``) [N,]. Default: ``0.``
        :param Optional[FloatVector] v_subtract:    [N,] If not ``None``, subtract provided values from neuron state after spike. Otherwise will reset. Default: ``8.``
        :param FloatVector refractory:              Vector of refractory times  [N,]
        :param float dt:                            Time step size in s. Default: ``0.1 ms``
        :param Optional[float] delay:               Time after which a spike within the layer arrives at the recurrent synapses of the receiving neurons within the network. Rounded down to multiple of `.dt`. Must be at least `.dt`. Default: ``None``, use `.dt`
        :param Optional[float] tTauBias:            Period for applying bias. Must be at least `.dt`. Is rounded down to multiple of `.dt`. If ``None``, will be set to `.dt`. Default: ``None``, use `.dt`
        :param Optional[ArrayLike] monitor_id:      IDs of neurons to be recorded. Default: ``[]``, do not monitor neurons. If ``True``, monitor all neurons
        :param type state_type:                     Data type for the membrane potential. Default: ``float``
        :param str name:                            Name of this layer. Default: ``'unnamed'``
        """

        # Call parent constructor
        super().__init__(
            weights_in=weights_in,
            bias=bias,
            v_thresh=v_thresh,
            v_reset=v_reset,
            v_subtract=v_subtract,
            dt=dt,
            monitor_id=monitor_id,
            name=name,
        )

        # - Set recurrent weights
        self.weights_rec = weights_rec
        self.delay = delay
        self.tTauBias = dt if tTauBias is None else tTauBias
        self.refractory = refractory
        self.state_type = state_type

        self.reset_state()

    def evolve(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        Function to evolve the states of this layer given an input

        :param Optional[TSEvent] ts_input:  Input spike trian
        :param Optional[float] duration:    Simulation/Evolution time
        :param Optional[int] num_timesteps: Number of evolution time steps
        :param bool verbose:      Show progress bar during evolution

        :return TSEvent:                    Output spike series
        """

        # - Generate input in rasterized form, get actual evolution duration
        __, inp_spike_raster, num_timesteps = self._prepare_input_events(
            ts_input, duration, num_timesteps
        )

        # Lists for recording spikes
        ts_spikes = []
        spike_ids = []

        # Local variables
        state = self.state
        v_thresh = self.v_thresh
        weights_rec = self.weights_rec
        weights_in = self.weights_in
        bias = self.bias
        size = self.size
        v_subtract = self.v_subtract
        v_reset = self.v_reset

        # - Deque of arrays with number of delayed spikes for each neuron for each time step
        num_rec_spikes_q = self._num_rec_spikes_q
        # - Array for storing new recurrent spikes
        num_rec_spikes = np.zeros(self.size, int)

        # - For each neuron store number time steps until refractoriness ends
        ts_until_refr_ends = self._ts_until_refr_ends
        ts_per_refr = self._ts_per_refr

        # - Indices of neurons to be monitored
        monitor_id = None if self.monitor_id.size == 0 else self.monitor_id

        # - Boolean array indicating evolution time steps where bias is applied
        is_bias = np.zeros(num_timesteps)
        # - Determine where bias is applied: Index i corresponds to bias taking effect at
        #   nTimeStep = self._timestep+1+i, want them when nTimeStep%_num_ts_per_bias == 0
        is_bias[
            -(self._timestep + 1) % self._num_ts_per_bias :: self._num_ts_per_bias
        ] = 1

        # - State type dependent variables
        state_type = self.state_type
        min_state = self._min_state
        max_state = self._max_state

        if monitor_id is not None:
            # States are recorded after update and after spike-triggered reset, i.e. twice per _timestep
            record = np.zeros((2 * num_timesteps + 1, monitor_id.size))
            # Record initial state of the network
            record[0, :] = state[monitor_id]

        if verbose:
            range_iter = tqdm(range(num_timesteps))
        else:
            range_iter = range(num_timesteps)

        # Iterate over all time steps
        for cur_time_step in range_iter:

            # - Spikes from input synapses
            is_inp_spike_raster = inp_spike_raster[cur_time_step]

            # Update neuron states
            update = (
                (is_inp_spike_raster @ weights_in)  # Input spikes
                + (num_rec_spikes_q.popleft() @ weights_rec)  # Recurrent spikes
                + (is_bias[cur_time_step] * bias)  # Bias
            )

            # - Only neurons that are not refractory can receive inputs and be updated
            is_refractory = ts_until_refr_ends > 0
            update[is_refractory] = 0

            # State update (write this way to avoid that type casting fails)
            state = np.clip(state + update, min_state, max_state).astype(state_type)

            if monitor_id is not None:
                # - Record state before reset
                record[2 * cur_time_step + 1] = state[monitor_id]

            # - Check threshold crossings for spikes
            is_spiking = state >= v_thresh

            # - Reset or subtract from membrane state after spikes
            if v_subtract is not None:  # - Subtract from potential
                if (
                    ts_per_refr == 0
                ).all():  # - No refractoriness - neurons can emit multiple spikes per time step
                    # - Reset recurrent spike counter
                    num_rec_spikes[:] = 0
                    while is_spiking.any():
                        # - Add to spike counter
                        num_rec_spikes[is_spiking] += 1
                        # - Subtract from states
                        state[is_spiking] = np.clip(
                            state[is_spiking] - v_subtract[is_spiking],
                            min_state,
                            max_state,
                        ).astype(state_type)
                        # - Neurons that are still above threshold will emit another spike
                        is_spiking = state >= v_thresh
                else:  # With refractoriness, at most one spike per time step is possible
                    # - Add to spike counter
                    num_rec_spikes = is_spiking.astype(int)
                    # - Reset neuron states
                    state[is_spiking] = np.clip(
                        state[is_spiking] - v_subtract[is_spiking], min_state, max_state
                    ).astype(state_type)
            else:  # - Reset potential
                # - Add to spike counter
                num_rec_spikes = is_spiking.astype(int)
                # - Reset neuron states
                state[is_spiking] = np.clip(
                    v_reset[is_spiking], min_state, max_state
                ).astype(state_type)

            if (ts_per_refr > 0).any():
                # - Update refractoryness
                ts_until_refr_ends = np.clip(ts_until_refr_ends - 1, 0, None)
                ts_until_refr_ends[is_spiking] = ts_per_refr[is_spiking]

            # - Store recurrent spikes in deque
            num_rec_spikes_q.append(num_rec_spikes)

            # - Record spikes
            ts_spikes += [cur_time_step] * np.sum(num_rec_spikes)
            spike_ids += list(np.repeat(np.arange(size), num_rec_spikes))

            if monitor_id is not None:
                # - Record state after reset
                record[2 * cur_time_step + 2] = state[monitor_id]

        # - Store IDs of neurons that would spike in furute time steps
        self._num_rec_spikes_q = num_rec_spikes_q

        # - Store refractoriness of neurons
        self._ts_until_refr_ends = ts_until_refr_ends

        # - Start and stop times for output time series
        t_start = self._timestep * self.dt
        t_stop = (self._timestep + num_timesteps) * self.dt

        # Generate output sime series with event times at middle of time bins
        spike_times = (np.array(ts_spikes) + 0.5 + self._timestep) * self.dt

        event_out = TSEvent(
            # Clip due to possible numerical errors,
            times=np.clip(spike_times, t_start, t_stop),
            channels=spike_ids,
            num_channels=self.size,
            t_start=t_start,
            t_stop=t_stop,
        )

        if monitor_id is not None:
            # - Store recorded data in timeseries
            record_times = np.repeat(
                (self._timestep + np.arange(num_timesteps + 1)) * self.dt, 2
            )[1:]
            self.ts_recorded = TSContinuous(record_times, record)

        # Update time
        self._timestep += num_timesteps

        # - Update state
        self._state = state

        return event_out

    def reset_state(self):
        """ Reset the internal state of this layer """
        # - Delete spikes that would arrive in recurrent synapses during future time steps
        #   by filling up deque with zeros
        num_ts_per_delay = self._num_rec_spikes_q.maxlen
        self._num_rec_spikes_q += [np.zeros(self.size) for _ in range(num_ts_per_delay)]

        # - Reset refractoriness
        self._ts_until_refr_ends = np.zeros(self.size, int)

        # - Reset neuron state to self.v_reset
        self.state = np.clip(self.v_reset, self._min_state, self._max_state).astype(
            self.state_type
        )

    def randomize_state(self):
        """ Randomize the internal state of this layer """
        # - Set state to random values between reset value and threshold
        self.state = np.clip(
            (np.amin(self.v_thresh) - np.amin(self.v_reset)) * np.random.rand(self.size)
            - np.amin(self.v_reset),
            self._min_state,
            self._max_state,
        ).astype(self.state_type)

    def to_dict(self) -> dict:
        """
        Convert parameters of `self` to a dict if they are relevant for reconstructing an identical layer.
        """
        config = super().to_dict()
        config["weights_in"] = self.weights_in.tolist()
        config["weights_rec"] = self.weights_rec.tolist()
        config["refractory"] = self.refractory.tolist()
        config["delay"] = self.delay
        config["tTauBias"] = self.tTauBias
        config["state_type"] = self.state_type
        return config

    ### --- Properties

    @property
    def weights(self):
        """ (np.ndarray) Recurrent weights for this layer """
        return self.weights_rec

    # - weights as synonym for weights_rec
    @weights.setter
    def weights(self, new_w):
        self.weights_rec = new_w

    @property
    def weights_rec(self):
        """ (np.ndarray) Recurrent weights for this layer """
        return self._weights_rec

    @weights_rec.setter
    def weights_rec(self, new_w):
        self._weights_rec = self._expand_to_weight_size(
            new_w, "weights_rec", allow_none=False
        )

    @property
    def tTauBias(self):
        """ (float) Time step on which to apply biases """
        return self._num_ts_per_bias * self._dt

    @tTauBias.setter
    def tTauBias(self, new_bias):
        assert (
            np.isscalar(new_bias) and new_bias >= self.dt
        ), "Layer `{}`: tTauBias must be a scalar greater than dt ({})".format(
            self.name, self.dt
        )
        # - new_bias is rounded to multiple of dt and at least dt
        self._num_ts_per_bias = int(np.floor(new_bias / self.dt))

    @property
    def delay(self):
        """ (float) Event transmission delay for the events in this layer, in s """
        return self._num_rec_spikes_q.maxlen * self._dt

    @delay.setter
    def delay(self, new_delay):
        if new_delay is None:
            num_ts_per_delay = 1
        else:
            assert (
                np.isscalar(new_delay) and new_delay >= self.dt
            ), "Layer `{}`: delay must be a scalar greater than dt ({})".format(
                self.name, self.dt
            )
            # - new_delay is rounded to multiple of dt and at least dt
            num_ts_per_delay = int(np.floor(new_delay / self.dt))

        ## -- Create a deque for holding delayed spikes
        # - Copy spikes from previous deque
        if hasattr(self, "_num_rec_spikes_q"):
            prev_spiken = list(self._num_rec_spikes_q)
            t_diff = num_ts_per_delay - len(prev_spiken)
            # - If new delay is less, some spikes will be lost
            self._num_rec_spikes_q = deque(prev_spiken, maxlen=num_ts_per_delay)
            if t_diff >= 0:
                self._num_rec_spikes_q = deque(
                    [np.zeros(self.size) for _ in range(t_diff)] + prev_spiken,
                    maxlen=num_ts_per_delay,
                )
            else:
                self._num_rec_spikes_q = deque(prev_spiken, maxlen=num_ts_per_delay)
                print(
                    "Layer `{}`: Last {} spikes in buffer have been lost due to reduction of delay.".format(
                        self.name, np.sum(np.array(prev_spiken[:-t_diff]))
                    )
                )
        else:
            self._num_rec_spikes_q = deque(
                [np.zeros(self.size) for _ in range(num_ts_per_delay)],
                maxlen=num_ts_per_delay,
            )

    @property
    def refractory(self):
        """ (float) Refractory period for the neurons in this layer """
        return None if self._ts_per_refr is None else self._ts_per_refr * self.dt

    @refractory.setter
    def refractory(self, new_refractory):
        if new_refractory is None:
            self._ts_per_refr = None
        else:
            refractory = self._expand_to_net_size(new_refractory, "refractory")
            # - refractory is rounded to multiple of dt and at least dt
            self._ts_per_refr = (np.floor(refractory / self.dt)).astype(int)

    @property
    def state(self):
        """ (np.ndarray) Internal state of the neurons in this layer [N,] """
        return self._state

    @state.setter
    def state(self, new_state):
        self._state = np.clip(
            self._expand_to_net_size(new_state, "state"),
            self._min_state,
            self._max_state,
        ).astype(self.state_type)

    @property
    def state_type(self):
        """ (type) Data type of the neuron state in this layer """
        return self._state_type

    @state_type.setter
    def state_type(self, new_type):
        if np.issubdtype(new_type, np.integer):
            # - Set limits for integer type states
            self._min_state = np.iinfo(new_type).min
            self._max_state = np.iinfo(new_type).max
        elif np.issubdtype(new_type, np.floating):
            self._min_state = np.finfo(new_type).min
            self._max_state = np.finfo(new_type).max
        else:
            raise ValueError(
                "Layer `{}`: state_type must be integer or float data type.".format(
                    self.name
                )
            )
        self._state_type = new_type
        # - Convert state to dtype
        if hasattr(self, "_state"):
            self.state = self.state
