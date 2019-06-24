###
# iaf_digital.py - Class implementing a recurrent layer consisting of
#                    digital neurons with constant leak and fixed-size
#                    integer as state. Event based.
###

# - Imports
# import sys
# strNetworkPath = sys.path[0] + '../../..'
# sys.path.insert(1, strNetworkPath)

from typing import Union, Optional, List, Tuple
import numpy as np
import heapq

from ...timeseries import TSEvent, TSContinuous

from ..layer import Layer

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Configure exports
__all__ = ["RecDIAF"]

# - Absolute tolerance, e.g. for comparing float values
tol_abs = 1e-10
# - Minimum refractory time
tMinRefractory = 1e-9


# - RecDIAF - Class: define a spiking recurrent layer based on digital IAF neurons


class RecDIAF(Layer):
    """ RecDIAF - Class: define a spiking recurrent layer based on digital IAF neurons
    """

    ## - Constructor
    def __init__(
        self,
        weights_in: np.ndarray,
        weights_rec: np.ndarray,
        dt: float = 1e-4,
        delay: float = 1e-8,
        tau_leak: float = 1e-3,
        refractory: Union[ArrayLike, float] = tMinRefractory,
        v_thresh: Union[ArrayLike, float] = 100,
        v_reset: Union[ArrayLike, float] = 0,
        v_rest: Union[ArrayLike, float, None] = None,
        leak: Union[ArrayLike, float] = 1,
        v_subtract: Union[ArrayLike, float, None] = None,
        state_type: Union[type, str] = "int8",
        monitor_id: Union[bool, int, None, ArrayLike] = [],
        name: str = "unnamed",
    ):
        """
        RecDIAF - Construct a spiking recurrent layer with digital IAF neurons

        :param weights_in:           np.array nSizeInxN input weight matrix.
        :param weights_rec:          np.array NxN weight matrix

        :param dt:             float Length of single time step

        :param delay:     float Time after which a spike within the
                                      layer arrives at the recurrent
                                      synapses of the receiving neurons
                                      within the network.
        :param tau_leak:        float Period for applying leak
        :param refractory:np.array Nx1 vector of refractory times.

        :param v_thresh:       np.array Nx1 vector of neuron thresholds.
        :param v_reset:        np.array Nx1 vector of neuron reset potentials.
        :param v_rest:         np.array Nx1 vector of neuron resting potentials.
                                Leak will change sign for neurons with state below this.
                                If None, leak will not change sign
        :param leak:         np.array Nx1 vector of leak values.

        :param v_subtract:     np.array If not None, subtract provided values
                                         from neuron state after spike.
                                         Otherwise will reset.

        :param state_type:      type data type for the membrane potential
        :param monitor_id:     array-like  IDs of neurons to be recorded

        :param name:         str Name for the layer. Default: 'unnamed'
        """

        # - Call super constructor
        super().__init__(weights=weights_in, dt=dt, name=name)

        # - Input weights must be provided
        assert (
            weights_rec is not None
        ), "Layer {}: Recurrent weights weights_rec must be provided.".format(self.name)

        # - Channel for leak
        self._leak_channel = self.size_in + self.size

        # - One large weight matrix to process input and recurrent connections
        #   as well as leak and multiple spikes if state after subtraction is
        #   still above threshold.
        self._weights_total = np.zeros((self._leak_channel + 2, self.size))

        # - Set minimum refractory time
        self._min_refractory = tMinRefractory

        # - Set neuron parameters
        self.weights = weights_rec
        self.weights_in = weights_in
        self.v_subtract = v_subtract
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.leak = leak
        self.delay = delay
        self.tau_leak = tau_leak
        self.refractory = refractory
        self.state_type = state_type
        # - Record states of these neurons
        self.monitor_id = monitor_id

        self.reset_state()

    def reset_state(self):
        """ .reset_state() - arguments:: reset the internal state of the layer
            Usage: .reset_state()
        """
        self.state = np.clip(self.v_reset, self._min_state, self._max_state).astype(
            self.state_type
        )
        # - Initialize heap and for events that are to be processed in future evolution
        self.heap_remaining_spikes = []

    def reset_time(self):
        """
        reset_time - Reset the internal clock of this layer
        """

        # - Adapt spike times in heap
        self.heap_remaining_spikes = [
            (t_time - self.t, iID) for t_time, iID in self.heap_remaining_spikes
        ]
        heapq.heapify(self.heap_remaining_spikes)
        self._timestep = 0

    ### --- State evolution
    def evolve(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: Optional[bool] = False,
    ) -> TSEvent:
        """
        evolve - Evolve the state of this layer

        :param ts_input:         TSEvent  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        :return: TimeSeries Output of this layer during evolution period
        """

        # - Prepare input and infer real duration of evolution
        event_times, event_channels, num_timesteps, t_final = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        ## -- Consider leak as periodic input spike with fixed weight

        # - Leak timings
        # First leak is at multiple of self.tau_leak
        t_first_leak = np.ceil(self.t / self.tau_leak) * self.tau_leak
        # Maximum possible number of leak steps in evolution period
        max_num_leaks = np.ceil((t_final - self.t) / self.tau_leak) + 1
        leak = np.arange(max_num_leaks) * self.tau_leak + t_first_leak
        # - Do not apply leak at t=self.t, assume it has already been applied previously
        leak = leak[np.logical_and(leak <= t_final + tol_abs, leak > self.t + tol_abs)]

        # - Include leaks in event trace, assign channel self.LeakChannel to leak
        event_channels = np.r_[event_channels, np.ones_like(leak) * self._leak_channel]
        event_times = np.r_[event_times, leak]

        # - Push spike timings and IDs to a heap, ordered by spike time
        # - Include spikes from previous evolution that might fall into this time interval
        heap_spikes = self.heap_remaining_spikes + list(
            zip(event_times, event_channels.astype(int))
        )
        heapq.heapify(heap_spikes)

        # - Store layer spike times and IDs in lists
        spike_times = []
        spike_ids = []

        # - Times when neurons are able to spike again
        t_refractory_ends = np.zeros(self.size)

        # print("prepared")

        # import time
        # t0 = time.time()

        t_time = self.t
        i = 0
        # - Iterate over spike times. Stop when t_final is exceeded.

        # - Copy instance variables to local variables
        state = self.state
        weights_total = self._weights_total
        min_state = self._min_state
        max_state = self._max_state
        leak_channel = self._leak_channel
        state_type = self.state_type
        v_rest = self.v_rest
        v_thresh = self.v_thresh
        v_reset = self.v_reset
        refractory = self.refractory
        delay = self.delay
        size_in = self.size_in
        v_subtract = self.v_subtract
        monitor_id = None if self._id_monitor.size == 0 else self._id_monitor
        name = self.name

        if monitor_id is not None:
            # - Lists for storing states, times and the channel from the heap
            states = [state[monitor_id].copy()]
            times = [t_time]
            channels = [np.nan]

        while t_time <= t_final:
            try:
                # - Iterate over spikes in temporal order
                t_time, channel = heapq.heappop(heap_spikes)
                # print(i, t_time, channel, "                       ", end="\r")
                if verbose:
                    print(
                        "Layer `{}`: Time passed: {:10.4f} of {} s.  Channel: {:4d}.  On heap: {:5d} events".format(
                            name, t_time, duration, channel, len(heap_spikes)
                        ),
                        end="\r",
                    )
            except IndexError:
                # - Stop if there are no spikes left
                break
            else:
                # print("update: ", self._weights_total[channel])

                if monitor_id is not None:
                    # - Record state before updates
                    times.append(t_time)
                    states.append(state[monitor_id].copy())
                    channels.append(channel)

                # - Only neurons that are not refractory can receive inputs
                is_not_refractory = t_refractory_ends <= t_time
                # - Resting potential: Sign of leat so that it drives neuron states to v_rest
                if v_rest is not None and channel == leak_channel:
                    state_below_rest = (
                        state[is_not_refractory] < v_rest[is_not_refractory]
                    )
                    # Flip sign of leak for corresponding neurons
                    sign = -2 * state_below_rest + 1
                    # Make sure leak is 0 when resting potential is reached
                    sign[state[is_not_refractory] == v_rest[is_not_refractory]] = 0
                else:
                    sign = 1
                # - State updates after incoming spike
                state[is_not_refractory] = np.clip(
                    state[is_not_refractory]
                    + weights_total[channel, is_not_refractory] * sign,
                    min_state,
                    max_state,
                ).astype(state_type)

                # - Neurons above threshold that are not refractory will spike
                is_spiking = np.logical_and(state >= v_thresh, is_not_refractory)

                if monitor_id is not None:
                    # - Record state after update but before subtraction/resetting
                    times.append(t_time)
                    states.append(state[monitor_id].copy())
                    channels.append(np.nan)

                if v_subtract is not None:
                    # - Subtract from states of spiking neurons
                    state[is_spiking] = np.clip(
                        state[is_spiking] - v_subtract[is_spiking], min_state, max_state
                    ).astype(state_type)
                    # - Check if among the neurons that are spiking there are still states above threshold
                    is_still_above_thresh = (state >= v_thresh) & is_spiking
                    # - Add the time(s) when they stop being refractory to the heap
                    #   on the last channel, where weights are 0, so that no neuron
                    #   states are updated but neurons that are still above threshold
                    #   can spike immediately after they stop being refractory
                    t_stop_refr = refractory[is_still_above_thresh] + t_time + tol_abs
                    # - Could use np.unique to only add each time once, but is very slow
                    # t_stop_refr = np.unique(refractory[is_still_above_thresh]) + t_time + tol_abs
                    for tStopRefr in t_stop_refr:
                        heapq.heappush(heap_spikes, (tStopRefr, -1))
                else:
                    # - Set states to reset potential
                    state[is_spiking] = v_reset[is_spiking].astype(state_type)

                if monitor_id is not None:
                    # - Record state after subtraction/resetting
                    times.append(t_time)
                    states.append(state[monitor_id].copy())
                    channels.append(np.nan)

                # - Determine times when refractory period will end for neurons that have just fired
                t_refractory_ends[is_spiking] = t_time + refractory[is_spiking]

                # - IDs of spiking neurons
                l_spike_ids = np.where(is_spiking)[0]
                # print("spiking: ", l_spike_ids)
                # - Append spike events to lists
                spike_times += [t_time] * np.sum(is_spiking)
                spike_ids += list(l_spike_ids)

                # - Append new spikes to heap
                for n_id in l_spike_ids:
                    # - Delay spikes by self.delay. Set IDs off by self.size_in in order
                    #   to distinguish them from spikes coming from the input
                    heapq.heappush(heap_spikes, (t_time + delay, n_id + size_in))
                # print("heap: ", heapq.nsmallest(5, heap_spikes))
            i += 1

        # - Update state variable
        self._state = state

        # print("finished loop")
        # print(time.time() - t0)

        # - Store remaining spikes (happening after t_final) for next call of evolution
        self.heap_remaining_spikes = heap_spikes

        # - Start and stop times for output time series
        t_start = self._timestep * self.dt
        t_stop = (self._timestep + num_timesteps) * self.dt

        # - Update time
        self._timestep += num_timesteps

        if monitor_id is not None:
            # - Store evolution of states in lists
            states = np.hstack((states, np.reshape(channels, (-1, 1))))
            self.ts_recorded = TSContinuous(times, states)

            # - Output time series
        return TSEvent(
            np.clip(spike_times, t_start, t_stop),
            spike_ids,
            num_channels=self.size,
            t_start=t_start,
            t_stop=t_stop,
        )

    def _prepare_input(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> (np.ndarray, np.ndarray, float, float):
        """
        _prepare_input - Sample input, set up time base

        :param ts_input:      TimeSeries TxM or Tx1 Input signals for this layer
        :param duration:    float Duration of the desired evolution, in seconds
        :param num_timesteps int Number of evolution time steps

        :return:
            event_times:     ndarray Event times
            event_channels:  ndarray Event channels
            num_timesteps:    int Number of evlution time steps
            t_final:           float End time of evolution
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
            num_timesteps = (duration + tol_abs) // self.dt
        else:
            assert isinstance(
                num_timesteps, int
            ), "Layer `{}`: num_timesteps must be of type int.".format(self.name)

        # - End time of evolution
        t_final = self.t + num_timesteps * self.dt

        # - Extract spike timings and channels
        if ts_input is not None:
            event_times, event_channels = ts_input(
                t_start=self.t, t_stop=(self._timestep + num_timesteps) * self.dt
            )
            if np.size(event_channels) > 0:
                # - Make sure channels are within range
                assert (
                    np.amax(event_channels) < self.size_in
                ), "Layer {}: Only channels between 0 and {} are allowed".format(
                    self.name, self.size_in - 1
                )
        else:
            event_times, event_channels = [], []

        return event_times, event_channels, num_timesteps, t_final

    def randomize_state(self):
        # - Set state to random values between reset value and theshold
        self.state = np.clip(
            (np.amin(self.v_thresh) - np.amin(self.v_reset)) * np.random.rand(self.size)
            - np.amin(self.v_reset),
            self._min_state,
            self._max_state,
        ).astype(self.state_type)

    def to_dict(self) -> dict:
        """
        to_dict - Convert parameters of `self` to a dict if they are relevant for
                  reconstructing an identical layer.
        """
        config = super().to_dict()
        config.pop("weights")
        config.pop("noise_std")
        config["weights_in"] = self.weights_in.tolist()
        config["weights_rec"] = self.weights_rec.tolist()
        config["refractory"] = self.refractory.tolist()
        config["delay"] = self.delay
        config["tau_leak"] = self.tau_leak
        config["leak"] = self.leak.tolist()
        config["v_subtract"] = self.v_subtract.tolist()
        config["v_thresh"] = self.v_thresh.tolist()
        config["v_reset"] = self.v_reset.tolist()
        config["v_rest"] = self.v_rest.tolist()
        config["state_type"] = self.state_type
        config["monitor_id"] = self.monitor_id.tolist()
        return config

    ### --- Properties

    @property
    def output_type(self):
        return TSEvent

    @property
    def input_type(self):
        return TSEvent

    @property
    def weights(self):
        return self.weights_rec

    @weights.setter
    def weights(self, new_w):
        self.weights_rec = new_w

    @property
    def weights_rec(self):
        return self._weights_total[self.size_in : self._leak_channel, :]

    @weights_rec.setter
    def weights_rec(self, new_w):

        self._weights_total[
            self.size_in : self._leak_channel, :
        ] = self._expand_to_weight_size(new_w, "weights_rec")

    @property
    def weights_in(self):
        return self._weights_total[: self.size_in, :]

    @weights_in.setter
    def weights_in(self, new_w):
        assert (
            np.size(new_w) == self.size_in * self.size
        ), "`new_w` must have [{}] elements.".format(self.size_in * self.size)

        self._weights_total[: self.size_in, :] = np.array(new_w)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        self._state = np.clip(
            self._expand_to_net_size(new_state, "state"),
            self._min_state,
            self._max_state,
        ).astype(self.state_type)

    @property
    def v_thresh(self):
        return self._v_thresh

    @v_thresh.setter
    def v_thresh(self, new_v_thresh):
        self._v_thresh = self._expand_to_net_size(new_v_thresh, "v_thresh")

    @property
    def v_reset(self):
        return self._v_reset

    @v_reset.setter
    def v_reset(self, new_v_reset):
        self._v_reset = self._expand_to_net_size(new_v_reset, "v_reset")

    @property
    def v_rest(self):
        return self._v_rest

    @v_rest.setter
    def v_rest(self, new_v_rest):
        if new_v_rest is None:
            self._v_rest = None
        else:
            self._v_rest = self._expand_to_net_size(new_v_rest, "v_rest")

    @property
    def leak(self):
        return -self._weights_total[self._leak_channel, :]

    @leak.setter
    def leak(self, new_leak):
        self._weights_total[self._leak_channel, :] = self._expand_to_net_size(
            -new_leak, "leak"
        )

    @property
    def v_subtract(self):
        return self._v_subtract

    @v_subtract.setter
    def v_subtract(self, new_v_state):
        if new_v_state is None:
            self._v_subtract = None
        else:
            self._v_subtract = self._expand_to_net_size(new_v_state, "v_subtract")

    @property
    def refractory(self):
        return self._refractory

    @refractory.setter
    def refractory(self, new_refractory):

        self._refractory = np.clip(
            self._expand_to_net_size(new_refractory, "refractory"),
            max(0, self._min_refractory),
            None,
        )

        if (np.array(new_refractory) < self._min_refractory).any():
            print(
                "Refractory times must be at least {}.".format(self._min_refractory)
                + " Lower values have been clipped. The minimum value can be"
                + " set by changing _min_refractory."
            )

    @Layer.dt.setter
    def dt(self, _):
        raise ValueError("The `dt` property cannot be set for this layer")

    @property
    def tau_leak(self):
        return self._tau_leak

    @tau_leak.setter
    def tau_leak(self, new_tau_leak):
        assert (
            np.isscalar(new_tau_leak) and new_tau_leak > 0
        ), "`new_tau_leak` must be a scalar greater than 0."

        self._tau_leak = new_tau_leak

    @property
    def delay(self):
        return self._delay

    @delay.setter
    def delay(self, new_delay):
        assert (
            np.isscalar(new_delay) and new_delay > 0
        ), "`new_delay` must be a scalar greater than 0."

        self._delay = new_delay

    @property
    def state_type(self):
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

    @property
    def monitor_id(self):
        return self._id_monitor

    @monitor_id.setter
    def monitor_id(self, new_ids):
        if new_ids is True:
            self._id_monitor = np.arange(self.size)
        elif new_ids is None or new_ids is False or np.size(new_ids) == 0:
            self._id_monitor = np.array([])
        else:
            self._id_monitor = np.array(new_ids)
