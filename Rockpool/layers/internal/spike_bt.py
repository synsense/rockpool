###
# spike_bt - Implement a back-tick precise spike time recurrent layer, with fast and slow synapses
###

### --- Imports

from ..layer import Layer
from ...timeseries import *
import numpy as np
from typing import Union, Callable
import copy

from numba import njit

# - Try to import holoviews
try:
    import holoviews as hv
except Exception:
    pass

# - Configure exports
__all__ = ["RecFSSpikeEulerBT"]


### --- Functions implementing membrane and synapse dynamics


@njit
def neuron_dot_v(
    t,
    V,
    dt,
    I_s_S,
    I_s_F,
    I_ext,
    I_bias,
    V_rest,
    V_reset,
    V_thresh,
    tau_V,
    tau_S,
    tau_F,
):
    return (V_rest - V + I_s_S + I_s_F + I_ext + I_bias) / tau_V


@njit
def syn_dot_I(t, I, dt, I_spike, tau_Syn):
    return -I / tau_Syn + I_spike / dt


@njit
def _backstep(vCurrent, vLast, tStep, tDesiredStep):
    return (vCurrent - vLast) / tStep * tDesiredStep + vLast


### --- RecFSSpikeEulerBT class implementation


class RecFSSpikeEulerBT(Layer):
    def __init__(
        self,
        weights_fast: np.ndarray = None,
        weights_slow: np.ndarray = None,
        bias: np.ndarray = 0.,
        noise_std: float = 0.,
        tau_mem: Union[np.ndarray, float] = 20e-3,
        tau_syn_r_fast: Union[np.ndarray, float] = 1e-3,
        tau_syn_r_slow: Union[np.ndarray, float] = 100e-3,
        v_thresh: Union[np.ndarray, float] = -55e-3,
        v_reset: Union[np.ndarray, float] = -65e-3,
        v_rest: Union[np.ndarray, float] = -65e-3,
        refractory: float = -np.finfo(float).eps,
        spike_callback: Callable = None,
        dt: float = None,
        name: str = None,
    ):
        """
        Implement a spiking reservoir with tight E/I balance This class does NOT use a Brian2 back-end. See the class code for possibilities to modify neuron and synapse dynamics. Currently uses leaky IAF neurons and exponential current synapses. Note that network parameters are tightly constrained for the reservoir to work as desired. See the documentation and source publications for details.

        :param ndarray weights_fast:            [NxN] Recurrent weight matrix (fast synapses)
        :param ndarray weights_slow:            [NxN] Recurrent weight matrix (slow synapses)
        :param Optional[ArrayLike[float]] bias: [Nx1] Bias currents for each neuron
        :param Optional[float] noise_std:       Noise Std. Dev.

        :param ArrayLike[float] tau_mem:        [Nx1] Neuron time constants
        :param ArrayLike[float] tau_syn_r_fast: [Nx1] Post-synaptic neuron fast synapse TCs
        :param ArrayLike[float] tau_syn_r_slow: [Nx1] Post-synaptic neuron slow synapse TCs

        :param ArrayLike[float] v_thresh:       [Nx1] Neuron firing thresholds
        :param ArrayLike[float] v_reset:        [Nx1] Neuron reset potentials
        :param ArrayLike[float] v_rest:         [Nx1] Neuron rest potentials

        :param Optional[float] refractory:      Post-spike refractory period

        :param Callable spike_callback:         Callable(lyrSpikeBT, t_time, nSpikeInd). Spike-based learning callback function. Default: None.

        :param Optional[float] dt:              Nominal time step (Euler solver). Default: `None`, choose a reasonable `.dt` as `min(tau)`
        :param Optional[str] name:              Name of this layer. Default: `None`
        """
        # - Initialise object and set properties
        super().__init__(weights=weights_fast, noise_std=noise_std, name=name)

        # - Check weight shape
        assert weights_slow.shape[0] == weights_slow.shape[1], \
            '`weights_slow` must be a square matrix'
        assert weights_fast.shape[0] == weights_fast.shape[1], \
            '`weights_fast` must be a square matrix'
        assert weights_slow.shape[0] == weights_fast.shape[0], \
            '`weights_fast` and `weights_slow` must be the same size'

        self.weights_slow = weights_slow
        self.bias = np.asarray(bias).astype("float")
        self.tau_mem = np.asarray(tau_mem).astype("float")
        self.tau_syn_r_fast = np.asarray(tau_syn_r_fast).astype("float")
        self.tau_syn_r_slow = np.asarray(tau_syn_r_slow).astype("float")
        self.v_thresh = np.asarray(v_thresh).astype("float")
        self.v_reset = np.asarray(v_reset).astype("float")
        self.v_rest = np.asarray(v_rest).astype("float")
        self.refractory = float(refractory)
        self.spike_callback = spike_callback

        # - Set a reasonable dt
        if dt is None:
            self.dt = self._min_tau / 10
        else:
            self.dt = np.asarray(dt).astype("float")

        # - Initialise network state
        self.reset_all()

    def reset_state(self):
        """
        reset_state() - Reset the internal state of the network
        """
        self.state = self.v_rest.copy()
        self.I_s_S = np.zeros(self.size)
        self.I_s_F = np.zeros(self.size)

    @property
    def _min_tau(self):
        """
        ._min_tau - Smallest time constant of the layer
        """
        return min(np.min(self.tau_syn_r_slow), np.min(self.tau_syn_r_fast))

    def evolve(
        self,
        ts_input: TimeSeries = None,
        duration: float = None,
        num_timesteps: int = None,
        verbose: bool = False,
        min_delta: float = None,
    ) -> TimeSeries:
        """
        evolve() - Simulate the spiking reservoir, using a precise-time spike detector
            This method implements an Euler integrator, coupled with precise spike time detection using a linear
            interpolation between integration steps. Time is then reset to the spike time, and integration proceeds.
            For this reason, the time steps returned by the integrator are not homogenous. A minimum time step can be set;
            by default this is 1/10 of the nominal time step.

        :param ts_input:         TimeSeries input for a given time t [TxN]
        :param duration:       float Duration of simulation in seconds. Default: 100ms
        :param num_timesteps    int Number of evolution time steps
        :param verbose:    bool Currently no effect, just for conformity
        :param min_delta:       float Minimum time step taken. Default: 1/10 nominal TC
        :param ts_input:         TimeSeries input for a given time t [TxN]
        :param min_delta:       float Minimum time step taken. Default: 1/10 nominal TC
        :param spike_callback  Callable(lyrSpikeBT, t_time, nSpikeInd). Spike-based learning callback function. Default: None.

        :return: TimeSeries containing the output currents of the reservoir
        """

        # - Work out reasonable default for nominal time step (1/10 fastest time constant)
        if min_delta is None:
            min_delta = self.dt / 10

        # - Check time step values
        assert min_delta < self.dt, "`min_delta` must be shorter than `dt`"

        # - Get discretised input and nominal time trace
        input_time_trace, static_input, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )
        final_time = input_time_trace[-1]

        # - Generate a noise trace
        noise_step = (
            np.random.randn(np.size(input_time_trace), self.size) * self.noise_std
        )
        static_input += noise_step

        # - Allocate state storage variables
        spike_pointer = 0
        times = full_nan(num_timesteps)
        v = full_nan((self.size, num_timesteps))
        s = full_nan((self.size, num_timesteps))
        f = full_nan((self.size, num_timesteps))
        dot_v = full_nan((self.size, num_timesteps))

        # - Allocate storage for spike times
        max_spike_pointer = num_timesteps * self.size
        spike_times = full_nan(max_spike_pointer)
        spike_indices = full_nan(max_spike_pointer)

        # - Refractory time variable
        vec_refractory = np.zeros(self.size)

        # - Initialise step and "previous step" variables
        t_time = self._t
        t_start = self._t
        step = 0
        t_last = 0.
        v_last = self._state.copy()
        I_s_S_Last = self.I_s_S.copy()
        I_s_F_Last = self.I_s_F.copy()

        zeros = np.zeros(self.size)
        # spike = np.nan
        # first_spike_id = 0

        # - Euler integrator loop
        while t_time < final_time:

            ### --- Numba-compiled inner function for speed
            # @njit
            def _evolve_backstep(
                t_time,
                weights,
                weights_slow,
                state,
                I_s_S,
                I_s_F,
                dt,
                v_last,
                I_s_S_Last,
                I_s_F_Last,
                v_reset,
                v_rest,
                v_thresh,
                bias,
                tau_mem,
                tau_syn_r_slow,
                tau_syn_r_fast,
                refractory,
                vec_refractory,
                zeros,
            ):
                # - Enforce refractory period by clamping membrane potential to reset
                state[vec_refractory > 0] = v_reset[vec_refractory > 0]

                ## - Back-tick spike detector

                # - Locate spiking neurons
                spike_ids = state > v_thresh
                spike_ids = argwhere(spike_ids)
                num_spikes = np.sum(spike_ids)

                # - Were there any spikes?
                if num_spikes > 0:
                    # - Predict the precise spike times using linear interpolation
                    spike_deltas = (
                        (v_thresh[spike_ids] - v_last[spike_ids])
                        * dt
                        / (state[spike_ids] - v_last[spike_ids])
                    )

                    # - Was there more than one neuron above threshold?
                    if num_spikes > 1:
                        # - Find the earliest spike
                        spike_delta, first_spike_id = min_argmin(spike_deltas)
                        first_spike_id = spike_ids[first_spike_id]
                    else:
                        spike_delta = spike_deltas[0]
                        first_spike_id = spike_ids[0]

                    # - Find time of actual spike
                    shortest_step = t_last + min_delta
                    spike = clip_scalar(t_last + spike_delta, shortest_step, t_time)
                    spike_delta = spike - t_last

                    # - Back-step time to spike
                    t_time = spike
                    vec_refractory = vec_refractory + dt - spike_delta

                    # - Back-step all membrane and synaptic potentials to time of spike (linear interpolation)
                    state = _backstep(state, v_last, dt, spike_delta)
                    I_s_S = _backstep(I_s_S, I_s_S_Last, dt, spike_delta)
                    I_s_F = _backstep(I_s_F, I_s_F_Last, dt, spike_delta)

                    # - Apply reset to spiking neuron
                    state[first_spike_id] = v_reset[first_spike_id]

                    # - Begin refractory period for spiking neuron
                    vec_refractory[first_spike_id] = refractory

                    # - Set spike currents
                    I_spike_slow = weights_slow[:, first_spike_id]
                    I_spike_fast = weights[:, first_spike_id]

                else:
                    # - Clear spike currents
                    first_spike_id = -1
                    I_spike_slow = zeros
                    I_spike_fast = zeros

                ### End of back-tick spike detector

                # - Save synapse and neuron states for previous time step
                v_last[:] = state
                I_s_S_Last[:] = I_s_S + I_spike_slow
                I_s_F_Last[:] = I_s_F + I_spike_fast

                # - Update synapse and neuron states (Euler step)
                dot_I_s_S = syn_dot_I(t_time, I_s_S, dt, I_spike_slow, tau_syn_r_slow)
                I_s_S += dot_I_s_S * dt

                dot_I_s_F = syn_dot_I(t_time, I_s_F, dt, I_spike_fast, tau_syn_r_fast)
                I_s_F += dot_I_s_F * dt

                int_time = int((t_time - t_start) // dt)
                I_ext = static_input[int_time, :]
                dot_v = neuron_dot_v(
                    t_time,
                    state,
                    dt,
                    I_s_S,
                    I_s_F,
                    I_ext,
                    bias,
                    v_rest,
                    v_reset,
                    v_thresh,
                    tau_mem,
                    tau_syn_r_slow,
                    tau_syn_r_fast,
                )
                state += dot_v * dt

                return (
                    t_time,
                    first_spike_id,
                    dot_v,
                    state,
                    I_s_S,
                    I_s_F,
                    dt,
                    v_last,
                    I_s_S_Last,
                    I_s_F_Last,
                    vec_refractory,
                )

            ### --- END of compiled inner function

            # - Call compiled inner function
            (
                t_time,
                first_spike_id,
                dot_v,
                self._state,
                self.I_s_S,
                self.I_s_F,
                self._dt,
                v_last,
                I_s_S_Last,
                I_s_F_Last,
                vec_refractory,
            ) = _evolve_backstep(
                t_time,
                self._weights,
                self.weights_slow,
                self._state,
                self.I_s_S,
                self.I_s_F,
                self._dt,
                v_last,
                I_s_S_Last,
                I_s_F_Last,
                self.v_reset,
                self.v_rest,
                self.v_thresh,
                self.bias,
                self.tau_mem,
                self.tau_syn_r_slow,
                self.tau_syn_r_fast,
                self.refractory,
                vec_refractory,
                zeros,
            )

            # - Call spike-based learning callback
            if first_spike_id > -1 and self.spike_callback is not None:
                self.spike_callback(self, t_time, first_spike_id)

            # - Extend spike record, if necessary
            if spike_pointer >= max_spike_pointer:
                extend = int(max_spike_pointer // 2)
                spike_times = np.append(spike_times, full_nan(extend))
                spike_indices = np.append(spike_indices, full_nan(extend))
                max_spike_pointer += extend

            # - Record spiking neuron
            spike_times[spike_pointer] = t_time
            spike_indices[spike_pointer] = first_spike_id
            spike_pointer += 1

            # - Extend state storage variables, if needed
            if step >= num_timesteps:
                extend = num_timesteps
                times = np.append(times, full_nan(extend))
                v = np.append(v, full_nan((self.size, extend)), axis=1)
                s = np.append(s, full_nan((self.size, extend)), axis=1)
                f = np.append(f, full_nan((self.size, extend)), axis=1)
                dot_v = np.append(dot_v, full_nan((self.size, extend)), axis=1)
                num_timesteps += extend

            # - Store the network states for this time step
            times[step] = t_time
            v[:, step] = self._state
            s[:, step] = self.I_s_S
            f[:, step] = self.I_s_F
            dot_v[:, step] = dot_v

            # - Next nominal time step
            t_last = copy.copy(t_time)
            t_time += self._dt
            step += 1
            vec_refractory -= self.dt
        ### End of Euler integration loop

        ## - Back-step to exact final time
        self.state = _backstep(self.state, v_last, self._dt, t_time - final_time)
        self.I_s_S = _backstep(self.I_s_S, I_s_S_Last, self._dt, t_time - final_time)
        self.I_s_F = _backstep(self.I_s_F, I_s_F_Last, self._dt, t_time - final_time)

        ## - Store the network states for final time step
        times[step - 1] = final_time
        v[:, step - 1] = self.state
        s[:, step - 1] = self.I_s_S
        f[:, step - 1] = self.I_s_F

        ## - Trim state storage variables
        times = times[:step]
        v = v[:, :step]
        s = s[:, :step]
        f = f[:, :step]
        dot_v = dot_v[:, :step]
        spike_times = spike_times[:spike_pointer]
        spike_indices = spike_indices[:spike_pointer]

        ## - Construct return time series
        resp = {
            "vt": times,
            "mfX": v,
            "a": s,
            "f": f,
            "mfFast": f,
            "dot_v": dot_v,
            "static_input": static_input,
        }

        use_hv, _ = get_global_ts_plotting_backend()
        if use_hv:
            spikes = {"times": spike_times, "vnNeuron": spike_indices}

            resp["spReservoir"] = hv.Points(
                spikes, kdims=["times", "vnNeuron"], label="Reservoir spikes"
            ).redim.range(
                times=(0, num_timesteps * self.dt), vnNeuron=(0, self.size)
            )
        else:
            resp["spReservoir"] = dict(times=spike_times, vnNeuron=spike_indices)

        # - Convert some elements to time series
        resp["tsX"] = TSContinuous(
            resp["vt"], resp["mfX"].T, name="Membrane potential"
        )
        resp["tsA"] = TSContinuous(
            resp["vt"], resp["a"].T, name="Slow synaptic state"
        )

        # - Store "last evolution" state
        self._last_evolve = resp
        self._timestep += num_timesteps

        # - Return output TimeSeries
        return TSEvent(spike_times, spike_indices)

    def to_dict(self) -> dict:
        NotImplemented

    @property
    def output_type(self):
        return TSEvent

    @property
    def tau_syn_r_f(self):
        return self.__tau_syn_r_f

    @tau_syn_r_f.setter
    def tau_syn_r_f(self, tau_syn_r_f):
        self.__tau_syn_r_f = self._expand_to_net_size(tau_syn_r_f, "tau_syn_r_f")

    @property
    def tau_syn_r_s(self):
        return self.__tau_syn_r_s

    @tau_syn_r_s.setter
    def tau_syn_r_s(self, tau_syn_r_s):
        self.__tau_syn_r_s = self._expand_to_net_size(tau_syn_r_s, "tau_syn_r_s")

    @property
    def v_thresh(self):
        return self.__thresh

    @v_thresh.setter
    def v_thresh(self, v_thresh):
        self.__thresh = self._expand_to_net_size(v_thresh, "v_thresh")

    @property
    def v_rest(self):
        return self.__rest

    @v_rest.setter
    def v_rest(self, v_rest):
        self.__rest = self._expand_to_net_size(v_rest, "v_rest")

    @property
    def v_reset(self):
        return self.__reset

    @v_reset.setter
    def v_reset(self, v_reset):
        self.__reset = self._expand_to_net_size(v_reset, "v_reset")

    @Layer.dt.setter
    def dt(self, new_dt):
        assert (
            new_dt <= self._min_tau / 10
        ), "`new_dt` must be shorter than 1/10 of the shortest time constant, for numerical stability."

        # - Call super-class setter
        super(RecFSSpikeEulerBT, RecFSSpikeEulerBT).dt.__set__(self, new_dt)


###### Convenience functions

# - Convenience method to return a nan array
def full_nan(shape: Union[tuple, int]):
    a = np.empty(shape)
    a.fill(np.nan)
    return a


### --- Compiled concenience functions


@njit
def min_argmin(data: np.ndarray):
    """
    min_argmin - Accelerated function to find minimum and location of minimum

    :param data:  np.ndarray of data

    :return:        min_val, min_loc
    """
    n = 0
    min_loc = -1
    min_val = np.inf
    for x in data:
        if x < min_val:
            min_loc = n
            min_val = x
        n += 1

    return min_val, min_loc


@njit
def argwhere(data: np.ndarray):
    """
    argwhere - Accelerated argwhere function

    :param data:  np.ndarray Boolean array

    :return:        list vnLocations where data = True
    """
    vnLocs = []
    n = 0
    for val in data:
        if val:
            vnLocs.append(n)
        n += 1

    return vnLocs


@njit
def clip_vector(v: np.ndarray, f_min: float, f_max: float):
    """
    clip_vector - Accelerated vector clip function

    :param v:
    :param min:
    :param max:

    :return: Clipped vector
    """
    v[v < f_min] = f_min
    v[v > f_max] = f_max
    return v


@njit
def clip_scalar(val: float, f_min: float, f_max: float):
    """
    clip_scalar - Accelerated scalar clip function

    :param val:
    :param min:
    :param max:

    :return: Clipped value
    """
    if val < f_min:
        return min
    elif val > f_max:
        return f_max
    else:
        return val


def rep_to_net_size(data, size):
    if np.size(data) == 1:
        return np.repeat(data, size)
    else:
        return data.flatten()
