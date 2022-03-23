"""
Implement the layer for the NetworkADS (Arbitrary Dynamical System), which is capable of learning
an arbitrary dynamical system.
"""

from rockpool.nn.layers.layer import Layer
from rockpool.timeseries import (
    TSEvent,
    TSContinuous,
    TimeSeries,
    get_global_ts_plotting_backend,
)
import numpy as np
from typing import Union, Any, Tuple, Optional
import copy

from numba import njit
from warnings import warn

FloatVector = Union[float, np.ndarray]

# - Try to import holoviews
try:
    import holoviews as hv
except Exception:
    pass

__all__ = ["RecFSSpikeADS"]


@njit
def _backstep(vCurrent, vLast, tStep, tDesiredStep):
    return (vCurrent - vLast) / tStep * tDesiredStep + vLast


@njit
def neuron_dot_v(t, V, dt, I_s_F, I_s_S, I_kDte, I_ext, V_rest, tau_V, bias):
    return (V_rest - V + I_s_F + I_s_S + I_kDte + I_ext + bias) / tau_V


@njit
def syn_dot_I_pre(I, dt_syn, I_spike):
    return -dt_syn * I + I_spike


def discretize(W, base_weight):
    tmp = np.round(W / base_weight)
    return base_weight * tmp, tmp


def quantize_weights_dynapse_II(N, M, num_synapses_available=None, use_dense=True):
    """
    @brief Function that discretizes a given continuous weight matrix
    respecting the constraints of the DYNAP-SE II.
    The constraints:
        - The development board has 1 chip with 4 cores and 256 neurons per core.
        - Each neuron on the core has 64 synapses with 2^4 distinctive weights and a sign bit.
        - A core can be configured to sacrifice 3 out of 4 neurons to allocate the
        synapses of the 192 freed neurons to the remaining 64 neurons, effectively giving
        each of the 64 neurons 256 synapses. This enables the implementation of more dense
        connection matrices
    The function first determines the number of cores in "dense mode" that can be used to
    satisfy the number of neurons needed in the matrix. Let this be denoted by X.
    After that, the weight matrix is discretized to 4 bits (plus sign bit). Following that,
    the X*64 neurons with the highest number of non-zero incoming connections are selected
    and the needed number of weights with lowest absolute value are set to 0.
    Following that, the rest of the neurons having only 64 synapses are considered. For these
    neurons, the lowest weights are set to zero so that the constraint is satisfied.
    @params N : int : Number of neurons
            M : np.ndarray : Matrix to be quantized
            num_synapses_available : [None, np.ndarray] Useful if a more complex architecture is used
                                        and specifying the FFwd matrix is not enough. This vector
                                        must hold N entries that each specify the number of synapses
                                        available per neuron.
            plot : bool : Indicates if the matrices should be plotted.
    """
    num_chips = 1
    num_cores_per_chip = 4
    num_neurons_per_core = 256
    num_cores_total = int(num_chips * num_cores_per_chip)  # 4
    num_neurons_total = num_neurons_per_core * num_cores_total  # 1024
    num_dense_core_neurons_sacrificed = int(3 / 4 * num_neurons_per_core)  # 192
    num_dense_core_neurons = num_neurons_per_core - num_dense_core_neurons_sacrificed

    # assert num_synapses_available is of type None, or np.ndarray(dtype=int)
    assert (num_synapses_available is None) or (
        np.array(list(num_synapses_available)).dtype == np.dtype("int")
    ), "Elements of num_synapses_available must be of type int"
    assert (
        N <= num_neurons_total
    ), "Number of neurons exceeds number of neurons on DYNAP-SE II"
    assert (
        M.shape[1] == N
    ), "Second matrix dimension does not fit number of neurons. Dimension must be [? x N]"

    if num_synapses_available is None:
        num_synapses_available = 64 * np.ones((N,), dtype=int)

    # - Matrix that gets returned
    M_disc = np.zeros(M.shape)

    # - Needs to be clipped because if N < 256 a value bigger than num_cores_total can be returned
    number_dense_cores = int(
        np.clip(
            int(
                num_cores_total
                - ((N - num_neurons_per_core) / (num_dense_core_neurons_sacrificed))
            ),
            0,
            num_cores_total,
        )
    )
    if not use_dense:
        number_dense_cores = 0
    number_sparse_cores = num_cores_total - number_dense_cores

    # - Quantize
    base_weight = (np.max(np.abs(M)) - np.min(np.abs(M))) / (2**5 - 1)
    if base_weight == 0.0:
        return np.zeros(M.shape)
    num_base_weights_needed = np.round(M / base_weight)

    num_dense_neurons = 0

    # - The following only matter if we have one dense core available
    if number_dense_cores > 0:
        # - Which neurons would loose the most incoming connections if we have to threshold?
        # - We also need to consider that some neurons had to sacrifice some synapses for other connections
        # - so they could benefit even greater from the additional 192 synapses if placed on a dense core.
        num_base_weights_lost = np.empty(shape=(N,), dtype=int)
        for i in range(N):
            incoming_weights = np.abs(num_base_weights_needed[i, :])
            incoming_weights[::-1].sort()  # - Sort in descending order
            # - Take the biggest weights first. Cutoff at the index that indicates the num. synapses we have left for this neuron
            num_base_weights_lost[i] = np.sum(
                incoming_weights[num_synapses_available[i] :]
            )

        # - Argsort the num_base_weights_lost vector in descending order and take the indices, maybe index oor but that is ok
        dense_neuron_indices = num_base_weights_lost[::-1].argsort()[
            : number_dense_cores * num_dense_core_neurons
        ]
        dense_neuron_indices.sort()
        num_dense_neurons = len(dense_neuron_indices)
        # - The number of synapses available increments by 192 for the dense indices
        num_synapses_available[dense_neuron_indices] += 192

    if num_dense_neurons > 0:
        # - The index array with the neurons that have 256 synapses will be filled by now
        # - First discretize the incoming weights of the dense neurons and keep track of the indices that have only 64 synapses
        sparse_neuron_indices = np.empty((N - num_dense_neurons,), dtype=int)
        c = 0
        for idx, i in enumerate(dense_neuron_indices):
            if idx == 0:
                for j in range(i):
                    sparse_neuron_indices[c] = j
                    c += 1
            elif idx == len(dense_neuron_indices) - 1:
                for j in range(dense_neuron_indices[idx - 1] + 1, N - 1):
                    sparse_neuron_indices[c] = j
                    c += 1
            else:
                for j in range(dense_neuron_indices[idx - 1] + 1, i):
                    sparse_neuron_indices[c] = j
                    c += 1

            # - Discretize incoming weights of i-th dense neuron
            indices = np.argsort(np.abs(num_base_weights_needed[i, :]))[::-1][
                : num_synapses_available[i]
            ]
            M_disc[i, indices] = base_weight * num_base_weights_needed[i, indices]
    else:
        sparse_neuron_indices = np.arange(0, N, 1)

    # - Discretize the rest
    for j in sparse_neuron_indices:
        indices = np.argsort(np.abs(num_base_weights_needed[j, :]))[::-1][
            : num_synapses_available[j]
        ]
        M_disc[j, indices] = base_weight * num_base_weights_needed[j, indices]

    return M_disc


class RecFSSpikeADS(Layer):
    """
    Implement the layer for the NetworkADS (Arbitrary Dynamical System), which is capable of learning
    an arbitrary dynamical system.
    See rockpool/docs/tutorials/network_ads_tutorial.ipynb for an example initialization of the Network.

    :param ndarray weights_fast: [NxN] matrix implementing the balanced network. Predefined given ffwd matrix (see tutorial)
    :param ndarray weights_slow: [NxN] learnable recurrent matrix implementing dynamics for the task
    :param ndarray weights_in: [NcxN] matrix that projects current input into the network [not trained]
    :param ndarray weights_out: [NxNc] matrix for reading out the target [not trained]
    :param float eta: Learning rate
    :param float k: Scaling factor determining the magnitude of error-current that is fed back into the system
    :param [ndarray,float] bias: Bias applied to the neurons membrane potential
    :param float noise_std: Standard deviation of Gaussian (zero mean) noise applied
    :param float dt: Euler integration timestep
    :param [ndarray,float] v_thresh: Spiking threshold typically at 1. Caution: Potentials are clipped at zero, meaning there can not be negative potentials
    :param [ndarray,float] v_reset: Reset potential typically at 0
    :param [ndarray,float] v_rest: Resting potential typically at v_thresh/2
    :param float tau_mem: Membrane time constant
    :param float tau_syn_r_fast: Synaptic time constant of fast connections 0.07s
    :param float tau_syn_r_slow: Synaptic time constant of slow connections typically 0.07s
    :param float tau_syn_r_out: Synaptic time constant of output filter typically 0.07s
    :param float refractory: Refractory period in seconds. Typically set to 0
    :param bool record: If set to true, records various states such as slow/fast currents, membrane potentials
    :param str name: Name of the layer
    :param int discretize: Number of distinctive weights used at all times. E.g. 8 would mean a 3 bit resolution. discretize_dynapse must be set False
    :param bool discretize_dynapse: If set to True, the constraints of the DYNAP-SE II are imposed on the slow recurrent weight matrix
    """

    def __init__(
        self,
        weights_fast: np.ndarray,
        weights_slow: np.ndarray,
        weights_in: np.ndarray,
        weights_out: np.ndarray,
        eta: float,
        k: float,
        bias: np.ndarray,
        noise_std: float,
        dt: float,
        v_thresh: Union[np.ndarray, float],
        v_reset: Union[np.ndarray, float],
        v_rest: Union[np.ndarray, float],
        tau_mem: float,
        tau_syn_r_fast: float,
        tau_syn_r_slow: float,
        tau_syn_r_out: float,
        refractory: float,
        record: bool,
        name: str,
        discretize: int,
        discretize_dynapse: bool,
    ):

        super().__init__(
            weights=np.zeros(
                (np.asarray(weights_fast).shape[0], np.asarray(weights_fast).shape[1])
            ),
            noise_std=noise_std,
            name=name,
        )

        # - Fast weights, noise_std and name are access. via self.XX or self._XX
        self.noise_std = noise_std
        self.weights_slow = np.asarray(weights_slow).astype("float")
        self.weights_slow_discretized = np.zeros(self.weights_slow.shape)
        self.weights_out = np.asarray(weights_out).astype("float")
        self.weights_in = np.asarray(weights_in).astype("float")
        self.weights_fast = np.asarray(weights_fast).astype("float")
        self.eta = eta
        self.k = k
        self.bias = np.asarray(bias).astype("float")
        self.v_thresh = np.asarray(v_thresh).astype("float")
        self.v_reset = np.asarray(v_reset).astype("float")
        self.v_rest = np.asarray(v_rest).astype("float")
        self.tau_mem = np.asarray(tau_mem).astype("float")
        self.tau_syn_r_fast = np.asarray(tau_syn_r_fast).astype("float")
        self.tau_syn_r_slow = np.asarray(tau_syn_r_slow).astype("float")
        self.tau_syn_r_out = np.asarray(tau_syn_r_out).astype("float")
        self.refractory = float(refractory)
        self.is_training = False
        self._ts_target = None
        self.static_target = None
        self.recorded_states = None
        self.record = bool(record)
        self.k_initial = k
        self.eta_initial = eta
        self.out_size = self.weights_out.shape[1]
        self.t_start_suppress = None
        self.t_stop_suppress = None
        self.percentage_suppress = None

        # - Discretization
        self.num_synapse_states = discretize  # How many distinct synapse weights to we allow? For example, 2**3 = 8 would be 3-bit precision
        self.discretize_dynapse = discretize_dynapse  # Do we want to respect the constraint of the DYNAP-SE neuromorphic hardware?

        # - Set a reasonable dt
        if dt is None:
            self.dt = self._min_tau / 10
        else:
            self.dt = np.asarray(dt).astype("float")

        # - Initialise the network
        self.reset_all()

    def reset_state(self):
        """
        Reset the internal state of the network
        """
        self.state = self.v_rest.copy()
        self.I_s_S = np.zeros(self.size)
        self.I_s_F = np.zeros(self.size)
        self.I_s_O = np.zeros(self.out_size)
        self.rate = np.zeros(self.size)
        self.num_training_iterations = 0

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
        min_delta: Optional[float] = None,
    ) -> TSEvent:
        """
        Evolve the function on the input c(t). This function simply feeds the input through the network and does not perform any learning

        :param TSContinuous ts_input: Corresponds to the input c in the simulations, shape: [int(duration/dt), N], eg [30001,100]
        :param float duration: Duration in seconds for the layer to be evolved, net_ads calls evolve with duration set to None, but passes num_timesteps
        :param int num_timesteps: Number of timesteps to be performed. Typically int(duration / dt) where duration is passed to net.evolve (not the layer evolve)
        :param bool verbose: Print verbose output
        :param bool min_delta: Minimal time set taken. Typically 1/10*dt . Must be strictly smaller than dt. This is used to determine the precise spike timing
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

        # - Assertions about training and the targets that were set
        if self.is_training:
            assert (
                self.ts_target is not None
            ), "Evolve called with learning flag set, but no target input provided"
            assert (
                input_time_trace.shape == self.target_time_trace.shape
            ), "Input and target time_trace shapes don't match"
            assert (
                static_input.shape[0] == self.static_target.shape[0]
            ), "Input and target lengths don't match"
            assert (
                num_timesteps == self.target_num_timesteps
            ), "Input and output num_timesteps don't match"

        # - Generate a noise trace
        noise_step = (
            np.random.randn(np.size(input_time_trace), self.size) * self.noise_std
        )
        static_input += noise_step

        record_length = num_timesteps
        spike_pointer = 0

        if verbose or self.record:
            times = full_nan(record_length)
            v = full_nan((self.size, record_length))
            s = full_nan((self.size, record_length))
            r = full_nan((self.size, record_length))
            f = full_nan((self.size, record_length))
            out = full_nan((self.out_size, record_length))
            err = full_nan((self.out_size, record_length))
            I_kDte_track = full_nan((self.size, record_length))
            dot_v_ts = full_nan((self.size, record_length))

        # - Allocate storage for spike times
        max_spike_pointer = record_length * self.size
        spike_times = full_nan(max_spike_pointer)
        spike_indices = full_nan(max_spike_pointer)

        # - Refractory time variable
        vec_refractory = np.zeros(self.size)

        # - Initialise step and "previous step" variables
        t_time = self._t
        t_start = self._t
        step = 0
        t_last = self._t
        v_last = self._state.copy()
        I_s_S_Last = self.I_s_S.copy()
        I_s_F_Last = self.I_s_F.copy()
        I_s_O_Last = self.I_s_O.copy()
        rate_Last = self.rate.copy()

        zeros = np.zeros(self.size)
        zeros_out = np.zeros(self.out_size)

        e = zeros_out.copy()

        # - Precompute dt/syn_tau for different taus
        dt_syn_slow = self.dt / self.tau_syn_r_slow
        dt_syn_fast = self.dt / self.tau_syn_r_fast
        dt_syn_out = self.dt / self.tau_syn_r_out
        dt_syn_rate = self.dt / np.mean(self.tau_syn_r_out)

        # - For getting spike-vectors
        eye = np.eye(self.size)

        # - Setup arrays for tracking
        E = full_nan((self.out_size, record_length))
        R = full_nan((self.size, record_length))
        step_counter = 0
        record_length_batched = num_timesteps

        # - Discretization
        if self.num_synapse_states == -1 and not self.discretize_dynapse:
            # - Should not discretize
            self.weights_slow_discretized = self.weights_slow
        elif self.discretize_dynapse:
            # - Meaning we should discretize using the DYNAP-SE II constraints
            self.weights_slow_discretized = quantize_weights_dynapse_II(
                self.size, self.weights_slow
            )
        else:
            # - Meaning we should discretize normally
            base_weight = (np.max(self.weights_slow) - np.min(self.weights_slow)) / (
                self.num_synapse_states - 1
            )
            if base_weight != 0:
                self.weights_slow_discretized, _ = discretize(
                    self.weights_slow, base_weight
                )

        def _evolve_backstep(
            t_last,
            t_time,
            weights,
            weights_slow,
            weights_in,
            weights_out,
            k,
            e_Last,
            is_training,
            state,
            I_s_S,
            I_s_F,
            I_s_O,
            rate,
            dt,
            v_last,
            I_s_S_Last,
            I_s_F_Last,
            I_s_O_Last,
            rate_Last,
            v_reset,
            v_rest,
            v_thresh,
            bias,
            tau_mem,
            tau_syn_r_slow,
            tau_syn_r_fast,
            tau_syn_r_out,
            refractory,
            vec_refractory,
            zeros,
            target,
        ):
            # - Enforce refractory period by clamping membrane potential to reset
            b = vec_refractory > 0
            state[b] = v_reset[b]

            if (
                self.percentage_suppress is not None
                and t_time > self.t_start_suppress
                and t_time < self.t_stop_suppress
            ):
                # - Suppress neurons
                state[: int(self.size * self.percentage_suppress)] = v_reset[
                    : int(self.size * self.percentage_suppress)
                ]

            # - Locate spiking neurons
            spike_ids = state > v_thresh
            spike_ids = np.asarray(argwhere(spike_ids))
            num_spikes = len(spike_ids)

            # - Were there any spikes?
            if num_spikes > 0:
                # - Predict the precise spike times using linear interpolation, returns a value between 0 and dt depending on whether V(t) or V(t-1) is closer to the threshold.
                # - We then have the precise spike time by adding spike_delta to the last time instance. If it is for example 0, we add the minimal time step, namely min_delta
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
                I_s_O = _backstep(I_s_O, I_s_O_Last, dt, spike_delta)
                rate = _backstep(rate, rate_Last, dt, spike_delta)

                # - Apply reset to spiking neuron
                state[first_spike_id] = v_reset[first_spike_id]

                # - Begin refractory period for spiking neuron
                vec_refractory[first_spike_id] = refractory

                # - Set spike currents
                I_spike_rate = eye[first_spike_id, :]
                I_spike_slow = weights_slow[first_spike_id, :]
                I_spike_fast = weights[:, first_spike_id]
                I_spike_out = weights_out[first_spike_id, :]

            else:
                # - Clear spike currents
                first_spike_id = -1
                I_spike_slow = zeros
                I_spike_fast = zeros
                I_spike_out = zeros_out
                I_spike_rate = zeros

            ### End of back-tick spike detector
            # - Save synapse and neuron states for previous time step
            v_last[:] = state
            I_s_S_Last[:] = I_s_S + I_spike_slow
            I_s_F_Last[:] = I_s_F + I_spike_fast
            I_s_O_Last[:] = I_s_O + I_spike_out
            rate_Last[:] = rate + I_spike_rate

            # - Update synapse and neuron states (Euler step)
            dot_I_s_S = syn_dot_I_pre(I_s_S, dt_syn_slow, I_spike_slow)
            I_s_S += dot_I_s_S

            dot_I_s_F = syn_dot_I_pre(I_s_F, dt_syn_fast, I_spike_fast)
            I_s_F += dot_I_s_F

            dot_I_s_O = syn_dot_I_pre(I_s_O, dt_syn_out, I_spike_out)
            I_s_O += dot_I_s_O

            dot_rate = syn_dot_I_pre(rate, dt_syn_rate, I_spike_rate)
            rate += dot_rate

            int_time = int((t_time - t_start) // dt)
            I_ext = static_input[int_time, :]

            if is_training:
                x = np.copy(target[int_time, :])
                x_hat = I_s_O
                e = x - x_hat
                I_kDte = k * (weights_in.T @ e)

            else:
                I_kDte = zeros
                assert (I_kDte == 0).all(), "I_kDte is not zero"
                e = zeros_out

            dot_v = neuron_dot_v(
                t=t_time,
                V=state,
                dt=dt,
                I_s_F=I_s_F,
                I_s_S=I_s_S,
                I_kDte=I_kDte,
                I_ext=I_ext,
                V_rest=v_rest,
                tau_V=tau_mem,
                bias=bias,
            )

            state += dot_v * dt
            state[state < 0] = 0.0

            return (
                t_time,
                first_spike_id,
                dot_v,
                state,
                I_s_S,
                I_s_F,
                I_s_O,
                rate,
                I_kDte,
                dt,
                v_last,
                I_s_S_Last,
                I_s_F_Last,
                I_s_O_Last,
                rate_Last,
                vec_refractory,
                e,
            )

        # - Euler integrator loop
        while t_time < final_time:

            ### --- END of compiled inner function
            (
                t_time,
                first_spike_id,
                dot_v,
                self._state,
                self.I_s_S,
                self.I_s_F,
                self.I_s_O,
                self.rate,
                I_kDte,
                self._dt,
                v_last,
                I_s_S_Last,
                I_s_F_Last,
                I_s_O_Last,
                rate_Last,
                vec_refractory,
                e,
            ) = _evolve_backstep(
                t_last=t_last,
                t_time=t_time,
                weights=self.weights_fast,
                weights_slow=self.weights_slow_discretized,  # This does not have to be discretized. If --discretize is set to -1, this is simply the continuous version
                weights_in=self.weights_in,
                weights_out=self.weights_out,
                k=self.k,
                e_Last=e,
                is_training=self.is_training,
                state=self._state,
                I_s_S=self.I_s_S,
                I_s_F=self.I_s_F,
                I_s_O=self.I_s_O,
                rate=self.rate,
                dt=self._dt,
                v_last=v_last,
                I_s_S_Last=I_s_S_Last,
                I_s_F_Last=I_s_F_Last,
                I_s_O_Last=I_s_O_Last,
                rate_Last=rate_Last,
                v_reset=self.v_reset,
                v_rest=self.v_rest,
                v_thresh=self.v_thresh,
                bias=self.bias,
                tau_mem=self.tau_mem,
                tau_syn_r_slow=self.tau_syn_r_slow,
                tau_syn_r_fast=self.tau_syn_r_fast,
                tau_syn_r_out=self.tau_syn_r_out,
                refractory=self.refractory,
                vec_refractory=vec_refractory,
                zeros=zeros,
                target=self.static_target,
            )

            if self.is_training and (np.abs(e) > 0).any():
                # - Check if tracking variables need extending
                if step_counter >= record_length_batched:
                    E = np.append(E, full_nan((self.out_size, num_timesteps)), axis=1)
                    R = np.append(R, full_nan((self.size, num_timesteps)), axis=1)
                    record_length_batched += num_timesteps
                # - Fill the tracking variables
                R[:, step_counter] = self.rate
                E[:, step_counter] = e
                step_counter += 1

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

            if verbose or self.record:
                # - Extend state storage variables, if needed
                if step >= record_length:
                    extend = num_timesteps
                    times = np.append(times, full_nan(extend))
                    v = np.append(v, full_nan((self.size, extend)), axis=1)
                    s = np.append(s, full_nan((self.size, extend)), axis=1)
                    f = np.append(f, full_nan((self.size, extend)), axis=1)
                    r = np.append(r, full_nan((self.size, extend)), axis=1)
                    err = np.append(err, full_nan((self.out_size, extend)), axis=1)
                    out = np.append(out, full_nan((self.out_size, extend)), axis=1)
                    I_kDte_track = np.append(
                        I_kDte_track, full_nan((self.size, extend)), axis=1
                    )
                    dot_v_ts = np.append(
                        dot_v_ts, full_nan((self.size, extend)), axis=1
                    )
                    record_length += extend

                # - Store the network states for this time step
                times[step] = t_time
                v[:, step] = self._state
                s[:, step] = self.I_s_S
                f[:, step] = self.I_s_F
                r[:, step] = self.rate
                err[:, step] = e
                out[:, step] = self.I_s_O
                I_kDte_track[:, step] = I_kDte
                dot_v_ts[:, step] = dot_v

            # - Next nominal time step
            t_last = copy.copy(t_time)
            t_time += self._dt
            step += 1
            vec_refractory -= self.dt

        ## END End of Euler integration loop

        ## - Back-step to exact final time
        self.state = _backstep(self.state, v_last, self._dt, t_time - final_time)
        self.I_s_S = _backstep(self.I_s_S, I_s_S_Last, self._dt, t_time - final_time)
        self.I_s_F = _backstep(self.I_s_F, I_s_F_Last, self._dt, t_time - final_time)
        self.I_s_O = _backstep(self.I_s_O, I_s_O_Last, self._dt, t_time - final_time)
        self.rate = _backstep(self.rate, rate_Last, self._dt, t_time - final_time)

        ## - Store the network states for final time step
        times[step - 1] = final_time
        v[:, step - 1] = self.state
        s[:, step - 1] = self.I_s_S
        f[:, step - 1] = self.I_s_F
        out[:, step - 1] = self.I_s_O
        r[:, step - 1] = self.rate
        err[:, step - 1] = e
        I_kDte_track[:, step - 1] = I_kDte

        ## - Trim state storage variables
        times = times[:step]
        v = v[:, :step]
        s = s[:, :step]
        f = f[:, :step]
        r = r[:, :step]
        err = err[:, :step]
        out = out[:, :step]
        I_kDte_track = I_kDte_track[:, :step]

        dot_v_ts = dot_v_ts[:, :step]

        spike_times = spike_times[:spike_pointer]
        spike_indices = spike_indices[:spike_pointer]

        R = R[:, :step_counter]
        E = E[:, :step_counter]

        if self.is_training:
            # - Compute the weight update here
            dot_W_slow_batched = R @ (self.weights_in.T @ E).T

            # - No learning along the diagonal
            np.fill_diagonal(dot_W_slow_batched, 0)

            # - Normalize the update to have frobenius norm 1.0
            dot_W_slow_batched /= np.sum(np.abs(dot_W_slow_batched)) / self.size**2

            # - Apply the learning rate
            dot_W_slow_batched *= self.eta

            # - Perform the update
            self.weights_slow += dot_W_slow_batched

        ## - Construct return time series
        valid_spikes = spike_indices > -1
        spike_times = spike_times[valid_spikes]
        spike_indices = spike_indices[valid_spikes]

        resp = {
            "vt": times,
            "mfX": v,
            "s": s,
            "f": f,
            "r": r,
            "out": out,
            "mfFast": f,
            "v": v,
            "dot_v": dot_v_ts,
            "static_input": static_input,
            "spike_times": spike_times,
            "spike_indices": spike_indices,
        }

        if verbose:
            backend = get_global_ts_plotting_backend()
            if backend is "holoviews":
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
                resp["vt"], resp["s"].T, name="Slow synaptic state"
            )

            # - Store "last evolution" state
            self._last_evolve = resp

        if self.record:
            self.recorded_states = resp

        self._timestep += num_timesteps

        # - Return output TimeSeries
        ts_event_return = TSEvent(
            spike_times,
            spike_indices,
            num_channels=self.size,
            t_start=t_start,
            t_stop=final_time,
        )

        if verbose:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(20, 20), constrained_layout=True)
            gs = fig.add_gridspec(7, 1)

            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(times, s[0:5, :].T)
            ax1.set_title(r"$I_{slow}$")

            ax2 = fig.add_subplot(gs[1, 0])
            ax2.plot(times, v[0:5, :].T)
            ax2.set_title(r"$V(t)$")

            ax3 = fig.add_subplot(gs[2, 0])
            ax3.plot(
                np.linspace(
                    0, len(static_input[:, 0]) / self.dt, len(static_input[:, 0])
                ),
                static_input[:, :],
            )
            ax3.set_title(r"$I_{ext}$")

            ax4 = fig.add_subplot(gs[3, 0])
            channels = ts_event_return.channels[ts_event_return.channels >= 0]
            times_tmp = ts_event_return.times[ts_event_return.channels >= 0]
            ax4.scatter(times_tmp, channels, color="k")
            ax4.set_xlim([0, final_time])

            ax5 = fig.add_subplot(gs[4, 0])
            ax5.plot(times, I_kDte_track[0:5, :].T)
            ax5.set_title(r"$I_{kD^Te}$")

            ax6 = fig.add_subplot(gs[5, 0])
            ax6.plot(times, out[0:5, :].T)
            ax6.set_title(r"$I_{out}$")

            ax7 = fig.add_subplot(gs[6, 0])
            ax7.plot(times, f[0:5, :].T)
            ax7.set_title(r"$I_{fast}$")

            plt.tight_layout()
            plt.draw()
            plt.waitforbuttonpress(0)  # this will wait for indefinite time
            plt.close(fig)

            # - Plot reconstruction
            fig = plt.figure(figsize=(20, 20))
            plot_num = self.weights_out.shape[1]
            stagger_out = np.ones((plot_num, out.shape[1]))
            stagger_target = np.ones((plot_num, self.static_target.shape[0]))
            for i in range(plot_num):
                stagger_out[i, :] *= i
                stagger_target[i, :] *= i

            colors = [("C%d" % i) for i in range(2, plot_num + 2)]
            l1 = plt.plot(times, (stagger_out + out[:plot_num, :]).T)
            for line, color in zip(l1, colors):
                line.set_color(color)
            l2 = plt.plot(
                np.linspace(
                    0,
                    self.static_target.shape[0] * self.dt,
                    self.static_target.shape[0],
                ),
                (stagger_target.T + self.static_target[:, :plot_num]),
                linestyle="--",
            )
            for line, color in zip(l2, colors):
                line.set_color(color)
            plt.title(r"Target vs reconstruction")
            lines = [l1[0], l2[0]]
            plt.legend(lines, ["Reconstruction", "Target"])
            plt.draw()
            plt.waitforbuttonpress(0)  # this will wait for indefinite time
            plt.close(fig)

            if self.is_training:
                fig = plt.figure()
                plt.subplot(121)
                im = plt.matshow(self.weights_slow, fignum=False)
                plt.xticks([], [])
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.title(r"$W_{slow}$")
                plt.subplot(122)
                im = plt.matshow(dot_W_slow_batched, fignum=False)
                plt.xticks([], [])
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.title(r"$\Delta W_{slow}$")
                plt.tight_layout()
                plt.draw()
                plt.waitforbuttonpress(0)  # this will wait for indefinite time
                plt.close(fig)

        return ts_event_return

    def to_dict(self):
        """
        Convert the parameters of this class to a dictionary

        :return dict:
        """
        config = {}
        config["class_name"] = "RecFSSpikeADS"
        config["N"] = self.size
        config["Nc"] = self.weights_in.shape[0]
        config["Nb"] = self.weights_slow.shape[0]
        config["weights_fast"] = self.weights_fast.tolist()
        config["weights_slow"] = self.weights_slow.tolist()
        config["weights_in"] = self.weights_in.tolist()
        config["weights_out"] = self.weights_out.tolist()
        config["eta"] = self.eta
        config["k"] = self.k
        config["bias"] = self.bias.tolist()
        config["noise_std"] = (
            self.noise_std if type(self.noise_std) is float else self.noise_std.tolist()
        )
        config["dt"] = self.dt
        config["v_thresh"] = self.v_thresh.tolist()
        config["v_reset"] = self.v_reset.tolist()
        config["v_rest"] = self.v_rest.tolist()
        config["tau_mem"] = self.tau_mem.tolist()
        config["tau_syn_r_fast"] = self.tau_syn_r_fast.tolist()
        config["tau_syn_r_slow"] = self.tau_syn_r_slow.tolist()
        config["tau_syn_r_out"] = self.tau_syn_r_out.tolist()
        config["refractory"] = self.refractory
        config["record"] = int(self.record)
        config["name"] = self.name
        config["discretize"] = self.num_synapse_states
        config["discretize_dynapse"] = self.discretize_dynapse

        return config

    @property
    def _min_tau(self):
        """
        (float) Smallest time constant of the layer
        """
        return min(np.min(self.tau_syn_r_slow), np.min(self.tau_syn_r_fast))

    @property
    def output_type(self):
        """(`TSEvent`) Output `TimeSeries` class (`TSEvent`)"""
        return TSEvent

    @property
    def tau_syn_r_f(self):
        """(float) Fast synaptic time constant (s)"""
        return self.__tau_syn_r_f

    @tau_syn_r_f.setter
    def tau_syn_r_f(self, tau_syn_r_f):
        self.__tau_syn_r_f = self._expand_to_net_size(tau_syn_r_f, "tau_syn_r_f")

    @property
    def tau_syn_r_s(self):
        """(float) Slow synaptic time constant (s)"""
        return self.__tau_syn_r_s

    @tau_syn_r_s.setter
    def tau_syn_r_s(self, tau_syn_r_s):
        self.__tau_syn_r_s = self._expand_to_net_size(tau_syn_r_s, "tau_syn_r_s")

    @property
    def v_thresh(self):
        """(float) Threshold potential"""
        return self.__thresh

    @v_thresh.setter
    def v_thresh(self, v_thresh):
        self.__thresh = self._expand_to_net_size(v_thresh, "v_thresh")

    @property
    def v_rest(self):
        """(float) Resting potential"""
        return self.__rest

    @v_rest.setter
    def v_rest(self, v_rest):
        self.__rest = self._expand_to_net_size(v_rest, "v_rest")

    @property
    def v_reset(self):
        """(float) Reset potential"""
        return self.__reset

    @v_reset.setter
    def v_reset(self, v_reset):
        self.__reset = self._expand_to_net_size(v_reset, "v_reset")

    @property
    def bias(self):
        """(float) Bias potential"""
        return self.__bias

    @bias.setter
    def bias(self, bias):
        self.__bias = self._expand_to_net_size(bias, "bias")

    @Layer.dt.setter
    def dt(self, new_dt):
        # - Call super-class setter
        super(RecFSSpikeADS, RecFSSpikeADS).dt.__set__(self, new_dt)

    @property
    def ts_target(self):
        """TSContinuous Target dynamics used during training to compute the error"""
        return self._ts_target

    @ts_target.setter
    def ts_target(self, t):
        if t is not None:
            (
                self.target_time_trace,
                self.static_target,
                self.target_num_timesteps,
            ) = self._prepare_input(t, is_target=True)
            self._ts_target = t
        else:
            self._ts_target = None

    def _prepare_input(
        self,
        ts_input: Optional[TimeSeries] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        is_target: Optional[bool] = False,
    ) -> (np.ndarray, np.ndarray, float):
        """
        Sample input, set up time base

        This function checks an input signal, and prepares a discretised time base according to the time step of the current layer

        :param Optional[TimeSeries] ts_input:   :py:class:`TimeSeries` of TxM or Tx1 Input signals for this layer
        :param Optional[float] duration:        Duration of the desired evolution, in seconds. If not provided, then either ``num_timesteps`` or the duration of ``ts_input`` will define the evolution time
        :param Optional[int] num_timesteps:     Integer number of evolution time steps, in units of ``.dt``. If not provided, then ``duration`` or the duration of ``ts_input`` will define the evolution time

        :return (ndarray, ndarray, float): (time_base, input_steps, duration)
            time_base:      T1 Discretised time base for evolution
            input_steps:    (T1xN) Discretised input signal for layer
            num_timesteps:  Actual number of evolution time steps, in units of ``.dt``
        """

        num_timesteps = self._determine_timesteps(ts_input, duration, num_timesteps)

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
                    warn(
                        "Layer `{}`: Evolution period (t = {} to {}) ".format(
                            self.name, time_base[0], time_base[-1]
                        )
                        + "is not fully contained in input signal (t = {} to {}).".format(
                            ts_input.t_start, ts_input.t_stop
                        )
                        + " You may need to use a `periodic` time series."
                    )

            if not is_target:
                # - Sample input trace and check for correct dimensions
                input_steps = self._check_input_dims(ts_input(time_base))
            else:
                input_steps = ts_input(time_base)
                if (
                    input_steps.ndim == 1
                    or (input_steps.ndim > 1 and input_steps.shape[1]) == 1
                ):
                    if self.size_in > 1:
                        warn(
                            f"Layer `{self.name}`: Only one channel provided in input - will "
                            + f"be copied to all {self.size_in} input channels."
                        )
                    # input_steps = np.repeat(input_steps.reshape((-1, 1)), self._size_in, axis=1)

            # - Treat "NaN" as zero inputs
            input_steps[np.where(np.isnan(input_steps))] = 0

        else:
            # - Assume zero inputs
            input_steps = np.zeros((np.size(time_base), self.size_in))

        return time_base, input_steps, num_timesteps


###### Convenience functions

# - Convenience method to return a nan array
def full_nan(shape: Union[tuple, int]):
    a = np.empty(shape)
    a.fill(np.nan)
    return a


### --- Compiled concenience functions


@njit
def min_argmin(data: np.ndarray) -> Tuple[float, int]:
    """
    Accelerated function to find minimum and location of minimum

    :param data:  np.ndarray of data

    :return (float, int):        min_val, min_loc
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
def argwhere(data: np.ndarray) -> list:
    """
    Accelerated argwhere function

    :param np.ndarray data:  Boolean array

    :return list:         vnLocations where data = True
    """
    vnLocs = []
    n = 0
    for val in data:
        if val:
            vnLocs.append(n)
        n += 1

    return vnLocs


@njit
def clip_vector(v: np.ndarray, f_min: float, f_max: float) -> np.ndarray:
    """
    Accelerated vector clip function

    :param np.ndarray v:
    :param float min:
    :param float max:

    :return np.ndarray: Clipped vector
    """
    v[v < f_min] = f_min
    v[v > f_max] = f_max
    return v


@njit
def clip_scalar(val: float, f_min: float, f_max: float) -> float:
    """
    Accelerated scalar clip function

    :param float val:
    :param float min:
    :param float max:

    :return float: Clipped value
    """
    if val < f_min:
        return f_min
    elif val > f_max:
        return f_max
    else:
        return val


def rep_to_net_size(data: Any, size: Tuple):
    """
    Repeat some data to match the layer size

    :param Any data:
    :param Tuple size:

    :return np.ndarray:
    """
    if np.size(data) == 1:
        return np.repeat(data, size)
    else:
        return data.flatten()
