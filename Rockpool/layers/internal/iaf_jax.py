##
# Spiking recurrent and FFwd layers with JAX backend
#

# - Imports
from ..layer import Layer
from ...timeseries import TimeSeries, TSContinuous, TSEvent

from jax import numpy as np

from jax import jit
from jax.lax import scan

import jax.random as rand
import numpy as onp
from typing import Optional, Tuple, Union
from warnings import warn

# - Define a float / array type
FloatVector = Union[float, np.ndarray]

# - Define module exports
__all__ = ["RecIAFExpJax", "RecIAFExpSpikeOutJax", "RecIAFExpWithIOJax"]

# - Raw evolution function
def _evolve_iaf_expsyn(
    v0: np.ndarray,
    syn0: np.ndarray,
    refractory0: np.ndarray,
    bias: np.ndarray,
    tau_m: np.ndarray,
    tau_s: np.ndarray,
    refractory: np.ndarray,
    w_rec: np.ndarray,
    inputs: np.ndarray,
    noise_std: float,
    key,
    dt: float,
) -> Tuple[Tuple, np.ndarray, np.ndarray, np.ndarray]:
    """
    Raw JAX-backed forward Euler evolution function for an IAF recurrent layer

    Neurons are implemented as integrate-and-fire spiking neurons, with exponential function synaptic output currents. Neurons have a membrane and a synaptic time constant. Resting potentials and reset potentials are 0.0; thresholds are 1.0 for all neurons. A forward Euler solver is used to solve the neuron dynamics.

    Neurons are optionally refractory. Membrane potentials are not clamped during the refractory period, and neurons are prevented from firing.

    On spiking, neurons undergo a subtractive reset of -1.

    Synaptic current variables undergo a positive step of +1 when their neuron spikes, followed by an exponential decay governed by `tau_s`.

    :param np.ndarray v0:           Initial membrane potential state [N,]
    :param np.ndarray syn0:         Initial synaptic output variable state [N,]
    :param np.ndarray refractory0:  Initial refractory state for each neuron [N,]
    :param np.ndarray bias:         Bias current for each neuron [N,]
    :param np.ndarray tau_m:        Membrane time constant for each neuron [N,]
    :param np.ndarray tau_s:        Output synaptic time constant for each neuron [N,]
    :param np.ndarray refractory:   Refractory period for each neuron [N,]
    :param np.ndarray w_rec:        Recurrent weights [N, N]
    :param np.ndarray inputs:       Raw rasterised time series of input currents for each neuron [T, N]
    :param float noise_std:         White noise standard deviation injected into each membrane
    :param key:                     pRNG key for JAX. Used to generate white noise
    :param float dt:                Time step for forward Euler solver

    :return (state, v_mem_ts, v_syn_ts, spike_raster_ts):
        state: Tuple(v_mem, I_syn, refractory)
        v_mem_ts: ndarray [T,N] Array of time series of membrane potentials per neuron
        v_syn_ts: ndarray [T,N] Array of time series of synaptic output currents per neuron
        spike_raster_ts: ndarrau [T,N] Boolean array of spiking activity per neuron
    """

    def step_iaf_exp(X, I):
        # - Unpack carry state and inputs
        (v_mem, v_syn, t_refractory) = X
        (I_ext, I_noise) = I

        # - Reshape input slices
        I_ext = I_ext.T
        I_noise = I_noise.T

        # - Compute recurrent input
        I_rec = np.dot(w_rec, v_syn.reshape((-1, 1))).reshape((-1))

        # - Voltage update equation
        dV = (I_ext + I_noise + I_rec + bias + 0 - v_mem) / tau_m

        # - Find refractory neurons
        vbRefractory = t_refractory > 0
        dV = dV * (1 - vbRefractory)

        # - Update membrane voltage
        v_mem = v_mem + dV * dt

        # - Discover spiking neurons
        vbSpikes = v_mem > 1

        # - Decay refractory period
        t_refractory -= dt

        # - Reset refractory period for spiking neurons
        t_refractory = t_refractory - vbSpikes * t_refractory + vbSpikes * refractory

        # - Subtractive reset of spiking neurons
        v_mem = v_mem - vbSpikes * (1 - 0)

        # - Update synaptic currents
        v_syn = v_syn * np.exp(-dt / tau_s) + vbSpikes

        return (v_mem, v_syn, t_refractory), v_mem, v_syn, vbSpikes

    # - Set up initial carry state
    carry0 = (v0, syn0, refractory0)

    # - Build noise trace
    # - Compute random numbers for reservoir noise
    __all__, subkey = rand.split(key)
    noise = noise_std * rand.normal(subkey, shape=(inputs.shape[0], np.size(v0)))

    # - Call scan to evaluate layer
    state, v_mem_ts, v_syn_ts, spike_raster_ts = scan(
        step_iaf_exp, carry0, (inputs, noise)
    )

    # - Return state and outputs
    return state, v_mem_ts, v_syn_ts, spike_raster_ts


def _evolve_iaf_expsyn_io(
    v0: np.ndarray,
    syn0: np.ndarray,
    refractory0: np.ndarray,
    bias: np.ndarray,
    tau_m: np.ndarray,
    tau_s: np.ndarray,
    refractory: np.ndarray,
    w_in: np.ndarray,
    w_rec: np.ndarray,
    w_out: np.ndarray,
    inputs: np.ndarray,
    noise_std: float,
    key,
    dt: float,
) -> Tuple[Tuple, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # - Call recurrent evolution function
    state, v_mem_ts, v_syn_ts, spike_raster_ts = _evolve_iaf_expsyn(
        v0,
        syn0,
        refractory0,
        bias,
        tau_m,
        tau_s,
        refractory,
        w_rec,
        np.dot(inputs, w_in),
        noise_std,
        key,
        dt,
    )

    # - Return state and outputs
    return state, v_mem_ts, v_syn_ts, spike_raster_ts, np.dot(v_syn_ts, w_out)


class RecIAFExpJax(Layer):
    """
    Recurrent spiking neuron layer (IAF), direct current input and synaptic current output. No input / output weights.

    `.RecIAFExpJax` is a basic recurrent spiking neuron layer, implemented with a JAX-backed Euler solver backend. Outputs are exponential synaptic currents generated by each layer neuron; no output weighting is provided. Inputs are provided by direct current injection onto the membrane of each neuron; no input weighting is provided. The layer is therefore N inputs -> N neurons -> N outputs.

    This layer can be used to implement gradient-based learning systems, using the JAX-provided automatic differentiation functionality of `jax.grad`.

    .. math::

        \\tau_m \\cdot \\dot{v_j} + v_j = I_{ext,j} + I_{rec,j} + bias_j + \\zeta_j(t)

        I_{rec} = W \\cdot I_{syn}

        \\tau_s \\cdot \\dot{I_{syn,j}} + I_{syn,j} = 0

        \\dot{r_j} = -1

    On spiking:

    .. math ::

        \\operatorname{if} v_j > 1 \\operatorname{and} r_j < 0

        \\rightarrow I_{syn,j} += 1

        \\rightarrow v_j -= 1

        \\rightarrow r_j = t_{ref}

    Each neuron has a membrane and synaptic time constant, :math:`\\tau_m` (`.tau_mem`) and :math:`\\tau_s` (`.tau_s`) respectively. Neurons share a common rest potential of 0, a firing threshold of 1, and a subtractive reset of -1. Neurons each have an optional bias current `.bias` (default: 0), and an optional refractory period :math:`{ref}` in seconds (`.refractory`, default: 0).

    On spiking, the synaptic variable :math:`I_{syn,j}` associated with each neuron is incremented by +1, and decays towards 0 with time constant :math:`\\tau_s` (`.tau_syn`).

    As output, this layer returns the synaptic current traces from the `.evolve` method. After each evolution, the attributes `.spikes_last_evolution` and `.v_mem_last_evolution` will be `.TimeSeries` objects containing the appropriate time series.
    """

    def __init__(
        self,
        w_recurrent: np.ndarray,
        tau_mem: FloatVector,
        tau_syn: FloatVector,
        bias: Optional[FloatVector] = 0.0,
        refractory: Optional[FloatVector] = 0.0,
        noise_std: Optional[float] = 0.0,
        dt: Optional[float] = None,
        name: Optional[str] = None,
        rng_key: Optional[int] = None,
    ):
        """
        A basic recurrent spiking neuron layer, with a JAX-implemented forward Euler solver

        :param ndarray w_recurrent:                     [N,N] Recurrent weight matrix
        :param ArrayLike[float] tau_mem:                [N,] Membrane time constants
        :param ArrayLike[float] tau_syn:                [N,] Output synaptic time constants
        :param Optional[ArrayLike[float]] bias:         [N,] Bias currents for each neuron (Default: 0)
        :param Optional[ArrayLike[float]] refractory:   [N,] Refractory period for each neuron (Default: 0)
        :param Optional[float] noise_std:               Std. dev. of white noise injected independently onto the membrane of each neuron (Default: 0)
        :param Optional[float] dt:                      Forward Euler solver time step. Default: min(tau_mem, tau_syn) / 10
        :param Optional[str] name:                      Name of this layer. Default: `None`
        :param Optional[int] rng_key:                   JAX pRNG key. Default: generate a new key
        """
        # - Ensure that weights are 2D
        w_recurrent = np.atleast_2d(w_recurrent)

        # - Transform arguments to JAX np.array
        tau_mem = np.array(tau_mem)
        tau_syn = np.array(tau_syn)
        bias = np.array(bias)
        refractory = np.array(refractory)

        if dt is None:
            dt = np.min(np.array((np.min(tau_mem), np.min(tau_syn)))) / 10.0

        # - Call super-class initialisation
        super().__init__(w_recurrent, dt, noise_std, name)

        # - Set properties
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.bias = bias
        self.refractory = refractory

        # - Get compiled evolution function
        self._evolve_jit = jit(_evolve_iaf_expsyn)

        # - Initialise "last evolution" attributes
        self.v_mem_last_evolution = None
        self.spikes_last_evolution = None

        # - Reset layer state
        self.reset_all()

        # - Seed RNG
        if rng_key is None:
            rng_key = rand.PRNGKey(onp.random.randint(0, 2 ** 63))
        _, self._rng_key = rand.split(rng_key)

    def reset_state(self):
        """
        Reset the membrane potentials, synaptic currents and refractory state for this layer
        """
        self._state = np.zeros((self._size,))
        self._state_syn = np.zeros((self._size,))
        self._state_refractory = -np.ones((self.size,))

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: Optional[bool] = False,
    ) -> TSContinuous:
        """
        Evolve the state of this layer given an input

        :param Optional[TSContinuous] ts_input: Input time series. Default: `None`, no stimulus is provided
        :param Optional[float] duration:        Simulation/Evolution time, in seconds. If not provided, then `num_timesteps` or the duration of `ts_input` is used to determine evolution time
        :param Optional[int] num_timesteps:     Number of evolution time steps, in units of `.dt`. If not provided, then `duration` or the duration of `ts_input` is used to determine evolution time
        :param Optional[bool]verbose:           Currently no effect, just for conformity

        :return TSContinuous:                   Output time series; the synaptic currents of each neuron
        """

        # - Prepare time base and inputs
        time_base, inps, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Call raw evolution function
        time_start = self.t
        v_mem_ts, v_syn_ts, spike_raster_ts = self._evolve_raw(inps)

        # - Record membrane traces
        self.v_mem_last_evolution = TSContinuous(time_base, onp.array(v_mem_ts))

        # - Record spike raster
        spikes_ids = onp.argwhere(onp.array(spike_raster_ts))
        self.spikes_last_evolution = TSEvent(
            spikes_ids[:, 0] * self.dt + time_start,
            spikes_ids[:, 1],
            t_start=time_start,
            t_stop=self.t,
            name="Spikes " + self.name,
            num_channels=self.size,
        )

        # - Wrap synaptic outputs as time series
        return TSContinuous(time_base, onp.array(v_syn_ts))

    def _evolve_raw(
        self, inps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Raw evolution over an input array

        :param ndarray inps:    Input matrix [T, I]

        :return:  (v_mem_ts, v_syn_ts, spike_raster_ts)
                v_mem_ts:        (np.ndarray) Time trace of neuron membrane potentials [T, N]
                v_syn_ts:        (np.ndarray) Time trace of output synaptic currents [T, N]
                spike_raster_ts: (np.ndarray) Boolean raster [T, N]; `True` if a spike occurred in time step `t`, from neuron `n`
        """
        # - Call compiled Euler solver to evolve reservoir
        state, v_mem_ts, v_syn_ts, spike_raster_ts = self._evolve_jit(
            self._state,
            self._state_syn,
            self._state_refractory,
            self._bias,
            self._tau_mem,
            self._tau_syn,
            self._refractory,
            self._weights,
            inps,
            self._noise_std,
            self._rng_key,
            self._dt,
        )

        # - Re-assign layer state
        self._state, self._state_syn, self._state_refractory = state

        # - Increment timesteps attribute
        self._timestep += inps.shape[0] - 1

        # - Return layer activity
        return v_mem_ts, v_syn_ts, spike_raster_ts

    def to_dict(self) -> dict:
        """
        Convert the configuration of this layer into a dictionary to assist in reconstruction

        :return: dict
        """
        config = super().to_dict()
        config["tau_mem"] = self.tau_mem.tolist()
        config["tau_syn"] = self.tau_syn.tolist()
        config["bias"] = self.bias.tolist()
        config["refractory"] = self.refractory.tolist()
        config["rng_key"] = self._rng_key.tolist()
        return config

    @property
    def w_recurrent(self) -> np.ndarray:
        """ (ndarray) Recurrent weight matrix [N,N] """
        return onp.array(self._weights)

    @w_recurrent.setter
    def w_recurrent(self, value: np.ndarray):
        assert np.ndim(value) == 2, "`w_recurrent` must be 2D"

        assert value.shape == (
            self._size,
            self._size,
        ), "`w_recurrent` must be [{:d}, {:d}]".format(self._size, self._size)

        self._weights = np.array(value).astype("float")

    @property
    def tau_mem(self) -> np.ndarray:
        """ (ndarray) Membrane time constant for each neuron [N,] """
        return onp.array(self._tau_mem)

    @tau_mem.setter
    def tau_mem(self, value: np.ndarray):
        # - Replicate `tau_mem` from a scalar value
        if np.size(value) == 1:
            value = np.repeat(value, self._size)

        assert (
            np.size(value) == self._size
        ), "`tau_mem` must have {:d} elements or be a scalar".format(self._size)

        self._tau_mem = np.reshape(value, self._size).astype("float")

    @property
    def tau_syn(self) -> np.ndarray:
        """ (ndarray) Output synaptic time constant for each neuron [N,] """
        return onp.array(self._tau_syn)

    @tau_syn.setter
    def tau_syn(self, value: np.ndarray):
        # - Replicate `tau_syn` from a scalar value
        if np.size(value) == 1:
            value = np.repeat(value, self._size)

        assert (
            np.size(value) == self._size
        ), "`tau_syn` must have {:d} elements or be a scalar".format(self._size)

        self._tau_syn = np.reshape(value, self._size).astype("float")

    @property
    def bias(self) -> np.ndarray:
        """ (ndarray) Bias current for each neuron [N,] """
        return onp.array(self._bias)

    @bias.setter
    def bias(self, value: np.ndarray):
        # - Replicate `bias` from a scalar value
        if np.size(value) == 1:
            value = np.repeat(value, self._size)

        assert (
            np.size(value) == self._size
        ), "`bias` must have {:d} elements or be a scalar".format(self._size)

        self._bias = np.reshape(value, self._size).astype("float")

    @property
    def refractory(self) -> np.ndarray:
        """ (ndarray) Refractory period for each neuron [N,] """
        return onp.array(self._refractory)

    @refractory.setter
    def refractory(self, value: np.ndarray):
        # - Replicate `refractory` from a scalar value
        if np.size(value) == 1:
            value = np.repeat(value, self._size)

        assert (
            np.size(value) == self._size
        ), "`refractory` must have {:d} elements or be a scalar".format(self._size)

        self._refractory = np.reshape(value, self._size).astype("float")

    @property
    def dt(self) -> float:
        """ (float) Forward Euler solver time step """
        return onp.array(self._dt).item(0)

    @dt.setter
    def dt(self, value: float):
        # - Ensure dt is numerically stable
        tau_min = np.min(self.tau) / 10.0
        if value is None:
            value = tau_min

        assert value >= tau_min, "`tau` must be at least {:.2e}".format(tau_min)

        self._dt = np.array(value).astype("float")


class RecIAFExpSpikeOutJax(RecIAFExpJax):
    """
    Recurrent spiking neuron layer (IAF), direct current input and spiking output. No input / output weights.

    `.RecIAFExpJax` is a basic recurrent spiking neuron layer, implemented with a JAX-backed Euler solver backend. Outputs are exponential synaptic currents generated by each layer neuron; no output weighting is provided. Inputs are provided by direct current injection onto the membrane of each neuron; no input weighting is provided. The layer is therefore N inputs -> N neurons -> N outputs.

    This layer can be used to implement gradient-based learning systems, using the JAX-provided automatic differentiation functionality of `jax.grad`.

    .. math::

        \\tau_m \\cdot \\dot{v_j} + v_j = I_{ext,j} + I_{rec,j} + bias_j + \\zeta_j(t)

        I_{rec} = W \\cdot I_{syn}

        \\tau_s \\cdot \\dot{I_{syn,j}} + I_{syn,j} = 0

        \\dot{r_j} = -1

    On spiking:

    .. math ::

        \\operatorname{if} v_j > 1 \\operatorname{and} r_j < 0

        \\rightarrow I_{syn,j} += 1

        \\rightarrow v_j -= 1

        \\rightarrow r_j = t_{ref}

    Each neuron has a membrane and synaptic time constant, :math:`\\tau_m` (`.tau_mem`) and :math:`\\tau_s` (`.tau_s`) respectively. Neurons share a common rest potential of 0, a firing threshold of 1, and a subtractive reset of -1. Neurons each have an optional bias current `.bias` (default: 0), and an optional refractory period :math:`{ref}` in seconds (`.refractory`, default: 0).

    On spiking, the synaptic variable :math:`I_{syn,j}` associated with each neuron is incremented by +1, and decays towards 0 with time constant :math:`\\tau_s` (`.tau_syn`).

    As output, this layer returns the synaptic current traces from the `.evolve` method. After each evolution, the attributes `.spikes_last_evolution` and `.v_mem_last_evolution` will be `.TimeSeries` objects containing the appropriate time series.
    """

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: Optional[bool] = False,
    ) -> TSEvent:
        """
        Evolve the state of this layer given an input

        :param Optional[TSContinuous] ts_input: Input time series. Default: `None`, no stimulus is provided
        :param Optional[float] duration:        Simulation/Evolution time, in seconds. If not provided, then `num_timesteps` or the duration of `ts_input` is used to determine evolution time
        :param Optional[int] num_timesteps:     Number of evolution time steps, in units of `.dt`. If not provided, then `duration` or the duration of `ts_input` is used to determine evolution time
        :param Optional[bool] verbose:          Currently no effect, just for conformity

        :return TSEvent:                   Output time series; spiking activity each neuron
        """

        # - Prepare time base and inputs
        time_base, inps, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Call raw evolution function
        time_start = self.t
        v_mem_ts, v_syn_ts, spike_raster_ts = self._evolve_raw(inps)

        # - Record membrane traces
        self.v_mem_last_evolution = TSContinuous(time_base, onp.array(v_mem_ts))

        # - Record synaptic output currents
        self.i_syn_last_evolution = TSContinuous(time_base, onp.array(v_syn_ts))

        # - Convert spike raster to TSEvent
        spikes_ids = onp.argwhere(onp.array(spike_raster_ts))
        spikes_last_evolution = TSEvent(
            spikes_ids[:, 0] * self.dt + time_start,
            spikes_ids[:, 1],
            t_start=time_start,
            t_stop=self.t,
            name="Spikes " + self.name,
            num_channels=self.size,
        )

        # - Wrap synaptic outputs as time series
        return spikes_last_evolution

    def cOutput(self):
        """ (TSEvent) Output `.TimeSeries` class: `.TSEvent` """
        return TSEvent


class RecIAFExpWithIOJax(RecIAFExpJax):
    """
    Recurrent spiking neuron layer (IAF), direct current input and synaptic current output. Input / output weights both supported.

    `.RecIAFExpJax` is a basic recurrent spiking neuron layer, implemented with a JAX-backed Euler solver backend. Outputs are exponential synaptic currents generated by each layer neuron; no output weighting is provided. Inputs are provided by direct current injection onto the membrane of each neuron; no input weighting is provided. The layer is therefore N inputs -> N neurons -> N outputs.

    This layer can be used to implement gradient-based learning systems, using the JAX-provided automatic differentiation functionality of `jax.grad`.

    .. math::

        \\tau_m \\cdot \\dot{v_j} + v_j = I_{ext,j} + I_{rec,j} + bias_j + \\zeta_j(t)

        I_{rec} = W \\cdot I_{syn}

        \\tau_s \\cdot \\dot{I_{syn,j}} + I_{syn,j} = 0

        \\dot{r_j} = -1

    On spiking:

    .. math ::

        \\operatorname{if} v_j > 1 \\operatorname{and} r_j < 0

        \\rightarrow I_{syn,j} += 1

        \\rightarrow v_j -= 1

        \\rightarrow r_j = t_{ref}

    Each neuron has a membrane and synaptic time constant, :math:`\\tau_m` (`.tau_mem`) and :math:`\\tau_s` (`.tau_s`) respectively. Neurons share a common rest potential of 0, a firing threshold of 1, and a subtractive reset of -1. Neurons each have an optional bias current `.bias` (default: 0), and an optional refractory period :math:`{ref}` in seconds (`.refractory`, default: 0).

    On spiking, the synaptic variable :math:`I_{syn,j}` associated with each neuron is incremented by +1, and decays towards 0 with time constant :math:`\\tau_s` (`.tau_syn`).

    As output, this layer returns the synaptic current traces from the `.evolve` method. After each evolution, the attributes `.spikes_last_evolution` and `.v_mem_last_evolution` will be `.TimeSeries` objects containing the appropriate time series.
    """

    def __init__(
        self,
        w_in: np.ndarray,
        w_recurrent: np.ndarray,
        w_out: np.ndarray,
        tau_mem: FloatVector,
        tau_syn: FloatVector,
        bias: Optional[FloatVector] = 0.0,
        refractory: Optional[FloatVector] = 0.0,
        noise_std: Optional[float] = 0.0,
        dt: Optional[float] = None,
        name: Optional[str] = None,
        rng_key: Optional[int] = None,
    ):
        # - Everything should be 2D
        w_in = np.atleast_2d(w_in)
        w_out = np.atleast_2d(w_out)

        # - Call super-class initialisation
        super().__init__(
            w_recurrent,
            tau_mem,
            tau_syn,
            bias,
            refractory,
            noise_std,
            dt,
            name,
            rng_key,
        )

        # - Get information about network size
        self._size_in = w_in.shape[0]
        self._size_out = w_out.shape[1]

        # - Get compiled evolution function
        self._evolve_jit = jit(_evolve_iaf_expsyn_io)

        # - Reset layer state
        self.reset_all()

        # - Seed RNG
        if rng_key is None:
            rng_key = rand.PRNGKey(onp.random.randint(0, 2 ** 63))
        _, self._rng_key = rand.split(rng_key)

        # - Store attribute values
        self.w_in = w_in
        self.w_out = w_out

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: Optional[bool] = False,
    ) -> TSContinuous:
        """
        Evolve the state of this layer given an input

        :param Optional[TSContinuous] ts_input: Input time series. Default: `None`, no stimulus is provided
        :param Optional[float] duration:        Simulation/Evolution time, in seconds. If not provided, then `num_timesteps` or the duration of `ts_input` is used to determine evolution time
        :param Optional[int] num_timesteps:     Number of evolution time steps, in units of `.dt`. If not provided, then `duration` or the duration of `ts_input` is used to determine evolution time
        :param Optional[bool]verbose:           Currently no effect, just for conformity

        :return TSContinuous:                   Output time series; the synaptic currents of each neuron
        """

        # - Prepare time base and inputs
        time_base, inps, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Call raw evolution function
        time_start = self.t
        v_mem_ts, v_syn_ts, spike_raster_ts, outputs = self._evolve_raw(inps)

        # - Record membrane traces
        self.v_mem_last_evolution = TSContinuous(time_base, onp.array(v_mem_ts))

        # - Record synaptic output currents
        self.i_syn_last_evolution = TSContinuous(time_base, onp.array(v_syn_ts))

        # - Convert spike raster to TSEvent and record
        spikes_ids = onp.argwhere(onp.array(spike_raster_ts))
        self.spikes_last_evolution = TSEvent(
            spikes_ids[:, 0] * self.dt + time_start,
            spikes_ids[:, 1],
            t_start=time_start,
            t_stop=self.t,
            name="Spikes " + self.name,
            num_channels=self.size,
        )

        # - Wrap synaptic outputs as time series
        return TSContinuous(time_base, onp.array(outputs))

    def _evolve_raw(
        self, inps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Raw evolution over an input array

        :param ndarray inps:    Input matrix [T, I]

        :return:  (v_mem_ts, v_syn_ts, spike_raster_ts, outputs)
                v_mem_ts:        (np.ndarray) Time trace of neuron membrane potentials [T, N]
                v_syn_ts:        (np.ndarray) Time trace of output synaptic currents [T, N]
                spike_raster_ts: (np.ndarray) Boolean raster [T, N]; `True` if a spike occurred in time step `t`, from neuron `n`
                outputs:         (np.ndarray) Time trace of output currents [T, O]
        """
        # - Call compiled Euler solver to evolve reservoir
        state, v_mem_ts, v_syn_ts, spike_raster_ts, outputs = self._evolve_jit(
            self._state,
            self._state_syn,
            self._state_refractory,
            self._bias,
            self._tau_mem,
            self._tau_syn,
            self._refractory,
            self._w_in,
            self._weights,
            self._w_out,
            inps,
            self._noise_std,
            self._rng_key,
            self._dt,
        )

        # - Re-assign layer state
        self._state, self._state_syn, self._state_refractory = state

        # - Increment timesteps attribute
        self._timestep += inps.shape[0] - 1

        # - Return layer activity
        return v_mem_ts, v_syn_ts, spike_raster_ts, outputs

    @property
    def w_in(self) -> np.ndarray:
        """ (ndarray) [M,N] Input weight matrix """
        return onp.array(self._w_in)

    @w_in.setter
    def w_in(self, value: np.ndarray):
        assert np.ndim(value) == 2, "`w_in` must be 2D"

        assert value.shape == (
            self._size_in,
            self._size,
        ), "`w_in` must be [{:d}, {:d}]".format(self._size_in, self._size)

        self._w_in = np.array(value).astype("float")

    @property
    def w_recurrent(self) -> np.ndarray:
        """ (ndarray) [N,N] Recurrent weight matrix """
        return onp.array(self._w_recurrent)

    @w_recurrent.setter
    def w_recurrent(self, value: np.ndarray):
        assert np.ndim(value) == 2, "`w_recurrent` must be 2D"

        assert value.shape == (
            self._size,
            self._size,
        ), "`w_recurrent` must be [{:d}, {:d}]".format(self._size, self._size)

        self._w_recurrent = np.array(value).astype("float")

    @property
    def w_out(self) -> np.ndarray:
        """ (ndarray) [N,O] Output weight matrix """
        return onp.array(self._w_out)

    @w_out.setter
    def w_out(self, value: np.ndarray):
        assert np.ndim(value) == 2, "`w_out` must be 2D"

        assert value.shape == (
            self._size,
            self._size_out,
        ), "`w_out` must be [{:d}, {:d}]".format(self._size, self._size_out)

        self._w_out = np.array(value).astype("float")

    def to_dict(self) -> dict:
        """
        Convert the configuration of this layer into a dictionary to assist in reconstruction

        :return: dict
        """
        config = super().to_dict()
        config["w_in"] = self.w_in.tolist()
        config["w_out"] = self.w_out.tolist()
        return config
