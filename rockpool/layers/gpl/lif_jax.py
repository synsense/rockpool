##
# Spiking layers with JAX backend
#

# - Imports
from ..layer import Layer
from ..training import JaxTrainer
from ...timeseries import TSContinuous, TSEvent, TimeSeries

from importlib import util

if (util.find_spec("jax") is None) or (util.find_spec("jaxlib") is None):
    raise ModuleNotFoundError(
        "'Jax' and 'Jaxlib' backend not found. Layers that rely on Jax will not be available."
    )

from jax import numpy as np
import numpy as onp

import jax
from jax import jit, custom_gradient
from jax.lax import scan
import jax.random as rand

from typing import Optional, Tuple, Union, Dict, Callable, Any

# - Define a float / array type
FloatVector = Union[float, np.ndarray]

# - Define a layer state type
State = Dict[
    str, np.ndarray
]  # TypedDict("LayerState", {"spikes": np.ndarray, "Isyn": np.ndarray, "Vmem": np.ndarray})

Params = Dict

# - Define module exports
__all__ = [
    "RecLIFJax",
    "RecLIFCurrentInJax",
    "RecLIFCurrentInJax_SO",
    "RecLIFJax_IO",
    "RecLIFCurrentInJax_IO",
    "FFLIFJax_IO",
    "FFLIFJax_SO",
    "FFLIFCurrentInJax_SO",
    "FFExpSynCurrentInJax",
    "FFExpSynJax",
]


# - Surrogate functions to use in learning
def sigmoid(x: FloatVector) -> FloatVector:
    """
    Sigmoid function

    :param FloatVector x: Input value

    :return FloatVector: Output value
    """
    return np.tanh(x + 1) / 2 + 0.5


@custom_gradient
def step_pwl(x: FloatVector) -> (FloatVector, Callable[[FloatVector], FloatVector]):
    """
    Heaviside step function with piece-wise linear derivative to use as spike-generation surrogate

    :param FloatVector x: Input value

    :return (FloatVector, Callable[[FloatVector], FloatVector]): output value and gradient function
    """
    s = np.clip(np.floor(x + 1.0), 0.0)
    return s, lambda g: (g * (x > -0.5),)


def _evolve_lif_jax(
    state0: State,
    w_in: np.ndarray,
    w_rec: np.ndarray,
    w_out_surrogate: np.ndarray,
    tau_mem: np.ndarray,
    tau_syn: np.ndarray,
    bias: np.ndarray,
    noise_std: float,
    sp_input_ts: np.ndarray,
    I_input_ts: np.ndarray,
    key: rand.PRNGKey,
    dt: float,
) -> Tuple[
    State,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    rand.PRNGKey,
]:
    """
    Raw JAX evolution function for an LIF spiking layer

    This function implements the dynamics:

    .. math ::

        \\tau_{syn} \\dot{I}_{syn} + I_{syn} = 0

        I_{syn} += S_{in}(t)

        \\tau_{syn} \\dot{V}_{mem} + V_{mem} = I_{syn} + b + \\sigma\\zeta(t)

    where :math:`S_{in}(t)` is a vector containing ``1`` for each input channel that emits a spike at time :math:`t`; :math:`b` is a :math:`N` vector of bias currents for each neuron; :math:`\\sigma\\zeta(t)` is a white-noise process with standard deviation :math:`\\sigma` injected independently onto each neuron's membrane; and :math:`\\tau_{mem}` and :math:`\\tau_{syn}` are the membrane and synaptic time constants, respectively.

    :On spiking:

    When the membrane potential for neuron :math:`j`, :math:`V_{mem, j}` exceeds the threshold voltage :math:`V_{thr} = 0`, then the neuron emits a spike.

    .. math ::

        V_{mem, j} > V_{thr} \\rightarrow S_{rec,j} = 1

        I_{syn} = I_{syn} + S_{rec} \\cdot w_{rec}

        V_{mem, j} = V_{mem, j} - 1

    Neurons therefore share a common resting potential of ``0``, a firing threshold of ``0``, and a subtractive reset of ``-1``. Neurons each have an optional bias current `.bias` (default: ``-1``).

    :Surrogate signals:

    To facilitate gradient-based training, a surrogate :math:`U(t)` is generated from the membrane potentials of each neuron.

    .. math ::

        U_j = \\textrm{tanh}(V_j + 1) / 2 + .5

    :param LayerState state0:           Layer state at start of evolution
    :param np.ndarray w_in:             Input weights [I, N]
    :param np.ndarray w_rec:            Recurrent weights [N, N]
    :param np.ndarray w_out_surrogate:  Output weights [N, O]
    :param np.ndarray tau_mem:          Membrane time constants for each neuron [N,]
    :param np.ndarray tau_syn:          Input synapse time constants for each neuron [N,]
    :param np.ndarray bias:             Bias values for each neuron [N,]
    :param float noise_std:             Noise injected onto the membrane of each neuron. Standard deviation at each time step.
    :param np.ndarray sp_input_ts:      Logical spike raster of input events on input channels [T, I]
    :param np.ndarray I_input_ts:       Time trace of currents injected on input channels (direct current injection) [T, I]
    :param jax.random.PRNGKey key:      pRNG key for JAX
    :param float dt:                    Time step in seconds

    :return: (state, Irec_ts, output_ts, surrogate_ts, spikes_ts, Vmem_ts, Isyn_ts)
        state:          (LayerState) Layer state at end of evolution
        Irec_ts:        (np.ndarray) Recurrent input received at each neuron over time [T, N]
        output_ts:      (np.ndarray) Weighted output surrogate over time [T, O]
        surrogate_ts:   (np.ndarray) Surrogate time trace for each neuron [T, N]
        spikes_ts:      (np.ndarray) Logical spiking raster for each neuron [T, N]
        Vmem_ts:        (np.ndarray) Membrane voltage of each neuron over time [T, N]
        Isyn_ts:        (np.ndarray) Synaptic input current received by each neuron over time [T, N]
    """

    # - Get evolution constants
    alpha = dt / tau_mem
    beta = np.exp(-dt / tau_syn)

    # - Single-step LIF dynamics
    def forward(
        state: State, inputs_t: Tuple[np.ndarray, np.ndarray]
    ) -> (
        State,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ):
        """
        Single-step LIF dynamics for a recurrent LIF layer

        :param LayerState state:
        :param Tuple[np.ndarray, np.ndarray] inputs_t: (spike_inputs_ts, current_inputs_ts)

        :return: (state, Irec_ts, output_ts, surrogate_ts, spikes_ts, Vmem_ts, Isyn_ts)
            state:          (LayerState) Layer state at end of evolution
            Irec_ts:        (np.ndarray) Recurrent input received at each neuron over time [T, N]
            output_ts:      (np.ndarray) Weighted output surrogate over time [T, O]
            surrogate_ts:   (np.ndarray) Surrogate time trace for each neuron [T, N]
            spikes_ts:      (np.ndarray) Logical spiking raster for each neuron [T, N]
            Vmem_ts:        (np.ndarray) Membrane voltage of each neuron over time [T, N]
            Isyn_ts:        (np.ndarray) Synaptic input current received by each neuron over time [T, N]
        """
        # - Unpack inputs
        (sp_in_t, I_in_t) = inputs_t
        sp_in_t = sp_in_t.reshape(-1)
        Iin = I_in_t.reshape(-1)

        # - Synaptic input
        Irec = np.dot(state["spikes"], w_rec)
        dIsyn = sp_in_t + Irec
        state["Isyn"] = beta * state["Isyn"] + dIsyn

        # - Apply subtractive reset
        state["Vmem"] = state["Vmem"] - state["spikes"]

        # - Membrane potentials
        dVmem = state["Isyn"] + Iin + bias - state["Vmem"]
        state["Vmem"] = state["Vmem"] + alpha * dVmem

        # - Detect next spikes (with custom gradient)
        state["spikes"] = step_pwl(state["Vmem"])

        # - Return state and outputs
        return state, (Irec, state["spikes"], state["Vmem"], state["Isyn"])

    # - Generate membrane noise trace
    # - Build noise trace
    # - Compute random numbers for reservoir noise
    num_timesteps = sp_input_ts.shape[0]
    key1, subkey = rand.split(key)
    noise_ts = noise_std * rand.normal(
        subkey, shape=(num_timesteps, np.size(state0["Vmem"]))
    )

    # - Evolve over spiking inputs
    state, (Irec_ts, spikes_ts, Vmem_ts, Isyn_ts) = scan(
        forward,
        state0,
        (np.dot(sp_input_ts, w_in), np.dot(I_input_ts, w_in) + noise_ts),
    )

    # - Generate output surrogate
    surrogate_ts = sigmoid(Vmem_ts * 20.0)

    # - Weighted output
    output_ts = np.dot(surrogate_ts, w_out_surrogate)

    # - Return outputs
    return state, Irec_ts, output_ts, surrogate_ts, spikes_ts, Vmem_ts, Isyn_ts, key1


# - Define default loss function
def loss_mse_reg_lif(
    params: Params,
    states_t: State,
    output_batch_t: np.ndarray,
    target_batch_t: np.ndarray,
    min_tau_mem: float,
    min_tau_syn: float,
    lambda_mse: float = 1.0,
    reg_tau: float = 10000.0,
    reg_l2_in: float = 0.1,
    reg_l2_rec: float = 1.0,
    reg_l2_out: float = 0.1,
    reg_act1: float = 2.0,
    reg_act2: float = 2.0,
) -> float:
    """
    Regularised MSE target-output loss function for Jax LIF layers

    This loss function computes the mean-squared error of the target signal versus the layer surrogate output. This loss is regularised by several terms to limit time constants, to control the weight spectra, and to control reservoir activity.

    .. math::
        L = \lambda_{mse}\\cdot L_{mse} + \\lambda_{\\tau}\\cdot L_{\\tau} + \\lambda_{L2}\\cdot L_{L2} + \\lambda_{act}\\cdot L_{act}

        L_{mse} = E|\\left(\\textbf{o}(t) - \\textbf{y}(t)\\right)^2| \\textrm{ : output versus target MSE}

        L_{\\tau} = l_{\\tau}(\\tau_{mem}, \\tau_{min}) + l_{\\tau}(\\tau_{syn}, \\tau_{min})

        l_{\\tau}(\\tau, \\tau_{min}) = \\sum \\exp(-(\\tau - \\tau_{min})) | \\tau < \\tau_{min}

        L_{L2} = l_{l2}(W_{in}) + l_{l2}(W_{rec}) + l_{l2}(W_{out})

        l_{l2}(W) = E|W^2|

        L_{act} = E|U(t)| + E|V_{mem}(t)^2|

        \textrm{where } E|\textbf{x}| = \\frac{1}{#\textbf{x}}\\sum{x}

    :param Params params:               Parameters of the LIF layer
    :param State states_t:              State time-series of the LIF layer
    :param np.ndarray output_batch_t:   Output time series of the layer
    :param np.ndarray target_batch_t:   Target time series of the layer
    :param float min_tau_mem:           Minimum permitted membrane time constant
    :param float min_tau_syn:           Minimum permitted synaptic time constant
    :param float lambda_mse:            Loss factor for MSE error. Default: 1.0
    :param float reg_tau:               Regularisation loss factor for time constant violations. Default: 10000.0
    :param float reg_l2_in:             Regularisation loss factor for input weight L2 norm. Default: 0.1
    :param float reg_l2_rec:            Regularisation loss factor for recurrent weight L2 norm. Default: 1.0
    :param float reg_l2_out:            Regularisation loss factor for output weight L2 norm. Default: 0.1
    :param float reg_act1:              Regularisation loss factor for activity (keeps activity low). Default: 2.0
    :param float reg_act2:              Regularisation loss factor for activity (keeps membranes near threshold). Default: 2.0

    :return float loss:                 Loss value
    """
    # - MSE between output and target
    dLoss = dict()
    dLoss["loss_mse"] = lambda_mse * np.mean((output_batch_t - target_batch_t) ** 2)

    # - Get loss for tau parameter constraints
    dLoss["loss_tau_syn"] = reg_tau * np.nansum(
        np.where(
            params["tau_syn"] < min_tau_syn,
            np.exp(-(params["tau_syn"] - min_tau_syn)),
            0,
        )
    )
    dLoss["loss_tau_mem"] = reg_tau * np.nansum(
        np.where(
            params["tau_mem"] < min_tau_mem,
            np.exp(-(params["tau_mem"] - min_tau_mem)),
            0,
        )
    )

    # - Regularisation for weights
    dLoss["loss_weights_l2"] = (
        reg_l2_in * np.mean(params["w_in"] ** 2)
        + reg_l2_rec * np.mean(params["w_recurrent"] ** 2)
        + reg_l2_out * np.mean(params["w_out"] ** 2)
    )

    # - Regularisation for activity
    dLoss["loss_activity1"] = reg_act1 * np.mean(states_t["surrogate"])
    dLoss["loss_activity2"] = reg_act2 * np.mean(states_t["Vmem"] ** 2)

    # - Return loss
    return sum(dLoss.values())


class RecLIFJax(Layer, JaxTrainer):
    """
    Recurrent spiking neuron layer (LIF), spiking input and spiking output. No input / output weights.

    `.RecLIFJax` is a basic recurrent spiking neuron layer, implemented with a JAX-backed Euler solver backend. Outputs are spikes generated by each layer neuron; no output weighting is provided. Inputs are provided by spiking through a synapse onto each layer neuron; no input weighting is provided. The layer is therefore N inputs -> N neurons -> N outputs.

    :Dynamics:

    The dynamics of the :math:`N` neurons' membrane potential :math:`V_{mem}` and the :math:`N` synaptic currents :math:`I_{syn}` evolve under the system

    .. math::

        \\tau_{syn} \\dot{I}_{syn} + I_{syn} = 0

        I_{syn} += S_{in}(t)

        \\tau_{syn} \\dot{V}_{mem} + V_{mem} = I_{syn} + b + \\sigma\\zeta(t)

    where :math:`S_{in}(t)` is a vector containing ``1`` for each input channel that emits a spike at time :math:`t`; :math:`b` is a :math:`N` vector of bias currents for each neuron; :math:`\\sigma\\zeta(t)` is a white-noise process with standard deviation :math:`\\sigma` injected independently onto each neuron's membrane; and :math:`\\tau_{mem}` and :math:`\\tau_{syn}` are the membrane and synaptic time constants, respectively.

    :On spiking:

    When the membane potential for neuron :math:`j`, :math:`V_{mem, j}` exceeds the threshold voltage :math:`V_{thr} = 0`, then the neuron emits a spike.

    .. math ::

        V_{mem, j} > V_{thr} \\rightarrow S_{rec,j} = 1

        I_{syn} = I_{syn} + S_{rec} \\cdot w_{rec}

        V_{mem, j} = V_{mem, j} - 1

    Neurons therefore share a common resting potential of ``0``, a firing threshold of ``0``, and a subtractive reset of ``-1``. Neurons each have an optional bias current `.bias` (default: ``-1``).

    :Surrogate signals:

    To facilitate gradient-based training, a surrogate :math:`U(t)` is generated from the membrane potentials of each neuron.

    .. math ::

        U_j = \\textrm{sig}(V_j)

    Where :math:`\\textrm{sig}(x) = \\left(1 + \\exp(-x)\\right)^{-1}`.

    :Outputs from evolution:

    As output, this layer returns the spiking activity of the :math:`N` neurons from the `.evolve` method. After each evolution, the attributes `.spikes_last_evolution`, `.i_rec_last_evolution` and `.v_mem_last_evolution` and `.surrogate_last_evolution` will be `.TimeSeries` objects containing the appropriate time series.
    """

    def __init__(
        self,
        w_recurrent: np.ndarray,
        tau_mem: FloatVector,
        tau_syn: FloatVector,
        bias: FloatVector = -1.0,
        noise_std: float = 0.0,
        dt: Optional[float] = None,
        name: Optional[str] = None,
        rng_key: Optional[list] = None,
        **kwargs,
    ):
        """
        A basic recurrent spiking neuron layer, with a JAX-implemented forward Euler solver.

        :param ndarray w_recurrent:                     [N,N] Recurrent weight matrix
        :param FloatVector tau_mem:                     [N,] Membrane time constants
        :param FloatVector tau_syn:                     [N,] Output synaptic time constants
        :param FloatVector bias:                        [N,] Bias currents for each neuron (Default: 0)
        :param float noise_std:                         Std. dev. of white noise injected independently onto the membrane of each neuron (Default: 0)
        :param Optional[float] dt:                      Forward Euler solver time step. Default: min(tau_mem, tau_syn) / 10
        :param Optional[str] name:                      Name of this layer. Default: ``None``
        :param Optional[list] rng_key:                  List of two integers representing the state of the Jax PRNG. Default: ``None``; generate a new key
        """
        # - Ensure that weights are 2D
        w_recurrent = onp.atleast_2d(w_recurrent)

        # - Transform arguments to JAX np.array
        tau_mem = np.array(tau_mem)
        tau_syn = np.array(tau_syn)
        bias = np.array(bias)

        if dt is None:
            dt = np.min(np.array((np.min(tau_mem), np.min(tau_syn)))) / 10.0

        # - Call super-class initialisation
        super().__init__(
            weights=w_recurrent, dt=dt, noise_std=noise_std, name=name, **kwargs
        )

        # - Set properties
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.bias = bias

        self._w_in = 1.0
        self._w_out = 1.0

        # - Get compiled evolution function
        self._evolve_jit = jit(_evolve_lif_jax)

        # - Reset layer state
        self.reset_all()

        # - Create RNG
        if rng_key is None:
            rng_key = rand.PRNGKey(onp.random.randint(0, 2 ** 63))
            _, self._rng_key = rand.split(rng_key)
        else:
            rng_key = np.array(onp.array(rng_key).astype(onp.uint32))
            self._rng_key = rng_key

        # - Define stored internal state properties
        self._v_mem_last_evolution = []
        self._surrogate_last_evolution = []
        self._spikes_last_evolution = []
        self._i_syn_last_evolution = []
        self._i_rec_last_evolution = []
        self._output_last_evolution = []

    # - Replace the default loss function
    @property
    def _default_loss(self) -> Callable[[Any], float]:
        return loss_mse_reg_lif

    @property
    def _default_loss_params(self) -> Dict:
        return {
            "min_tau_syn": self._dt * 11.0,
            "min_tau_mem": self._dt * 11.0,
        }

    def _pack(self) -> Params:
        """
        Return a packed form of the tunable parameters for this layer

        :return Params: params: All parameters as a Dict
        """
        return {
            "w_in": self._w_in,
            "w_recurrent": self._weights,
            "w_out": self._w_out,
            "bias": self._bias,
            "tau_mem": self._tau_mem,
            "tau_syn": self._tau_syn,
        }

    def _unpack(self, params: Params):
        """
        Set the parameters for this layer, given a parameter dictionary

        :param Params params:  Set of parameters for this layer
        """
        (
            self._w_in,
            self._weights,
            self._w_out,
            self._bias,
            self._tau_mem,
            self._tau_syn,
        ) = (
            params["w_in"],
            params["w_recurrent"],
            params["w_out"],
            params["bias"],
            params["tau_mem"],
            params["tau_syn"],
        )

    # - Define stored state properties
    @property
    def v_mem_last_evolution(self):
        """(TSContinuous) Membrane potential traces saved during the most recent evolution"""
        return self._v_mem_last_evolution

    @v_mem_last_evolution.setter
    def v_mem_last_evolution(self, **_):
        "Some blah"
        raise ValueError("Setting the evolution properties is not permitted.")

    @property
    def surrogate_last_evolution(self):
        """(TSContinuous) Surrogate traces saved during the most recent evolution"""
        return self._surrogate_last_evolution

    @property
    def spikes_last_evolution(self):
        """(TSEvent) Spike trains emitted by the layer neurons, saved during the most recent evolution"""
        return self._spikes_last_evolution

    @property
    def i_syn_last_evolution(self):
        """(TSContinuous) Synaptic external input current traces saved during the most recent evolution"""
        return self._i_syn_last_evolution

    @property
    def i_rec_last_evolution(self):
        """(TSContinuous) Recurrent synaptic input current traces saved during the most recent evolution"""
        return self._i_rec_last_evolution

    @property
    def output_last_evolution(self):
        """(TSContinuous) Weighted surrogate output saved during the most recent evolution"""
        return self._output_last_evolution

    def reset_state(self):
        """
        Reset the membrane potentials, synaptic currents and refractory state for this layer
        """
        self._state = {
            "Vmem": np.ones((self._size,)) * self._bias,
            "Isyn": np.zeros((self._size,)),
            "spikes": np.zeros((self._size,)),
        }

    @property
    def state(self) -> State:
        """
        Internal state of the neurons in this layer
        :return: dict{"Vmem", "Isyn", "spikes"}
        """
        return {k: np.array(v) for k, v in self._state.items()}

    @state.setter
    def state(self, new_state: State):
        """
        Setter for state values. Verifies that new state dict contains correct keys and sizes.
        `new_state` must be a dict{"Vmem", "Isyn", "spikes"}
        """
        # - Verify that `new_state` has the correct sizes
        for k, v in new_state.items():
            if np.size(v) != self.size:
                raise ValueError(
                    self.start_print
                    + "New state values must have {} elements".format(self.size)
                )

        # - Verify that `new_state` contains the correct keys
        if (
            "Vmem" not in new_state.keys()
            or "Isyn" not in new_state.keys()
            or "spikes" not in new_state.keys()
        ):
            raise ValueError(
                "`new_state` must be a dict containing keys 'Vmem', 'Isyn' and 'spikes'."
            )

        # - Update state dictionary
        self._state.update(new_state)

    @property
    def _evolve_functional(self):
        """
        Return a functional form of the evolution function for this layer

        Returns a function ``evol_func`` with the signature::

            def evol_func(params, state, inputs) -> (outputs, new_state):

        :return Callable[[Params, State, np.ndarray], Tuple[np.ndarray, State, Dict[str, np.ndarray]]]:
        """

        def evol_func(
            params: Params,
            state: State,
            sp_input_ts: np.ndarray,
        ) -> Tuple[np.ndarray, State, Dict[str, np.ndarray]]:
            # - Call the jitted evolution function for this layer
            (
                new_state,
                Irec_ts,
                output_ts,
                surrogate_ts,
                spikes_ts,
                Vmem_ts,
                Isyn_ts,
                key1,
            ) = self._evolve_jit(
                state,
                params["w_in"],
                params["w_recurrent"],
                params["w_out"],
                params["tau_mem"],
                params["tau_syn"],
                params["bias"],
                self._noise_std,
                sp_input_ts,
                sp_input_ts * 0.0,
                self._rng_key,
                self._dt,
            )

            # - Maintain RNG key, if not under compilation
            if not isinstance(key1, jax.core.Tracer):
                self._rng_key = key1

            # - Return the outputs from this layer, and the final layer state
            states_t = {
                "Vmem": Vmem_ts,
                "Isyn": Isyn_ts,
                "Irec": Irec_ts,
                "surrogate": surrogate_ts,
                "spikes": spikes_ts,
                "output": output_ts,
            }
            return spikes_ts, new_state, states_t

        # - Return the evolution function
        return evol_func

    def _get_outputs_from_state(self, state):
        surrogate = sigmoid(state["Vmem"] * 20.0)
        output = np.dot(surrogate, self._w_out)
        Irec = np.dot(state["Isyn"], self.w_recurrent)
        spikes = state["spikes"]

        return output, surrogate, Irec, spikes

    def evolve(
        self,
        ts_input: Optional[TimeSeries] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        Evolve the state of this layer given an input

        :param Optional[TimeSeries] ts_input:      Input time series. Default: `None`, no stimulus is provided
        :param Optional[float] duration:        Simulation/Evolution time, in seconds. If not provided, then `num_timesteps` or the duration of `ts_input` is used to determine evolution time
        :param Optional[int] num_timesteps:     Number of evolution time steps, in units of `.dt`. If not provided, then `duration` or the duration of `ts_input` is used to determine evolution time
        :param bool verbose:           Currently no effect, just for conformity

        :return TSContinuous:                   Output time series; the synaptic currents of each neuron
        """

        # - Prepare time base and inputs
        time_base_inp, inps, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Call raw evolution function and update state
        (
            __,
            self._state,
            states_t,
        ) = self._evolve_functional(self._pack(), self._state, inps)

        # - Record spike raster
        self._spikes_last_evolution = TSEvent.from_raster(
            raster=onp.array(states_t["spikes"]),
            dt=self.dt,
            t_start=self.t,
            periodic=False,
            num_channels=self.size,
            spikes_at_bin_start=False,
            name="Spikes " + self.name,
        )

        # - Record membrane traces
        self._v_mem_last_evolution = TSContinuous.from_clocked(
            onp.array(states_t["Vmem"]),
            t_start=self.t,
            dt=self.dt,
            name="V_mem " + self.name,
        )

        # - Record neuron surrogates
        self._surrogate_last_evolution = TSContinuous.from_clocked(
            onp.array(states_t["surrogate"]),
            t_start=self.t,
            dt=self.dt,
            name="$U$ " + self.name,
        )

        # - Record recurrent inputs
        self._i_rec_last_evolution = TSContinuous.from_clocked(
            onp.array(states_t["Irec"]),
            t_start=self.t,
            dt=self.dt,
            name="$I_{rec}$ " + self.name,
        )

        # - Record synaptic currents
        self._i_syn_last_evolution = TSContinuous.from_clocked(
            onp.array(states_t["Isyn"]),
            t_start=self.t,
            dt=self.dt,
            name="$I_{syn}$ " + self.name,
        )

        self._output_last_evolution = TSContinuous.from_clocked(
            onp.array(states_t["output"]),
            t_start=self.t,
            dt=self.dt,
            name="$O$ " + self.name,
        )

        # - Update time
        self._timestep += num_timesteps

        # - Wrap spiking outputs as time series
        return self._spikes_last_evolution

    def randomize_state(self):
        """
        Randomize the internal state of this layer.

        `.state['Isyn']` will be drawn from a Normal distribution with std. dev. 1/10. `.state['Vmem']` will be uniformly distributed between -1. and 1. `.state['spikes']` will be zeroed.
        """
        _, subkey = rand.split(self._rng_key)
        self._state["Isyn"] = rand.normal(subkey, (self.size,)) / 10.0
        _, subkey = rand.split(self._rng_key)
        self._state["Vmem"] = rand.uniform(
            subkey, (self.size,), minval=-1.0, maxval=0.0
        )
        self._state["spikes"] = np.zeros((self.size,))

    def to_dict(self) -> dict:
        """
        Convert the configuration of this layer into a dictionary to assist in reconstruction

        :return: dict
        """
        config = super().to_dict()
        config.pop("weights")
        config["w_recurrent"] = onp.array(self.w_recurrent).tolist()
        config["tau_mem"] = onp.array(self.tau_mem).tolist()
        config["tau_syn"] = onp.array(self.tau_syn).tolist()
        config["bias"] = onp.array(self.bias).tolist()
        config["rng_key"] = onp.array(self._rng_key).tolist()

        return config

    @property
    def w_recurrent(self) -> np.ndarray:
        """ (ndarray) Recurrent weight matrix [N, N] """
        return onp.array(self._weights)

    @w_recurrent.setter
    def w_recurrent(self, value: np.ndarray):
        if np.ndim(value) != 2:
            raise ValueError(self.start_print + "`w_recurrent` must be 2D")

        if value.shape != (self._size, self._size):
            raise ValueError(
                self.start_print
                + "`w_recurrent` must be [{:d}, {:d}]".format(self._size, self._size)
            )

        self._weights = np.array(value).astype("float32")

    @property
    def tau_mem(self) -> np.ndarray:
        """ (ndarray) Membrane time constant for each neuron [N,] """
        return onp.array(self._tau_mem)

    @tau_mem.setter
    def tau_mem(self, value: np.ndarray):
        # - Replicate `tau_mem` from a scalar value
        if np.size(value) == 1:
            value = np.repeat(value, self._size)

        if np.size(value) != self._size:
            raise ValueError(
                self.start_print
                + "`tau_mem` must have {:d} elements or be a scalar".format(self._size)
            )

        # - Check for valid time constant
        if np.any(value <= 0.0):
            raise ValueError(self.start_print + "`tau_mem` must be larger than zero")

        if hasattr(self, "dt"):
            tau_min = self.dt * 10.0
            numeric_eps = 1e-8
            if np.any(value - tau_min + numeric_eps < 0):
                raise ValueError(
                    self.start_print
                    + "`tau_mem` must be larger than {:4f}".format(tau_min)
                )

        self._tau_mem = np.reshape(value, self._size).astype("float32")

    @property
    def tau_syn(self) -> np.ndarray:
        """ (ndarray) Output synaptic time constant for each neuron [N,] """
        return onp.array(self._tau_syn)

    @tau_syn.setter
    def tau_syn(self, value: np.ndarray):
        # - Replicate `tau_syn` from a scalar value
        if np.size(value) == 1:
            value = np.repeat(value, self._size)

        if np.size(value) != self._size:
            raise ValueError(
                self.start_print
                + "`tau_syn` must have {:d} elements or be a scalar".format(self._size)
            )

        # - Check for valid time constant
        if np.any(value <= 0.0):
            raise ValueError(self.start_print + "`tau_syn` must be larger than zero")

        if hasattr(self, "dt"):
            tau_min = self.dt * 10.0
            numeric_eps = 1e-8
            if np.any(value - tau_min + numeric_eps < 0):
                raise ValueError(
                    self.start_print
                    + "`tau_syn` must be larger than {:4f}".format(tau_min)
                )

        self._tau_syn = np.reshape(value, self._size).astype("float32")

    @property
    def bias(self) -> np.ndarray:
        """ (ndarray) Bias current for each neuron [N,] """
        return onp.array(self._bias)

    @bias.setter
    def bias(self, value: np.ndarray):
        # - Replicate `bias` from a scalar value
        if np.size(value) == 1:
            value = np.repeat(value, self._size)

        if np.size(value) != self._size:
            raise ValueError(
                self.start_print
                + "`bias` must have {:d} elements or be a scalar".format(self._size)
            )

        self._bias = np.reshape(value, self._size).astype("float32")

    @property
    def dt(self) -> float:
        """ (float) Forward Euler solver time step """
        return onp.array(self._dt).item(0)

    @dt.setter
    def dt(self, value: float):
        """ (float) Time step in seconds """
        # - Ensure dt is numerically stable
        tau_min = np.min(np.min(self._tau_mem), np.min(self._tau_syn)) / 10.0
        if value is None:
            value = tau_min

        # - Check for valid time constant
        if np.any(value <= 0.0):
            raise ValueError(self.start_print + "`dt` must be larger than zero")
        if value < tau_min:
            raise ValueError(
                self.start_print + "`dt` must be at least {:.2e}".format(tau_min)
            )

        self._dt = np.array(value).astype("float32")

    @property
    def output_type(self):
        """ (TSEvent) Output `.TimeSeries` class: `.TSEvent` """
        return TSEvent

    @property
    def input_type(self):
        """ (TSEvent) Input `.TimeSeries` class: `.TSEvent` """
        return TSEvent


class RecLIFCurrentInJax(RecLIFJax):
    """
    Recurrent spiking neuron layer (LIF), current injection input and spiking output. No input / output weights.

    `.RecLIFCurrentInJax` is a basic recurrent spiking neuron layer, implemented with a JAX-backed Euler solver backend. Outputs are spikes generated by each layer neuron; no output weighting is provided. Inputs are provided by direct current injection onto each neuron membrane; no input weighting is provided. The layer is therefore N inputs -> N neurons -> N outputs.

    :Dynamics:

    The dynamics of the :math:`N` neurons' membrane potential :math:`V_{mem}` and the :math:`N` synaptic currents :math:`I_{syn}` evolve under the system

    .. math::

        \\tau_{syn} \\dot{I}_{syn} + I_{syn} = 0

        \\tau_{syn} \\dot{V}_{mem} + V_{mem} = I_{syn} + I_{in}(t) + b + \\sigma\\zeta(t)

    where :math:`S_{in}(t)` is a vector containing ``1`` for each input channel that emits a spike at time :math:`t`; :math:`I_{in}(t)` is a vector of input currents injected directly onto the neuron membranes; :math:`b` is a :math:`N` vector of bias currents for each neuron; :math:`\\sigma\\zeta(t)` is a white-noise process with standard deviation :math:`\\sigma` injected independently onto each neuron's membrane; and :math:`\\tau_{mem}` and :math:`\\tau_{syn}` are the membrane and synaptic time constants, respectively.

    :On spiking:

    When the membane potential for neuron :math:`j`, :math:`V_{mem, j}` exceeds the threshold voltage :math:`V_{thr} = 0`, then the neuron emits a spike.

    .. math ::

        V_{mem, j} > V_{thr} \\rightarrow S_{rec,j} = 1

        I_{syn} = I_{syn} + S_{rec} \\cdot w_{rec}

        V_{mem, j} = V_{mem, j} - 1

    Neurons threfore share a common resting potential of ``0``, a firing threshold of ``0``, and a subtractive reset of ``-1``. Neurons each have an optional bias current `.bias` (default: ``-1``).

    :Surrogate signals:

    To facilitate gradient-based training, a surrogate :math:`U(t)` is generated from the membrane potentials of each neuron.

    .. math ::

        U_j = \\textrm{sig}(V_j)

    Where :math:`\\textrm{sig}(x) = \\left(1 + \\exp(-x)\\right)^{-1}`.

    :Outputs from evolution:

    As output, this layer returns the spiking activity of the :math:`N` neurons from the `.evolve` method. After each evolution, the attributes `.spikes_last_evolution`, `.i_rec_last_evolution` and `.v_mem_last_evolution` and `.surrogate_last_evolution` will be `.TimeSeries` objects containing the appropriate time series.
    """

    @property
    def _evolve_functional(self):
        """
        Return a functional form of the evolution function for this layer

        Returns a function ``evol_func`` with the signature::

            def evol_func(params, state, inputs) -> (outputs, new_state):

        :return Callable[[Params, State, np.ndarray], Tuple[np.ndarray, State]]:
        """

        def evol_func(
            params: Params,
            state: State,
            I_input_ts: np.ndarray,
        ):
            # - Call the jitted evolution function for this layer
            (
                new_state,
                Irec_ts,
                output_ts,
                surrogate_ts,
                spikes_ts,
                Vmem_ts,
                Isyn_ts,
                key1,
            ) = self._evolve_jit(
                state,
                params["w_in"],
                params["w_recurrent"],
                params["w_out"],
                params["tau_mem"],
                params["tau_syn"],
                params["bias"],
                self._noise_std,
                I_input_ts * 0.0,
                I_input_ts,
                self._rng_key,
                self._dt,
            )

            # - Maintain RNG key, if not under compilation
            if not isinstance(key1, jax.core.Tracer):
                self._rng_key = key1

            # - Return the outputs from this layer, and the final layer state
            states_t = {
                "Vmem": Vmem_ts,
                "Isyn": Isyn_ts,
                "Irec": Irec_ts,
                "surrogate": surrogate_ts,
                "spikes": spikes_ts,
                "output": output_ts,
            }
            return spikes_ts, new_state, states_t

        # - Return the evolution function
        return evol_func

    @property
    def output_type(self):
        """ (TSEvent) Output `.TimeSeries` class: `.TSEvent` """
        return TSEvent

    @property
    def input_type(self):
        """ (TSContinuous) Input `.TimeSeries` class: `.TSContinuous` """
        return TSContinuous


class RecLIFCurrentInJax_SO(RecLIFCurrentInJax):
    """
    Recurrent spiking neuron layer (LIF), current injection input and spiking output. No input / output weights.

    `.RecLIFCurrentInJax` is a basic recurrent spiking neuron layer, implemented with a JAX-backed Euler solver backend. Outputs are spikes generated by each layer neuron; no output weighting is provided. Inputs are provided by direct current injection onto each neuron membrane; no input weighting is provided. The layer is therefore N inputs -> N neurons -> N outputs.

    :Dynamics:

    The dynamics of the :math:`N` neurons' membrane potential :math:`V_{mem}` and the :math:`N` synaptic currents :math:`I_{syn}` evolve under the system

    .. math::

        \\tau_{syn} \\dot{I}_{syn} + I_{syn} = 0

        \\tau_{syn} \\dot{V}_{mem} + V_{mem} = I_{syn} + I_{in}(t) + b + \\sigma\\zeta(t)

    where :math:`S_{in}(t)` is a vector containing ``1`` for each input channel that emits a spike at time :math:`t`; :math:`I_{in}(t)` is a vector of input currents injected directly onto the neuron membranes; :math:`b` is a :math:`N` vector of bias currents for each neuron; :math:`\\sigma\\zeta(t)` is a white-noise process with standard deviation :math:`\\sigma` injected independently onto each neuron's membrane; and :math:`\\tau_{mem}` and :math:`\\tau_{syn}` are the membrane and synaptic time constants, respectively.

    :On spiking:

    When the membane potential for neuron :math:`j`, :math:`V_{mem, j}` exceeds the threshold voltage :math:`V_{thr} = 0`, then the neuron emits a spike.

    .. math ::

        V_{mem, j} > V_{thr} \\rightarrow S_{rec,j} = 1

        I_{syn} = I_{syn} + S_{rec} \\cdot w_{rec}

        V_{mem, j} = V_{mem, j} - 1

    Neurons threfore share a common resting potential of ``0``, a firing threshold of ``0``, and a subtractive reset of ``-1``. Neurons each have an optional bias current `.bias` (default: ``-1``).

    :Surrogate signals:

    To facilitate gradient-based training, a surrogate :math:`U(t)` is generated from the membrane potentials of each neuron.

    .. math ::

        U_j = \\textrm{sig}(V_j)

    Where :math:`\\textrm{sig}(x) = \\left(1 + \\exp(-x)\\right)^{-1}`.

    :Outputs from evolution:

    As output, this layer returns the spiking activity of the :math:`N` neurons from the `.evolve` method. After each evolution, the attributes `.spikes_last_evolution`, `._i_rec_last_evolution` and `._v_mem_last_evolution` and `._surrogate_last_evolution` will be `.TimeSeries` objects containing the appropriate time series.
    """

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSContinuous:
        """
        Evolve the state of this layer given an input

        :param Optional[TSContinuous] ts_input: Input time series. Default: `None`, no stimulus is provided
        :param Optional[float] duration:        Simulation/Evolution time, in seconds. If not provided, then `num_timesteps` or the duration of `ts_input` is used to determine evolution time
        :param Optional[int] num_timesteps:     Number of evolution time steps, in units of `.dt`. If not provided, then `duration` or the duration of `ts_input` is used to determine evolution time
        :param bool verbose:                    Currently no effect, just for conformity

        :return TSContinuous:                   Output time series; surrogate activity of each neuron
        """

        # - Call evolution function
        super().evolve(ts_input, duration, num_timesteps, verbose)

        # - Return spiking outputs
        return self._surrogate_last_evolution

    @property
    def output_type(self):
        """ (TSContinuous) Output `.TimeSeries` class: `.TSContinuous` """
        return TSContinuous


class RecLIFJax_IO(RecLIFJax):
    """
    Recurrent spiking neuron layer (LIF), spiking input and weighted surrogate output. Input and output weights.

    `.RecLIFJax` is a basic recurrent spiking neuron layer, implemented with a JAX-backed Euler solver backend. Outputs are surrogates generated by each layer neuron, weighted by a set of output weights. Inputs are provided by spiking through a synapse onto each layer neuron via a set of input weights. The layer is therefore M inputs -> N neurons -> O outputs.

    This layer can be used to implement gradient-based learning systems, using the JAX-provided automatic differentiation functionality of `jax.grad`. See :py:meth:`.train_output_target` for information on training.

    :Dynamics:

    The dynamics of the :math:`N` neurons' membrane potential :math:`V_{mem}` and the :math:`N` synaptic currents :math:`I_{syn}` evolve under the system

    .. math::

        \\tau_{syn} \\dot{I}_{syn} + I_{syn} = 0

        I_{syn} += S_{in}(t) \\cdot w_{in}

        \\tau_{syn} \\dot{V}_{mem} + V_{mem} = I_{syn} + I_{in}(t) \\cdot w_{in} + b + \\sigma\\zeta(t)

    where :math:`S_{in}(t)` is a vector containing ``1`` for each input channel that emits a spike at time :math:`t`; :math:`w_{in}` is a :math:`[N_{in} \\times N]` matrix of input weights; :math:`I_{in}(t)` is a vector of input currents injected directly onto the neuron membranes; :math:`b` is a :math:`N` vector of bias currents for each neuron; :math:`\\sigma\\zeta(t)` is a white-noise process with standard deviation :math:`\\sigma` injected independently onto each neuron's membrane; and :math:`\\tau_{mem}` and :math:`\\tau_{syn}` are the membrane and synaptic time constants, respectively.

    :On spiking:

    When the membrane potential for neuron :math:`j`, :math:`V_{mem, j}` exceeds the threshold voltage :math:`V_{thr} = 0`, then the neuron emits a spike.

    .. math ::

        V_{mem, j} > V_{thr} \\rightarrow S_{rec,j} = 1

        I_{syn} = I_{syn} + S_{rec} \\cdot w_{rec}

        V_{mem, j} = V_{mem, j} - 1

    Neurons therefore share a common resting potential of ``0``, a firing threshold of ``0``, and a subtractive reset of ``-1``. Neurons each have an optional bias current `.bias` (default: ``-1``).

    :Surrogate signals:

    To facilitate gradient-based training, a surrogate :math:`U(t)` is generated from the membrane potentials of each neuron. This is used to provide a weighted output :math:`O(t)`.

    .. math ::

        U_j = \\textrm{sig}(V_j)

        O(t) = U(t) \\cdot w_{out}

    Where :math:`w_{out}` is a :math:`[N \\times N_{out}]` matrix of output weights, and :math:`\\textrm{sig}(x) = \\left(1 + \\exp(-x)\\right)^{-1}`.

    :Outputs from evolution:

    As output, this layer returns the weighted surrogate activity of the :math:`N` neurons from the `.evolve` method. After each evolution, the attributes `.spikes_last_evolution`, `.i_rec_last_evolution` and `.v_mem_last_evolution` and `.surrogate_last_evolution` will be `.TimeSeries` objects containing the appropriate time series.
    """

    def __init__(
        self,
        w_in: np.ndarray,
        w_recurrent: np.ndarray,
        w_out: np.ndarray,
        tau_mem: FloatVector,
        tau_syn: FloatVector,
        bias: FloatVector = -1.0,
        noise_std: float = 0.0,
        dt: Optional[float] = None,
        name: Optional[str] = None,
        rng_key: Optional[list] = None,
        **kwargs,
    ):
        """
        Build a spiking recurrent layer with weighted spiking inputs and weighted surrogate outputs, and a JAX backend.

        :param np.ndarray w_in:              Input weights [M, N]
        :param np.ndarray w_recurrent:       Recurrent weights [N, N]
        :param np.ndarray w_out:             Output weights [N, O]
        :param FloatVector tau_mem:          Membrane time constants [N,]
        :param FloatVector tau_syn:          Synaptic time constants [N,]
        :param FloatVector bias:             Neuron biases [N,]
        :param float noise_std:              Std. dev. of noise injected onto neuron membranes. Default: ``0.``, no noise
        :param Optional[float] dt:           Time step for simulation, in s. Default: ``None``, will be determined automatically from ``tau_...``
        :param Optional[str] name:           Name of this layer. Default: ``None``
        :param Optional[list] rng_key:       List of two integers representing the state of the Jax PRNG. Default: generate a new key
        """
        # - Convert arguments to arrays
        w_in = np.array(w_in)
        w_out = np.array(w_out)

        # - Call superclass constructor
        super().__init__(
            w_recurrent=w_recurrent,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            bias=bias,
            noise_std=noise_std,
            dt=dt,
            name=name,
            rng_key=rng_key,
            **kwargs,
        )

        # - Set correct information about network size
        self._size_in = w_in.shape[0]
        self._size = w_in.shape[1]
        self._size_out = w_out.shape[1]

        # -- Set properties
        self.w_in = w_in
        self.w_out = w_out

    @property
    def _evolve_functional(self):
        """
        Return a functional form of the evolution function for this layer

        Returns a function ``evol_func`` with the signature::

            def evol_func(params, state, inputs) -> (outputs, new_state):

        :return Callable[[Params, State, np.ndarray], Tuple[np.ndarray, State, Dict[str, np.ndarray]]]:
        """

        def evol_func(
            params: Params,
            state: State,
            sp_input_ts: np.ndarray,
        ) -> Tuple[np.ndarray, State, Dict[str, np.ndarray]]:
            # - Call the jitted evolution function for this layer
            (
                new_state,
                Irec_ts,
                output_ts,
                surrogate_ts,
                spikes_ts,
                Vmem_ts,
                Isyn_ts,
                key1,
            ) = self._evolve_jit(
                state,
                params["w_in"],
                params["w_recurrent"],
                params["w_out"],
                params["tau_mem"],
                params["tau_syn"],
                params["bias"],
                self._noise_std,
                sp_input_ts,
                sp_input_ts * 0.0,
                self._rng_key,
                self._dt,
            )

            # - Maintain RNG key, if not under compilation
            if not isinstance(key1, jax.core.Tracer):
                self._rng_key = key1

            # - Return the outputs from this layer, and the final layer state
            states_t = {
                "Vmem": Vmem_ts,
                "Isyn": Isyn_ts,
                "Irec": Irec_ts,
                "surrogate": surrogate_ts,
                "spikes": spikes_ts,
                "output": output_ts,
            }
            return output_ts, new_state, states_t

        # - Return the evolution function
        return evol_func

    def evolve(
        self,
        ts_input: Optional[TimeSeries] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSContinuous:
        """
        Evolve the state of this layer given an input

        :param Optional[TSEvent] ts_input:      Input time series. Default: `None`, no stimulus is provided
        :param Optional[float] duration:        Simulation/Evolution time, in seconds. If not provided, then `num_timesteps` or the duration of `ts_input` is used to determine evolution time
        :param Optional[int] num_timesteps:     Number of evolution time steps, in units of `.dt`. If not provided, then `duration` or the duration of `ts_input` is used to determine evolution time
        :param bool verbose:                    Currently no effect, just for conformity

        :return TSContinuous:                   Output time series; the weighted surrogates of each neuron
        """

        super().evolve(ts_input, duration, num_timesteps)

        # - Return weighted output
        return self._output_last_evolution

    @property
    def w_in(self) -> np.ndarray:
        """ (np.ndarray) [M,N] input weights """
        return onp.array(self._w_in)

    @w_in.setter
    def w_in(self, value: np.ndarray):
        if np.ndim(value) != 2:
            raise ValueError(self.start_print + "`w_in` must be 2D")

        if value.shape != (self._size_in, self._size):
            raise ValueError(
                self.start_print
                + "`win` must be [{:d}, {:d}]".format(self._size_in, self._size)
            )

        self._w_in = np.array(value).astype("float32")

    @property
    def w_out(self) -> np.ndarray:
        """ (np.ndarray) [N,O] output weights """
        return onp.array(self._w_out)

    @w_out.setter
    def w_out(self, value: np.ndarray):
        if np.ndim(value) != 2:
            raise ValueError(self.start_print + "`w_out` must be 2D")

        if value.shape != (
            self._size,
            self._size_out,
        ):
            raise ValueError(
                self.start_print
                + "`w_out` must be [{:d}, {:d}]".format(self._size, self._size_out)
            )

        self._w_out = np.array(value).astype("float32")

    @property
    def output_type(self):
        """ (TSContinuous) Output `.TimeSeries` class: `.TSContinuous` """
        return TSContinuous

    def to_dict(self) -> dict:
        """
        Convert parameters of this layer to a dict if they are relevant for reconstructing an identical layer

        :return Dict:   A dictionary that can be used to reconstruct the layer
        """
        config = super().to_dict()
        config["w_in"] = onp.array(self.w_in).tolist()
        config["w_out"] = onp.array(self.w_out).tolist()

        return config


class RecLIFCurrentInJax_IO(RecLIFJax_IO):
    """
    Recurrent spiking neuron layer (LIF), weighted current input and weighted surrogate output. Input / output weighting provided.

    `.RecLIFJax` is a basic recurrent spiking neuron layer, implemented with a JAX-backed Euler solver backend. Outputs are surrogates generated by each layer neuron, via a set of output weights. Inputs are provided by weighted current injection to each layer neuron, via a set of input weights. The layer is therefore M inputs -> N neurons -> O outputs.

    This layer can be used to implement gradient-based learning systems, using the JAX-provided automatic differentiation functionality of `jax.grad`. See :py:meth:`.train_output_target` for information on training.

    :Dynamics:

    The dynamics of the :math:`N` neurons' membrane potential :math:`V_{mem}` and the :math:`N` synaptic currents :math:`I_{syn}` evolve under the system

    .. math::

        \\tau_{syn} \\dot{I}_{syn} + I_{syn} = 0

        I_{syn} += S_{in}(t) \\cdot w_{in}

        \\tau_{syn} \\dot{V}_{mem} + V_{mem} = I_{syn} + I_{in}(t) \\cdot w_{in} + b + \\sigma\\zeta(t)

    where :math:`S_{in}(t)` is a vector containing ``1`` for each input channel that emits a spike at time :math:`t`; :math:`w_{in}` is a :math:`[N_{in} \\times N]` matrix of input weights; :math:`I_{in}(t)` is a vector of input currents injected directly onto the neuron membranes; :math:`b` is a :math:`N` vector of bias currents for each neuron; :math:`\\sigma\\zeta(t)` is a white-noise process with standard deviation :math:`\\sigma` injected independently onto each neuron's membrane; and :math:`\\tau_{mem}` and :math:`\\tau_{syn}` are the membrane and synaptic time constants, respectively.

    :On spiking:

    When the membane potential for neuron :math:`j`, :math:`V_{mem, j}` exceeds the threshold voltage :math:`V_{thr} = 0`, then the neuron emits a spike.

    .. math ::

        V_{mem, j} > V_{thr} \\rightarrow S_{rec,j} = 1

        I_{syn} = I_{syn} + S_{rec} \\cdot w_{rec}

        V_{mem, j} = V_{mem, j} - 1

    Neurons therefore share a common resting potential of ``0``, a firing threshold of ``0``, and a subtractive reset of ``-1``. Neurons each have an optional bias current `.bias` (default: ``-1``).

    :Surrogate signals:

    To facilitate gradient-based training, a surrogate :math:`U(t)` is generated from the membrane potentials of each neuron. This is used to provide a weighted output :math:`O(t)`.

    .. math ::

        U_j = \\textrm{sig}(V_j)

        O(t) = U(t) \\cdot w_{out}

    Where :math:`w_{out}` is a :math:`[N \\times N_{out}]` matrix of output weights, and :math:`\\textrm{sig}(x) = \\left(1 + \\exp(-x)\\right)^{-1}`.

    :Outputs from evolution:

    As output, this layer returns the weighted surrogate activity of the :math:`N` neurons from the `.evolve` method. After each evolution, the attributes `.spikes_last_evolution`, `.i_rec_last_evolution` and `.v_mem_last_evolution` and `.surrogate_last_evolution` will be `.TimeSeries` objects containing the appropriate time series.
    """

    @property
    def _evolve_functional(self):
        """
        Return a functional form of the evolution function for this layer

        Returns a function ``evol_func`` with the signature::

            def evol_func(params, state, inputs) -> (outputs, new_state):

        :return Callable[[Params, State, np.ndarray], Tuple[np.ndarray, State]]:
        """

        def evol_func(
            params: Params,
            state: State,
            I_input_ts: np.ndarray,
        ):
            # - Call the jitted evolution function for this layer
            (
                new_state,
                Irec_ts,
                output_ts,
                surrogate_ts,
                spikes_ts,
                Vmem_ts,
                Isyn_ts,
                key1,
            ) = self._evolve_jit(
                state,
                params["w_in"],
                params["w_recurrent"],
                params["w_out"],
                params["tau_mem"],
                params["tau_syn"],
                params["bias"],
                self._noise_std,
                I_input_ts * 0.0,
                I_input_ts,
                self._rng_key,
                self._dt,
            )

            # - Maintain RNG key, if not under compilation
            if not isinstance(key1, jax.core.Tracer):
                self._rng_key = key1

            # - Return the outputs from this layer, and the final layer state
            states_t = {
                "Vmem": Vmem_ts,
                "Isyn": Isyn_ts,
                "Irec": Irec_ts,
                "surrogate": surrogate_ts,
                "spikes": spikes_ts,
                "output": output_ts,
            }
            return output_ts, new_state, states_t

        # - Return the evolution function
        return evol_func

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        Evolve the state of this layer given an input

        :param Optional[TSContinuous] ts_input: Input time series. Default: `None`, no stimulus is provided
        :param Optional[float] duration:        Simulation/Evolution time, in seconds. If not provided, then `num_timesteps` or the duration of `ts_input` is used to determine evolution time
        :param Optional[int] num_timesteps:     Number of evolution time steps, in units of `.dt`. If not provided, then `duration` or the duration of `ts_input` is used to determine evolution time
        :param bool verbose:          Currently no effect, just for conformity

        :return TSEvent:                   Output time series; spiking activity each neuron
        """

        # - Call evolution function
        super().evolve(ts_input, duration, num_timesteps, verbose)

        # - Return weighted output
        return self._output_last_evolution

    @property
    def input_type(self):
        """ (TSContinuous) Output `.TimeSeries` class: `.TSContinuous` """
        return TSContinuous


class FFLIFJax_IO(RecLIFJax_IO):
    """
    Feed-forward spiking neuron layer (LIF), spiking input and weighted surrogate output. Input and output weights.

    `.FFLIFJax_IO` is a basic feed-forward spiking neuron layer, implemented with a JAX-backed Euler solver backend. Outputs are surrogates generated by each layer neuron, weighted by a set of output weights. Inputs are provided by spiking through a synapse onto each layer neuron via a set of input weights. The layer is therefore M inputs -> N neurons -> O outputs.

    This layer can be used to implement gradient-based learning systems, using the JAX-provided automatic differentiation functionality of `jax.grad`. See :py:meth:`.train_output_target` for information on training.

    :Dynamics:

    The dynamics of the :math:`N` neurons' membrane potential :math:`V_{mem}` and the :math:`N` synaptic currents :math:`I_{syn}` evolve under the dynamics

    .. math::

        \\tau_{syn} \\dot{I}_{syn} + I_{syn} = 0

        I_{syn} += S_{in}(t) \\cdot w_{in}

        \\tau_{syn} \\dot{V}_{mem} + V_{mem} = I_{syn} + I_{in}(t) \\cdot w_{in} + b + \\sigma\\zeta(t)

    where :math:`S_{in}(t)` is a vector containing ``1`` for each input channel that emits a spike at time :math:`t`; :math:`w_{in}` is a :math:`[N_{in} \\times N]` matrix of input weights; :math:`I_{in}(t)` is a vector of input currents injected directly onto the neuron membranes; :math:`b` is a :math:`N` vector of bias currents for each neuron; :math:`\\sigma\\zeta(t)` is a white-noise process with standard deviation :math:`\\sigma` injected independently onto each neuron's membrane; and :math:`\\tau_{mem}` and :math:`\\tau_{syn}` are the membrane and synaptic time constants, respectively.

    :On spiking:

    When the membrane potential for neuron :math:`j`, :math:`V_{mem, j}` exceeds the threshold voltage :math:`V_{thr} = 0`, then the neuron emits a spike.

    .. math ::

        V_{mem, j} > V_{thr} \\rightarrow S_j(t) = 1

        V_{mem, j} = V_{mem, j} - 1

    Neurons therefore share a common resting potential of ``0``, a firing threshold of ``0``, and a subtractive reset of ``-1``. Neurons each have an optional bias current `.bias` (default: ``-1``).

    :Surrogate signals:

    To facilitate gradient-based training, a surrogate :math:`U(t)` is generated from the membrane potentials of each neuron. This is used to provide a weighted output :math:`O(t)`.

    .. math ::

        U_j = \\textrm{sig}(V_j)

        O(t) = U(t) \\cdot w_{out}

    Where :math:`w_{out}` is a :math:`[N \\times N_{out}]` matrix of output weights, and :math:`\\textrm{sig}(x) = \\left(1 + \\exp(-x)\\right)^{-1}`.

    :Outputs from evolution:

    As output, this layer returns the weighted surrogate activity of the :math:`N` neurons from the `.evolve` method. After each evolution, the attributes `.spikes_last_evolution`, `.i_rec_last_evolution` and `.v_mem_last_evolution` and `.surrogate_last_evolution` will be `.TimeSeries` objects containing the appropriate time series.
    """

    def __init__(
        self,
        w_in: FloatVector,
        w_out: FloatVector,
        tau_mem: FloatVector,
        tau_syn: FloatVector,
        bias: FloatVector = -1.0,
        noise_std: float = 0.0,
        dt: Optional[float] = None,
        name: Optional[str] = None,
        rng_key: Optional[list] = None,
        **kwargs,
    ):
        """
        Create a feedforward spiking LIF layer, with a JAX-accelerated backend.

        :param FloatVector w_in:             Input weight matrix for this layer [M, N]
        :param FloatVector w_out:            Output weight matrix for this layer [N, O]
        :param FloatVector tau_mem:          Membrane time constants for each neuron in this layer. Can be provided as a scalar, which is then used for all neurons
        :param FloatVector tau_syn:          Synaptic time constants for each neuron in this layer. Can be provided as a scalar, which is then used for all neurons
        :param FloatVector bias:             Bias currents for each neuron in this layer. Can be provided as a scalar, which is then used for all neurons
        :param float noise_std:              Standard deviation of a noise current which is injected onto the membrane of each neuron
        :param float dt:                     Euler solver time-step. Must be at least 10 times smaller than the smallest time constant, for numerical stability
        :param Optional[str] name:           A string to use as the name of this layer
        :param Optional[list] rng_key:       List of two integers representing the state of the Jax PRNG. Default: generate a new key
        """
        # - Determine network shape
        w_in = onp.atleast_2d(w_in)
        w_out = onp.atleast_2d(w_out)
        net_size = w_in.shape[1]

        # - Initialise layer object
        super().__init__(
            w_in=w_in,
            w_recurrent=np.zeros((net_size, net_size)),
            w_out=w_out,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            bias=bias,
            noise_std=noise_std,
            dt=dt,
            name=name,
            rng_key=rng_key,
            **kwargs,
        )

        # - Set recurrent weights to zero
        self._weights = 0.0

    @property
    def i_rec_last_evolution(self):
        """Not defined for `.FFLIFJax_IO`"""
        raise ValueError("Recurrent currents do not exist for a feedforward layer")

    def to_dict(self) -> dict:
        """
        Convert parameters of this layer to a dict if they are relevant for reconstructing an identical layer

        :return Dict:   A dictionary that can be used to reconstruct the layer
        """
        config = super().to_dict()
        config.pop("w_recurrent")

        return config


class FFLIFJax_SO(FFLIFJax_IO):
    def __init__(
        self,
        w_in,
        tau_mem,
        tau_syn,
        bias=-1.0,
        noise_std=0.0,
        dt=None,
        rng_key=None,
        name=None,
        **kwargs,
    ):
        super().__init__(
            w_in=w_in,
            w_out=onp.array(w_in).T,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            bias=bias,
            noise_std=noise_std,
            dt=dt,
            rng_key=rng_key,
            name=name,
            **kwargs,
        )

        self._size_out = self._w_in.shape[1]

    def evolve(
        self,
        ts_input: Optional[TimeSeries] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSContinuous:
        super().evolve(ts_input, duration, num_timesteps, verbose)

        return self._surrogate_last_evolution


class FFLIFCurrentInJax_SO(FFLIFJax_IO):
    """
    Feed-forward spiking neuron layer (LIF), current input and surrogate output. Input weighting, no output weights.

    `.FFLIFCurrentInJax_SO` is a basic feed-forward spiking neuron layer, implemented with a JAX-backed Euler solver backend. Outputs are surrogates generated by each layer neuron. Inputs are provided by spiking through a synapse onto each layer neuron via a set of input weights. The layer is therefore M inputs -> N neurons -> N outputs.

    This layer can be used to implement gradient-based learning systems, using the JAX-provided automatic differentiation functionality of `jax.grad`. See :py:meth:`~FFLIFCurrentInJax_SO.train_output_target` for information on training.

    :Dynamics:

    The dynamics of the :math:`N` neurons' membrane potential :math:`V_{mem}` and the :math:`N` synaptic currents :math:`I_{syn}` evolve under the dynamics

    .. math::

        \\tau_{syn} \\dot{I}_{syn} + I_{syn} = 0

        \\tau_{syn} \\dot{V}_{mem} + V_{mem} = I_{syn} + I_{in}(t) \\cdot w_{in} + b + \\sigma\\zeta(t)

    where :math:`S_{in}(t)` is a vector containing ``1`` for each input channel that emits a spike at time :math:`t`; :math:`w_{in}` is a :math:`[N_{in} \\times N]` matrix of input weights; :math:`I_{in}(t)` is a vector of input currents injected directly onto the neuron membranes; :math:`b` is a :math:`N` vector of bias currents for each neuron; :math:`\\sigma\\zeta(t)` is a white-noise process with standard deviation :math:`\\sigma` injected independently onto each neuron's membrane; and :math:`\\tau_{mem}` and :math:`\\tau_{syn}` are the membrane and synaptic time constants, respectively.

    :On spiking:

    When the membrane potential for neuron :math:`j`, :math:`V_{mem, j}` exceeds the threshold voltage :math:`V_{thr} = 0`, then the neuron emits a spike.

    .. math ::

        V_{mem, j} > V_{thr} \\rightarrow S_j(t) = 1

        V_{mem, j} = V_{mem, j} - 1

    Neurons therefore share a common resting potential of ``0``, a firing threshold of ``0``, and a subtractive reset of ``-1``. Neurons each have an optional bias current `.bias` (default: ``-1``).

    :Surrogate signals:

    To facilitate gradient-based training, a surrogate :math:`U(t)` is generated from the membrane potentials of each neuron.

    .. math ::

        U_j = \\textrm{sig}(V_j)

    Where :math:`\\textrm{sig}(x) = \\left(1 + \\exp(-x)\\right)^{-1}`.

    :Outputs from evolution:

    As output, this layer returns the weighted surrogate activity of the :math:`N` neurons from the `.evolve` method. After each evolution, the attributes `.spikes_last_evolution`, `.i_rec_last_evolution` and `.v_mem_last_evolution` and `.surrogate_last_evolution` will be `.TimeSeries` objects containing the appropriate time series.
    """

    def __init__(
        self,
        w_in: FloatVector,
        tau_mem: FloatVector,
        tau_syn: FloatVector,
        bias: FloatVector = -1.0,
        noise_std: float = 0.0,
        dt: Optional[float] = None,
        name: Optional[str] = None,
        rng_key: Optional[list] = None,
        **kwargs,
    ):
        """
        Create a feedforward spiking LIF layer, with a JAX-accelerated backend.

        :param FloatVector w_in:             Input weight matrix for this layer [M, N]
        :param FloatVector tau_mem:          Membrane time constants for each neuron in this layer. Can be provided as a scalar, which is then used for all neurons
        :param FloatVector tau_syn:          Synaptic time constants for each neuron in this layer. Can be provided as a scalar, which is then used for all neurons
        :param FloatVector bias:             Bias currents for each neuron in this layer. Can be provided as a scalar, which is then used for all neurons
        :param float noise_std:              Standard deviation of a noise current which is injected onto the membrane of each neuron
        :param float dt:                     Euler solver time-step. Must be at least 10 times smaller than the smallest time constant, for numerical stability
        :param Optional[str] name:           A string to use as the name of this layer
        :param Optional[list] rng_key:       List of two integers representing the state of the Jax PRNG. Default: generate a new key
        """
        # - Determine network shape
        w_in = onp.atleast_2d(w_in)
        net_size = w_in.shape[1]

        # - Initialise layer
        super().__init__(
            w_in=w_in,
            w_out=np.zeros((net_size, net_size)),
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            bias=bias,
            noise_std=noise_std,
            dt=dt,
            name=name,
            rng_key=rng_key,
            **kwargs,
        )

        # - Set recurrent weights to zero
        self._weights = 0.0

        # - Set output weights to unity
        self._w_out = 1.0

    def evolve(
        self,
        ts_input: Optional[TimeSeries] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSContinuous:
        """
        Evolve the state of this layer given an input

        :param Optional[TSEvent] ts_input:      Input time series. Default: `None`, no stimulus is provided
        :param Optional[float] duration:        Simulation/Evolution time, in seconds. If not provided, then `num_timesteps` or the duration of `ts_input` is used to determine evolution time
        :param Optional[int] num_timesteps:     Number of evolution time steps, in units of `.dt`. If not provided, then `duration` or the duration of `ts_input` is used to determine evolution time
        :param bool verbose:                    Currently no effect, just for conformity

        :return TSContinuous:                   Output time series; the weighted surrogates of each neuron
        """

        super().evolve(ts_input, duration, num_timesteps)

        # - Return weighted output
        return self._surrogate_last_evolution

    @property
    def i_rec_last_evolution(self):
        """Not defined for `.FFLIFCurrentInJax_SO`"""
        raise ValueError("Recurrent currents do not exist for a feedforward layer")

    def to_dict(self) -> dict:
        """
        Convert parameters of this layer to a dict if they are relevant for reconstructing an identical layer

        :return Dict:   A dictionary that can be used to reconstruct the layer
        """
        config = super().to_dict()
        config.pop("w_out")

        return config

    @property
    def input_type(self):
        """ (TSContinuous) Output `.TimeSeries` class: `.TSContinuous` """
        return TSContinuous


# - Define a State type for the exponential synapses
StateExpSyn = np.ndarray
ParamsExpSyn = Dict[str, np.ndarray]


def _evolve_expsyn_jax(
    state0: StateExpSyn,
    tau: np.ndarray,
    noise_std: float,
    sp_input_ts: np.ndarray,
    I_input_ts: np.ndarray,
    key: rand.PRNGKey,
    dt: float,
) -> Tuple[StateExpSyn, np.ndarray, rand.PRNGKey]:
    """
    Jax-backed evolution function for exponential synapses

    This function implements the simple dynamics

    .. math::
        \\tau \\dot\\I_{syn} = -I_{syn} + W_{in} \\cdot s(t) + W_{in} \\cdot I_{in}(t) + \\zeta \\sigma(t)

    where :math:`\\tau` is the time constant for each node; :math:`I_{syn}(t)` is the synaptic current at time :math:`t`; :math:`s(t)` is the input spike train; :math:`I_{in}(t)` is the input current; :math:`W_{in}` is the input weight matrix with shape ``[MxN]`` for ``M`` input channels and ``N`` nodes; and :math:`\\zeta \\sigma(t)` is a white noise process with std. dev :math:`\\sigma`.

    :param StateExpSyn state0:      Initial state for the layer
    :param np.ndarray w_out:        Output weights for the layer :math:`W_{out}` with shape ``[N, O]``
    :param np.ndarray tau:          Time constants for the layer nodes :math:`\\tau` with shape ``[N,]``
    :param float noise_std:         Std. dev. of noise to inject into node currents
    :param np.ndarray sp_input_ts:  Rasterised time series ``[T, M]`` of input spikes on each input channel
    :param np.ndarray I_input_ts:   Rasterised time series ``[T, M]`` of input currents on each input channel
    :param rand.PRNGKey key:        Jax RNG key to use when generating randomness
    :param float dt:                Time step

    :return (new_state, Isyn_ts, new_key): ``new_state`` is the layer state after evolution. ``Isyn_ts`` is the rasterised time series ``[T, N]`` of synaptic currents associated with each node. ``new_key`` is the new Jax RNG key after splitting to generate randomness.
    """
    # - Get evolution constants
    beta = np.exp(-dt / tau)

    # - Single-step dynamics
    def forward(
        state: State, inputs_t: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[State, np.ndarray,]:
        # - Unpack inputs
        (sp_in_t, I_in_t) = inputs_t
        sp_in_t = sp_in_t.reshape(-1)
        I_in_t = I_in_t.reshape(-1)

        # - Synaptic input
        dIsyn = sp_in_t + I_in_t
        state["Isyn"] = beta * state["Isyn"] + dIsyn

        # - Return state and outputs
        return state, state["Isyn"]

    # - Generate synapse current noise trace
    num_timesteps = sp_input_ts.shape[0]
    key1, subkey = rand.split(key)
    noise_ts = noise_std * rand.normal(
        subkey, shape=(num_timesteps, np.size(state0["Isyn"]))
    )

    # - Evolve over spiking inputs
    state, Isyn_ts = scan(
        forward,
        state0,
        (
            sp_input_ts,
            I_input_ts + noise_ts,
        ),
    )

    # - Return outputs
    return state, Isyn_ts, key1


def loss_mse_reg_expsyn(
    params: ParamsExpSyn,
    states_t: StateExpSyn,
    output_batch_t: np.ndarray,
    target_batch_t: np.ndarray,
    min_tau: float,
    lambda_mse: float = 1.0,
    reg_tau: float = 10000.0,
    reg_l2_weights: float = 1.0,
) -> float:
    """
    Regularised loss function for Jax Exponential Synapse layers

    This loss function computes the mean-squared error of the target signal versus the layer synaptic currents. This loss is regularised by several terms to limit time constants and to control the weight spectra.

    .. math::
        L = \lambda_{mse}\\cdot L_{mse} + \\lambda_{\\tau}\\cdot L_{\\tau} + \\lambda_{L2}\\cdot L_{L2}

        L_{mse} = E|\\left(\\textbf{o}(t) - \\textbf{y}(t)\\right)^2| \\textrm{ : output versus target MSE}

        L_{\\tau} = l_{\\tau}(\\tau_{syn}, \\tau_{min})

        l_{\\tau}(\\tau, \\tau_{min}) = \\sum \\exp(-(\\tau - \\tau_{min})) | \\tau < \\tau_{min}

        L_{L2} = l_{l2}(W_{in})

        l_{l2}(W) = E|W^2|

        \textrm{where } E|\textbf{x}| = \\frac{1}{#\textbf{x}}\\sum{x}

    :param ParamsExpSyn params:         Parameters of the ExpSyn layer
    :param StateExpSyn states_t:        State time-series of the ExpSyn layer
    :param np.ndarray output_batch_t:   Output time series of the layer
    :param np.ndarray target_batch_t:   Target time series of the layer
    :param float min_tau:               Minimum permitted synaptic time constant
    :param float lambda_mse:            Loss factor for MSE error. Default: 1.0
    :param float reg_tau:               Regularisation loss factor for time constant violations. Default: 10000.0
    :param float reg_l2_weights:        Regularisation loss factor for input weight L2 norm. Default: 1.0

    :return:
    """
    # - MSE between output and target
    dLoss = dict()
    dLoss["loss_mse"] = lambda_mse * np.mean((output_batch_t - target_batch_t) ** 2)

    # - Get loss for tau parameter constraints
    dLoss["loss_tau"] = reg_tau * np.nansum(
        np.where(
            params["tau"] < min_tau,
            np.exp(-(params["tau"] - min_tau)),
            0,
        )
    )

    # - Regularisation for weights
    dLoss["loss_weights_l2"] = reg_l2_weights * np.mean(params["weights"] ** 2)

    # - Return loss
    return sum(dLoss.values())


class FFExpSynCurrentInJax(Layer, JaxTrainer):
    """
    Feed-forward layer of exponential current synapses, receiving current inputs. Input weighting provided

    This layer implements an array of exponential synaptic filters, driven by current inputs, passing through a weight matrix :math:`W_{in}`.

    The dynamics of each node are given by

    .. math::

        \\tau \\dot{I}_{syn} = -I_{syn} + W_{in} \\cdot I_{in}(t) + \\zeta \\sigma(t)

    where :math:`\\tau` is the time constant for each node; :math:`I_{syn}(t)` is the synaptic current at time :math:`t`; :math:`I_{in}(t)` is the input current; :math:`W_{in}` is the input weight matrix with shape ``[MxN]`` for ``M`` input channels and ``N`` nodes; and :math:`\\zeta \\sigma(t)` is a white noise process with std. dev :math:`\\sigma`.

    This layer supports the :py:class:`JaxTrainer` interface, permitting gradient-descent training using the method :py:meth:`~.FFExpSynCurrentInJax.train_output_target`.
    """

    def __init__(
        self,
        w_out: np.ndarray,
        tau: np.ndarray,
        dt: float,
        noise_std: float = 0.0,
        name: str = None,
        rng_key: rand.PRNGKey = None,
        **kwargs,
    ) -> None:
        """
        Initialise a Jax-backed exponential synapse layer

        :param np.ndarray w_out:                The input weights ``[N, O]`` of this layer
        :param np.ndarray tau:                  The time constants ``[N,]`` of the ``N`` nodes in this layer
        :param float dt:                        Simulation time-step to use for this layer
        :param float noise_std:                 Std. dev. of the noise to inject into each node during evolution. Default: ``0.0``, no noise.
        :param Optional[str] name:              Optional name to use when describing this layer. Default: ``None``
        :param Optional[rand.PRNGKey] rng_key:  pRNG key to use when generating randomness. Default: ``None``, generate a new key
        """
        # - Ensure that weights are 2D
        w_out = np.atleast_2d(w_out)

        # - Transform arguments to JAX np.array
        tau = np.array(tau)

        if dt is None:
            dt = np.min(np.array((np.min(tau)))) / 10.0

        # - Call super-class initialisation
        super().__init__(
            weights=np.identity(w_out.shape[0]), dt=dt, noise_std=noise_std, name=name
        )

        # - Set properties
        self._size_out = w_out.shape[1]
        self._size_in = w_out.shape[0]
        self._w_out = w_out
        self.tau = tau
        self._weights = 0.0

        # - Create RNG
        if rng_key is None:
            rng_key = rand.PRNGKey(onp.random.randint(0, 2 ** 63))
            _, self._rng_key = rand.split(rng_key)
        else:
            rng_key = np.array(onp.array(rng_key).astype(onp.uint32))
            self._rng_key = rng_key

        # - Get evolution function
        self._evolve_jit = jit(_evolve_expsyn_jax)

        # - Initialise state
        self.reset_all()

    # - Replace the default loss function
    @property
    def _default_loss(self) -> Callable[[Any], float]:
        return loss_mse_reg_expsyn

    @property
    def _default_loss_params(self) -> Dict:
        return {
            "min_tau": self._dt * 11.0,
        }

    def reset_state(self):
        """
        Reset the membrane potentials, synaptic currents and refractory state for this layer
        """
        self._state = {
            "Isyn": np.zeros((self._size,)),
        }

    def randomize_state(self):
        self._rng_key, subkey = rand.split(self._rng_key)
        self._state = {"Isyn": rand.normal(subkey, shape=(self.size,))}

    def _pack(self) -> Params:
        """
        Return a packed form of the tunable parameters for this layer

        :return Params: params: All parameters as a Dict
        """
        return {
            "w_out": self._w_out,
            "tau_syn": self._tau,
            "tau_mem": np.inf,
        }

    def _unpack(self, params: Params):
        """
        Set the parameters for this layer, given a parameter dictionary

        :param Params params:  Set of parameters for this layer
        """
        (self._w_out, self._tau,) = (
            params["w_out"],
            params["tau_syn"],
        )

    @property
    def _evolve_functional(self):
        """
        Return a functional form of the evolution function for this layer

        Returns a function ``evol_func`` with the signature::

            def evol_func(params, state, inputs) -> (outputs, new_state):

        :return Callable[[Params, State, np.ndarray], Tuple[np.ndarray, State, Dict[str, np.ndarray]]]:
        """

        def evol_func(
            params: ParamsExpSyn,
            state: StateExpSyn,
            I_input_ts: np.ndarray,
        ) -> Tuple[np.ndarray, StateExpSyn, Dict[str, np.ndarray]]:
            # - Call the jitted evolution function for this layer
            (new_state, Isyn_ts, key1,) = self._evolve_jit(
                state,
                params["tau_syn"],
                self._noise_std,
                I_input_ts * 0.0,
                I_input_ts,
                self._rng_key,
                self._dt,
            )

            # - Maintain RNG key, if not under compilation
            if not isinstance(key1, jax.core.Tracer):
                self._rng_key = key1

            # - Return the outputs from this layer, and the final layer state
            states_t = {"Isyn": Isyn_ts}

            return np.dot(Isyn_ts, params["w_out"]), new_state, states_t

        # - Return the evolution function
        return evol_func

    def evolve(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSContinuous:
        """
        Evolve the state of this layer given an input

        :param Optional[TSContinuous] ts_input:      Input time series. Default: `None`, no stimulus is provided
        :param Optional[float] duration:        Simulation/Evolution time, in seconds. If not provided, then `num_timesteps` or the duration of `ts_input` is used to determine evolution time
        :param Optional[int] num_timesteps:     Number of evolution time steps, in units of `.dt`. If not provided, then `duration` or the duration of `ts_input` is used to determine evolution time
        :param bool verbose:           Currently no effect, just for conformity

        :return TSContinuous:                   Output time series; the synaptic currents of each neuron
        """

        # - Prepare time base and inputs
        time_base, inps, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Call raw evolution function
        (Isyn_ts, self._state, state_ts,) = self._evolve_functional(
            self._pack(),
            self._state,
            inps,
        )

        # - Record synaptic currents
        self._i_syn_last_evolution = TSContinuous.from_clocked(
            onp.array(Isyn_ts),
            t_start=self.t,
            dt=self.dt,
            name="$I_{syn}$ " + self.name,
        )

        # - Increment layer time
        self._timestep += num_timesteps

        # - Augment time base
        time_base = onp.append(time_base, self.t)

        # - Advance time
        self._timestep += num_timesteps

        # - Return output currents
        return self._i_syn_last_evolution

    def to_dict(self) -> Dict:
        config = super().to_dict()
        config.pop("weights")
        config["w_out"] = onp.array(self._w_out).tolist()
        config["rng_key"] = onp.array(self._rng_key).tolist()
        config["tau"] = onp.array(self.tau).tolist()
        return config

    @classmethod
    def load_from_dict(cls: Any, config: Dict, **kwargs) -> "FFExpSynCurrentIn":
        config.pop("class_name")
        config = dict(config, **kwargs)
        return cls(**config)

    @property
    def tau(self):
        return onp.array(self._tau)

    @tau.setter
    def tau(self, value):
        self._tau = np.array(self._expand_to_net_size(value, "tau", False))

    @property
    def noise_std(self):
        return onp.array(self._noise_std).item()

    @noise_std.setter
    def noise_std(self, value):
        self._noise_std = np.array(value).item()


class FFExpSynJax(FFExpSynCurrentInJax):
    """
    Feed-forward layer of exponential current synapses, receiving spiking inputs. Input weighting provided

    This layer implements an array of exponential synaptic filters, driven by spiking inputs, passing through a weight matrix :math:`W_{in}`.

    The dynamics of each node are given by

    .. math::

        \\tau \\dot{I}_{syn} = -I_{syn} + W_{in} \\cdot s(t) + \\zeta \\sigma(t)

    where :math:`\\tau` is the time constant for each node; :math:`I_{syn}(t)` is the synaptic current at time :math:`t`; :math:`s(t)` is the input spike train; :math:`W_{in}` is the input weight matrix with shape ``[MxN]`` for ``M`` input channels and ``N`` nodes; and :math:`\\zeta \\sigma(t)` is a white noise process with std. dev :math:`\\sigma`.

    This layer supports the :py:class:`JaxTrainer` interface, permitting gradient-descent training using the method :py:meth:`~.FFExpSynCurrentInJax.train_output_target`.
    """

    @property
    def input_type(self):
        return TSEvent

    @property
    def _evolve_functional(self):
        """
        Return a functional form of the evolution function for this layer

        Returns a function ``evol_func`` with the signature::

            def evol_func(params, state, inputs) -> (outputs, new_state):

        :return Callable[[Params, State, np.ndarray], Tuple[np.ndarray, State, Dict[str, np.ndarray]]]:
        """

        def evol_func(
            params: Params,
            state: State,
            sp_input_ts: np.ndarray,
        ) -> Tuple[np.ndarray, State, Dict[str, np.ndarray]]:
            # - Call the jitted evolution function for this layer
            (new_state, Isyn_ts, key1,) = self._evolve_jit(
                state,
                params["tau_syn"],
                self._noise_std,
                sp_input_ts,
                sp_input_ts * 0.0,
                self._rng_key,
                self._dt,
            )

            # - Maintain RNG key, if not under compilation
            if not isinstance(key1, jax.core.Tracer):
                self._rng_key = key1

            # - Return the outputs from this layer, and the final layer state
            states_t = {
                "Isyn": Isyn_ts,
            }
            return np.dot(Isyn_ts, params["w_out"]), new_state, states_t

        # - Return the evolution function
        return evol_func
