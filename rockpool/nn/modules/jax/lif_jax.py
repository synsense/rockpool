"""
Implements a leaky integrate-and-fire neuron module with a Jax backend
"""
import jax

from rockpool.nn.modules.jax.jax_module import JaxModule
from rockpool.nn.modules.native.linear import kaiming
from rockpool.parameters import Parameter, State, SimulationParameter
from rockpool import TSContinuous, TSEvent
from rockpool.graph import (
    GraphModuleBase,
    as_GraphHolder,
    LIFNeuronWithSynsRealValue,
    LinearWeights,
)

import numpy as onp

from jax import numpy as np
from jax.tree_util import Partial
from jax.lax import scan
import jax.random as rand

from typing import Optional, Tuple, Union, Callable
from rockpool.typehints import FloatVector, P_ndarray, JaxRNGKey, P_float, P_int

__all__ = ["LIFJax"]


# - Surrogate functions to use in learning
def sigmoid(x: FloatVector, threshold: FloatVector) -> FloatVector:
    """
    Sigmoid function

    :param FloatVector x: Input value

    :return FloatVector: Output value
    """
    return np.tanh(x + 1 - threshold) / 2 + 0.5


@jax.custom_jvp
def step_pwl(
    x: FloatVector,
    threshold: FloatVector,
    window: FloatVector = 0.5,
    max_spikes_per_dt: int = np.inf,
) -> FloatVector:
    """
    Heaviside step function with piece-wise linear derivative to use as spike-generation surrogate

    Args:
        x (float):          Input value
        threshold (float):  Firing threshold
        window (float): Learning window around threshold. Default: 0.5
        max_spikes_per_dt (int): Maximum number of spikes that may be produced each dt. Default: ``np.inf``, do not clamp spikes

    Returns:
        float: Number of output events for each input value
    """
    spikes = (x >= threshold) * np.floor(x / threshold)
    return np.clip(spikes, 0.0, max_spikes_per_dt)


@step_pwl.defjvp
def step_pwl_jvp(primals, tangents):
    x, threshold, window, max_spikes_per_dt = primals
    x_dot, threshold_dot, window_dot, max_spikes_per_dt_dot = tangents
    primal_out = step_pwl(*primals)
    tangent_out = (x >= (threshold - window)) * (
        x_dot / threshold - threshold_dot * x / (threshold**2)
    )
    return primal_out, tangent_out


class LIFJax(JaxModule):
    """
    A leaky integrate-and-fire spiking neuron model, with a Jax backend

    This module implements the update equations:

    .. math ::

        I_{syn} += S_{in}(t) + S_{rec} \\cdot W_{rec}
        I_{syn} *= \exp(-dt / \tau_{syn})
        V_{mem} *= \exp(-dt / \tau_{mem})
        V_{mem} += I_{syn} + b + \sigma \zeta(t)

    where :math:`S_{in}(t)` is a vector containing ``1`` (or a weighed spike) for each input channel that emits a spike at time :math:`t`; :math:`b` is a :math:`N` vector of bias currents for each neuron; :math:`\\sigma\\zeta(t)` is a Wiener noise process with standard deviation :math:`\\sigma` after 1s; and :math:`\\tau_{mem}` and :math:`\\tau_{syn}` are the membrane and synaptic time constants, respectively. :math:`S_{rec}(t)` is a vector containing ``1`` for each neuron that emitted a spike in the last time-step. :math:`W_{rec}` is a recurrent weight matrix, if recurrent weights are used. :math:`b` is an optional bias current per neuron (default 0.).

    :On spiking:

    When the membrane potential for neuron :math:`j`, :math:`V_{mem, j}` exceeds the threshold voltage :math:`V_{thr}`, then the neuron emits a spike. The spiking neuron subtracts its own threshold on reset.

    .. math ::

        V_{mem, j} > V_{thr} \\rightarrow S_{rec,j} = 1

        V_{mem, j} = V_{mem, j} - V_{thr}

    Neurons therefore share a common resting potential of ``0``, have individual firing thresholds, and perform subtractive reset of ``-V_{thr}``.
    """

    def __init__(
        self,
        shape: Union[Tuple, int],
        tau_mem: Optional[FloatVector] = None,
        tau_syn: Optional[FloatVector] = None,
        bias: Optional[FloatVector] = None,
        w_rec: Optional[FloatVector] = None,
        has_rec: bool = False,
        weight_init_func: Optional[Callable[[Tuple], np.ndarray]] = kaiming,
        threshold: Optional[FloatVector] = None,
        noise_std: float = 0.0,
        max_spikes_per_dt: P_int = np.inf,
        dt: float = 1e-3,
        rng_key: Optional[JaxRNGKey] = None,
        spiking_input: bool = False,
        spiking_output: bool = True,
        *args,
        **kwargs,
    ):
        """
        Instantiate an LIF module

        Args:
            shape (tuple): Either a single dimension ``(Nout,)``, which defines a feed-forward layer of LIF modules with equal amounts of synapses and neurons, or two dimensions ``(Nin, Nout)``, which defines a layer of ``Nin`` synapses and ``Nout`` LIF neurons.
            tau_mem (Optional[np.ndarray]): An optional array with concrete initialisation data for the membrane time constants. If not provided, 20ms will be used by default.
            tau_syn (Optional[np.ndarray]): An optional array with concrete initialisation data for the synaptic time constants. If not provided, 20ms will be used by default.
            bias (Optional[np.ndarray]): An optional array with concrete initialisation data for the neuron bias currents. If not provided, 0.0 will be used by default.
            w_rec (Optional[np.ndarray]): If the module is initialised in recurrent mode, you can provide a concrete initialisation for the recurrent weights, which must be a square matrix with shape ``(Nout, Nin)``.
            has_rec (bool): If ``True``, module provides a recurrent weight matrix. Default: ``False``, no recurrent connectivity.
            weight_init_func (Optional[Callable[[Tuple], np.ndarray]): The initialisation function to use when generating weights. Default: ``None`` (Kaiming initialisation)
            threshold (FloatVector): An optional array specifying the firing threshold of each neuron. If not provided, ``1.`` will be used by default.
            noise_std (float): The std. dev. after 1s of the noise added to membrane state variables. Default: ``0.0`` (no noise).
            max_spikes_per_dt (int): The maximum number of events that will be produced in a single time-step. Default: ``np.inf``; do not clamp spiking.
            dt (float): The time step for the forward-Euler ODE solver. Default: 1ms
            rng_key (Optional[Any]): The Jax RNG seed to use on initialisation. By default, a new seed is generated.
        """
        # - Check shape argument
        if np.size(shape) == 1:
            shape = (np.array(shape).item(), np.array(shape).item())

        if np.size(shape) > 2:
            raise ValueError(
                "`shape` must be a one- or two-element tuple `(Nin, Nout)`."
            )

        # - Call the superclass initialiser
        super().__init__(
            shape=shape,
            spiking_input=spiking_input,
            spiking_output=spiking_output,
            *args,
            **kwargs,
        )

        # - Seed RNG
        if rng_key is None:
            rng_key = rand.PRNGKey(onp.random.randint(0, 2**63))
        _, rng_key = rand.split(np.array(rng_key, dtype=np.uint32))

        # - Initialise state
        self.rng_key: Union[np.ndarray, State] = State(
            rng_key, init_func=lambda _: rng_key
        )

        self.n_synapses = shape[0] // shape[1]
        """ (int) Number of input synapses per neuron """

        if self.n_synapses * shape[1] != self.size_in:
            raise ValueError(
                "You must specify an integer number of synapses per neuron."
            )

        # - Should we be recurrent or FFwd?
        if isinstance(has_rec, jax.core.Tracer) or has_rec:
            self.w_rec: P_ndarray = Parameter(
                w_rec,
                shape=(self.size_out, self.size_in),
                init_func=weight_init_func,
                family="weights",
                cast_fn=np.array,
            )
            """ (Tensor) Recurrent weights `(Nout, Nin)` """
        else:
            self.w_rec = np.zeros((self.size_out, self.size_in))

        # - Set parameters
        self.tau_mem: P_ndarray = Parameter(
            tau_mem,
            shape=[(self.size_out,), ()],
            init_func=lambda s: np.ones(s) * 20e-3,
            family="taus",
            cast_fn=np.array,
        )
        """ (np.ndarray) Membrane time constants `(Nout,)` or `()` """

        self.tau_syn: P_ndarray = Parameter(
            tau_syn,
            "taus",
            init_func=lambda s: np.ones(s) * 20e-3,
            shape=[
                (
                    self.size_out,
                    self.n_synapses,
                ),
                (
                    1,
                    self.n_synapses,
                ),
                (),
            ],
            cast_fn=np.array,
        )
        """ (np.ndarray) Synaptic time constants `(Nout,)` or `()` """

        self.bias: P_ndarray = Parameter(
            bias,
            "bias",
            init_func=lambda s: np.zeros(s),
            shape=[(self.size_out,), ()],
            cast_fn=np.array,
        )
        """ (np.ndarray) Neuron bias currents `(Nout,)` or `()` """

        self.threshold: P_ndarray = Parameter(
            threshold,
            "threshold",
            shape=[(self.size_out,), ()],
            init_func=np.ones,
            cast_fn=np.array,
        )
        """ (np.ndarray) Firing threshold for each neuron `(Nout,)` or `()`"""

        self.dt: P_float = SimulationParameter(dt)
        """ (float) Simulation time-step in seconds """

        self.noise_std: P_float = SimulationParameter(noise_std)
        """ (float) Noise injected on each neuron membrane per time-step """

        # - Specify state
        self.spikes: P_ndarray = State(shape=(self.size_out,), init_func=np.zeros)
        """ (np.ndarray) Spiking state of each neuron `(Nout,)` """

        self.isyn: P_ndarray = State(
            shape=(self.size_out, self.n_synapses), init_func=np.zeros
        )
        """ (np.ndarray) Synaptic current of each neuron `(Nout, Nsyn)` """

        self.vmem: P_ndarray = State(shape=(self.size_out,), init_func=np.zeros)
        """ (np.ndarray) Membrane voltage of each neuron `(Nout,)` """

        self.max_spikes_per_dt: P_int = SimulationParameter(max_spikes_per_dt)
        """ (int) Maximum number of events that can be produced in each time-step """

        # - Define additional arguments required during initialisation
        self._init_args = {
            "has_rec": has_rec,
            "weight_init_func": Partial(weight_init_func),
        }

    def evolve(
        self,
        input_data: np.ndarray,
        record: bool = False,
    ) -> Tuple[np.ndarray, dict, dict]:
        """

        Args:
            input_data (np.ndarray): Input array of shape ``(T, Nin)`` to evolve over
            record (bool): If ``True``,

        Returns:
            (np.ndarray, dict, dict): output, new_state, record_state
            ``output`` is an array with shape ``(T, Nout)`` containing the output data produced by this module. ``new_state`` is a dictionary containing the updated module state following evolution. ``record_state`` will be a dictionary containing the recorded state variables for this evolution, if the ``record`` argument is ``True``.
        """
        # - Get input shapes, add batch dimension if necessary
        input_data, (vmem, spikes, isyn) = self._auto_batch(
            input_data,
            (self.vmem, self.spikes, self.isyn),
            (
                (self.size_out,),
                (self.size_out,),
                (self.size_out, self.n_synapses),
            ),
        )
        n_batches, n_timesteps, _ = input_data.shape

        # - Reshape data over separate input synapses
        input_data = input_data.reshape(
            n_batches, n_timesteps, self.size_out, self.n_synapses
        )

        # - Get evolution constants
        alpha = np.exp(-self.dt / self.tau_mem)
        beta = np.exp(-self.dt / self.tau_syn)
        noise_zeta = self.noise_std * np.sqrt(self.dt)

        # - Generate membrane noise trace
        key1, subkey = rand.split(self.rng_key)
        noise_ts = noise_zeta * rand.normal(
            subkey, shape=(n_batches, n_timesteps, self.size_out)
        )

        # - Single-step LIF dynamics
        def forward(
            state: Tuple[np.ndarray, np.ndarray, np.ndarray],
            inputs_t: Tuple[np.ndarray, np.ndarray],
        ) -> (
            Tuple[np.ndarray, np.ndarray, np.ndarray],
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
                state:          (Tuple[np.ndarray, np.ndarray, np.ndarray]) Layer state at end of evolution
                Irec_ts:        (np.ndarray) Recurrent input received at each neuron over time [T, N]
                output_ts:      (np.ndarray) Weighted output surrogate over time [T, O]
                surrogate_ts:   (np.ndarray) Surrogate time trace for each neuron [T, N]
                spikes_ts:      (np.ndarray) Logical spiking raster for each neuron [T, N]
                Vmem_ts:        (np.ndarray) Membrane voltage of each neuron over time [T, N]
                Isyn_ts:        (np.ndarray) Synaptic input current received by each neuron over time [T, N]
            """
            # - Unpack inputs
            (sp_in_t, noise_in_t) = inputs_t

            # - Unpack state
            spikes, isyn, vmem = state

            # - Apply synaptic and recurrent input
            isyn = isyn + sp_in_t
            irec = np.dot(spikes, self.w_rec).reshape(self.size_out, self.n_synapses)
            isyn = isyn + irec

            # - Decay synaptic and membrane state
            vmem *= alpha
            isyn *= beta

            # - Integrate membrane potentials
            vmem = vmem + isyn.sum(1) + noise_in_t + self.bias

            # - Detect next spikes (with custom gradient)
            spikes = step_pwl(vmem, self.threshold, 0.5, self.max_spikes_per_dt)

            # - Apply subtractive membrane reset
            vmem = vmem - spikes * self.threshold

            # - Return state and outputs
            return (spikes, isyn, vmem), (irec, spikes, vmem, isyn)

        # - Map over batches
        @jax.vmap
        def scan_time(spikes, isyn, vmem, input_data, noise_ts):
            return scan(forward, (spikes, isyn, vmem), (input_data, noise_ts))

        # - Evolve over spiking inputs
        state, (irec_ts, spikes_ts, vmem_ts, isyn_ts) = scan_time(
            spikes, isyn, vmem, input_data, noise_ts
        )

        # - Generate output surrogate
        surrogate_ts = sigmoid(vmem_ts * 20.0, self.threshold)

        # - Generate return arguments
        outputs = spikes_ts
        states = {
            "spikes": spikes_ts[0, -1],
            "isyn": isyn_ts[0, -1],
            "vmem": vmem_ts[0, -1],
            "rng_key": key1,
        }

        record_dict = {
            "irec": irec_ts,
            "spikes": spikes_ts,
            "isyn": isyn_ts,
            "vmem": vmem_ts,
            "U": surrogate_ts,
        }

        # - Return outputs
        return outputs, states, record_dict

    def as_graph(self) -> GraphModuleBase:
        # - Generate a GraphModule for the neurons
        neurons = LIFNeuronWithSynsRealValue._factory(
            self.size_in,
            self.size_out,
            f"{type(self).__name__}_{self.name}_{id(self)}",
            self,
            self.tau_mem,
            self.tau_syn,
            self.threshold,
            self.bias,
            self.dt,
        )

        # - Include recurrent weights if present
        if len(self.attributes_named("w_rec")) > 0:
            # - Weights are connected over the existing input and output nodes
            w_rec_graph = LinearWeights(
                neurons.output_nodes,
                neurons.input_nodes,
                f"{type(self).__name__}_recurrent_{self.name}_{id(self)}",
                self,
                self.w_rec,
            )

        # - Return a graph containing neurons and optional weights
        return as_GraphHolder(neurons)

    def _wrap_recorded_state(self, state_dict: dict, t_start: float = 0.0) -> dict:
        args = {"dt": self.dt, "t_start": t_start}

        return {
            "vmem": TSContinuous.from_clocked(
                np.squeeze(state_dict["vmem"][0]), name="$V_{mem}$", **args
            ),
            "isyn": TSContinuous.from_clocked(
                np.squeeze(state_dict["isyn"][0]), name="$I_{syn}$", **args
            ),
            "irec": TSContinuous.from_clocked(
                np.squeeze(state_dict["irec"][0]), name="$I_{rec}$", **args
            ),
            "U": TSContinuous.from_clocked(
                np.squeeze(state_dict["U"][0]), name="Surrogate $U$", **args
            ),
            "spikes": TSEvent.from_raster(
                np.squeeze(state_dict["spikes"][0]), name="Spikes", **args
            ),
        }
