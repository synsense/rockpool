"""
Implements a leaky integrate-and-fire neuron module with a numpy backend
"""

from rockpool.nn.modules.module import Module
from rockpool.parameters import Parameter, State, SimulationParameter
from rockpool.nn.modules.native.linear import kaiming
from rockpool import TSContinuous, TSEvent

import numpy as np

from typing import Optional, Tuple, Union, Dict, Callable, Any
from rockpool.typehints import (
    FloatVector,
    P_float,
    P_tensor,
    P_ndarray,
    P_int,
    P_Callable,
)

from rockpool.graph import (
    GraphModuleBase,
    as_GraphHolder,
    LIFNeuronWithSynsRealValue,
    LinearWeights,
)


__all__ = ["LIF", "spike_subtract_threshold"]

# - Surrogate functions to use in learning
def sigmoid(x: FloatVector, threshold: FloatVector) -> FloatVector:
    """
    Sigmoid function

    :param FloatVector x: Input value

    :return FloatVector: Output value
    """
    return np.tanh(x + 1 - threshold) / 2 + 0.5


def spike_subtract_threshold(
    x: FloatVector,
    threshold: FloatVector,
    window: FloatVector = 0.5,
    max_spikes_per_dt: int = np.inf,
) -> FloatVector:
    """
    Spike production function

    Number of spikes is equal to floor(x / threshold)

    Args:
        x (float):          Input value
        threshold (float):  Firing threshold
        window (float): Unused
        max_spikes_per_dt (int): Maximum number of events that may be produced in each time-step. Default: ``np.inf`` (do not impose a maximum)

    Returns:
        float: Number of output events for each input value
    """
    return np.clip((x >= threshold) * np.floor(x / threshold), None, max_spikes_per_dt)


class LIF(Module):
    """
    A leaky integrate-and-fire spiking neuron model

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
            max_spikes_per_dt (int): The maximum number of events that will be produced in a single time-step. Default: ``np.inf``; do not clamp spiking
            dt (float): The time step for the forward-Euler ODE solver. Default: 1ms
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

        self.n_synapses: P_int = shape[0] // shape[1]
        """ (int) Number of input synapses per neuron """

        # - Should we be recurrent or FFwd?
        if has_rec:
            self.w_rec: P_ndarray = Parameter(
                w_rec,
                shape=(self.size_out, self.size_in),
                init_func=weight_init_func,
                family="weights",
                cast_fn=np.array,
            )
            """ (Tensor) Recurrent weights `(Nout, Nin)` """
        else:
            if w_rec is not None:
                raise ValueError("`w_rec` may not be provided if `has_rec` is `False`")

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
        )
        """ (np.ndarray) Synaptic time constants `(Nout,)` or `()` """

        self.bias: P_ndarray = Parameter(
            bias,
            "bias",
            init_func=lambda s: np.zeros(s),
            shape=[(self.size_out,), ()],
        )
        """ (np.ndarray) Neuron bias currents `(Nout,)` or `()` """

        self.threshold: P_ndarray = Parameter(
            threshold,
            "threshold",
            init_func=np.ones,
            shape=[(self.size_out,), ()],
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
        input_data, (vmem, spikes, isyn) = self._auto_batch(
            input_data,
            (self.vmem, self.spikes, self.isyn),
            (
                (self.size_out,),
                (self.size_out,),
                (self.size_out, self.n_synapses),
            ),
        )
        batches, num_timesteps, _ = input_data.shape

        # - Reshape data over separate input synapses
        input_data = input_data.reshape(
            batches, num_timesteps, self.size_out, self.n_synapses
        )

        # - Get evolution constants
        alpha = np.exp(-self.dt / self.tau_mem)
        beta = np.exp(-self.dt / self.tau_syn)
        noise_zeta = self.noise_std * np.sqrt(self.dt)

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

            # - Unpack state
            spikes, Isyn, Vmem = state

            # - Apply synaptic and recurrent input
            Irec = (
                np.dot(spikes, self.w_rec).reshape(self.size_out, self.n_synapses)
                if hasattr(self, "w_rec")
                else np.zeros((self.size_out, self.n_synapses))
            )
            Isyn += sp_in_t + Irec

            # - Decay synaptic and membrane state
            Vmem *= alpha
            Isyn *= beta

            # - Integrate membrane potentials
            Vmem += Isyn.sum(1) + I_in_t + self.bias

            # - Detect next spikes (with custom gradient)
            spikes = spike_subtract_threshold(
                Vmem, self.threshold, None, self.max_spikes_per_dt
            )

            # - Apply subtractive membrane reset
            Vmem = Vmem - spikes * self.threshold

            # - Return state and outputs
            return (spikes, Isyn, Vmem), (Irec, spikes, Vmem, Isyn)

        # - Generate membrane noise trace
        noise_ts = noise_zeta * np.random.randn(batches, num_timesteps, self.size_out)

        Irec_ts = np.zeros((batches, num_timesteps, self.size_out, self.n_synapses))
        spikes_ts = np.zeros((batches, num_timesteps, self.size_out))
        Vmem_ts = np.zeros((batches, num_timesteps, self.size_out))
        Isyn_ts = np.zeros((batches, num_timesteps, self.size_out, self.n_synapses))

        for b in range(batches):
            for t in range(num_timesteps):
                # - Solve layer dynamics for this timestep
                (
                    (spikes[b], isyn[b], vmem[b]),
                    (
                        Irec_ts[b, t, :, :],
                        spikes_ts[b, t, :],
                        Vmem_ts[b, t, :],
                        Isyn_ts[b, t, :, :],
                    ),
                ) = forward(
                    (spikes[b], isyn[b], vmem[b]),
                    (input_data[b, t, :], noise_ts[b, t, :]),
                )

        # - Generate output surrogate
        surrogate_ts = sigmoid(Vmem_ts * 20.0, self.threshold)

        self.spikes = spikes[0]
        self.isyn = isyn[0]
        self.vmem = vmem[0]

        # - Generate return arguments
        outputs = spikes_ts

        record_dict = {
            "irec": Irec_ts,
            "spikes": spikes_ts,
            "isyn": Isyn_ts,
            "vmem": Vmem_ts,
            "U": surrogate_ts,
        }

        # - Return outputs
        return outputs, self.state(), record_dict

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
