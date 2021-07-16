"""
Implements a leaky integrate-and-fire neuron module with a Jax backend
"""

from rockpool.nn.modules.jax.jax_module import JaxModule
from rockpool.parameters import Parameter, State, SimulationParameter

import numpy as onp

from jax import numpy as np
from jax import custom_gradient
from jax.lax import scan
import jax.random as rand

from typing import Optional, Tuple, Union, Dict, Callable, Any

# - Define a float / array type
FloatVector = Union[float, np.ndarray]

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


class LIFJax(JaxModule):
    """
    A leaky integrate-and-fire spiking neuron model

    This module implements the dynamics:

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

    Neurons therefore share a common resting potential of ``0``, a firing threshold of ``0``, and a subtractive reset of ``-1``. Neurons each have an optional bias current `.bias` (default: ``0.``).

    :Surrogate signals:

    To facilitate gradient-based training, a surrogate :math:`U(t)` is generated from the membrane potentials of each neuron.

    .. math ::

        U_j = \\textrm{tanh}(20 * V_j + 1) / 2 + .5
    """

    def __init__(
        self,
        shape: tuple = None,
        tau_mem: Optional[FloatVector] = None,
        tau_syn: Optional[FloatVector] = None,
        bias: Optional[FloatVector] = None,
        w_rec: Optional[FloatVector] = None,
        dt: float = 1e-3,
        noise_std: float = 0.0,
        rng_key: Optional[Any] = None,
        spiking_input: bool = True,
        spiking_output: bool = True,
        *args,
        **kwargs,
    ):
        """
        Instantiate an LIF module

        Args:
            shape (tuple): Either a single dimension ``N``, which defines a feed-forward layer of LIF neurons, or two dimensions ``(N, N)``, which defines a recurrent layer of LIF neurons.
            tau_mem (Optional[np.ndarray]): An optional array with concrete initialisation data for the membrane time constants. If not provided, 100ms will be used by default.
            tau_syn (Optional[np.ndarray]): An optional array with concrete initialisation data for the synaptic time constants. If not provided, 50ms will be used by default.
            bias (Optional[np.ndarray]): An optional array with concrete initialisation data for the neuron bias currents. If not provided, 0.0 will be used by default.
            w_rec (Optional[np.ndarray]): If the module is initialised in recurrent mode, you can provide a concrete initialisation for the recurrent weights, which must be a square matrix with shape ``(N, N)``. If the model is not initialised in recurrent mode, then you may not provide ``w_rec``.
            dt (float): The time step for the forward-Euler ODE solver. Default: 1ms
            noise_std (float): The std. dev. of the noise added to membrane state variables at each time-step. Default: 0.0
            rng_key (Optional[Any]): The Jax RNG seed to use on initialisation. By default, a new seed is generated.
        """
        # - Work out the shape of this module
        if shape is None:
            assert (
                tau_mem is not None
            ), "You must provide either `shape` or else specify parameters."
            shape = np.array(tau_mem).shape

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
            rng_key = rand.PRNGKey(onp.random.randint(0, 2 ** 63))
        _, rng_key = rand.split(np.array(rng_key, dtype=np.uint32))

        # - Initialise state
        self.rng_key: Union[np.ndarray, State] = State(
            rng_key, init_func=lambda _: rng_key
        )

        # - Should we be recurrent or FFwd?
        if len(self.shape) == 1:
            # - Feed-forward mode
            if w_rec is not None:
                raise ValueError(
                    "If `shape` is unidimensional, then `w_rec` may not be provided as an argument."
                )

            self.w_rec: float = 0.0

        else:
            # - Recurrent mode
            # - Check that `shape` is correctly specified
            if len(self.shape) > 2:
                raise ValueError("`shape` may not specify more than two dimensions.")

            if self.size_out != self.size_in:
                raise ValueError(
                    "`shape[0]` and `shape[1]` must be equal for a recurrent module."
                )

            self.w_rec: Union[np.ndarray, Parameter] = Parameter(
                w_rec,
                family="weights",
                init_func=lambda s: rand.normal(
                    rand.split(self.rng_key)[0], shape=self.shape
                ),
                shape=self.shape,
            )

        # - Set parameters
        self.tau_mem: Union[np.ndarray, Parameter] = Parameter(
            tau_mem,
            "taus",
            init_func=lambda s: np.ones(s) * 100e-3,
            shape=(self.size_out,),
        )
        # - Set parameters
        self.tau_syn: Union[np.ndarray, Parameter] = Parameter(
            tau_syn,
            "taus",
            init_func=lambda s: np.ones(s) * 50e-3,
            shape=(self.size_out,),
        )
        self.bias: Union[np.ndarray, Parameter] = Parameter(
            bias,
            "bias",
            init_func=lambda s: np.zeros(s),
            shape=(self.size_out,),
        )
        self.dt: Union[float, SimulationParameter] = SimulationParameter(dt)
        self.noise_std: Union[float, SimulationParameter] = SimulationParameter(
            noise_std
        )

        # - Specify state
        self.spikes: Union[np.ndarray, State] = State(
            shape=(self.size_out,), init_func=np.zeros
        )
        self.Isyn: Union[np.ndarray, State] = State(
            shape=(self.size_out,), init_func=np.zeros
        )
        self.Vmem: Union[np.ndarray, State] = State(
            shape=(self.size_out,), init_func=np.zeros
        )

    def evolve(
        self,
        input_data: np.ndarray,
        record: bool = False,
    ) -> Tuple[np.ndarray, dict, dict]:
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

            U_j = \\textrm{tanh}(20 * V_j + 1) / 2 + .5

        Args:
            input_data (np.ndarray): Input array of shape ``(T, Nin)`` to evolve over
            record (bool): If ``True``,

        Returns:
            (np.ndarray, dict, dict): output, new_state, record_state
            ``output`` is an array with shape ``(T, Nout)`` containing the output data produced by this module. ``new_state`` is a dictionary containing the updated module state following evolution. ``record_state`` will be a dictionary containing the recorded state variables for this evolution, if the ``record`` argument is ``True``.
        """

        # - Get evolution constants
        alpha = self.dt / self.tau_mem
        beta = np.exp(-self.dt / self.tau_syn)

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

            spikes, Isyn, Vmem = state

            # - Synaptic input
            Irec = np.dot(spikes, self.w_rec)
            dIsyn = sp_in_t + Irec
            Isyn = beta * Isyn + dIsyn

            # - Apply subtractive reset
            Vmem = Vmem - spikes

            # - Membrane potentials
            dVmem = Isyn + Iin + self.bias - Vmem
            Vmem = Vmem + alpha * dVmem

            # - Detect next spikes (with custom gradient)
            spikes = step_pwl(Vmem)

            # - Return state and outputs
            return (spikes, Isyn, Vmem), (Irec, spikes, Vmem, Isyn)

        # - Generate membrane noise trace
        num_timesteps = input_data.shape[0]
        key1, subkey = rand.split(self.rng_key)
        noise_ts = self.noise_std * rand.normal(
            subkey, shape=(num_timesteps, self.size_out)
        )

        # - Evolve over spiking inputs
        state, (Irec_ts, spikes_ts, Vmem_ts, Isyn_ts) = scan(
            forward,
            (self.spikes, self.Isyn, self.Vmem),
            (input_data, noise_ts),
        )

        # - Generate output surrogate
        surrogate_ts = sigmoid(Vmem_ts * 20.0)

        # - Generate return arguments
        outputs = spikes_ts
        states = {
            "spikes": spikes_ts[-1],
            "Isyn": Isyn_ts[-1],
            "Vmem": Vmem_ts[-1],
            "rng_key": key1,
        }

        record_dict = {
            "Irec": Irec_ts,
            "spikes": spikes_ts,
            "Isyn": Isyn_ts,
            "Vmem": Vmem_ts,
            "U": surrogate_ts,
        }

        # - Return outputs
        return outputs, states, record_dict
