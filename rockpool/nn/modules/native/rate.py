"""
Contains an implementation of a non-spiking rate module
"""

# - Rockpool imports
from rockpool.nn.modules.module import Module
from rockpool.parameters import Parameter, State, SimulationParameter
from .linear import kaiming, unit_eigs

from rockpool.graph import (
    RateNeuronWithSynsRealValue,
    LinearWeights,
    GraphModuleBase,
    as_GraphHolder,
)

# -- Imports
import numpy as np
import numpy.random as rand

from typing import Optional, Union, Any, Callable, Tuple
from rockpool.typehints import FloatVector, P_float, P_int, P_tensor

__all__ = ["Rate"]

# -- Define useful neuron transfer functions
def H_ReLU(x: FloatVector, threshold: FloatVector) -> FloatVector:
    return (x - threshold) * ((x - threshold) > 0.0)


def H_tanh(x: FloatVector, threshold: FloatVector) -> FloatVector:
    return np.tanh(x - threshold)


def H_sigmoid(x: FloatVector, threshold: FloatVector) -> FloatVector:
    return (np.tanh(x - threshold) + 1) / 2


class Rate(Module):
    """
    Encapsulates a population of rate neurons, supporting feed-forward and recurrent modules

    Examples:
        Instantiate a feed-forward module with 8 neurons:

        >>> mod = Rate(8,)
        RateEulerJax 'None' with shape (8,)

        Instantiate a recurrent module with 12 neurons:

        >>> mod_rec = Rate(12, has_rec = True)
        RateEulerJax 'None' with shape (12,)

        Instantiate a feed-forward module with defined time constants:

        >>> mod = Rate(7, tau = np.arange(7,) * 10e-3)
        RateEulerJax 'None' with shape (7,)

    This module implements the update equations:

    .. math::

        \dot{X} = -X + i(t) + W_{rec} H(X) + bias + \sigma \zeta_t
        X = X + \dot{x} * dt / \tau

        H(x, t) = relu(x, t) = (x - t) * ((x - t) > 0)
    """

    def __init__(
        self,
        shape: Union[int, Tuple[int, int], Tuple[int]],
        tau: Optional[FloatVector] = None,
        bias: Optional[FloatVector] = None,
        threshold: Optional[FloatVector] = None,
        w_rec: Optional[np.ndarray] = None,
        weight_init_func: Callable = unit_eigs,
        has_rec: bool = False,
        activation_func: Union[str, Callable] = H_ReLU,
        dt: float = 1e-3,
        noise_std: float = 0.0,
        *args: list,
        **kwargs: dict,
    ):
        """
        Instantiate a non-spiking rate module, either feed-forward or recurrent.

        Args:
            shape (Tuple[int]): A tuple containing the numer  of this module.
            tau (float): A scalar or vector defining the initialisation time constants for the module. If a vector is provided, it must match the output size of the module. Default: ``20ms`` for each unit
            bias (float): A scalar or vector defining the initialisation bias values for the module. If a vector is provided, it must match the output size of the module. Default: ``0.``
            w_rec (np.ndarray): An optional matrix defining the initialisation recurrent weights for the module.
            weight_init_func (Callable): A function used to initialise the recurrent weights, if used. Default: :py:func:`.unit_eigs`; initialise such that recurrent feedback has eigenvalues distributed within the unit circle.
            has_rec (bool): A flag parameter indicating whether the module has recurrent connections or not. Default: ``False``, no recurrent connections.
            activation_func (Callable): The activation function of the neurons. This can be provided as a string ``['ReLU', 'sigmoid', 'tanh']``, or as a function that accepts a vector of neural states and returns the vector of output activations. Default: ``'ReLU'``.
            dt (float): The Euler solver time-step. Default: ``1e-3``
            noise_std (float): The std. dev. of normally-distributed noise added to the neural state at each time step. Default: ``0.``
            rng_key (Any): A Jax PRNG key to initialise the module with. Default: not provided, the module PRNG will be initialised with a random number.
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        # - Call the superclass initialiser
        super().__init__(
            shape=shape, spiking_input=False, spiking_output=False, *args, **kwargs
        )

        if self.size_out != self.size_in:
            raise ValueError("Rate module must have `size_out` == `size_in`.")

        # - Initialise state
        self.x: Union[np.ndarray, State] = State(
            shape=self.size_out, init_func=np.zeros
        )
        """A vector ``(N,)`` of the internal state of each neuron"""

        # - Should we be recurrent or FFwd?
        if not has_rec:
            # - Feed-forward mode
            if w_rec is not None:
                raise ValueError(
                    "If `shape` is unidimensional, then `w_rec` may not be provided as an argument."
                )

        else:
            # - Recurrent mode
            self.w_rec: P_tensor = Parameter(
                w_rec,
                family="weights",
                init_func=weight_init_func,
                shape=(self.size_out, self.size_in),
            )
            """The recurrent weight matrix ``(N, N)`` for this module"""

        # - Set parameters
        self.tau: P_tensor = Parameter(
            tau,
            family="taus",
            init_func=lambda s: np.ones(s) * 20e-3,
            shape=[(self.size_out,), ()],
        )
        """ The vector ``(N,)`` of time constants :math:`\\tau` for each neuron"""

        self.bias: P_tensor = Parameter(
            bias,
            "bias",
            init_func=lambda s: np.zeros(s),
            shape=[(self.size_out,), ()],
        )
        """The vector ``(N,)`` of bias currents for each neuron"""

        self.threshold: P_tensor = Parameter(
            threshold,
            family="thresholds",
            shape=[(self.size_out,), ()],
            init_func=np.zeros,
        )
        """ (Tensor) Unit thresholds `(Nout,)` or `()` """

        self.dt: P_float = SimulationParameter(dt)
        """The Euler solver time step for this module"""

        self.noise_std: P_float = SimulationParameter(noise_std)
        """The std. dev. :math:`\\sigma` of noise added to internal neuron states at each time step"""

        # - Check and assign the activation function
        if isinstance(activation_func, str):
            # - Handle a string argument
            if activation_func.lower() in ["relu", "r"]:
                act_fn = H_ReLU
            elif activation_func.lower() in ["sigmoid", "sig", "s"]:
                act_fn = H_sigmoid
            elif activation_func.lower() in ["tanh", "t"]:
                act_fn = H_tanh
            else:
                raise ValueError(
                    'If `activation_func` is provided as a string argument, it must be one of ["ReLU", "sigmoid", "tanh"].'
                )

        elif callable(activation_func):
            # - Handle a callable function
            act_fn = activation_func
            """The activation function of the neurons in the module"""

        else:
            raise ValueError(
                "Argument `activation_func` must be a string or a function."
            )

        # - Assign activation function
        self.act_fn: Union[Callable, SimulationParameter] = SimulationParameter(act_fn)

    def evolve(
        self,
        input_data: np.ndarray,
        record: bool = False,
    ):
        input_data, (x,) = self._auto_batch(input_data, (self.x,))
        batches, num_timesteps, n_inputs = input_data.shape

        # - Get evolution constants
        alpha = self.dt / np.clip(self.tau, 10 * self.dt, np.inf)
        noise_zeta = self.noise_std * np.sqrt(self.dt)

        w_rec = self.w_rec if hasattr(self, "w_rec") else None

        # - Reservoir state step function (forward Euler solver)
        def forward(x, inp):
            """
            reservoir_step() - Single step of recurrent reservoir

            :param x:       np.ndarray Current state and activation of reservoir units
            :param inp:    np.ndarray Inputs to each reservoir unit for the current step

            :return:    (new_state, new_activation), (rec_input, activation)
            """
            state, activation = x

            rec_input = np.dot(activation, w_rec) if w_rec is not None else 0.0
            dstate = -state + inp + self.bias + rec_input
            state = state + dstate * alpha
            activation = self.act_fn(state, self.threshold)

            return (state, activation), (rec_input, state, activation)

        # - Compute random numbers for reservoir noise
        noise = noise_zeta * rand.normal(size=input_data.shape)
        inputs = input_data + noise

        # - Loop over time
        rec_inputs = np.zeros((batches, num_timesteps, self.size_out))
        res_state = np.zeros((batches, num_timesteps, self.size_out))
        outputs = np.zeros((batches, num_timesteps, self.size_out))

        for b in range(batches):
            for t in range(num_timesteps):
                # - Solve layer dynamics for this time-step
                (
                    (x[b], _),
                    (
                        this_rec_i,
                        this_r_s,
                        this_out,
                    ),
                ) = forward((x[b], self.act_fn(x[b], self.threshold)), inputs[b, t, :])

                # - Keep a record of the layer dynamics
                rec_inputs[b, t, :] = this_rec_i
                res_state[b, t, :] = this_r_s
                outputs[b, t, :] = this_out

        self.x = x[0]

        record_dict = {"rec_input": rec_inputs, "x": res_state} if record else {}

        return outputs, self.state(), record_dict

    def as_graph(self) -> GraphModuleBase:
        # - Generate a GraphModule for the neurons
        neurons = RateNeuronWithSynsRealValue._factory(
            self.size_in,
            self.size_out,
            f"{type(self).__name__}_{self.name}_{id(self)}",
            self,
            self.tau,
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
