"""
Contains an implementation of a non-spiking rate module, with a Jax backend
"""

# - Rockpool imports
from rockpool.nn.modules.jax.jax_module import JaxModule
from rockpool.parameters import Parameter, State, SimulationParameter
from ..native.linear import unit_eigs, kaiming
from rockpool.graph import (
    RateNeuronWithSynsRealValue,
    LinearWeights,
    GraphModuleBase,
    as_GraphHolder,
)

# -- Imports
from importlib import util

import jax.numpy as np
import jax
from jax.lax import scan
import jax.random as rand
from jax.tree_util import Partial
from jax.nn import relu, leaky_relu
from functools import partial

import numpy as onp

from typing import Optional, Union, Any, Callable, Tuple
from rockpool.typehints import FloatVector, P_Callable, P_ndarray, P_float

__all__ = ["RateJax", "H_tanh", "H_ReLU", "H_sigmoid", "H_LReLU"]

# -- Define useful neuron transfer functions
def H_ReLU(x: FloatVector, threshold: FloatVector) -> FloatVector:
    return (x - threshold) * ((x - threshold) > 0)


def H_tanh(x: FloatVector, threshold: FloatVector) -> FloatVector:
    return np.tanh(x - threshold)


def H_sigmoid(x: FloatVector, threshold: FloatVector) -> FloatVector:
    return (np.tanh(x - threshold) + 1) / 2


def H_LReLU(
    x: FloatVector, threshold: FloatVector, negative_slope: float = 1e-2
) -> FloatVector:
    return (x - threshold) * ((x - threshold) >= 0) + negative_slope * (
        (x - threshold) < 0
    )


class RateJax(JaxModule):
    """
    Encapsulates a population of rate neurons, supporting feed-forward and recurrent modules, with a Jax backend

    Examples:
        Instantiate a feed-forward module with 8 neurons:

        >>> mod = RateJax(8,)
        RateEulerJax 'None' with shape (8,)

        Instantiate a recurrent module with 12 neurons:

        >>> mod_rec = RateJax(12, has_rec = True)
        RateEulerJax 'None' with shape (12,)

        Instantiate a feed-forward module with defined time constants:

        >>> mod = RateJax(7, tau = np.arange(7,) * 10e-3)
        RateEulerJax 'None' with shape (7,)

    This module implements the update equations:

    .. math::

        \dot{X} = -X + i(t) + W_{rec} H(X) + bias + \sigma \zeta_t
        X = X + \dot{x} * dt / \tau

        H(x, t) = relu(x, t) = (x - t) * ((x - t) > 0)
    """

    # @partial(
    #     jax.jit,
    #     static_argnames=("self", "has_rec", "activation_func", "weight_init_func"),
    # )
    def __init__(
        self,
        shape: Union[int, Tuple[np.ndarray]],
        tau: Optional[FloatVector] = None,
        bias: Optional[FloatVector] = None,
        threshold: Optional[FloatVector] = None,
        w_rec: Optional[np.ndarray] = None,
        weight_init_func: Callable = unit_eigs,
        has_rec: bool = False,
        activation_func: Union[str, Callable] = H_ReLU,
        noise_std: float = 0.0,
        dt: float = 1e-3,
        rng_key: Optional[int] = None,
        *args: list,
        **kwargs: dict,
    ):
        """
        Instantiate a non-spiking rate module, either feed-forward or recurrent.

        Args:
            shape (Tuple[np.ndarray]): A tuple containing the shape of this module. If one dimension is provided ``(N,)``, it will define the number of neurons in a feed-forward layer. If two dimensions are provided, a recurrent layer will be defined. In that case the two dimensions must be identical ``(N, N)``.
            tau (float): A scalar or vector defining the initialisation time constants for the module. If a vector is provided, it must match the output size of the module. Default: ``20ms``
            bias (float): A scalar or vector defining the initialisation bias values for the module. If a vector is provided, it must match the output size of the module. Default: ``0.``
            w_rec (np.ndarray): An optional matrix defining the initialisation recurrent weights for the module. Default: ``Normal / sqrt(N)``
            has_rec (bool): Iff ``True``, the module operates in recurrent mode. Default: ``False``, operate in feed-forward mode.
            weight_init_func (Callable): A function used to initialise the recurrent weights, if used. Default: :py:func:`.unit_eigs`; initialise such that recurrent feedback has eigenvalues distributed within the unit circle.
            activation_func (Callable): The activation function of the neurons. This can be provided as a string ``['ReLU', 'sigmoid', 'tanh']``, or as a function that accepts a vector of neural states and returns the vector of output activations. This function must use `jax.numpy` math functions, and *not* `numpy` math functions. Default: ``'ReLU'``.
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

        # if self.size_out != self.size_in:
        #     raise ValueError("RateJax module must have `size_out` == `size_in`.")

        # - Seed RNG
        if rng_key is None:
            rng_key = rand.PRNGKey(onp.random.randint(0, 2**63))
        _, rng_key = rand.split(np.array(rng_key, dtype=np.uint32))

        self.rng_key: Union[np.ndarray, State] = State(
            rng_key, init_func=lambda _: rng_key
        )
        """The Jax PRNG key for this module"""

        # - Initialise state
        self.x: P_ndarray = State(
            shape=self.size_out,
            init_func=np.zeros,
            cast_fn=np.array,
        )
        """A vector ``(N,)`` of the internal state of each unit"""

        if isinstance(has_rec, jax.core.Tracer) or has_rec:
            self.w_rec: P_ndarray = Parameter(
                w_rec,
                family="weights",
                init_func=weight_init_func,
                shape=(self.size_out, self.size_in),
                cast_fn=np.array,
            )
            """The recurrent weight matrix ``(N, N)`` for this module """
        else:
            self.w_rec = 0.0

        # - Set parameters
        self.tau: P_ndarray = Parameter(
            tau,
            family="taus",
            init_func=lambda s: np.ones(s) * 20e-3,
            shape=[(self.size_out,), ()],
            cast_fn=np.array,
        )
        """ The vector ``(N,)`` of time constants :math:`\\tau` for each unit """

        self.bias: P_ndarray = Parameter(
            bias,
            "bias",
            init_func=lambda s: np.zeros(s),
            shape=[(self.size_out,), ()],
            cast_fn=np.array,
        )
        """The vector ``(N,)`` of bias currents for each unit """

        self.threshold: P_ndarray = Parameter(
            threshold,
            family="thresholds",
            shape=[(self.size_out,), ()],
            init_func=np.zeros,
            cast_fn=np.array,
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
        self.act_fn: P_Callable = SimulationParameter(Partial(act_fn))
        """ (Callable) Activation function """

        # - Define additional arguments required during initialisation
        self._init_args = {
            "has_rec": has_rec,
            "weight_init_func": Partial(weight_init_func),
            "activation_func": Partial(act_fn),
        }

    def evolve(
        self,
        input_data: np.ndarray,
        record: bool = False,
    ):
        # - Expand over batches
        input_data, (x0,) = self._auto_batch(input_data, (self.x,))

        # - Get evolution constants
        alpha = self.dt / self.tau
        noise_zeta = self.noise_std * np.sqrt(self.dt)

        # - Reservoir state step function (forward Euler solver)
        def forward(x, inp):
            """
            forward() - Single step of recurrent reservoir

            :param x:       np.ndarray Current state and activation of reservoir units
            :param inp:    np.ndarray Inputs to each reservoir unit for the current step

            :return:    (new_state, new_activation), (rec_input, activation)
            """
            state, activation = x

            rec_input = np.dot(activation, self.w_rec)
            state += alpha * (-state + inp + self.bias + rec_input)
            activation = self.act_fn(state, self.threshold)

            return (state, activation), (rec_input, state, activation)

        # - Generate noise trace
        key1, subkey = rand.split(self.rng_key)
        noise = noise_zeta * rand.normal(subkey, shape=input_data.shape)
        inputs = input_data + noise

        # - Map over batches
        @jax.vmap
        def scan_time(state0, act0, inputs):
            return scan(forward, (state0, act0), inputs)

        # - Use `scan` to evaluate reservoir
        (x1, _), (rec_inputs, res_state, outputs) = scan_time(
            x0, self.act_fn(x0, self.threshold), inputs
        )

        new_state = {
            "x": x1[0],
            "rng_key": key1,
        }

        record_dict = {
            "rec_input": rec_inputs,
            "x": res_state,
        }

        return outputs, new_state, record_dict

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


RateEulerJax = RateJax
