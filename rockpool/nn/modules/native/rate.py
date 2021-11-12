"""
Contains an implementation of a non-spiking rate module, with a Jax backend
"""

# - Rockpool imports
from rockpool.nn.modules.module import Module
from rockpool.parameters import Parameter, State, SimulationParameter

# -- Imports
from importlib import util

import numpy as np
import numpy.random as rand

from typing import Optional, Union, Any, Callable, Tuple
from rockpool.typehints import FloatVector, P_float, P_int, P_tensor

__all__ = ["Rate"]

# -- Define useful neuron transfer functions
def H_ReLU(x: FloatVector) -> FloatVector:
    return x * (x > 0.0)


H_tanh = np.tanh


def H_sigmoid(x: FloatVector) -> FloatVector:
    return (np.tanh(x) + 1) / 2


class Rate(Module):
    """
    Encapsulates a population of rate neurons, supporting feed-forward and recurrent modules.

    Examples:
        Instantiate a feed-forward module with 8 neurons:

        >>> mod = Rate(8,)
        Rate 'None' with shape (8,)

        Instantiate a recurrent module with 12 neurons:

        >>> mod_rec = Rate((12, 12))
        Rate 'None' with shape (12, 12)

        Instantiate a feed-forward module with defined time constants:

        >>> mod = Rate(tau = np.arange(7,) * 10e-3)
        Rate 'None' with shape (7,)

        ``mod`` will contain 7 neurons, taking the dimensionlity of `tau`.

    Notes:
        Each neuron follows the dynamics

        .. math::
            \\tau \\cdot \\dot{x} + x = b + i(t) + \\sigma\\eta(t)

        where :math:`x` is the neuron state; :math:`\\tau` is the neuron time constant; :math:`b` is the neuron bias; :math:`i(t)`$` is the input current at time :math:`t`$`; and :math:`\\sigma\\eta(t)`$` is a white noise process with std. dev. :math:`\\eta`.
    """

    def __init__(
        self,
        shape: Optional[Union[int, Tuple[int, int], Tuple[int]]] = None,
        tau: Optional[FloatVector] = None,
        bias: Optional[FloatVector] = None,
        w_rec: Optional[np.ndarray] = None,
        activation_func: Union[str, Callable] = H_ReLU,
        dt: float = 1e-3,
        noise_std: float = 1e-3,
        *args: list,
        **kwargs: dict,
    ):
        """
        Instantiate a non-spiking rate module, either feed-forward or recurrent.

        Args:
            shape (Tuple[int]): A tuple containing the shape of this module. If one dimension is provided ``(N,)``, it will define the number of neurons in a feed-forward layer. If two dimensions are provided, a recurrent layer will be defined. In that case the two dimensions must be identical ``(N, N)``. If not provided, `shape` will be inferred from other provided arguments.
            tau (float): A scalar or vector defining the initialisation time constants for the module. If a vector is provided, it must match the output size of the module. Default: ``1.``
            bias (float): A scalar or vector defining the initialisation bias values for the module. If a vector is provided, it must match the output size of the module. Default: ``0.``
            w_rec (np.ndarray): An optional matrix defining the initialisation recurrent weights for the module. Default: ``Normal / sqrt(N)``
            activation_func (Callable): The activation function of the neurons. This can be provided as a string ``['ReLU', 'sigmoid', 'tanh']``, or as a function that accepts a vector of neural states and returns the vector of output activations. Default: ``'ReLU'``.
            dt (float): The Euler solver time-step. Default: ``1e-3``
            noise_std (float): The std. dev. of normally-distributed noise added to the neural state at each time step. Default: ``0.``
            rng_key (Any): A Jax PRNG key to initialise the module with. Default: not provided, the module PRNG will be initialised with a random number.
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        # - Work out the shape of this module
        if shape is None:
            assert (
                tau is not None
            ), "You must provide either `shape` or else specify concrete parameters."
            shape = np.array(tau).shape

        # - Call the superclass initialiser
        super().__init__(
            shape=shape, spiking_input=False, spiking_output=False, *args, **kwargs
        )

        # - Initialise state
        self.neur_state: Union[np.ndarray, State] = State(
            shape=self.size_out, init_func=np.zeros
        )
        """A vector ``(N,)`` of the internal state of each neuron"""

        # - Should we be recurrent or FFwd?
        if len(self.shape) == 1:
            # - Feed-forward mode
            if w_rec is not None:
                raise ValueError(
                    "If `shape` is unidimensional, then `w_rec` may not be provided as an argument."
                )

        else:
            # - Recurrent mode
            # - Check that `shape` is correctly specified
            if len(self.shape) > 2:
                raise ValueError("`shape` may not specify more than two dimensions.")

            if self.size_out != self.size_in:
                raise ValueError(
                    "`shape[0]` and `shape[1]` must be equal for a recurrent module."
                )

            self.w_rec: P_tensor = Parameter(
                w_rec,
                family="weights",
                init_func=lambda s: rand.normal(self.shape) / np.sqrt(self.shape[0]),
                shape=self.shape,
            )
            """The recurrent weight matrix ``(N, N)`` for this module, if in recurrent mode"""

        # - Set parameters
        self.tau: P_tensor = Parameter(
            tau,
            family="taus",
            init_func=lambda s: np.ones(s) * 100e-3,
            shape=(self.size_out,),
        )
        """ The vector ``(N,)`` of time constants :math:`\\tau` for each neuron"""

        self.bias: P_tensor = Parameter(
            bias,
            "bias",
            init_func=lambda s: np.zeros(s),
            shape=(self.size_out,),
        )
        """The vector ``(N,)`` of bias currents for each neuron"""

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
        dt_tau = self.dt / self.tau
        if hasattr(self, "w_rec"):
            w_rec = self.w_rec
        else:
            w_rec = 0.0

        # - Reservoir state step function (forward Euler solver)
        def reservoir_step(x, inp):
            """
            reservoir_step() - Single step of recurrent reservoir

            :param x:       np.ndarray Current state and activation of reservoir units
            :param inp:    np.ndarray Inputs to each reservoir unit for the current step

            :return:    (new_state, new_activation), (rec_input, activation)
            """
            state, activation = x
            rec_input = np.dot(activation, w_rec)
            state += dt_tau * (-state + inp + self.bias + rec_input)
            activation = self.act_fn(state)

            return (state, activation), (rec_input, state, activation)

        # - Evaluate passthrough input layer
        res_inputs = input_data

        # - Compute random numbers for reservoir noise
        noise = self.noise_std * rand.normal(res_inputs.shape)
        inputs = res_inputs + noise
        this_act = self.act_fn(self.neur_state)

        # - Loop over time
        rec_inputs = []
        res_state = []
        outputs = []

        for t in range(inputs.shape[0]):
            # - Solve layer dynamics for this time-step
            (self.neur_state, this_act), (
                this_rec_i,
                this_r_s,
                this_out,
            ) = reservoir_step((self.neur_state, this_act), inputs[t, :])

            # - Keep a record of the layer dynamics
            rec_inputs.append(this_rec_i)
            res_state.append(this_r_s)
            outputs.append(this_out)

        # (neur_state1, activation1), (rec_inputs, res_state, res_acts) = scan(
        #     reservoir_step, (self.neur_state, H_tanh(self.neur_state)), inputs
        # )

        record_dict = (
            {
                "res_inputs": res_inputs,
                "rec_inputs": np.array(rec_inputs),
                "res_state": np.array(res_state),
            }
            if record
            else {}
        )

        return outputs, self.state(), record_dict
