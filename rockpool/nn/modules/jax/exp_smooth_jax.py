"""
Exponential smoothing module in Jax.
"""

from rockpool.nn.modules.jax.jax_module import JaxModule
from rockpool.parameters import Parameter, SimulationParameter, State

import jax
import jax.numpy as np
from jax.tree_util import Partial

from typing import Tuple, Union, Any, Callable, Optional

__all__ = ["ExpSmoothJax"]

import warnings

warnings.warn("The module ExpSmoothJax is deprecated. Use ExpSynJax instead.")


class ExpSmoothJax(JaxModule):
    """
    Low-pass smoothing module using an exponential filter.

    This module implements a low-pass filter with exponential dynamics:

    .. math::

        \\tau \\dot{I} + I = i(t)

        y = H(I)

    where :math:`H(\cdot)` is an activation function for the module.

    Examples:
        >>> import jax.numpy as jnp
        >>> mod = ExpSmoothJax((N,), activation_fun=jnp.tanh)
    """

    def __init__(
        self,
        shape: Tuple,
        tau: float = 100e-3,
        dt: float = 1e-3,
        activation_fun: Callable[[np.ndarray], np.ndarray] = lambda x: x,
        *args,
        **kwargs,
    ):
        """
        Instantiate a smoothing filter module.

        Args:
            shape (tuple): The shape of this module ``(N,)``
            tau (float): The smoothing time constant :math:`\\tau` in seconds. Default: 100 ms.
            dt (float): Simulation time-step in seconds. Default: 1 ms.
            activation_fun (Callable[[np.ndarray], np.ndarray]): Activation function of the state of this module. Default: ``lambda x: x``.
        """
        # - Check the shape input
        assert np.size(shape) == 1, "`shape` must have one dimension."

        # - Initialise superclass
        super().__init__(
            shape=shape, spiking_input=True, spiking_output=False, *args, **kwargs
        )

        self.tau: Union[Parameter, float] = Parameter(tau, "taus")
        """ (float) Time constant for the exponential synapse. """

        self.dt: Union[SimulationParameter, float] = SimulationParameter(dt)
        """ (float) Time step of the simulation. """

        self.activation: Union[State, np.ndarray] = State(
            shape=shape, init_func=lambda s: np.zeros(s[-1])
        )
        """ (np.ndarray) Internal activation of each unit. """

        self.activation_fun: Union[
            SimulationParameter, Callable[[float], float]
        ] = SimulationParameter(Partial(activation_fun))
        """ (Callable[[np.ndarray], np.ndarray]) Activation function of this module. """

        self._init_args = {
            "activation_fun": Partial(activation_fun),
        }

    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[Any, Any, Any]:
        # - Compute alpha
        alpha = np.exp(-self.dt / self.tau)

        # - Define forward dynamics
        def forward(activation, input_t):
            # - Integrate the synaptic input
            activation = activation * alpha + input_t

            # - Perform the softmax and return
            return (
                activation,
                (activation, self.activation_fun(activation)),
            )

        # - Evolve dynamics
        new_activation, (activation_ts, output) = jax.lax.scan(
            forward, self.activation, input_data
        )

        # - Return outputs, new state and recorded state
        return output, {"activation": new_activation}, {"activation": activation_ts}
