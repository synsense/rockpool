"""
A spiking softmax module.
"""

from rockpool.nn.modules.jax.jax_module import JaxModule
from rockpool.nn.modules.native.linear import LinearJax
from rockpool.nn.modules.jax.exp_smooth_jax import ExpSmoothJax

import jax.numpy as np

from typing import Tuple, Any, Optional, Callable

__all__ = ["softmax", "logsoftmax", "SoftmaxJax", "LogSoftmaxJax"]


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Implements the softmax function

    .. math::

        S(x, \\tau) = \\exp(l / \\tau) / { \\Sigma { \\exp(l / \\tau)} }
        l = x - \\max(x)

    Args:
        x (np.ndarray): Input vector of scores
        temperature (float): Temperature :math:`\\tau` of the softmax. As :math:`\\tau \\rightarrow 0`, the function becomes a hard :math:``\\max`` operation.

    Returns:
        np.ndarray: The output of the softmax.
    """
    logits = x - np.max(x)
    eta = np.exp(logits / temperature)
    return eta / np.sum(eta)


def logsoftmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Efficient implementation of the log softmax function

    .. math ::

        log S(x, \\tau) = (l / \\tau) - \\log \\Sigma { \\exp (l / \\tau) }
        l = x - \\max (x)

    Args:
        x (np.ndarray): Input vector of scores
        temperature (float): Temperature :math:`\\tau` of the softmax. As :math:`\\tau \\rightarrow 0`, the function becomes a hard :math:``\\max`` operation.

    Returns:
        np.ndarray: The output of the logsoftmax.
    """
    logits = x - np.max(x)
    return (logits / temperature) - np.log(np.sum(np.exp(logits / temperature)))


class WeightedSmoothBase(JaxModule):
    """
    A weighted smoothing Jax-backed module.
    """

    def __init__(
        self,
        shape: Optional[tuple] = None,
        weight: Optional[np.ndarray] = None,
        tau: float = 100e-3,
        dt: float = 1e-3,
        activation_fun: Callable[[np.ndarray], np.ndarray] = lambda x: x,
        *args,
        **kwargs,
    ):
        """
        Initialise the module.

        Args:
            shape (Optional[tuple]): Defines the module shape ``(Nin, Nout)``. If not provided, the shape of ``weight`` will be used.
            weight (Optional[tuple]): Concrete initialisation data for the weights. If not provided, will be initialised to ``U[-sqrt(1 / Nin), sqrt(1 / Nin)]``.
            tau (float): Smoothing time constant :math:`\\tau`. Default: 100 ms.
            dt (float): Simulation tme-step in seconds. Default: 1 ms.
            activation_fun (Callable): Activation function to apply to each neuron
        """

        # - Check `shape` argument
        if shape is None:
            if weight is None:
                raise ValueError(
                    "One of `shape` or `weight` parameters must be provided."
                )

            shape = weight.shape

        # - Initialise super-class
        super().__init__(shape=shape, *args, **kwargs)

        # - Define the submodules
        self.linear = LinearJax(shape=shape, weight=weight)
        self.smooth = ExpSmoothJax(
            shape=(shape[-1],), tau=tau, dt=dt, activation_fun=activation_fun
        )

    def evolve(self, input_data, record: bool = False) -> Tuple[Any, Any, Any]:
        # - Initialise return dictionaries
        record_dict = {}
        new_state = {}

        # - Pass data through modules
        x, new_state["linear"], record_dict["linear"] = self.linear(input_data, record)
        x, new_state["smooth"], record_dict["smooth"] = self.smooth(x, record)

        # - Return data
        return x, new_state, record_dict


class SoftmaxJax(WeightedSmoothBase):
    """
    A Jax-backed module implementing a smoothed weighted softmax, compatible with spiking inputs

    This module implements synaptic dynamics::

        \\tau \dot{I}_{syn} + I_{syn} = i(t) \\cdot W

    The softmax function is given by::

        S(x, \\tau) = \\exp(l) / { \\Sigma { \\exp(l)} }
        l = x - \\max(x)

    and is applied to the synaptic currents :math:`I_{syn}`. Input weighting :math:`W` is provided, and the exponential smoothing kernel is paramterised by time constant :math:`\\tau`.
    """

    def __init__(
        self,
        shape: Optional[tuple] = None,
        weight: Optional[np.ndarray] = None,
        tau: float = 100e-3,
        dt: float = 1e-3,
        *args,
        **kwargs,
    ):
        """
        Initialise a soft-max module.

        Args:
            shape (Optional[tuple]): Defines the module shape ``(Nin, Nout)``. If not provided, the shape of ``weight`` will be used.
            weight (Optional[tuple]): Concrete initialisation data for the weights. If not provided, will be initialised to ``U[-sqrt(1 / Nin), sqrt(1 / Nin)]``.
            tau (float): Smoothing time constant :math:`\\tau`. Default: 100 ms.
            dt (float): Simulation tme-step in seconds. Default: 1 ms.
        """
        # - Initialise super-class
        super().__init__(
            shape=shape,
            weight=weight,
            tau=tau,
            dt=dt,
            activation_fun=lambda x: softmax(x),
            *args,
            **kwargs,
        )


class LogSoftmaxJax(JaxModule):
    """
    A Jax-backed module implementing a smoothed weighted softmax, compatible with spiking inputs

    This module implements synaptic dynamics::

        \\tau \dot{I}_{syn} + I_{syn} = i(t) \\cdot W

    The log softmax function is given by::

        log S(x, \\tau) = (l) - \\log \\Sigma { \\exp (l) }
        l = x - \\max (x)

    and is applied to the synaptic currents :math:`I_{syn}`. Input weighting :math:`W` is provided, and the exponential smoothing kernel is paramterised by time constant :math:`\\tau`.
    """

    def __init__(
        self,
        shape: Optional[tuple] = None,
        weight: Optional[np.ndarray] = None,
        tau: float = 100e-3,
        dt: float = 1e-3,
        *args,
        **kwargs,
    ):
        """
        Initialise a soft-max module.

        Args:
            shape (Optional[tuple]): Defines the module shape ``(Nin, Nout)``. If not provided, the shape of ``weight`` will be used.
            weight (Optional[tuple]): Concrete initialisation data for the weights. If not provided, will be initialised to ``U[-sqrt(1 / Nin), sqrt(1 / Nin)]``.
            tau (float): Smoothing time constant :math:`\\tau`. Default: 100 ms.
            dt (float): Simulation tme-step in seconds. Default: 1 ms.
        """
        # - Initialise super-class
        super().__init__(
            shape=shape,
            weight=weight,
            tau=tau,
            dt=dt,
            activation_fun=lambda x: logsoftmax(x),
            *args,
            **kwargs,
        )
