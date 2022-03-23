"""
Spiking softmax modules, with Jax backends.
"""

from rockpool.nn.modules.jax.jax_module import JaxModule
from rockpool.nn.modules.jax.linear_jax import LinearJax
from rockpool.nn.modules.jax.exp_syn_jax import ExpSynJax
from rockpool.training.jax_loss import softmax, logsoftmax

import jax.numpy as np
from jax.tree_util import Partial

from rockpool.parameters import SimulationParameter

from typing import Tuple, Any, Optional, Callable
from rockpool.typehints import P_Callable

__all__ = ["SoftmaxJax", "LogSoftmaxJax"]


class WeightedSmoothBase(JaxModule):
    """
    A weighted smoothing Jax-backed module.
    """

    def __init__(
        self,
        shape: Optional[tuple] = None,
        weight: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
        has_bias: bool = True,
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
            weight (Optional[np.ndarray]): Concrete initialisation data for the weights. If not provided, will be initialised to ``U[-sqrt(2 / Nin), sqrt(2 / Nin)]``.
            bias (Optonal[np.ndarray]): Concrete initialisation data for the biases. If not provided, will be initialised to ``U[-sqrt(2 / Nin), sqrt(2 / Nin)]``.
            has_bias (bool): Iff ``True``, the module will include a set of biases. Default: ``True``.
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
        self.linear = LinearJax(
            shape=shape, weight=weight, bias=bias, has_bias=has_bias
        )
        self.smooth = ExpSynJax(
            shape=(shape[-1],),
            tau=tau,
            dt=dt,
        )

        self.activation_fn: P_Callable = SimulationParameter(Partial(activation_fun))
        """ (Callable) Activation function """

    def evolve(self, input_data, record: bool = False) -> Tuple[Any, Any, Any]:
        # - Initialise return dictionaries
        record_dict = {}
        new_state = {}

        # - Pass data through modules
        x, new_state["linear"], record_dict["linear"] = self.linear(input_data, record)
        x, new_state["smooth"], record_dict["smooth"] = self.smooth(x, record)

        # - Return data
        return self.activation_fn(x), new_state, record_dict


class SoftmaxJax(WeightedSmoothBase):
    """
    A Jax-backed module implementing a smoothed weighted softmax, compatible with spiking inputs

    This module implements synaptic dynamics:

    .. math::

        \\tau \dot{I}_{syn} + I_{syn} = i(t) \\cdot W

    The softmax function is given by:

    .. math ::

        S(x, \\tau) = \\exp(l) / { \\Sigma { \\exp(l)} }

        l = x - \\max(x)

    and is applied to the synaptic currents :math:`I_{syn}`. Input weighting :math:`W` is provided, and the exponential smoothing kernel is paramterised by time constant :math:`\\tau`.
    """

    def __init__(
        self,
        shape: Optional[tuple] = None,
        weight: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
        has_bias: bool = True,
        tau: float = 100e-3,
        dt: float = 1e-3,
        *args,
        **kwargs,
    ):
        """
        Instantiate a soft-max module.

        Args:
            shape (Optional[tuple]): Defines the module shape ``(Nin, Nout)``. If not provided, the shape of ``weight`` will be used.
            weight (Optional[tuple]): Concrete initialisation data for the weights. If not provided, will be initialised using Kaiming initialization: :math:`W \sim U[\pm\sqrt(2 / N_{in})]`.
            bias (Optonal[np.ndarray]): Concrete initialisation data for the biases. If not provided, will be initialised to ``U[-sqrt(2 / Nin), sqrt(2 / Nin)]``.
            has_bias (bool): Iff ``True``, the module will include a set of biases. Default: ``True``.
            tau (float): Smoothing time constant :math:`\\tau`. Default: 100 ms.
            dt (float): Simulation tme-step in seconds. Default: 1 ms.
        """
        # - Initialise super-class
        super().__init__(
            shape=shape,
            weight=weight,
            bias=bias,
            has_bias=has_bias,
            tau=tau,
            dt=dt,
            activation_fun=lambda x: softmax(x),
            *args,
            **kwargs,
        )


class LogSoftmaxJax(WeightedSmoothBase):
    """
    A Jax-backed module implementing a smoothed weighted softmax, compatible with spiking inputs

    This module implements synaptic dynamics:

    .. math::

        \\tau \dot{I}_{syn} + I_{syn} = i(t) \\cdot W

    The log softmax function is given by:

    .. math::

        log S(x, \\tau) = (l) - \\log \\Sigma { \\exp (l) }

        l = x - \\max (x)

    and is applied to the synaptic currents :math:`I_{syn}`. Input weighting :math:`W` is provided, and the exponential smoothing kernel is paramterised by time constant :math:`\\tau`.
    """

    def __init__(
        self,
        shape: Optional[tuple] = None,
        weight: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
        has_bias: bool = True,
        tau: float = 100e-3,
        dt: float = 1e-3,
        *args,
        **kwargs,
    ):
        """
        Initialise a soft-max module.

        Args:
            shape (Optional[tuple]): Defines the module shape ``(Nin, Nout)``. If not provided, the shape of ``weight`` will be used.
            weight (Optional[tuple]): Concrete initialisation data for the weights. If not provided, will be initialised using Kaiming initialization: :math:`W \sim U[\pm\sqrt(2 / N_{in})]`.
            bias (Optonal[np.ndarray]): Concrete initialisation data for the biases. If not provided, will be initialised to ``U[-sqrt(2 / Nin), sqrt(2 / Nin)]``.
            has_bias (bool): Iff ``True``, the module will include a set of biases. Default: ``True``.
            tau (float): Smoothing time constant :math:`\\tau`. Default: 100 ms.
            dt (float): Simulation tme-step in seconds. Default: 1 ms.
        """
        # - Initialise super-class
        super().__init__(
            shape=shape,
            weight=weight,
            bias=bias,
            has_bias=has_bias,
            tau=tau,
            dt=dt,
            activation_fun=lambda x: logsoftmax(x),
            *args,
            **kwargs,
        )
