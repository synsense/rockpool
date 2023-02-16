"""
Implements linear weight matrix modules
"""

from __future__ import annotations
from rockpool.nn.modules.module import Module
from rockpool.parameters import Parameter
from rockpool.graph import GraphModuleBase, LinearWeights, as_GraphHolder

import numpy as onp
from warnings import warn

from typing import Tuple, Any, Callable

from abc import ABC

__all__ = ["unit_eigs", "kaiming", "xavier", "Linear"]


def unit_eigs(s):
    return onp.random.randn(*s) / onp.sqrt(s[0])


def uniform_sqrt(s):
    lim = onp.sqrt(1 / s[0])
    return onp.random.uniform(-lim, lim, s)


def kaiming(s):
    lim = onp.sqrt(6 / s[0])
    return onp.random.uniform(-lim, lim, s)


def xavier(s):
    lim = onp.sqrt(6 / onp.sum(s))
    return onp.random.uniform(-lim, lim, s)


class LinearMixin(ABC):
    """
    Encapsulate a linear weight matrix
    """

    _dot = None

    def __init__(
        self,
        shape: tuple,
        weight=None,
        bias=None,
        has_bias: bool = False,
        weight_init_func: Callable = kaiming,
        bias_init_func: Callable = uniform_sqrt,
        *args,
        **kwargs,
    ):
        """
        Encapsulate a linear weight matrix, with optional biases

        `.Linear` wraps a single weight matrix, and passes data through by using the matrix as a set of weights. The shape of the matrix must be specified as a tuple ``(Nin, Nout)``. `.Linear` provides optional biases.

        A weight initialisation function may be specified. By default the weights will use Kaiming initialisation (:py:func:`.kaiming`).

        A bias initialisation function may be specified, if used. By default the biases will be initialised as uniform random over the range :math:`(-\\sqrt(1/N), \\sqrt(1/N))`.

        Warnings:
            Standard DNN libraries by default include a bias on linear layers. These are usually not used for SNNs, where the bias is configured on the spiking neuron module. :py:class:`.Linear` layers in Rockpool use a default of ``has_bias = False``. You can force the presence of a bias on the linear layer with ``has_bias = True`` on initialisation.

        Examples:

            Build a linear weight matrix with shape ``(3, 4)``, and no biases:

            >>> Linear((3, 4))
            Linear  with shape (3, 4)

            Build a linear weight matrix with shape ``(2, 5)``, which will be initialised with zeros:

            >>> Linear((2, 5), weight_init_func = lambda s: np.zeros(s))
            Linear  with shape (2, 5)

            Provide a concrete initialisation for the linear weights:

            >>> Linear((2, 2), weight = np.array([[1, 2], [3, 4]]))
            Linear  with shape (2, 2)

            Build a linear layer including biases:

            >>> mod = Linear((2, 2), has_bias = True)
            >>> mod.parameters()
            Out:
            {'weight': array([[ 0.56655314,  0.64411151],
                    [-1.43016068, -1.538719  ]]),
             'bias': array([-0.58513867, -0.32314069])}

        Args:
            shape (tuple): The desired shape of the weight matrix. Must have two entries ``(Nin, Nout)``
            weight_init_func (Callable): The initialisation function to use for the weights. Default: Kaiming initialization; uniform on the range :math:`(-\\sqrt(6/Nin), \\sqrt(6/Nin))`
            weight (Optional[np.array]): A concrete weight matrix to assign to the weights on initialisation. ``weight.shape`` must match the ``shape`` argument
            has_bias (bool): A boolean flag indicating that this linear layer should have a bias parameter. Default: ``False``, no bias parameter
            bias_init_func (Callable): The initialisation function to use for the biases. Default: Uniform / sqrt(N); uniform on the range :math:`(-\\sqrt(1/N), \\sqrt(1/N))`
            bias (Optional[np.array]): A concrete bias vector to assign to the biases on initialisation. ``bias.shape`` must be ``(N,)``
        """
        # - Base class must be `Module`
        if not isinstance(self, Module):
            raise TypeError(
                "`LinearMixin` mix-in class may only be used with `Module` classes."
            )

        # - Call superclass init
        super().__init__(shape=shape, *args, **kwargs)

        if len(self.shape) != 2:
            raise ValueError("`shape` must specify input and output sizes for Linear.")

        # - Specify weight parameter
        self.weight = Parameter(
            weight, shape=self.shape, init_func=weight_init_func, family="weights"
        )
        """ Weight matrix of this module """

        # - Specify bias parameter
        if has_bias or bias is not None:
            self.bias = Parameter(
                bias, shape=self.size_out, init_func=bias_init_func, family="biases"
            )
            """ Bias vector of this module """
            self._has_bias = True
        else:
            self.bias = 0
            self._has_bias = False

    def evolve(self, input_data, record: bool = False) -> Tuple[Any, Any, Any]:
        return self._dot(input_data, self.weight) + self.bias, {}, {}

    def as_graph(self) -> GraphModuleBase:
        return LinearWeights._factory(
            self.size_in,
            self.size_out,
            f"{type(self).__name__}_{self.name}_{id(self)}",
            self,
            self.weight,
            self.bias if self._has_bias else None,
        )

    @classmethod
    def from_graph(cls, graph: LinearWeights) -> LinearMixin:
        """
        from_graph constructs a LinearMixin object from the comptutational graph

        :param graph: the reference computational graph to restore computational module
        :type graph: LinearWeights
        :return: a LinearMixin object
        :rtype: LinearMixin
        """
        if not isinstance(graph, LinearWeights):
            graph = LinearWeights._convert_from(graph)

        return cls(
            shape=(len(graph.input_nodes), len(graph.output_nodes)),
            weight=graph.weights,
            bias=graph.biases,
            has_bias=False if graph.biases is None else True,
        )


class Linear(LinearMixin, Module):
    """
    Encapsulates a linear weight matrix
    """

    _dot = staticmethod(onp.dot)
    pass
