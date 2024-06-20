"""
Implement a linear module, using a Torch backend
"""

import math
import warnings
from typing import Union, Optional, Callable
import numpy as np
from rockpool.nn.modules.torch.torch_module import TorchModule
from rockpool.graph import GraphModuleBase, LinearWeights, as_GraphHolder
from rockpool.typehints import P_tensor

import torch
import torch.nn.init as init
import torch.nn.functional as F
import rockpool.parameters as rp

__all__ = ["LinearTorch"]

# - Define a float / array type
FloatVector = Union[float, np.ndarray, torch.Tensor]


class LinearTorch(TorchModule):
    """
    Applies a linear transformation to the incoming data: :math:`y = xA + b`

    This module supports `TensorFloat32`.

    Examples::
        >>> m = LinearTorch((20, 30))
        >>> input = torch.randn(128, 20)
        >>> output, _, _ = m(input)
        >>> print(output.size())
        torch.Size([128, 30])

        >>> m = LinearTorch((2, 3), has_bias = False)
        >>> m.parameters()
        {'weight': tensor([[ 0.6433, -0.7139, -0.2954],
         [ 0.9858,  0.3872,  0.6614]])}
    """

    def __init__(
        self,
        shape: tuple,
        weight=None,
        bias=None,
        has_bias: bool = False,
        weight_init_func: Callable = lambda s: init.kaiming_uniform_(
            torch.empty((s[1], s[0]))
        ).T,
        bias_init_func: Callable = lambda s: init.uniform_(
            torch.empty(s[-1]),
            -math.sqrt(1 / s[0]),
            math.sqrt(1 / s[0]),
        ),
        *args,
        **kwargs,
    ) -> None:
        """
        Initialise a LinearTorch layer

        Warnings:
            Standard DNN libraries by default include a bias on linear layers. These are usually not used for SNNs, where the bias is configured on the spiking neuron module. :py:class:`.Linear` layers in Rockpool use a default of ``has_bias = False``. You can force the presence of a bias on the linear layer with ``has_bias = True`` on initialisation.

        Examples:

            Build a linear weight matrix with shape ``(3, 4)``, with no biases:

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
        # - Initialise superclass
        super().__init__(shape=shape, *args, **kwargs)

        # - Check arguments
        if len(self.shape) != 2:
            raise ValueError(
                "`shape` must specify input and output sizes for LinearTorch."
            )

        # - Set up parameters
        w_rec_shape = (self.size_in, self.size_out)
        self.weight: P_tensor = rp.Parameter(
            torch.as_tensor(weight) if weight is not None else None,
            shape=w_rec_shape,
            init_func=weight_init_func,
            family="weights",
        )
        """ (torch.Tensor) Weight matrix with shape ``(Nin, Nout)`` """

        if has_bias:
            self.bias: Union[torch.Tensor, rp.Parameter] = rp.Parameter(
                bias,
                shape=[(self.size_out,), ()],
                init_func=bias_init_func,
                family="biases",
            )
            """ (torch.Tensor) Bias vector with shape ``(Nout,)`` """
        else:
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input, _ = self._auto_batch(input)

        return (
            F.linear(
                input,
                self.weight.T,
                self.bias,
            )
            if self.bias is not None
            else F.linear(input, self.weight.T)
        )

    def _extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.shape[0], self.shape[1], self.bias is not None
        )

    def as_graph(self) -> GraphModuleBase:
        return LinearWeights._factory(
            self.size_in,
            self.size_out,
            f"{type(self).__name__}_{self.name}_{id(self)}",
            self,
            self.weight,
            self.bias,
        )
