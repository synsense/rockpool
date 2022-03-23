"""
Implement a linear module, using a Torch backend
"""
import math
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

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

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
        has_bias: bool = True,
        weight_init_func: Optional[Callable] = lambda s: init.kaiming_uniform_(
            torch.empty((s[1], s[0]))
        ).T,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialise a LinearTorch layer

        Args:
            shape (tuple): The shape of this layer ``(Nin, Nout)``
            weight (Tensor): Concrete initialisation data for the weights ``(Nin, Nout)``
            bias (Tensor): Concrete initialisation data for the biases ``(Nout,)`` Default: ``0.0``
            has_bias (bool): Iff ``True``, this layer includes a bias. Default: ``True``
            weight_init_func (Callable): Random initialisation function for weights. Default: Kaiming initialisation
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
            weight,
            shape=w_rec_shape,
            init_func=weight_init_func,
            family="weights",
        )
        """ (torch.Tensor) Weight matrix with shape ``(Nin, Nout)`` """

        if has_bias:
            self.bias: Union[torch.Tensor, rp.Parameter] = rp.Parameter(
                bias,
                shape=[(self.size_out,), ()],
                init_func=lambda s: init.uniform_(
                    torch.empty(s[-1]),
                    -math.sqrt(1 / s[0]),
                    math.sqrt(1 / s[0]),
                ),
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
            self.weight.detach().numpy(),
        )
