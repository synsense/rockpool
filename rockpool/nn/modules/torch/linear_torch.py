"""
Implement a linear module, using a Torch backend
"""

from importlib import util

if util.find_spec("torch") is None:
    raise ModuleNotFoundError(
        "'Torch' backend not found. Modules that rely on Torch will not be available."
    )

import math
from typing import Union, Optional
import numpy as np
from rockpool.nn.modules.torch.torch_module import TorchModule
from rockpool.graph import GraphModuleBase, LinearWeights, as_GraphHolder
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
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> None:
        """
        Initialise a LinearTorch layer

        Args:
            shape (tuple): The shape of this layer ``(Nin, Nout)``
            weight (Tensor): Concrete initialisation data for the weights ``(Nin, Nout)``
            bias (Tensor): Concrete initialisation data for the biases ``(Nout,)``
            has_bias (bool): Iff ``True``, this layer includes a bias. Default: ``True``
            device (Optional[str]): Initialise the tensors on the supplied device.
            dtype (Optional[str]): Initialise the tensors with the supplied dtype.
        """
        # - Initialise superclass
        super().__init__(shape=shape)

        # - Check arguments
        if len(self.shape) != 2:
            raise ValueError(
                "`shape` must specify input and output sizes for LinearTorch."
            )

        # - Set up parameters
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight: Union[torch.Tensor, rp.Parameter] = rp.Parameter(
            weight,
            shape=shape,
            init_func=lambda s: init.uniform_(
                torch.empty(s, **factory_kwargs),
                -math.sqrt(2 / s[0]),
                math.sqrt(2 / s[0]),
            ),
            family="weights",
        )
        self.weight.requires_grad = True
        """ (torch.Tensor) Weight matrix with shape ``(Nin, Nout)`` """

        if has_bias:
            self.bias: Union[torch.Tensor, rp.Parameter] = rp.Parameter(
                bias,
                shape=shape[-1],
                init_func=lambda s: init.uniform_(
                    torch.empty(s[-1], **factory_kwargs),
                    -math.sqrt(2 / s[0]),
                    math.sqrt(2 / s[0]),
                ),
                family="biases",
            )
            """ (torch.Tensor) Bias vector with shape ``(Nout,)`` """
            self.bias.requires_grad = True
        else:
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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
        return as_GraphHolder(
            LinearWeights._factory(
                self.size_in,
                self.size_out,
                f"{type(self).__name__}_{self.name}_{id(self)}",
                self.weight,
            )
        )
