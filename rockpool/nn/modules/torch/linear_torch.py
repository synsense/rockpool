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

    Args:
        shape (tuple): ``(in_features, out_features)``
        bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of additional dimensions and :math:`H_{in} = \\text{in_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension are the same shape as the input and :math:`H_{out} = \\text{out_features}`.

    Attributes:
        weight: the learnable weights of the module of shape :math:`(\\text{out_features}, \\text{in_features})`. The values are initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`, where :math:`k = \\frac{2}{\\text{in_features}}`
        bias:   the learnable bias of the module of shape :math:`(\\text{out_features})`. If :attr:`bias` is ``True``, the values are initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where :math:`k = \\frac{2}{\\text{in_features}}`

    Examples::
        >>> m = LinearTorch((20, 30))
        >>> input = torch.randn(128, 20)
        >>> output, _, _ = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
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
        else:
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.T, self.bias)

    def _extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.shape[0], self.shape[1], self.bias is not None
        )
