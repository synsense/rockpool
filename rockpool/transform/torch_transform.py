"""
Defines the parameter and activation transformation-in-training pipeline for `TorchModule` s
"""

from rockpool.utilities.backend_management import torch_version_satisfied

if not torch_version_satisfied(1, 12):
    raise ModuleNotFoundError(
        "torch version 1.12.0 or greater is required. The `torch_transform` package is not available."
    )

import torch

from rockpool.nn.modules.module import Module, ModuleBase
from rockpool.nn.modules.torch.torch_module import TorchModule
from rockpool.typehints import Tensor, P_Callable, Tree
from typing import Optional, Tuple, List, Callable, Dict

from rockpool.graph import GraphModuleBase

import copy

import rockpool.utilities.tree_utils as tu

__all__ = [
    "stochastic_rounding",
    "stochastic_channel_rounding",
    "deterministic_rounding",
    "dropout",
    "make_param_T_config",
    "make_param_T_network",
    "make_backward_passthrough",
    "make_act_T_config",
    "make_act_T_network",
]


def make_backward_passthrough(function: Callable) -> Callable:
    """
    Wrap a function to pass the gradient directly through in the backward pass

    Args:
        function (Callable): A function to wrap

    Returns:
        Callable: A function wrapped in the backward pass
    """

    class Wrapper(torch.autograd.Function):
        """A torch.autograd.Function that wraps a function with a pass-through gradient."""
        @staticmethod
        def forward(self, x):
            self.save_for_backward(x)
            return function(x)

        @staticmethod
        def backward(self, grad_output):
            grad_x = grad_output if self.needs_input_grad[0] else None
            return grad_x

    return Wrapper.apply


# - Make a passthrough version of the floor function
floor_passthrough = make_backward_passthrough(torch.floor)
round_passthrough = make_backward_passthrough(torch.round)


def stochastic_rounding(
    value: Tensor,
    input_range: Optional[List] = None,
    output_range: Optional[List] = None,
    num_levels: int = 2**8,
    maintain_zero: bool = True,
):
    """
    Perform floating-point stochastic rounding on a tensor, with detailed control over quantisation levels
    
    Stochastic rounding randomly pushes values up or down probabilistically, depending on the original value.
    Values will round with greater probability to their nearest quantised level, and with lower probability to their next-nearest quantised level.
    
    For example, if we are rounding to integers, then a value of 0.1 will round down to 0.0 with 90% probability; it will round to 1.0 with 10% probability.
    
    If we are rounding to arbitrary floating point levels, then the same logic holds, but the quantised output values will not be round numbers.

    :py:func:`stochastic_rounding` permits the input space to be re-scaled during rounding to an output space, which will be quantised over a specified number of quantisation levels (``2**8`` by default).
    By default, the input and output space are defined to be the full input range from minimum to maximum.
    
    :py:func:`stochastic_rounding` permits careful handling of symmetric or asymmetric spaces.
    By default, values of zero in the input space will map to zero in the output space (i.e. ``maintain_zero = True``).
    In this case the output range is defined as ``max(abs(input_range)) * [-1, 1]``.
    
    Quantisation and rounding is always to equally-spaced levels.
    
    Examples
    
        >>> stochastic_rounding(torch.tensor([-1., -0.5, 0., 0.5, 1.]), num_levels = 3)
        tensor([-1.,  0.,  0.,  1.,  1.])
        
        >>> stochastic_rounding(torch.tensor([-1., -0.5, 0., 0.5, 1.]), num_levels = 3)
        tensor([-1., -1.,  0.,  0.,  1.])
        
        Quantise to round integers over a defined space, changing the input scale (0..1) to (0..10).
        
        >>> stochastic_rounding(torch.rand(10), input_range = [0., 1.], output_range = [0., 10.], num_levels = 11)
        tensor([1., 9., 2., 2., 7., 1., 7., 9., 6., 1.])
        
        Quantise to floating point levels, without changing the scale of the values.
        >>> stochastic_rounding(torch.rand(10)-.5, num_levels = 3)
        tensor([ 0.0000,  0.0000,  0.0000, -0.4701, -0.4701,  0.4701, -0.4701, -0.4701,
         0.0000,  0.4701])
         
        >>> stochastic_rounding(torch.rand(10)-.5, num_levels = 3)
        tensor([ 0.0000,  0.0000, -0.4316,  0.0000,  0.0000,  0.4316, -0.4316,  0.0000,
         0.4316,  0.4316])
         
         Note that the scale is defined by the observed range of the random values, in this case.
    
    Args:
        value (torch.Tensor): A Tensor of values to round 
        input_range (Optional[List]): If defined, a specific input range to use (``[min_value, max_value]``), as floating point numbers. If ``None`` (default), use the range of input values to define the input range.
        output_range (Optional[List]): If defined, a specific output range to use (``[min_value, max_value]``), as floating point numbers. If ``None`` (default), use the range of input values to define the input range.
        num_levels (int): The number of output levels to quantise to (Default: ``2**8``)  
        maintain_zero (bool): Iff ``True``, ensure that input values of zero map to zero in the output space (Default: ``True``). If ``False``, the output range may shift zero w.r.t. the input range.

    Returns:
        torch.Tensor: Floating point stochastically rounded values
    """
    if maintain_zero:
        # - By default, input range is whatever the current data range is
        max_range = torch.max(torch.abs(value))
        input_range = (
            [-max_range, max_range] if input_range is None else list(input_range)
        )

        # - By default, the output range is the same as the input range
        output_range = input_range if output_range is None else list(output_range)

    else:
        # - By default, input range is whatever the current data range is
        input_range = (
            [torch.min(value), torch.max(value)]
            if input_range is None
            else list(input_range)
        )

        # - By default, the output range is the same as the input range
        output_range = input_range if output_range is None else list(output_range)

    # - Compute input and output quanta
    input_quantum = (input_range[1] - input_range[0]) / (num_levels - 1)
    output_quantum = (output_range[1] - output_range[0]) / (num_levels - 1)

    # - Perform quantisation
    levels = (value - input_range[0]) / input_quantum
    levels_floor = floor_passthrough(levels)
    levels_round = levels_floor + (
        (levels - levels_floor) > torch.rand(*value.shape).to(levels_floor.device)
    )

    output_param = levels_round * output_quantum + output_range[0]

    return output_param


def stochastic_channel_rounding(
    value: Tensor,
    output_range: List[float],
    num_levels: int = 2**8,
    maintain_zero: bool = True,
):
    """
    Perform stochastic rounding of a matrix, but with the input range defined automatically for each column independently
    
    This function performs the same quantisation approach as :py:func:`stochastic_rounding`, but considering each column of a matrix independently. ie. per-channel.
    
    Args:
        value (torch.Tensor): A tensor of values to quantise 
        output_range (List[float]): Defines the destination quantisation space ``[min_value, max_value]``. 
        num_levels (int): The number of quantisation values to round to (Default: ``2**8``) 
        maintain_zero (bool): Iff ``True`` (default), input values of zero map to zero in the output space. If ``False``, the output space may shift w.r.t. the input space. Note that the output space must be symmetric for the zero mapping to work as expected.

    Returns:
        torch.Tensor: The rounded values
    """
    def round_vector(vector: Tensor):
        if maintain_zero:
            max_range = torch.max(torch.abs(vector))
            input_range = [-max_range, max_range]
        else:
            input_range = [torch.min(vector), torch.max(vector)]

        # - Compute input and output quanta
        input_quantum = (input_range[1] - input_range[0]) / (num_levels - 1)
        output_quantum = (output_range[1] - output_range[0]) / (num_levels - 1)

        # - Perform quantisation
        levels = (vector - input_range[0]) / input_quantum
        levels_floor = floor_passthrough(levels)
        levels_round = levels_floor + (
            (levels - levels_floor) > torch.rand(*vector.shape)
        )
        output_param = levels_round * output_quantum + output_range[0]

        return output_param

    for i in range(value.shape[1]):
        value[:, i] = round_vector(value[:, i])

    return value


def deterministic_rounding(
    value: Tensor,
    input_range: Optional[List] = None,
    output_range: Optional[List] = None,
    num_levels: int = 2**8,
    maintain_zero: bool = True,
):
    """
    Quantise values by shifting them to the closest quantisation level
    
    This is a floating-point equivalent to standard integer rounding (e.g. using ``torch.round()``).
    :py:func:`deterministic_rounding` provides fine control over input and output spaces, as well as numbers of levels to quantise to, and can round to floating point levels instead of round numbers.
    :py:func:`deterministic_rounding` always leaves values as floating point.
    
    For example, if we are rounding to integers, then a value of ``0.1`` will round down to ``0.``.
    A value of ``0.5`` will round up to ``1.``.
    A value of ``0.9`` will round up to ``1.``.
    
    If we are rounding to arbitrary floating point levels, then the same logic holds, but the quantised output values will not be round numbers, but will be the nearest floating point quantisation level.

    :py:func:`deterministic_rounding` permits the input space to be re-scaled during rounding to an output space, which will be quantised over a specified number of quantisation levels (``2**8`` by default).
    By default, the input and output space are defined to be the full input range from minimum to maximum.
    
    :py:func:`deterministic_rounding` permits careful handling of symmetric or asymmetric spaces.
    By default, values of zero in the input space will map to zero in the output space (i.e. ``maintain_zero = True``).
    In this case the output range is defined as ``max(abs(input_range)) * [-1, 1]``.
    
    Quantisation and rounding is always to equally-spaced levels.
    
    Examples
    
        >>> deterministic_rounding(torch.tensor([-1., -0.5, 0., 0.5, 1.]), num_levels = 3)
        tensor([-1.,  -1.,  0.,  1.,  1.])
        
        >>> deterministic_rounding(torch.tensor([-1., -0.5, 0., 0.5, 1.]), num_levels = 3)
        tensor([-1., -1.,  0.,  0.,  1.])
        
        Round to integer values (-10..10)
        
        >>> deterministic_rounding(torch.rand(10)-.5, output_range=[-10., 10.], num_levels = 21)
        tensor([ 10.,  -3.,   3., -10.,   5.,   0.,   9.,  -5.,   7.,   8.])
        
        value (torch.Tensor): A Tensor of values to round 
        input_range (Optional[List]): If defined, a specific input range to use (``[min_value, max_value]``), as floating point numbers. If ``None`` (default), use the range of input values to define the input range.
        output_range (Optional[List]): If defined, a specific output range to use (``[min_value, max_value]``), as floating point numbers. If ``None`` (default), use the range of input values to define the input range.
        num_levels (int): The number of output levels to quantise to (Default: ``2**8``)  
        maintain_zero (bool): Iff ``True``, ensure that input values of zero map to zero in the output space (Default: ``True``). If ``False``, the output range may shift zero w.r.t. the input range.

    Returns:
        torch.Tensor: Floating point rounded values
    """
    if maintain_zero:
        # - By default, input range is whatever the current data range is
        max_range = torch.max(torch.abs(value))
        input_range = (
            [-max_range, max_range] if input_range is None else list(input_range)
        )

        # - By default, the output range is the same as the input range
        output_range = input_range if output_range is None else list(output_range)

    else:
        # - By default, input range is whatever the current data range is
        input_range = (
            [torch.min(value), torch.max(value)]
            if input_range is None
            else list(input_range)
        )

        # - By default, the output range is the same as the input range
        output_range = input_range if output_range is None else list(output_range)

    # - Compute input and output quanta
    input_quantum = (input_range[1] - input_range[0]) / (num_levels - 1)
    output_quantum = (output_range[1] - output_range[0]) / (num_levels - 1)

    # - Perform quantisation
    levels_round = round_passthrough((value - input_range[0]) / input_quantum)
    output_param = levels_round * output_quantum + output_range[0]

    return output_param


def dropout(param: Tensor, dropout_prob: float = 0.5):
    """
    Randomly set values of a tensor to ``0.``, with a defined probability
    
    Dropout is used to improve the robustness of a network, by reducing the dependency of a network on any given parameter value.
    This is accomplished by randomly setting parameters (usually weights) to zero during training, such that the parameters are ignored.
    
    For a ``dropout_prob = 0.8``, each parameters is randomly set to zero with 80\% probability.
    
    Examples
    
        >>> dropout(torch.ones(10))
        tensor([0., 0., 0., 1., 0., 1., 1., 1., 1., 0.])
    
        >>> dropout(torch.ones(10), dropout_prob = 0.8)
        tensor([1., 0., 0., 0., 0., 0., 0., 1., 0., 0.])
    
    Args:
        param (torch.Tensor): A tensor of values to dropout
        dropout_prob (float): The probability of zeroing each parameter value (Default: ``0.5``, 50\%)

    Returns:
        torch.Tensor: The tensor of values 
    """
    mask = torch.rand(param.shape, device=param.device) > dropout_prob
    return param * mask


class TWrapper(TorchModule):
    """
    A wrapper for a Rockpool TorchModule, implementing a parameter transformation in the forward pass 
    """
    def __init__(
        self, mod: TorchModule, T_config: Optional[Tree] = None, *args, **kwargs
    ):
        """
        Initialise a parameter transformer wrapper module
        
        
        Args:
            mod: 
            T_config: 
            *args: 
            **kwargs: 
        """
        # - Initialise Module superclass
        super().__init__(*args, shape=mod.shape, **kwargs)

        # - Copy module attributes
        self._name = mod._name
        self._mod = mod
        self._spiking_input = mod.spiking_input
        self._spiking_output = mod.spiking_output

        # - Generate null transformation config
        attributes, _ = self._mod._get_attribute_registry()
        T_config_null = {k: None for k in attributes.keys()}

        # - Default: null transformation config
        T_config = {} if T_config is None else T_config

        # - Update transformation config to cover all attributes
        self._T_config = T_config_null
        self._T_config.update(T_config)

    def forward(self, *args, **kwargs):
        # - Get transformed attributes
        transformed_attrs = self._T()

        # - Call module with torch functional API
        out = torch.nn.utils.stateless.functional_call(
            self._mod, transformed_attrs, *args, **kwargs
        )

        if not self._has_torch_api:
            return out[0]
        else:
            return out

    def as_graph(self) -> GraphModuleBase:
        return self._mod.as_graph()

    def _T(self):
        # - Transform parameters
        return {
            k: T_fn(getattr(self._mod, k))
            if T_fn is not None
            else getattr(self._mod, k)
            for k, T_fn in self._T_config.items()
        }

    def apply_T(self) -> TorchModule:
        # - Get transformed attributes
        transformed_attrs = self._T()
        self._mod = self._mod.set_attributes(transformed_attrs)
        return self


def make_param_T_config(
    net: ModuleBase, T_fn: Callable, param_family: Optional[str] = None
):
    return tu.tree_map(net.parameters(param_family), lambda _: T_fn)


def make_param_T_network(
    net: ModuleBase, T_config_tree: Tree, inplace: bool = False
) -> TorchModule:
    if not inplace:
        net = copy.deepcopy(net)

    if len(net.modules()) == 0:
        net = TWrapper(net, T_config_tree)
    else:
        _, modules = net._get_attribute_registry()

        for k, mod in modules.items():
            if k in T_config_tree:
                setattr(net, k, make_param_T_network(mod[0], T_config_tree[k]))

    return net


def apply_T(net: TorchModule, inplace: bool = False) -> TorchModule:
    if not inplace:
        net = copy.deepcopy(net)

    if isinstance(net, TWrapper):
        net = net.apply_T()

    _, modules = net._get_attribute_registry()

    if len(net.modules()) > 0:
        for k, mod in modules.items():
            setattr(net, k, apply_T(mod[0]))

    return net


def remove_T_net(net: TorchModule, inplace: bool = False) -> TorchModule:
    if not inplace:
        net = copy.deepcopy(net)

    if isinstance(net, TWrapper):
        net._mod._name = net.name
        net = net._mod

    else:
        _, modules = net._get_attribute_registry()

        for k, mod in modules.items():
            setattr(net, k, remove_T_net(mod[0]))

    return net


class ActWrapper(TorchModule):
    def __init__(
        self,
        mod: TorchModule,
        trans_Fn: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        # - Initialise superclass
        super().__init__(*args, shape=mod.shape, **kwargs)

        # - Record module attributes
        self._name = mod._name
        self._mod = mod
        self._spiking_input = mod.spiking_input
        self._spiking_output = mod.spiking_output

        # - Record transformation function
        self._trans_Fn = (lambda x: x) if trans_Fn is None else trans_Fn

    def forward(self, *args, **kwargs):
        out = self._mod(*args, **kwargs)

        if self._mod._has_torch_api:
            return self._trans_Fn(out)
        else:
            return self._trans_Fn(out[0])

    def as_graph(self) -> GraphModuleBase:
        return self._mod.as_graph()


def make_act_T_config(
    net: TorchModule, T_fn: Callable = None, ModuleClass: type = None
) -> Dict:
    # - Define a transformation function for this module, optionally matching a Module class
    if ModuleClass is not None:
        act_T_config = {"": T_fn} if isinstance(net, ModuleClass) else {"": None}
    else:
        act_T_config = {"": T_fn}

    if len(net.modules()) > 0:
        for k, mod in net.modules().items():
            act_T_config[k] = make_act_T_config(mod, T_fn, ModuleClass)

    return act_T_config


def make_act_T_network(
    net: TorchModule, act_T_config: Tree, inplace: bool = False
) -> TorchModule:
    if not inplace:
        net = copy.deepcopy(net)

    if len(net.modules()) == 0:
        if "" in act_T_config and act_T_config[""] is not None:
            net = ActWrapper(net, act_T_config[""])
    else:
        _, modules = net._get_attribute_registry()

        for k, mod in modules.items():
            if k in act_T_config:
                setattr(net, k, make_act_T_network(mod[0], act_T_config[k]))

    return net
