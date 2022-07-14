"""
Defines the parameter and activation transformation-in-training pipeline for `TorchModule` s
"""

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
    Wrap a function to pass the gradient directly through in teh backward pass

    Args:
        function (Callable): A function to wrap

    Returns:
        Callable: A function wrapped in the backward pass
    """

    class Wrapper(torch.autograd.Function):
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
    levels_round = levels_floor + ((levels - levels_floor) > torch.rand(*value.shape))
    output_param = levels_round * output_quantum + output_range[0]

    return output_param


def stochastic_channel_rounding(
    value: Tensor,
    output_range: Optional[List] = None,
    num_levels: int = 2**8,
    maintain_zero: bool = True,
):
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
    mask = torch.rand(param.shape, device=param.device) > dropout_prob
    return param * mask


class TWrapper(TorchModule):
    def __init__(
        self, mod: TorchModule, T_config: Optional[Tree] = None, *args, **kwargs
    ):
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
