"""
Defines the parameter and activation transformation-in-training pipeline for `TorchModule` s

See Also:
    :ref:`/advanced/QuantTorch.ipynb`

Examples:
    Construct a network, and patch it to round each weight parameter:

    >>> net = Sequential(...)
    >>> T_fn = lambda p: stochastic_rounding(p, num_levels = 2**num_bits)
    >>> T_config = make_param_T_config(net, T_fn, 'weights')
    >>> T_net = make_param_T_network(net, T_config)

    Train here. To burn-in and remove the transformations:

    >>> burned_in_net = apply_T(T_net)
    >>> unpatched_net = remove_T_net(burned_in_net)

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
    "int_quant",
    "t_decay",
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


def int_quant(
    value: Tensor,
    maintain_zero: bool = True,
    map_to_max: bool = False,
    n_bits: int = 8,
    max_n_bits: int = 8,
):
    """
    Transforms a tensor to a quantized space with a range of integer values defined by n_bits

    Examples

        >>> int_quant(torch.tensor([-1, -0.50, 0.25, 0, 0.25, 0.5, 1]), n_bits = 3,  map_to_max = False)
        tensor([-3., -2., -1.,  0.,  1.,  2.,  3.])
        >>> int_quant(torch.tensor([-1, -0.50, 0.25, 0, 0.25, 0.5, 1]), n_bits = 3, map_to_max = True)
        tensor([-127.,  -64.,   32.,    0.,   32.,   64.,  127.])
    Args:
        value (torch.Tensor): A Tensor of values to be quantized
        maintain_zero (bool): If ``True``, ensure that input values of zero map to zero in the output space (Default: ``True``). If ``False``, the output range may shift zero w.r.t. the input range.
        map_to_max (bool): If True the integer values in the quantized output are mapped to the extremes of bitdepth
        nbits (int): defines the maximum integer in the quantized tensor
        max_n_bits (int): maximum allowed bitdepth


    Returns: torch.tensor: (q_value) quantized values
    """
    if not maintain_zero:
        l = value.max() - value.min()
        value = value - (l / 2)

    max_value = torch.max(torch.abs(value))
    max_value_quant = 2 ** (n_bits - 1) - 1

    if max_value != 0:
        scale = max_value_quant / max_value
    else:
        scale = 1

    if map_to_max:
        max_range = 2 ** (max_n_bits - 1) - 1
        scale *= max_range / max_value_quant

    q_value = round_passthrough(scale * value)

    return q_value


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
        tensor([ 0.0000,  0.0000,  0.0000, -0.4701, -0.4701,  0.4701, -0.4701, -0.4701, 0.0000,  0.4701])

        >>> stochastic_rounding(torch.rand(10)-.5, num_levels = 3)
        tensor([ 0.0000,  0.0000, -0.4316,  0.0000,  0.0000,  0.4316, -0.4316,  0.0000, 0.4316,  0.4316])

        Note that the scale is defined by the observed range of the random values, in this case.

    Args:
        value (torch.Tensor): A Tensor of values to round
        input_range (Optional[List]): If defined, a specific input range to use (``[min_value, max_value]``), as floating point numbers. If ``None`` (default), use the range of input values to define the input range.
        output_range (Optional[List]): If defined, a specific output range to use (``[min_value, max_value]``), as floating point numbers. If ``None`` (default), use the range of input values to define the input range.
        num_levels (int): The number of output levels to quantise to (Default: ``2**8``)
        maintain_zero (bool): If ``True``, ensure that input values of zero map to zero in the output space (Default: ``True``). If ``False``, the output range may shift zero w.r.t. the input range.

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

    Examples:

        >>> dropout(torch.ones(10))
        tensor([0., 0., 0., 1., 0., 1., 1., 1., 1., 0.])

        >>> dropout(torch.ones(10), dropout_prob = 0.8)
        tensor([1., 0., 0., 0., 0., 0., 0., 1., 0., 0.])

    Args:
        param (torch.Tensor): A tensor of values to dropout
        dropout_prob (float): The probability of zeroing each parameter value (Default: ``0.5``, 50\%)

    Returns:
        torch.Tensor: The tensor of values, with elements dropped out probabilistically
    """
    mask = torch.rand(param.shape, device=param.device) > dropout_prob
    return param * mask


class TWrapper(TorchModule):
    """
    A wrapper for a Rockpool TorchModule, implementing a parameter transformation in the forward pass

    This module is not designed to be be user-facing; you should probably use the helper functions :py:func:`make_param_T_config` and :py:func:`make_param_T_network` to patch a Rockpool network. This will insert :py:class:`TWrapper` modules into the network architecture as required.

    See Also:
        :ref:`/advanced/QuantTorch.ipynb`
    """

    def __init__(
        self, mod: TorchModule, T_config: Optional[Tree] = None, *args, **kwargs
    ):
        """
        Initialise a parameter transformer wrapper module

        ``mod`` is a Rockpool module with some set of attributes. ``T_config`` is a dictionary, with keys optionally matching the attributes of ``mod``. Each value must be a callable ``T_Fn(a) -> a`` which can transform the associated attribute ``a``.

        A :py:class:`.TWrapper` module will be created, with ``mod`` as a sub-module. The :py:class:`.TWrapper` will apply the specified transformations to all the attributes of ``mod`` at the beginning of the forward-pass of evolution, then evolve ``mod`` with the transformed attributes.

        Users should use the helper functions :py:func:`.make_param_T_config` and :py:func:`.make_param_T_network`.

        See Also:
            :ref:`/advanced/QuantTorch.ipynb`

        Args:
            mod (TorchModule): A Rockpool module to apply parameter transformations to
            T_config (Optional[Tree]): A nested dictionary specifying which transofmration transformations to apply to specific parameters. Each transformation function must be specified as a Callable with a key identical to a parameter of ``mod``. If ``None``, do not apply any transformation to ``mod``.
        """
        # - Initialise Module superclass
        super().__init__(*args, shape=mod.shape, **kwargs)

        # - Copy module attributes
        self._name = mod._name
        self._mod = mod
        self._spiking_input = mod.spiking_input
        self._spiking_output = mod.spiking_output

        # - Default: null transformation config
        self._T_config = {} if T_config is None else T_config

    def forward(self, *args, **kwargs):
        # - Get transformed attributes
        transformed_attrs = self._T()

        if self._record and not self._has_torch_api:
            kwargs["record"] = self._record

        # - Call module with torch functional API
        out = torch.nn.utils.stateless.functional_call(
            self._mod, transformed_attrs, args, kwargs
        )

        if not self._has_torch_api:
            self._record_dict = out[2]
        elif hasattr(self._mod, "_record_dict"):
            self._record_dict = self._mod._record_dict

        if not self._has_torch_api:
            return out[0]
        else:
            return out

    def as_graph(self) -> GraphModuleBase:
        return self._mod.as_graph()

    def _T(self):
        # - Transform parameters
        return {
            k: (
                T_fn(getattr(self._mod, k))
                if T_fn is not None
                else getattr(self._mod, k)
            )
            for k, T_fn in self._T_config.items()
        }

    def apply_T(self, inplace: bool = True) -> TorchModule:
        # - Get transformed attributes
        transformed_attrs = self._T()
        self._mod = self._mod.set_attributes(transformed_attrs)
        return self


def make_param_T_config(
    net: ModuleBase, T_fn: Callable, param_family: Optional[str] = None
) -> Tree:
    """
    Helper function to build parameter transformation configuration trees

    This function builds a parameter transformation nested configuration tree, based on an existing network ``net``.

    You can use :py:func:`.tree_utils.tree_update` to merge two configuration trees for different parameter families.

    The resulting configuration tree can be passed to :py:func:`.make_param_T_network` to patch the network ``net``.

    Examples:
        >>> T_config = make_param_T_config(net, lambda p: p**2, 'weights')
        >>> T_net = make_param_T_network(net, T_config)

    Args:
        net (Module): A Rockpool network to use as a template for the transformation configuration tree
        T_fn (Callable): A transformation function to apply to a parameter. Must have the signature ``T_fn(x) -> x``.
        param_family (Optional[str]): An optional argument to specify a parameter family. Only parameters matching this family within ``net`` will be specified in the configuration tree.
    """
    return tu.tree_map(net.parameters(param_family), lambda _: T_fn)


def make_param_T_network(
    net: ModuleBase, T_config_tree: Tree, inplace: bool = False
) -> TorchModule:
    """
    Patch a Rockpool network to apply parameter transformations in the forward pass

    This helper function inserts :py:class:`.TWrapper` modules into the network tree, where required, to apply transformations to each module as defined by a configuration tree ``T_config_tree``. Use the helper function :py:func:`.make_param_T_config` to build configuration trees.

    The resulting network will have analogous structure and behaviour to the original network, but the transformations will be applied before the forward pass of each module.

    Network parameters will remain "held" by the original modules, un-transformed.

    You can use the :py:func:`.remove_T_net` function to undo this patching behaviour, restoring the original network structure but keeping any parameter modifications (e.g. training) in place.

    You can use the :py:func:`.apply_T` function to "burn in" the parameter transformation.

    Args:
        net (Module): A Rockpool network to use as a template for the transformation configuration tree
        T_config_tree (Tree): A nested dictionary, mimicing the structure of ``net``, specifying which parameters should be transformed and which transformation function to apply to each parameter.
        inplace (bool): If ``False`` (default), a deep copy of ``net`` will be created, transformed and returned. If ``True``, the network will be patched in place.
    """
    if not inplace:
        net = copy.deepcopy(net)

    if len(net.modules()) == 0:
        # - Patch a single module
        net = TWrapper(net, T_config_tree)
    else:
        # - Patch a network tree or sub-tree
        #   Get a list of sub-modules
        _, modules = net._get_attribute_registry()

        for k, mod in modules.items():
            # - If there are transformations specified for this module
            if k in T_config_tree:
                # - Then recurse to patch the module
                setattr(
                    net,
                    k,
                    make_param_T_network(mod[0], T_config_tree[k], inplace=inplace),
                )

    return net


def apply_T(T_net: TorchModule, inplace: bool = False) -> TorchModule:
    """
    "Burn in" a set of parameter transformations, applying each transformation and storing the resulting transformed parameters

    This function takes a transformer-patched network ``net``, and applies the pre-specified transformations to each parameter. The resulting transformed parameters are then stored within the parameters of the network.

    This is a useful step **after** training, as part of extracting the transformed parameters from the trained network.

    The helper function :py:func:`.remove_T_net` can be used afterwards to remove the transformer patches from the network.

    Examples:
        >>> T_net = make_param_T_network(net, T_config)

        At this point, ``T_net.parameters()`` will return un-transformed parameters.

        >>> T_net.apply_T()

        Now ``T_net.parameters()`` will contain the results of applying the transformation to each parameter.

    Args:
        T_net (TorchModule): A transformer-patched network, obtained with :py:func:`make_param_T_network`
        inplace (bool): If ``False`` (default), a deep copy of ``net`` will be created, transformed and returned. If ``True``, the network will be transformed in place.
    """
    if not inplace:
        T_net = copy.deepcopy(T_net)

    if isinstance(T_net, TWrapper):
        T_net = T_net.apply_T(inplace=inplace)

    _, modules = T_net._get_attribute_registry()

    if len(T_net.modules()) > 0:
        for k, mod in modules.items():
            setattr(T_net, k, apply_T(mod[0], inplace=inplace))

    return T_net


def remove_T_net(T_net: TorchModule, inplace: bool = False) -> TorchModule:
    """
    Un-patch a transformed-patched network

    This function will iterate through a network patched using :py:func:`make_param_T_network`, and remove all patching modules. The resulting network should have the same network architecture as the original un-patched network.

    Any parameter values applied within ``T_net`` will be retained in the unpatched network.

    Args:
        T_net (TorchModule): A transformer-patched network, obtained with :py:func:`make_param_T_network`
        inplace (bool): If ``False`` (default), a deep copy of ``net`` will be created, transformed and returned. If ``True``, the network will be un-patched in place. Warning: in-place operation cannot work for single instances of :py:class:`TWrapper`

    Returns:
        TorchModule: A network matching ``T_net``, but with transformers removed.
    """
    if not inplace:
        T_net = copy.deepcopy(T_net)

    if isinstance(T_net, (TWrapper, ActWrapper)):
        T_net._mod._name = T_net.name
        T_net = T_net._mod

    else:
        _, modules = T_net._get_attribute_registry()

        for k, mod in modules.items():
            setattr(T_net, k, remove_T_net(mod[0], inplace=inplace))

    return T_net


class ActWrapper(TorchModule):
    """
    A wrapper module that applies an output activity transformation after evolution

    This module is not designed to be user-facing. Users should use the helper functions :py:func:`make_act_T_config` and :py:func:`make_act_T_network`. This approach will insert :py:class:`ActWrapper` modules into the network as required.

    See Also:
        :ref:`/advanced/QuantTorch.ipynb`
    """

    def __init__(
        self,
        mod: TorchModule,
        trans_Fn: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        """
        Instantiate an ActWrapper object

        ``mod`` is a Rockpool module. An :py:class:`ActWrapper` will be created to wrap ``mod``. The transformation function ``trans_Fn`` will be applied to the outputs of ``mod`` during evolution.

        See Also:
            :ref:`/advanced/QuantTorch.ipynb`

        Args:
            mod (TorchModule): A module to patch
            trans_Fn (Optional(Callable)): A transformation function to apply to the outputs of ``mod``. If ``None``, no transformation will be applied.
        """
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
    net: TorchModule,
    T_fn: Optional[Callable] = None,
    ModuleClass: Optional[type] = None,
) -> Tree:
    """
    Create an activity transformation configuration for a network

    This helper function assists in defining an activity transformation configuration tree. It allows to to search a predefined network ``net`` to find modules of a chosen class, and specify an activity transformation for those modules.

    ``net`` is a pre-defined Rockpool network.

    ``T_fn`` is a `Callable` with signature ``f(x) -> x``, transforming the output ``x`` of a module.

    ``ModuleClass`` optionally specifies the class of module to search for in ``net``.

    You can use :py:func:`.tree_utils.tree_update` to merge two configuration trees for different parameter families.

    The resulting configuration tree will match the structure of ``net`` (or will be a sub-tree of ``net``, including modules of type ``ModuleClass``). You can pass this configuration tree to :py:func:`.make_act_T_network` to build an activity transformer tree.

    Args:
        net (TorchModule): A Rockpool network to build a configuration tree for
        T_fn (Callable): A function ``f(x) -> x`` to apply as a transformation to module output. If ``None``, no transformation will be applied.
        ``ModuleClass`` (Optional[type]): A :py:class:`~.modules.Module` subclass to search for. The configuration tree will include only modules matching this class.

    Returns:
        Tree: An activity transformer configuration tree, to pass to :py:func:`.make_act_T_network`
    """
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
    """
    Patch a Rockpool network with activity transformers

    This helper function inserts :py:class:`ActWrapper` modules into a pre-defined network ``net``, to apply an activity transformation configuration ``act_T_config``.

    Args:
        net (TorchModule): A Rockpool network to patch
        act_T_config (Tree): A configuration tree from :py:func:`make_act_T_config`
        inplace (bool): If ``False`` (default), create a deep copy of ``net`` to patch. If ``True``, patch the network in place. This in place operation does not work when patching single modules.

    Returns:
        TorchModule: The patched network
    """
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


class class_calc_q_decay:
    """
    function used to calculate bitshift equivalent of decay (\exp(-dt/tau))

    Args:
    decay (torch.Tensor) : alpha and beta parameters from decay parameter family of LIF neurons
    dt (float) : Euler simulator time-step in seconds

    Returns:
    q_decay (torch.Tensor)

    """

    def __init__(self, dt):
        self.dt = dt

    def calc_bitshift_decay(self, decay):
        tau = -self.dt / torch.log(decay)

        bitsh = torch.round(torch.log2(tau / self.dt)).int()
        bitsh[bitsh < 0] = 0
        q_alpha = torch.tensor(1 - (1 / (2**bitsh)))

        return q_alpha

    def __call__(self, decay):
        return self.calc_bitshift_decay(decay)


def t_decay(decay: Tensor, dt: float = 1e-3):
    """
    quantizes decay factor (\exp (-dt/tau)) of LIF neurons: alpha and beta respectively for Vmem and Isyn
    the quantization is done based one converting the decay to bitshoft subtraction and reconstructing decay.
    the trasnformation is passed to make_backward_passthrough function that applies it in the forward pass of Torch module

    note: this transform can be used only if decay_training = True atleast one of LIF modules in teh network
    Examples:
        applying on tensors:
        >>> alpha = torch.rand(10)
        tensor([0.4156, 0.8306, 0.1557, 0.9475, 0.4532, 0.3585, 0.4014, 0.9380, 0.9426, 0.7212])
        >>> tt.t_decay(alpha)
        tensor([0.0000, 0.7500, 0.0000, 0.9375, 0.0000, 0.0000, 0.0000, 0.9375, 0.9375,
        0.7500])

        applying to decay parameter family of a network:

        >>> net = Sequential(LinearTorch((2,2)), LIFTorch(2, decay_training=True))
        >>> tconfig = tt.make_param_T_config(net, lambda p : tt.t_decay(p), 'decays')
        : {'1_LIFTorch': {'alpha': <function <lambda> at 0x7fb32efbb280>, 'beta': <function <lambda> at 0x7fb32efbb280>}}
        >>> T_net = make_param_T_network(net, T_config)
            : TorchSequential  with shape (2, 2) {
                LinearTorch '0_LinearTorch' with shape (2, 2)
                TWrapper '1_LIFTorch' with shape (2, 2) {
                    LIFTorch '_mod' with shape (2, 2) }}

    Args:
    decay (torch.Tensor) : alpha and beta parameters from decay parameter family of LIF neurons
    dt (float) : Euler simulator time-step in seconds

    Returns:
    q_decay (torch.Tensor)

    """
    fn = class_calc_q_decay(dt=dt)
    decay_passthrough = make_backward_passthrough(fn)
    q_decay = decay_passthrough(decay)
    return q_decay
