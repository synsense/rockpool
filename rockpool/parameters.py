"""
Classes to manage registered Module attributes in Rockpool
"""

from typing import Callable, Iterable, Any, Union, List, Tuple, Optional
from copy import deepcopy

from itertools import compress
from functools import partial

import numpy as np

__all__ = ["Parameter", "State", "SimulationParameter", "Constant"]


from rockpool.utilities.backend_management import backend_available

if backend_available("torch"):
    from torch import Tensor
else:

    class Tensor:
        pass


class RP_Constant:
    """
    Represent a concrete initialisation value as a constant parameter, which should not be trained


    See Also:
         Use :py:func:`Constant` to wrap an intialisation as a constant argument.
    """

    pass


def Constant(obj: Any) -> RP_Constant:
    """
    Identify an initialisation argument as a constant (non-trainable) parameter

    Examples
        >>> mod = LIFJax(1)
        >>> mod.parameters('taus')
        {'tau_mem': DeviceArray([0.02], dtype=float32),
         'tau_syn': DeviceArray([[0.02]], dtype=float32)}
        >>> mod.simulation_parameters('taus')
        {}

        >>> mod = LIFJax(1, tau_mem = Constant(10e-3))
        >>> mod.parameters('taus')
        {'tau_syn': DeviceArray([[0.02]], dtype=float32)}
        >>> mod.simulation_parameters('taus')
        {'tau_mem': DeviceArray(0.01, dtype=float32)}

    Args:
        obj (Any): The initialisation object to wrap

    Returns:
        A wrapped object, of the same class as ``obj``.
    """

    class ConstantPatch(obj.__class__, RP_Constant):
        pass

    ConstantPatch.__name__ = obj.__class__.__name__

    try:
        obj.__class__ = ConstantPatch
    except TypeError:
        if isinstance(obj, np.ndarray):
            obj = obj.view(ConstantPatch)
        else:
            obj = ConstantPatch(obj)

    return obj


# -- Parameter classes
class ParameterBase:
    """
    Base class for Rockpool registered attributes

    See Also:
        See :py:class:`.Parameter` for representing the configuration of a module, :py:class:`.State` for representing the transient internal state of a neuron or module, and :py:class:`.SimulationParameter` for representing simulation- or solver-specific parameters that are not important for network configuration.
    """

    def __init__(
        self,
        data: Any = None,
        family: str = None,
        init_func: Callable[[Any], Any] = None,
        shape: Optional[Union[List[Tuple], Tuple, int]] = None,
        permit_reshape: bool = True,
        cast_fn: Callable[[Any], Any] = lambda x: x,
    ):
        """
        Instantiate a Rockpool registered attribute

        Args:
            data (Optional[Any]): Concrete initialisation data for this attribute. The shape of ``data`` will specify the allowable shape of the attribute data, unless the ``shape`` argument is provided.
            family (Optional[str]): An arbitrary string to specify the "family" of this attribute. You should use ``'weights'``, ``'taus'``, ``'biases'`` if you can; otherwise you can use whatever you like. These are used by the :py:meth:`.Module.parameters`, :py:meth:`.Module.state` and :py:meth:`.Module.simulation_parameters` methods to group and select attributes.
            init_func (Optional[Callable]): A function that initialises this attributed. Called by :py:meth:`.Module.reset_parameters` and :py:meth:`.Module.reset_state`. The signature is ``f(shape: tuple) -> np.ndarray``.
            shape (Optional[Union[List[Tuple], Tuple, int]]): A list of permisable shapes for the parameter, or a tuple specifying the permitted shape, or an integer specifying the number of elements. If not provided, the shape of the concrete initialisation data will be used as the attribute shape. The first item in the list will be used as the concrete shape, if ``data`` is not provided and ``init_func`` should be used.
            permit_reshape (bool): If ``True``, the input data will be reshaped to a matching permitted shape. If ``False``, then an error will be raised if the shapes do not match exactly.
            cast_fn (Optional[Callable]): A function to call to cast the data for this parameter. Will only be called once on initialisation.
        """
        if data is None and shape is None:
            raise ValueError(f"One of `data` or `shape` must be provided.")

        # - Check type and configuration of `shape` argument
        if shape is not None:
            if not isinstance(
                shape,
                (List, Tuple, int),
            ):
                raise TypeError(
                    f"`shape` must be a list, a tuple or an integer. Instead `shape` was a {type(shape).__name__}."
                )

            # - Convert a single tuple to a list
            if isinstance(shape, (Tuple, int)):
                shape = [shape]

            # - Check each list element in turn
            for i, st in enumerate(shape):
                # - Convert non-tuples to tuples
                if not isinstance(st, tuple):
                    shape[i] = (st,)
                    st = shape[i]

                # - Check each element of each tuple
                for elem in st:
                    if not isinstance(elem, int):
                        raise TypeError(
                            f"All elements in a shape tuple must be integers. Instead I found an element of type {type(elem).__name__}."
                        )

        # - Assign attributes
        self.family: str = family
        self.data: Union[np.ndarray, Iterable, float, int] = data
        self.init_func: Callable = init_func
        self.shape: Optional[List] = shape
        self.cast_fn: Callable = cast_fn

        class_name = type(self).__name__

        # - Check that the initialisation function is callable
        if self.init_func is not None and not callable(self.init_func):
            raise ValueError(
                f"The `init_func` for a {class_name} must be a callable that accepts a shape tuple."
            )

        # - Force object to be a SimulationParameter, if training should be disabled
        if isinstance(self.data, RP_Constant):
            self.__class__ = SimulationParameter

        def numel(x):
            if isinstance(x, np.ndarray):
                return x.size
            elif isinstance(x, Tensor):
                return x.numel()
            else:
                return np.size(x)

        # - Get the shape from the data, if not provided explicitly
        if self.data is not None:
            if self.shape is not None:
                # - Check that the concrete data matches the shape
                if not any([np.shape(self.data) == st for st in self.shape]):
                    # - Check if the concrete and desired sizes match for any permitted shape
                    matching_sizes = [
                        numel(self.data) == int(np.prod(st)) for st in self.shape
                    ]

                    # - Can we reshape the concrete data to match a shape?
                    if not any(matching_sizes) or not permit_reshape:
                        raise ValueError(
                            f"The shape provided for this {class_name} does not match the provided initialisation data.\n"
                            + f"    self.shape = {self.shape}; data.shape = {np.shape(self.data)}"
                        )
                    elif permit_reshape and any(matching_sizes):
                        # - Reshape input data to first matching size
                        target_shape = list(compress(self.shape, matching_sizes))[0]
                        self.data = np.array(self.data).reshape(target_shape)

                self.shape = None

            if self.shape is None:
                # - Record the shape of the data as the concrete shape
                self.shape = np.shape(self.data)

        # - Initialise data, if not provided
        if self.data is None:
            # - Get the concrete shape to use (by default: first shape option in the list)
            self.shape = self.shape[0]

            if self.init_func is None:
                raise ValueError(
                    f"If concrete initialisation `data` is not provided for a {class_name} then `init_func` must be provided.\nParameter was {self.data, self.family, self.init_func, self.shape, self.cast_fn}"
                )

            # - Call the `init_func`
            self.data = self.init_func(self.shape)
        else:
            # - If concrete initialisation data is provided, then override the `init_func`
            data_copy = deepcopy(data)
            self.init_func = lambda _: data_copy

        # - Cast the data using the cast function
        if self.cast_fn is not None:
            self.data = self.cast_fn(self.data)

    def __repr__(self):
        return f"{type(self).__name__}(data={self.data}, family={self.family}, init_func={self.init_func}, shape={self.shape})"

    def _tree_flatten(self) -> Tuple[tuple, tuple]:
        """FLatten this parameter / state for Jax"""
        return (
            (
                self.data,
                self.family,
                partial(self.init_func),
                self.shape,
                partial(self.cast_fn),
            ),
            (),
        )

    @classmethod
    def _tree_unflatten(cls, _, children):
        """Unflatten a tree of parameter from Jax to Rockpool"""
        data, family, init_func, shape, cast_fn = children
        obj = cls(data=data, family=family, init_func=init_func)

        return obj


class Parameter(ParameterBase):
    """
    Represent a module parameter

    A :py:class:`.Parameter` in Rockpool is a configuration value that is important for communicating the configuration of a network. For example, network weights; network time constants; neuron biases; etc. These are likely to be your set of trainable parameters for a module or network.

    See Also:
        See :py:class:`.State` for representing the transient internal state of a neuron or module, and :py:class:`.SimulationParameter` for representing simulation- or solver-specific parameters that are not important for network configuration.
    """

    pass


class State(ParameterBase):
    """
    Represent a module state

    A :py:class:`.State` in Rockpool is a transient value which is required to maintain the dynamics of a stateful module. For example the membrane potential of a neuron; the synaptic current; the refractory state of a neuron; etc.

    See Also:
        See :py:class:`.Parameter` for representing the configuration of a module, and :py:class:`.SimulationParameter` for representing simulation- or solver-specific parameters that are not important for network configuration.
    """

    pass


class SimulationParameter(ParameterBase):
    """
    Represent a module simulation parameter

    A :py:class:`.SimulationParameter` in Rockpool is a simulation-specific configuration value, which is only needed to control the simulation of a network, but is **not** needed to communicate your network configuration to someone else. For example, the simulation time-step your solver uses to simulate the dynamics of a module. :py:class:`.SimulationParameter` s are basically never trainable parameters.

    See Also:
        See :py:class:`.Parameter` for representing the configuration of a module, and :py:class:`.State` for representing the transient internal state of a neuron or module.
    """

    pass
