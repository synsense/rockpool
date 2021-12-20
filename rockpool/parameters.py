"""
Classes to manage registered Module attributes in Rockpool
"""

from typing import Callable, Iterable, Any, Union, List, Tuple, Optional
from copy import deepcopy

from collections import abc

import numpy as np

__all__ = ["Parameter", "State", "SimulationParameter"]

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
        init_func: Callable[[Iterable], Any] = None,
        shape: Optional[Union[List[Tuple], Tuple, int]] = None,
    ):
        """
        Instantiate a Rockpool registered attribute

        Args:
            data (Optional[Any]): Concrete initialisation data for this attribute. The shape of ``data`` will specify the allowable shape of the attribute data, unless the ``shape`` argument is provided.
            family (Optional[str]): An arbitrary string to specify the "family" of this attribute. You should use ``'weights'``, ``'taus'``, ``'biases'`` if you can; otherwise you can use whatever you like. These are used by the :py:meth:`.Module.parameters`, :py:meth:`.Module.state` and :py:meth:`.Module.simulation_parameters` methods to group and select attributes.
            init_func (Optional[Callable]): A function that initialises this attributed. Called by :py:meth:`.Module.reset_parameters` and :py:meth:`.Module.reset_state`. The signature is ``f(shape: tuple) -> np.ndarray``.
            shape (Optional[Union[List[Tuple], Tuple, int]]): A list of permisable shapes for the parameter, or a tuple specifying the permitted shape, or an integer specifying the number of elements. If not provided, the shape of the concrete initialisation data will be used as the attribute shape. The first item in the list will be used as the concrete shape, if ``data`` is not provided and ``init_func`` should be used.
        """
        # - Be generous if a scalar shape was provided instead of a tuple
        if not isinstance(shape, (List, Tuple, int)):
            raise TypeError(
                f"`shape` must be a list, a tuple or an integer. Instead `shape` was a {type(shape).__name__}."
            )

        if isinstance(shape, Tuple):
            shape = [shape]

        if isinstance(shape, int):
            shape = [(shape,)]

        for st in shape:
            for elem in st:
                if not isinstance(elem, int):
                    raise TypeError(
                        f"All elements in a shape tuple must be integers. Instead I found an elements of type {type(elem).__name__}."
                    )

        # - Assign attributes
        self.family = family
        self.data = data
        self.init_func = init_func
        self.shape = shape

        class_name = type(self).__name__

        # - Check that the initialisation function is callable
        if self.init_func is not None and not callable(self.init_func):
            raise ValueError(
                f"The `init_func` for a {class_name} must be a callable that accepts a shape tuple."
            )

        # - Get the shape from the data, if not provided explicitly
        if self.data is not None:
            if self.shape is not None and self.shape != np.shape(self.data):
                raise ValueError(
                    f"The shape provided for this {class_name} does not match the provided initialisation data.\n"
                    + f"    self.shape = {self.shape}; data.shape = {np.shape(self.data)}"
                )

            if self.shape is None:
                self.shape = np.shape(self.data)

        # - Initialise data, if not provided
        if self.data is None:
            if self.init_func is None:
                raise ValueError(
                    f"If concrete initialisation `data` is not provided for a {class_name} then `init_func` must be provided."
                )

            # - Call the `init_func`
            self.data = self.init_func(self.shape)
        else:
            # - If concrete initialisation data is provided, then override the `init_func`
            data_copy = deepcopy(data)
            self.init_func = lambda _: data_copy

    def __repr__(self):
        return f"{type(self).__name__}(data={self.data}, family={self.family}, init_func={self.init_func}, shape={self.shape})"


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
