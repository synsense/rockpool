"""
Contains the module base classes for Rockpool
"""
# - Rockpool imports
import collections

from rockpool.parameters import ParameterBase
from rockpool.timeseries import TimeSeries

try:
    from rockpool.graph.graph_base import GraphModuleBase
except:
    GraphModuleBase = "GraphModuleBase"

# - Other imports
from abc import ABC, abstractmethod
from warnings import warn
from collections import ChainMap
from typing import Tuple, Any, Iterable, Dict, Optional, List, Union
import numpy as np

__all__ = ["Module"]


class ModuleBase(ABC):
    """
    Base class for all `Module` subclasses in Rockpool
    """

    def __init__(
        self,
        shape: Optional[Union[Tuple, int]] = None,
        spiking_input: bool = False,
        spiking_output: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialise this module

        Args:
            shape (Optional[Union[Tuple, int]]): The shape of the defined module
            spiking_input (bool): Whether this module receives spiking input. Default: False
            spiking_output (bool): Whether this module produces spiking output. Default: False
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        # - Set flag to specify that we are in the `__init__()` method
        self._in_Module_init = True
        """ (bool) If exists and ``True``, indicates that the module is in the ``__init__`` chain."""

        self._force_set_attributes = False
        """ (bool) If ``True``, do not sanity-check attributes when setting. """

        # - Initialise co-classes etc.
        super().__init__(*args, **kwargs)

        # - Initialise Module attributes
        self._submodulenames: List[str] = []
        """Registry of sub-module names"""

        self._name: Optional[str] = None
        """Name of this module, if assigned"""

        self._spiking_input: bool = spiking_input
        """Whether this module receives spiking input"""

        self._spiking_output: bool = spiking_output
        """Whether this module produces spiking output"""

        # - Be generous if a scalar was provided instead of a tuple
        if isinstance(shape, Iterable):
            self._shape = tuple(shape)
            """The shape of this module"""
        else:
            self._shape = (shape,)
            """The shape of this module"""

    def __repr__(self, indent: str = "") -> str:
        """
        Produce a string representation of this module

        Args:
            indent (str): The indent to prepend to each line of output

        Returns:
            str: A string representation of this module
        """
        # - String representation
        repr = f"{indent}{self.full_name} with shape {self._shape}"

        # - Add submodules
        if self._submodulenames:
            repr += " {"
            for mod_name in self._submodulenames:
                repr += "\n" + ModuleBase.__repr__(
                    getattr(self, mod_name), indent=indent + "    "
                )

            repr += f"\n{indent}" + "}"

        return repr

    def _get_attribute_registry(self) -> Tuple[Dict, Dict]:
        """
        Return or initialise the attribute registry for this module

        Returns:
            (tuple): registered_attributes, registered_modules
        """
        if not hasattr(self, "_ModuleBase__registered_attributes") or not hasattr(
            self, "_ModuleBase__modules"
        ):
            super().__setattr__("_ModuleBase__registered_attributes", {})
            super().__setattr__("_ModuleBase__modules", {})

        # - Get the attribute and modules dictionaries in a safe way
        __registered_attributes = self.__dict__.get(
            "_ModuleBase__registered_attributes"
        )
        __modules = self.__dict__.get("_ModuleBase__modules")

        return __registered_attributes, __modules

    def __setattr__(self, name: str, val: Any):
        """
        Set an attribute for this module

        Args:
            name (str): The name of the attribute to set
            val (Any): The value to assign to the attribute
        """
        # - Get attribute registry
        __registered_attributes, __modules = self._get_attribute_registry()

        # - Check if this is a new rockpool Parameter
        if isinstance(val, ParameterBase):
            if (
                hasattr(self, name)
                and hasattr(self, "_in_Module_init")
                and not self._in_Module_init
            ):
                raise ValueError(
                    f'Cannot assign a new Parameter or State to an existing attribute "{name}".'
                )

            # - Register the attribute
            self._register_attribute(name, val)
            val = val.data

        # - Are we assigning a sub-module?
        if isinstance(val, ModuleBase):
            self._register_module(name, val)

        # - Check if this is an already registered attribute
        if name in __registered_attributes:
            if hasattr(self, name):
                (_, _, _, _, shape) = __registered_attributes[name]

                if val is not None:
                    # - Should we force-set the attribute?
                    if self._force_set_attributes:
                        __registered_attributes[name][4] = np.shape(val)

                    elif np.shape(val) != shape:
                        # - Check that shapes are identical
                        raise ValueError(
                            f"The new value assigned to {name} must be of shape {shape} (got {np.shape(val)})."
                        )

            # - Assign the value to the __registered_attributes dictionary
            __registered_attributes[name][0] = val

        # - Assign attribute to self
        super().__setattr__(name, val)

    def __delattr__(self, name: str):
        """
        Delete an attribute from this module, and remove from the attribute registry if present

        Args:
            name (str): The name of the attribute to delete
        """
        # - Get attribute registry
        __registered_attributes, __modules = self._get_attribute_registry()

        # - Remove attribute from registered attributes
        if name in __registered_attributes:
            del __registered_attributes[name]

        # - Remove name from modules
        if name in __modules:
            del __modules[name]
            self._submodulenames.remove(name)

        # - Remove attribute
        super().__delattr__(name)

    def _register_attribute(self, name: str, val: ParameterBase):
        """
        Record an attribute in the attribute registry

        Args:
            name (str): The name of the attribute to register
            val (ParameterBase): The `ParameterBase` subclass object to register. e.g. `Parameter`, `SimulationParameter` or `State`.
        """
        # - Get attribute registry
        __registered_attributes, __modules = self._get_attribute_registry()

        # - Record attribute in attribute registry
        __registered_attributes[name]: dict = [
            val.data,
            type(val).__name__,
            val.family,
            val.init_func,
            val.shape,
        ]
        """The attribute registry for this module"""

    def _register_module(self, name: str, mod: "ModuleBase"):
        """
        Register a sub-module in the module registry

        Args:
            name (str): The name of the module to register
            mod (ModuleBase): The `ModuleBase` object to register
        """
        # - Get attribute registry
        __registered_attributes, __modules = self._get_attribute_registry()

        if not isinstance(mod, ModuleBase):
            raise ValueError(
                f"You may only assign a `Module` subclass as a sub-module."
            )

        # - Assign module name to module
        mod._name = name

        # - Assign to appropriate attribute dictionary
        __modules[name] = [mod, type(mod).__name__]
        self._submodulenames.append(name)

    def set_attributes(self, new_attributes: dict) -> "ModuleBase":
        """
        Set the attributes and sub-module attributes from a dictionary

        This method can be used with the dictionary returned from module evolution to set the new state of the module. It can also be used to set multiple parameters of a module and submodules.

        Examples:
            Use the functional API to evolve, obtain new states, and set those states:

            >>> _, new_state, _ = mod(input)
            >>> mod = mod.set_attributes(new_state)

            Obtain a parameter dictionary, modify it, then set the parameters back:

            >>> params = mod.parameters()
            >>> params['w_input'] *= 0.
            >>> mod.set_attributes(params)

        Args:
            new_attributes (dict): A nested dictionary containing parameters of this module and sub-modules.
        """
        # - Get attribute registry
        __registered_attributes, __modules = self._get_attribute_registry()

        # - Set self attributes
        for (k, v) in __registered_attributes.items():
            if k in new_attributes:
                self.__setattr__(k, new_attributes[k])

        # - Set submodule attributes
        for (k, m) in __modules.items():
            if k in new_attributes:
                m[0].set_attributes(new_attributes[k])

        # - Return the module, for compatibility with the functional interface
        return self

    def _get_attribute_family(
        self, type_name: str, family: Union[str, Tuple, List] = None
    ) -> dict:
        """
        Search for attributes of this module and submodules that match a given family

        This method can be used to conveniently get all weights for a network; or all time constants; or any other family of parameters. Parameter families are defined simply by a string: ``"weights"`` for weights; ``"taus"`` for time constants, etc. These strings are arbitrary, but if you follow the conventions then future developers will thank you (that includes you in six month's time).

        Args:
            type_name (str): The class of parameters to search for. Must be one of ``["Parameter", "SimulationParameter", "State"]`` or another future subclass of :py:class:`.ParameterBase`
            family (Union[str, Tuple[str]]): A string or list or tuple of strings, that define one or more attribute families to search for

        Returns:
            dict: A nested dictionary of attributes that match the provided `type_name` and `family`
        """
        # - Get attribute registry
        __registered_attributes, __modules = self._get_attribute_registry()

        # - Filter own attribute dictionary by type key
        matching_attributes = {
            k: v for (k, v) in __registered_attributes.items() if v[1] == type_name
        }

        # - Filter by family
        if family is not None:
            if not isinstance(family, (tuple, list)):
                family = (family,)

            list_attributes = [
                {k: v for (k, v) in matching_attributes.items() if v[2] is f}
                for f in family
            ]
            matching_attributes = dict(ChainMap(*list_attributes))

        # - Just take values using getattr
        matching_attributes = {k: getattr(self, k) for k in matching_attributes.keys()}

        # - Append sub-module attributes as nested dictionaries
        submodule_attributes = {}
        for (k, m) in __modules.items():
            mod_attributes = m[0]._get_attribute_family(type_name, family)

            if (family and mod_attributes) or (not family):
                submodule_attributes[k] = mod_attributes

        # - Push submodule attributes into dictionary
        if family and submodule_attributes or not family:
            matching_attributes.update(submodule_attributes)

        # - Return nested attributes
        return matching_attributes

    def attributes_named(self, name: Union[Tuple[str], List[str], str]) -> dict:
        """
        Search for attributes of this or submodules by time

        Args:
            name (Union[str, Tuple[str]): The name of the attribute to search for

        Returns:
            dict: A nested dictionary of attributes that match `name`
        """
        # - Get attribute registry
        __registered_attributes, __modules = self._get_attribute_registry()

        # - Check if we were given a tuple or not
        if not isinstance(name, (tuple, list)):
            name = (name,)

        # - Filter own attribute dictionary by name keys
        list_attributes = [
            {k: v for (k, v) in __registered_attributes.items() if k == n} for n in name
        ]
        matching_attributes = dict(ChainMap(*list_attributes))

        # - Just take values
        matching_attributes = {k: v[0] for (k, v) in matching_attributes.items()}

        # - Append sub-module attributes as nested dictionaries
        submodule_attributes = {}
        for (k, m) in __modules.items():
            mod_attributes = m[0].attributes_named(name)

            if mod_attributes:
                submodule_attributes[k] = mod_attributes

        # - Push submodule attributes into dictionary
        if submodule_attributes:
            matching_attributes.update(submodule_attributes)

        # - Return nested attributes
        return matching_attributes

    def parameters(self, family: Union[str, Tuple, List] = None) -> Dict:
        """
        Return a nested dictionary of module and submodule Parameters

        Use this method to inspect the Parameters from this and all submodules. The optional argument `family` allows you to search for Parameters in a particular family — for example ``"weights"`` for all weights of this module and nested submodules.

        Although the `family` argument is an arbitrary string, reasonable choises are ``"weights"``, ``"taus"`` for time constants, ``"biases"`` for biases...

        Examples:
            Obtain a dictionary of all Parameters for this module (including submodules):

            >>> mod.parameters()
            dict{ ... }

            Obtain a dictionary of Parameters from a particular family:

            >>> mod.parameters("weights")
            dict{ ... }

        Args:
            family (str): The family of Parameters to search for. Default: ``None``; return all parameters.

        Returns:
            dict: A nested dictionary of Parameters of this module and all submodules
        """
        return self._get_attribute_family("Parameter", family)

    def simulation_parameters(self, family: Union[str, Tuple, List] = None) -> Dict:
        """
        Return a nested dictionary of module and submodule SimulationParameters

        Use this method to inspect the SimulationParameters from this and all submodules. The optional argument `family` allows you to search for SimulationParameters in a particular family.

        Examples:
            Obtain a dictionary of all SimulationParameters for this module (including submodules):

            >>> mod.simulation_parameters()
            dict{ ... }

        Args:
            family (str): The family of SimulationParameters to search for. Default: ``None``; return all SimulationParameter attributes.

        Returns:
            dict: A nested dictionary of SimulationParameters of this module and all submodules

        """
        return self._get_attribute_family("SimulationParameter", family)

    def state(self, family: Union[str, Tuple, List] = None) -> Dict:
        """
        Return a nested dictionary of module and submodule States

        Use this method to inspect the States from this and all submodules. The optional argument `family` allows you to search for States in a particular family.

        Examples:
            Obtain a dictionary of all States for this module (including submodules):

            >>> mod.state()
            dict{ ... }

        Args:
            family (str): The family of States to search for. Default: ``None``; return all State attributes.

        Returns:
            dict: A nested dictionary of States of this module and all submodules
        """
        return self._get_attribute_family("State", family)

    def modules(self) -> Dict:
        """
        Return a dictionary of all sub-modules of this module

        Returns:
            dict: A dictionary containing all sub-modules. Each item will be named with the sub-module name.
        """
        # - Get attribute registry
        __registered_attributes, __modules = self._get_attribute_registry()

        return collections.OrderedDict([(k, m[0]) for (k, m) in __modules.items()])

    def _reset_attribute(self, name: str) -> "ModuleBase":
        """
        Reset an attribute to its initialisation value

        Args:
            name (str): The name of the attribute to reset

        Returns:
            self (`Module`): For compatibility with the functional API
        """
        # - Get attribute registry
        __registered_attributes, __modules = self._get_attribute_registry()

        # - Check that the attribute is registered
        if name not in __registered_attributes:
            raise KeyError(f"{name} is not a registered attribute.")

        # - Get the initialisation function from the registry
        (_, _, family, init_func, shape) = __registered_attributes[name]

        # - Use the registered initialisation function, if present
        if init_func is not None:
            setattr(self, name, init_func(shape))

        return self

    def _has_registered_attribute(self, name: str) -> bool:
        """
        Check if the module has a registered attribute

        Args:
            name (str): The name of the attribute to check

        Returns:
            bool: ``True`` if the attribute `name` is in the attribute registry, ``False`` otherwise.
        """
        __registered_attributes, _ = self._get_attribute_registry()
        return name in __registered_attributes

    def reset_state(self) -> "ModuleBase":
        """
        Reset the state of this module

        Returns:
            Module: The updated module is returned for compatibility with the functional API

        """
        # - Get attribute registry
        __registered_attributes, __modules = self._get_attribute_registry()

        # - Get a list of states
        states = self.state()

        # - Set self attributes
        for (k, v) in __registered_attributes.items():
            if k in states:
                self._reset_attribute(k)

        # - Reset submodule states
        for (k, m) in __modules.items():
            m[0] = m[0].reset_state()

        return self

    def reset_parameters(self):
        """
        Reset all parameters in this module

        Returns:
            Module: The updated module is returned for compatibility with the functional API
        """
        # - Get attribute registry
        __registered_attributes, __modules = self._get_attribute_registry()

        # - Get a list of parameters
        parameters = self.parameters()

        # - Set self attributes
        for (k, v) in __registered_attributes.items():
            if k in parameters:
                self._reset_attribute(k)

        # - Reset submodule states
        for (k, m) in __modules.items():
            m[0] = m[0].reset_parameters()

        return self

    @property
    def class_name(self) -> str:
        """str: Class name of ``self``"""
        # - Determine class name by removing "<class '" and "'>" and the package information
        return type(self).__name__

    @property
    def name(self) -> str:
        """str: The name of this module, or an empty string if ``None``"""
        return f"'{self._name}'" if hasattr(self, "_name") and self._name else ""

    @property
    def full_name(self) -> str:
        """str: The full name of this module (class plus module name)"""
        return f"{self.class_name} {self.name}"

    @property
    def spiking_input(self) -> bool:
        """bool: If ``True``, this module receives spiking input. If ``False``, this module expects continuous input."""
        return self._spiking_input

    @property
    def spiking_output(self):
        """bool: If ``True``, this module sends spiking output. If ``False``, this module sends continuous output."""
        return self._spiking_output

    @property
    def shape(self) -> tuple:
        """tuple: The shape of this module"""
        return self._shape

    @property
    def size(self) -> int:
        """int: (DEPRECATED) The output size of this module"""
        warn(
            "The `size` property is deprecated. Please use `size_out` instead.",
            DeprecationWarning,
        )
        return self._shape[-1]

    @property
    def size_out(self) -> int:
        """int: The output size of this module"""
        return self._shape[-1]

    @property
    def size_in(self) -> int:
        """int: The input size of this module"""
        return self._shape[0]

    @abstractmethod
    def evolve(self, input_data, record: bool = False) -> Tuple[Any, Any, Any]:
        """
        Evolve the state of this module over input data

        NOTE: THIS MODULE CLASS DOES NOT PROVIDE DOCUMENTATION FOR ITS EVOLVE METHOD. PLEASE UPDATE THE DOCUMENTATION FOR THIS MODULE.

        Args:
            input_data: The input data with shape ``(T, size_in)`` to evolve with
            record (bool): If ``True``, the module should record internal state during evolution and return the record. If ``False``, no recording is required. Default: ``False``.

        Returns:
            tuple: (output, new_state, record)
                output (np.ndarray): The output response of this module with shape ``(T, size_out)``
                new_state (dict): A dictionary containing the updated state of this and all submodules after evolution
                record (dict): A dictionary containing recorded state of this and all submodules, if requested using the `record` argument
        """
        return None, None, None

    def __call__(self, input_data, *args, **kwargs):
        """
        Evolve the state of this module over input data

        Args:
            input_data: The input data with shape ``(T, size_in)`` to evolve this module with
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:

        """
        # - Catch the case where we have been called with the raw output of a previous call
        if isinstance(input_data, tuple) and len(input_data) == 3:
            input_data, new_state, recorded_state = input_data
            outputs, this_new_state, this_recorded_state = self.evolve(
                input_data, *args, **kwargs
            )
            new_state.update({self.name: this_new_state})
            recorded_state.update({self.name: this_recorded_state})
        else:
            outputs, new_state, recorded_state = self.evolve(
                input_data, *args, **kwargs
            )

        return outputs, new_state, recorded_state

    def _wrap_recorded_state(
        self, recorded_dict: dict, t_start: float
    ) -> Dict[str, TimeSeries]:
        """
        Convert a recorded dictionary to a :py:class:`TimeSeries` representation

        This method is optional, and is provided to make the :py:meth:`timed` conversion to a :py:class:`TimedModule` work better. You should override this method in your custom :py:class:`Module`, to wrap each element of your recorded state dictionary as a :py:class:`TimeSeries`

        Args:
            state_dict (dict): A recorded state dictionary as returned by :py:meth:`.evolve`
            t_start (float): The initial time of the recorded state, to use as the starting point of the time series

        Returns:
            Dict[str, TimeSeries]: The mapped recorded state dictionary, wrapped as :py:class:`TimeSeries` objects
        """
        return recorded_dict

    def as_graph(self) -> GraphModuleBase:
        """
        Convert this module to a computational graph

        Returns:
            GraphModuleBase: The computational graph corresponding to this module

        Raises:
            NotImplementedError: If :py:meth:`.as_graph` is not implemented for this subclass
        """
        raise NotImplementedError(
            f"The Module class '{type(self).__name__}' used by module [{self}] does not implement the graph serialisation API"
        )


class PostInitMetaMixin(type(ModuleBase)):
    """
    A mixin base class that adds a ``__post_init__()`` method to a class. ``__post_init__()`` is called after the ``__init__()`` method, on instantiation of an object.
    """

    def __call__(cls, *args, **kwargs):
        # - Instantiate the object
        obj = super().__call__(*args, **kwargs)

        # - Check for a `__post_init__` method
        if hasattr(cls, "__post_init__"):
            cls.__post_init__(obj)

        # - Clear the init flag
        obj._in_Module_init = False
        return obj


class Module(ModuleBase, ABC, metaclass=PostInitMetaMixin):
    """
    The native Python / numpy :py:class:`.Module` base class for Rockpool

    This class acts as the base class for all "native" modules in Rockpool. To get started with using and writing your own Rockpool modules, see :ref:`/basics/getting_started.ipynb` and :ref:`/in-depth/api-low-level.ipynb`.

    If you plan to write modules using Jax or Torch backends, you should use either :py:class:`.JaxModule` or :py:class:`.TorchModule` as base classes, respectively.

    To get started with the Jax backend, see :ref:`/in-depth/api-functional.ipynb` and :ref:`/in-depth/jax-training.ipynb`.

    To get started with the Torch backend, see :ref:`/in-depth/torch-api.ipynb` and :ref:`/in-depth/torch-training.ipynb`.
    """

    def timed(self, output_num: int = 0, dt: float = None, add_events: bool = False):
        """
        Convert this module to a :py:class:`.TimedModule`

        Args:
            output_num (int): Specify which output of the module to take, if the module returns multiple output series. Default: ``0``, take the first (or only) output.
            dt (float): Used to provide a time-step for this module, if the module does not already have one. If ``self`` already defines a time-step, then ``self.dt`` will be used. Default: ``None``
            add_events (bool): Iff ``True``, the :py:class:`.TimedModule` will add events occurring on a single timestep on input and output. Default: ``False``, don't add time steps.

        Returns: :py:class:`.TimedModule`: A timed module that wraps this module
        """
        from rockpool.nn.modules.timed_module import TimedModuleWrapper

        return TimedModuleWrapper(
            self, output_num=output_num, dt=dt, add_events=add_events
        )

    def _auto_batch(
        self,
        data: np.ndarray,
        states: Tuple = (),
        target_shapes: Tuple = None,
    ) -> (np.ndarray, Tuple[np.ndarray]):
        """
        Automatically replicate states over batches and verify input dimensions

        Examples:
            >>> data, (state0, state1, state2) = self._auto_batch(data, (self.state0, self.state1, self.state2))

            This will verify that ``data`` has the correct final dimension (i.e. ``self.size_in``).

            If ``data`` has only two dimensions ``(T, Nin)``, then it will be augmented to ``(1, T, Nin)``. The individual states will be replicated out from shape ``(a, b, c, ...)`` to ``(n_batches, a, b, c, ...)`` and returned.

            If ``data`` has only a single dimension ``(T,)``, it will be expanded to ``(1, T, self.size_in)``.

        Args:
            data (np.ndarray): Input data tensor. Either ``(batches, T, Nin)`` or ``(T, Nin)``
            states (Tuple): Tuple of state variables. Each will be replicated out over batches by prepending a batch dimension

        Returns:
            (np.ndarray, Tuple[np.ndarray]) data, states
        """
        # - Ensure data is a float tensor
        data = np.array(data, "float")

        # - Verify input data shape
        if len(data.shape) == 1:
            data = np.expand_dims(data, 0)
            data = np.expand_dims(data, 2)
        elif len(data.shape) == 2:
            data = np.expand_dims(data, 0)

        if data.shape[-1] == 1:
            data = np.broadcast_to(data, (data.shape[0], data.shape[1], self.size_in))

        # - Get shape of input
        (n_batches, time_steps, n_connections) = data.shape

        # - Check input dimensions
        if n_connections != self.size_in:
            raise ValueError(
                "Input has wrong neuron dimension. It is {}, must be {}".format(
                    n_connections, self.size_in
                )
            )

        # - Get target shapes
        if target_shapes is None:
            target_shapes = tuple(s.shape for s in states)
        else:
            target_shapes = tuple(
                s.shape if shape is None else shape
                for s, shape in zip(states, target_shapes)
            )

        # - Replicate shapes and return
        states = tuple(
            np.ones((n_batches, *shape)) * s for s, shape in zip(states, target_shapes)
        )
        return data, states
