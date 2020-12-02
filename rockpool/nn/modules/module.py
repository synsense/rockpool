from copy import deepcopy
from abc import ABC, abstractmethod

from warnings import warn

from rockpool.parameters import ParameterBase

from collections import ChainMap

from typing import Tuple, Any, Iterable, Dict, Optional, List, Union

import numpy as np


class ModuleBase(ABC):
    def __init__(
        self,
        shape=None,
        spiking_input: bool = False,
        spiking_output: bool = False,
        *args,
        **kwargs,
    ):
        # - Initialise co-classes etc.
        super().__init__(*args, **kwargs)

        # - Initialise Module attributes
        self._submodulenames: List[str] = []
        self._name: Optional[str] = None
        self._spiking_input: bool = spiking_input
        self._spiking_output: bool = spiking_output

        # - Be generous if a scalar was provided instead of a tuple
        if isinstance(shape, Iterable):
            self._shape = tuple(shape)
        else:
            self._shape = (shape,)

    def __repr__(self, indent="") -> str:
        # - String representation
        repr = f"{indent}{self.full_name} with shape {self._shape}"

        # - Add submodules
        if self.modules():
            repr += " {"
            for mod in self.modules().values():
                repr += "\n" + mod.__repr__(indent + "   ")

            repr += f"\n{indent}" + "}"

        return repr

    def _get_attribute_registry(self):
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
        # - Get attribute registry
        __registered_attributes, __modules = self._get_attribute_registry()

        # - Check if a module has already been assigned
        if name in __modules:
            raise ValueError(
                f'Cannot reassign a new sub-module "{name}" to an already initialised object.'
            )

        # - Check if this is a new rockpool Parameter
        if isinstance(val, ParameterBase):
            if hasattr(self, name):
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
            # - Check that shapes are identical
            if hasattr(self, name):
                (_, _, _, _, shape) = self.__registered_attributes[name]
                if np.shape(val) != shape and val is not None:
                    raise ValueError(
                        f"The new value assigned to {name} must be of shape {shape}."
                    )

            # - Assign the value to the __registered_attributes dictionary
            __registered_attributes[name][0] = val

        # - Assign attribute to self
        super().__setattr__(name, val)

    def __delattr__(self, name):
        # - Remove attribute from registered attributes
        if name in self.__registered_attributes:
            del self.__registered_attributes[name]

        # - Remove name from modules
        if name in self.__modules:
            del self.__modules[name]
            self._submodulenames.remove(name)

        # - Remove attribute
        super().__delattr__(name)

    def _register_attribute(self, name: str, val: ParameterBase):
        # - Record attribute in attribute registry
        self.__registered_attributes[name] = [
            val.data,
            type(val).__name__,
            val.family,
            val.init_func,
            val.shape,
        ]

    def _register_module(self, name: str, mod):
        if not isinstance(mod, ModuleBase):
            raise ValueError(f"You may only assign a Module subclass as a sub-module.")

        # - Assign module name to module
        mod._name = name

        # - Assign to appropriate attribute dictionary
        self.__modules[name] = [mod, type(mod).__name__]
        self._submodulenames.append(name)

    def set_attributes(self, new_attributes: dict):
        # - Set self attributes
        for (k, v) in self.__registered_attributes.items():
            if k in new_attributes:
                self.__setattr__(k, new_attributes[k])

        # - Set submodule attributes
        for (k, m) in self.__modules.items():
            if k in new_attributes:
                m[0].set_attributes(new_attributes[k])

    def _get_attribute_family(
        self, type_name: str, family: Union[str, Tuple, List] = None
    ) -> dict:
        # - Filter own attribute dictionary by type key
        matching_attributes = {
            k: v for (k, v) in self.__registered_attributes.items() if v[1] == type_name
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

        # - Just take values
        matching_attributes = {k: getattr(self, k) for (k, v) in matching_attributes.items()}

        # - Append sub-module attributes as nested dictionaries
        submodule_attributes = {}
        for (k, m) in self.__modules.items():
            mod_attributes = m[0]._get_attribute_family(type_name, family)

            if (family and mod_attributes) or (not family):
                submodule_attributes[k] = mod_attributes

        # - Push submodule attributes into dictionary
        if family and submodule_attributes or not family:
            matching_attributes.update(submodule_attributes)

        # - Return nested attributes
        return matching_attributes

    def attributes_named(self, name: Union[Tuple[str], List[str], str]) -> dict:
        # - Check if we were given a tuple or not
        if not isinstance(name, (tuple, list)):
            name = (name,)

        # - Filter own attribute dictionary by name keys
        list_attributes = [
            {k: v for (k, v) in self.__registered_attributes.items() if k == n}
            for n in name
        ]
        matching_attributes = dict(ChainMap(*list_attributes))

        # - Just take values
        matching_attributes = {k: v[0] for (k, v) in matching_attributes.items()}

        # - Append sub-module attributes as nested dictionaries
        submodule_attributes = {}
        for (k, m) in self.__modules.items():
            mod_attributes = m[0].attributes_named(name)

            if mod_attributes:
                submodule_attributes[k] = mod_attributes

        # - Push submodule attributes into dictionary
        if submodule_attributes:
            matching_attributes.update(submodule_attributes)

        # - Return nested attributes
        return matching_attributes

    def parameters(self, family: Union[str, Tuple, List] = None):
        return self._get_attribute_family("Parameter", family)

    def simulation_parameters(self, family: Union[str, Tuple, List] = None):
        return self._get_attribute_family("SimulationParameter", family)

    def state(self, family: Union[str, Tuple, List] = None):
        return self._get_attribute_family("State", family)

    def modules(self):
        return {k: m[0] for (k, m) in self.__modules.items()}

    def _reset_attribute(self, name: str):
        # - Check that the attribute is registered
        if name not in self.__registered_attributes:
            raise KeyError(f"{name} is not a registered attribute.")

        # - Get the initialisation function from the registry
        (_, _, family, init_func, shape) = self.__registered_attributes[name]

        # - Use the registered initialisation function, if present
        if init_func is not None:
            setattr(self, name, init_func(shape))

    def _has_registered_attribute(self, name: str):
        return name in self.__registered_attributes

    def reset_state(self):
        # - Get a list of states
        states = self.state()

        # - Set self attributes
        for (k, v) in self.__registered_attributes.items():
            if k in states:
                self._reset_attribute(k)

        # - Reset submodule states
        for (k, m) in self.__modules.items():
            m[0].reset_state()

    def reset_parameters(self):
        # - Get a list of parameters
        parameters = self.parameters()

        # - Set self attributes
        for (k, v) in self.__registered_attributes.items():
            if k in parameters:
                self._reset_attribute(k)

        # - Reset submodule states
        for (k, m) in self.__modules.items():
            m[0].reset_parameters()

    @property
    def class_name(self) -> str:
        """
        (str) Class name of ``self``
        """
        # - Determine class name by removing "<class '" and "'>" and the package information
        return type(self).__name__

    @property
    def name(self) -> str:
        return f"'{self._name}'" if hasattr(self, "_name") else "[Unnamed]"

    @property
    def full_name(self) -> str:
        return f"{self.class_name} {self.name}"

    @property
    def spiking_input(self):
        return self._spiking_input

    @property
    def spiking_output(self):
        return self._spiking_output

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        warn(
            "The `size` property is deprecated. Please use `size_out` instead.",
            DeprecationWarning,
        )
        return self._shape[-1]

    @property
    def size_out(self):
        return self._shape[-1]

    @property
    def size_in(self):
        return self._shape[0]

    @abstractmethod
    def evolve(self, input_data, record: bool = False) -> Tuple[Any, Any, Any]:
        return None, None, None

    def __call__(self, input_data, *args, **kwargs):
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


class Module(ModuleBase, ABC):
    def _register_module(self, name: str, mod):
        # - Register the module
        super()._register_module(name, mod)

        # - Do we even have a `dt` attribute?
        if hasattr(self, "dt"):
            # - Check that the submodule `dt` is the same as mine
            if hasattr(mod, "dt"):
                if mod.dt != getattr(self, "dt"):
                    raise ValueError(
                        f"The submodule {mod.name} must have the same `dt` as the parent module {self.name}"
                    )
            else:
                # - Add `dt` as an attribute to the module (not a registered attribute)
                mod.dt = getattr(self, "dt")

        else:
            # - We should inherit the first `dt` of a submodule
            if hasattr(mod, "dt"):
                setattr(self, "dt", mod.dt)


class PostInitMetaMixin(type(ModuleBase)):
    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        if hasattr(cls, "__post_init__"):
            cls.__post_init__(obj)

        return obj
