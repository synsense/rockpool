"""
Defines the `JaxModule` base class, for Jax support in Rockpool.
"""

# - Import Rockpool Module base class
from rockpool.nn.modules.module import Module

from importlib import util

# - Check that jax is installed
if (util.find_spec("jax") is None) or (util.find_spec("jaxlib") is None):
    raise ModuleNotFoundError(
        "'Jax' and 'Jaxlib' backend not found. Layers that rely on Jax will not be available."
    )


# - Jax imports
from jax.tree_util import register_pytree_node

# - Other imports
from copy import deepcopy
import operator as op
from abc import ABC
from typing import Optional, Tuple


class JaxModule(Module, ABC):
    """
    Base class for `Module` subclasses that use a Jax backend.

    All modules in Rockpool that require Jax support must inherit from this base class. For compatibility with Jax, all `JaxModule` subclasses must use the functional API for evolution and setting state and attributes.

    Examples:
        Functional evolution of a module:

        >>> output, new_state, recorded_state = mod(input)
        >>> mod = mod.set_attributes(new_state)
    """

    _rockpool_pytree_registry = []
    """The internal registry of registered `JaxModule` s"""

    def __init__(
        self, shape: tuple = None, *args, **kwargs,
    ):
        """

        Args:
            shape (Optional[Tuple]): The shape of this module
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        # - Call the superclass initialiser
        super().__init__(shape, *args, **kwargs)

        # - Register this class as a pytree for Jax
        cls = type(self)
        if cls not in JaxModule._rockpool_pytree_registry:
            register_pytree_node(
                cls, op.methodcaller("tree_flatten"), cls.tree_unflatten
            )
            JaxModule._rockpool_pytree_registry.append(cls)

    def tree_flatten(self) -> Tuple[tuple, tuple]:
        """Flatten this module tree for Jax"""
        return (
            (
                self.parameters(),
                self.simulation_parameters(),
                self.state(),
                self.modules(),
            ),
            (self._name, self._shape, self._submodulenames),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten a tree of modules from Jax to Rockpool"""
        params, sim_params, state, modules = children
        _name, _shape, _submodulenames = aux_data
        obj = cls(**params, **sim_params)
        obj._name = _name

        for (name, mod) in modules.items():
            if not hasattr(obj, name):
                setattr(obj, name, mod)

        obj.set_attributes(state)

        return obj

    def _register_module(self, name: str, mod):
        """
        Add a submodule to the module registry

        Args:
            name (str): The name of the submodule, extracted from the assigned attribute name
            mod (JaxModule): The submodule to register
        """
        # - Check that the submodule is also Jax compatible
        if not isinstance(mod, JaxModule):
            raise ValueError(
                f"Submodules of a `JaxModule` must themselves all be `JaxModule`s. Trying to assign a {mod.class_name} as a submodule of a {self.class_name}"
            )

        # - Register the module
        super()._register_module(name, mod)

    def set_attributes(self, new_attributes: dict):
        """
        Assign new attributes to this module and submodules

        Args:
            new_attributes (dict): The dictionary of new attributes to assign to this module tree
        """
        mod = deepcopy(self)
        Module.set_attributes(self, new_attributes)
        Module.set_attributes(mod, new_attributes)
        return mod

    def __setattr__(self, name, value):
        mod = deepcopy(self)
        Module.__setattr__(mod, name, value)
        Module.__setattr__(self, name, value)
        return mod

    def __delattr__(self, item):
        mod = deepcopy(self)
        Module.__delattr__(mod, item)
        Module.__delattr__(self, item)
        return mod

    def reset_state(self):
        mod = deepcopy(self)
        Module.reset_state(self)
        Module.reset_state(mod)
        return mod

    def reset_parameters(self):
        mod = deepcopy(self)
        Module.reset_parameters(self)
        Module.reset_parameters(mod)
        return mod
