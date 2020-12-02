from rockpool.nn.modules.module import Module

from importlib import util

if (util.find_spec("jax") is None) or (util.find_spec("jaxlib") is None):
    raise ModuleNotFoundError(
        "'Jax' and 'Jaxlib' backend not found. Layers that rely on Jax will not be available."
    )

from jax.tree_util import register_pytree_node

from copy import deepcopy

import operator as op

from abc import ABC


class JaxModule(Module, ABC):
    _rockpool_pytree_registry = []

    def __init__(
        self,
        shape=None,
        *args,
        **kwargs,
    ):
        # - Call the superclass initialiser
        super().__init__(shape, *args, **kwargs)

        # - Register this class as a pytree for Jax
        cls = type(self)
        if cls not in JaxModule._rockpool_pytree_registry:
            register_pytree_node(
                cls, op.methodcaller("tree_flatten"), cls.tree_unflatten
            )
            JaxModule._rockpool_pytree_registry.append(cls)

    def tree_flatten(self):
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
        # - Check that the submodule is also Jax compatible
        if not isinstance(mod, JaxModule):
            raise ValueError(
                f"Submodules of a `JaxModule` must themselves all be `JaxModule`s. Trying to assign a {mod.class_name} as a submodule of a {self.class_name}"
            )

        # - Register the module
        super()._register_module(name, mod)

    def set_attributes(self, new_attributes: dict):
        mod = deepcopy(self)
        Module.set_attributes(self, new_attributes)
        Module.set_attributes(mod, new_attributes)
        Module.set_attributes(self, new_attributes)
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
