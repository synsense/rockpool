"""
Defines the `JaxModule` base class, for Jax support in Rockpool.
"""
__all__ = ["JaxModule"]

# - Import Rockpool Module base class
from rockpool.nn.modules.module import Module

# - Jax imports
from jax.tree_util import register_pytree_node
import jax.numpy as np

# - Other imports
from copy import deepcopy
import operator as op
from abc import ABC
from typing import Optional, Tuple, Union
from rockpool.typehints import Tree


class JaxModule(Module, ABC):
    """
    Base class for `Module` subclasses that use a Jax backend.

    All modules in Rockpool that require Jax support must inherit from this base class. For compatibility with Jax, all :py:class:`.JaxModule` subclasses must use the functional API for evolution and setting state and attributes.

    To get started with the Jax backend, see :ref:`/in-depth/api-functional.ipynb` and :ref:`/in-depth/jax-training.ipynb`.

    Examples:
        Functional evolution of a module:

        >>> output, new_state, recorded_state = mod(input)
        >>> mod = mod.set_attributes(new_state)
    """

    _rockpool_pytree_registry = []
    """The internal registry of registered `JaxModule` s"""

    def __init__(
        self,
        shape: Optional[Union[int, Tuple]] = None,
        *args,
        **kwargs,
    ):
        """

        Args:
            shape (Optional[Tuple]): The shape of this module
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        # - Call the superclass initialiser
        super().__init__(shape, *args, **kwargs)

        # - Initialise initialisation args
        self._init_args = {}

        # - Register this class as a pytree for Jax
        cls = type(self)
        if cls not in JaxModule._rockpool_pytree_registry:
            register_pytree_node(
                cls, op.methodcaller("tree_flatten"), cls.tree_unflatten
            )
            JaxModule._rockpool_pytree_registry.append(cls)

    def _auto_batch(
        self,
        data: np.ndarray,
        states: Tuple = (),
        target_shapes: Tuple = None,
    ) -> (np.ndarray, Tuple[np.ndarray]):
        """
        Automatically replicate states over batches and verify input dimensions

        Usage:
            >>> data, (state0, state1, state2) = self._auto_batch(data, (self.state0, self.state1, self.state2))

            This will verify that `data` has the correct final dimension (i.e. `self.size_in`). If `data` has only two dimensions `(T, Nin)`, then it will be augmented to `(1, T, Nin)`. The individual states will be replicated out from shape `(a, b, c, ...)` to `(n_batches, a, b, c, ...)` and returned.

        Args:
            data (np.ndarray): Input data tensor. Either ``(batches, T, Nin)`` or ``(T, Nin)``
            states (Tuple): Tuple of state variables. Each will be replicated out over batches by prepending a batch dimension

        Returns:
            (np.ndarray, Tuple[np.ndarray]) data, states
        """
        # - Ensure data is a float32 tensor
        data = np.array(data, "float32")

        # - Verify input data shape
        if len(data.shape) == 0:
            data = np.expand_dims(data, 0)

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

    def tree_flatten(self) -> Tuple[tuple, tuple]:
        """Flatten this module tree for Jax"""
        return (
            (
                self.parameters(),
                self.simulation_parameters(),
                self.state(),
                self.modules(),
                self._init_args,
            ),
            (self._name, self._shape, self._submodulenames),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten a tree of modules from Jax to Rockpool"""
        params, sim_params, state, modules, init_args = children
        _name, _shape, _submodulenames = aux_data
        obj = cls(shape=_shape, **init_args)
        obj._name = _name

        # - Assign modules if necessary
        for (name, mod) in modules.items():
            if not hasattr(obj, name):
                setattr(obj, name, mod)

        # - Restore configuration
        obj._force_set_attributes = True
        obj = obj.set_attributes(params)
        obj = obj.set_attributes(state)
        obj = obj.set_attributes(sim_params)
        obj._force_set_attributes = False

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

    def set_attributes(self, new_attributes: Tree) -> "JaxModule":
        """
        Assign new attributes to this module and submodules

        Args:
            new_attributes (Tree): The tree of new attributes to assign to this module tree

        Returns:
            `.JaxModule`:
        """
        mod = deepcopy(self)

        # Module.set_attributes(self, new_attributes)
        # mod = Module.set_attributes(mod, new_attributes)

        __registered_attributes, __modules = mod._get_attribute_registry()

        # - Set self attributes
        for (k, v) in __registered_attributes.items():
            if k in new_attributes:
                mod.__setattr__(k, new_attributes[k])

        # - Set submodule attributes
        for (k, m) in __modules.items():
            if k in new_attributes:
                sub_mod = m[0].set_attributes(new_attributes[k])
                mod.__setattr__(k, sub_mod)

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

    def reset_state(self) -> "JaxModule":
        """
        Reset the state of this module

        Returns:
            Module: The updated module is returned for compatibility with the functional API

        """
        # - Copy module and get registry
        mod = deepcopy(self)
        __registered_attributes, __modules = mod._get_attribute_registry()

        # - Get a list of states
        states = mod.state()

        # - Set self attributes
        for (k, v) in __registered_attributes.items():
            if k in states:
                mod = mod._reset_attribute(k)

        # - Reset submodule states
        for (k, m) in __modules.items():
            sub_mod = m[0].reset_state()
            mod.__setattr__(k, sub_mod)

        return mod

    def reset_parameters(self):
        mod = deepcopy(self)
        Module.reset_parameters(self)
        Module.reset_parameters(mod)
        return mod
