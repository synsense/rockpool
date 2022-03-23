"""
Provide a Module wrapper that transforms parameters before evolution
"""

from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter, State

from typing import Union, List, Tuple, Iterable, Any, Callable, Optional

from copy import deepcopy
import warnings
import numpy as onp

from collections import abc

from abc import ABC, abstractmethod

Tree = Union[abc.Iterable, abc.MutableMapping]

__all__ = ["ParameterTransformerMixin", "JaxParameterTransformerMixin"]


def deep_update(
    source: abc.MutableMapping, overrides: abc.Mapping
) -> abc.MutableMapping:
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        if isinstance(value, abc.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


class ParameterTransformerMixin(ABC):
    def __init__(
        self,
        module: Module,
        families: Optional[Union[Tuple, List, str]] = None,
        transform_params: Optional[Union[Tuple, List, Any]] = None,
        *args,
        **kwargs,
    ):
        # - Base class must be `Module`
        if not isinstance(self, Module):
            raise TypeError(
                "`ParameterTransformerMixin` mix-in class may only be used with `Module` classes."
            )

        # - Wrapped object must be a Module
        if not isinstance(module, Module):
            raise TypeError(
                f"{type(self).__name__} may only be used with a `Module` object."
            )

        # - Call superclass __init__()
        super().__init__(*args, **kwargs)

        # - Record submodule
        self.module = module

        # - Record module parameters
        self._shape = module.shape
        self._spiking_input = module.spiking_input
        self._spiking_output = module.spiking_output

        # - Be lenient if iterables were not provided
        if not isinstance(families, tuple) and not isinstance(families, list):
            families = [families]

        if not isinstance(transform_params, tuple) and not isinstance(
            transform_params, list
        ):
            transform_params = [transform_params] * len(families)

        # - Record which parameter families to transform
        self.families: Iterable = SimulationParameter(families)
        self.transform_params: Iterable = SimulationParameter(transform_params)

    def __repr__(self):
        return f"{type(self).__name__}({self.module})"

    def transformed_parameters(
        self: Union[Module, "ParameterTransformerMixin"]
    ) -> dict:
        # - Get the base set of parameters
        params = self.parameters()

        # - Loop over desired parameter families to transform
        for family, tf_params in zip(self.families, self.transform_params):
            # - Get this parameter family
            these_params = self.parameters(family)

            # - Transform these parameters
            these_params = self._tree_map(
                lambda p: self._transform(p, tf_params), these_params
            )

            # - Update the parameter dictionary
            params = deep_update(params, these_params)

        # - Return the set of parameters, including transformed parameters
        return params

    def evolve(
        self: Union[Module, "ParameterTransformerMixin"],
        input_data,
        record: bool = False,
    ):
        # - Get a copy of the module and set the transformed parameters before evolve
        orig_params = self.parameters()
        self.set_attributes(self.transformed_parameters())

        # - Evolve the sub-module
        output_data, mod_state, record_dict = self.module.evolve(input_data, record)

        # - Restore the original parameters
        self.set_attributes(orig_params)

        # - Return the module state
        return output_data, {"module": mod_state}, {"module": record_dict}

    @abstractmethod
    def _transform(self, param: Any, tf_params: Any) -> Any:
        raise NotImplemented

    def _tree_map(self, func: Callable[[Any], Any], tree: Tree) -> Tree:
        if isinstance(tree, dict):  # if dict, apply to each key
            return {k: self._tree_map(func, v) for k, v in tree.items()}

        elif isinstance(tree, list):  # if list, apply to each element
            return [self._tree_map(func, elem) for elem in tree]

        elif isinstance(tree, tuple):  # if tuple, apply to each element
            return tuple([self._tree_map(func, elem) for elem in tree])

        else:
            #  - Apply function
            return func(tree)


try:
    from rockpool.nn.modules.jax.jax_module import JaxModule
    import jax.tree_util as tu
    import jax.random as rand
    import jax.numpy as jnp

    class JaxParameterTransformerMixin(ParameterTransformerMixin):
        def __init__(
            self,
            module: JaxModule,
            families: Optional[Union[Tuple, list, str]] = None,
            transform_params: Optional[Union[Tuple, list, Any]] = None,
            rng_key: Any = None,
            *args,
            **kwargs,
        ):
            # - Initialise superclass
            super().__init__(module, families, transform_params, *args, **kwargs)

            # - Seed RNG
            if rng_key is None:
                rng_key = rand.PRNGKey(onp.random.randint(0, 2**63))
            _, rng_key = rand.split(jnp.array(rng_key, dtype=jnp.uint32))

            # - Initialise state
            self.rng_key: jnp.ndarray = State(rng_key, init_func=lambda _: rng_key)

        @abstractmethod
        def _transform(self, param: Any, tf_params: Any, *args, **kwargs) -> Any:
            raise NotImplemented

        def _tree_map(self, func: Callable[[Any], Any], tree: Tree) -> Tree:
            return tu.tree_map(func, tree)

        def evolve(self, input_data, record: bool = False):
            # - Take a copy of the module
            mod = deepcopy(self)

            # - Split rng key before evolve
            mod.rng_key = rand.split(mod.rng_key)[0]

            # - Call evolve
            output_data, new_state, record_dict = ParameterTransformerMixin.evolve(
                mod, input_data, record
            )

            # - Make sure we return the new rng key for self as well
            new_state.update({"rng_key": mod.rng_key})
            return output_data, new_state, record_dict

except (ImportError, ModuleNotFoundError) as err:
    from rockpool.utilities.backend_management import missing_backend_shim

    JaxParameterTransformerMixin = missing_backend_shim(
        "JaxParameterTransformerMixin", "jax"
    )
