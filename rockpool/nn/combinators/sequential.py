from .jax_module import JaxModule
from .module import Module
from .parameters import Parameter

from typing import Tuple, Any

from jax import numpy as jnp
import numpy as onp

from abc import ABC


class SequentialMixin(ABC):
    _dot = None

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # - Check that `shape` wasn't provided as a keyword argument
        if "shape" in kwargs:
            raise ValueError(
                "You may not provide a `shape` argument when building a Sequential module."
            )

        if "spiking_input" in kwargs:
            raise ValueError(
                "You may not provide a `spiking_input` argument when building a Sequential module."
            )

        if "spiking_output" in kwargs:
            raise ValueError(
                "You may not provide a `spiking_output` argument when building a Sequential module."
            )

        # - Collect the submodules
        submods = []
        submod_names = []
        other_args = []
        mod_index = 0
        for item in args:
            if isinstance(item, Module):
                # - Collect the module and define a name
                submods.append(item)
                submod_names.append(f"{mod_index}_{item.class_name}")
                mod_index += 1
            else:
                other_args.append(item)

        if len(submods) < 2:
            raise ValueError("Sequential expects at least two modules to combine.")

        # - Work out shape of each submodule
        shape_in = [mod.size_in for mod in submods]
        shape_out = [mod.size_out for mod in submods]

        # - Check that shapes are compatible
        for mod_index in range(len(submods) - 1):
            if shape_out[mod_index] != shape_in[mod_index + 1]:
                raise ValueError(
                    f"The output of submodule {mod_index} "
                    + "({type(submods[mod_index]).__name__}) "
                    + "does not match the input shape of submodule "
                    + "{mod_index+1} ({type(submods[mod_index+1]).__name__})"
                )

        # - Call superclass __init__
        super().__init__(
            shape=(shape_in[0], shape_out[-1]),
            spiking_input=submods[0].spiking_input,
            spiking_output=submods[-1].spiking_output,
            *other_args,
            **kwargs,
        )

        # - Assign modules as submodules
        for (mod_name, submod) in zip(submod_names, submods):
            setattr(
                self,
                mod_name,
                submod,
            )

        # - Record module and weight lists
        self._submodule_names = submod_names

    def evolve(self, input_data, record: bool = False) -> Tuple[Any, Any, Any]:
        # - Initialise state and record dictionaries
        new_state_dict = {}
        record_dict = {}

        # - Loop through submodules
        for submod_name in self._submodule_names:
            # - Get this submodule and weight
            mod = getattr(self, submod_name)

            # - Push data through submodule
            input_data, substate, subrec = mod(input_data, record=record)
            new_state_dict.update({submod_name: substate})
            record_dict.update(
                {
                    submod_name: subrec,
                    f"{submod_name}_output": input_data,
                }
            )

        # - Return output, state and record
        return input_data, new_state_dict, record_dict


class ModSequential(SequentialMixin, Module):
    _dot = staticmethod(onp.dot)
    pass


class JaxSequential(SequentialMixin, JaxModule):
    _dot = staticmethod(jnp.dot)
    pass


def Sequential(*args, **kwargs) -> SequentialMixin:
    # - Check for Jax submodules
    use_jax = False
    for item in args:
        if isinstance(item, JaxModule):
            use_jax = True

    # - Use either the JaxSequential or ModSequential classes
    if use_jax:
        return JaxSequential(*args, **kwargs)
    else:
        return ModSequential(*args, **kwargs)
