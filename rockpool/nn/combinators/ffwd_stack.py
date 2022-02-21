"""
Implement a combinator that creates feed-forward module stacks, by placing a linear module in between each module
"""

from rockpool.nn.modules.module import Module
from rockpool.parameters import Parameter


from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)

from typing import Tuple, Any

import numpy as onp

from abc import ABC

__all__ = ["FFwdStack"]


class FFwdStackMixin(ABC):
    """
    Assemble modules into a feed-forward linear stack, with linear weights in between
    """

    _dot = None

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # - Check that `shape` wasn't provided as a keyword argument
        if "shape" in kwargs:
            raise ValueError(
                "You may not provide a `shape` argument when building a FFwdStack module."
            )

        if "spiking_input" in kwargs:
            raise ValueError(
                "You may not provide a `spiking_input` argument when building a FFwdStack module."
            )

        if "spiking_output" in kwargs:
            raise ValueError(
                "You may not provide a `spiking_output` argument when building a FFwdStack module."
            )

        if "weight_init_func" not in kwargs:
            raise ValueError(
                "`weight_init_func` must be provided on constructing a FFwdStack module."
            )
        weight_init_func = kwargs.pop("weight_init_func")

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
            raise ValueError("FFwdStack expects at least two modules to combine.")

        # - Work out shape of each layer
        shape_in = [mod.shape[0] for mod in submods]
        shape_out = [mod.shape[-1] for mod in submods]

        # - Generate weight shapes
        weight_shapes = list(zip(shape_out[:-1], shape_in[1:]))

        # - Generate weight names
        weight_names = [f"{n}_{n+1}_weight" for n in range(len(weight_shapes))]

        # - Call superclass __init__
        super().__init__(
            shape=(shape_in[0], shape_out[-1]),
            spiking_input=submods[0].spiking_input,
            spiking_output=submods[-1].spiking_output,
            *other_args,
            **kwargs,
        )

        # - Generate weight parameters
        for (w_name, w_shape) in zip(weight_names, weight_shapes):
            setattr(
                self,
                w_name,
                Parameter(
                    shape=w_shape,
                    family="weights",
                    init_func=weight_init_func,
                ),
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
        self._weight_names = weight_names

    def evolve(self, input_data, record: bool = False) -> Tuple[Any, Any, Any]:
        # - Initialise state and record dictionaries
        new_state_dict = {}
        record_dict = {}

        # - Loop through submodules and weights
        for (submod_name, weight_name) in zip(
            self._submodule_names[:-1], self._weight_names
        ):
            # - Get this submodule and weight
            mod = getattr(self, submod_name)
            weight = getattr(self, weight_name)

            # - Push data through submodule
            input_data, substate, subrec = mod(input_data, record=record)
            new_state_dict.update({submod_name: substate})
            record_dict.update(
                {
                    submod_name: subrec,
                    f"{submod_name}_output": input_data,
                }
            )

            # - Push data through weight
            if isinstance(input_data, tuple):
                input_data = input_data[0]
            input_data = self._dot(input_data, weight)

        # - Push data through final module
        mod = getattr(self, self._submodule_names[-1])
        input_data, substate, subrec = mod(input_data, record=record)
        new_state_dict.update({self._submodule_names[-1]: substate})
        record_dict.update({self._submodule_names[-1]: subrec})

        # - Return output, state and record
        return input_data, new_state_dict, record_dict


class ModFFwdStack(FFwdStackMixin, Module):
    _dot = staticmethod(onp.dot)
    pass


if backend_available("jax"):
    from jax import numpy as jnp
    from rockpool.nn.modules.jax.jax_module import JaxModule

    class JaxFFwdStack(FFwdStackMixin, JaxModule):
        _dot = staticmethod(jnp.dot)
        pass

else:
    JaxFFwdStack = missing_backend_shim("JaxFFwdStack", "jax")

    class JaxModule:
        pass


if backend_available("torch"):
    from rockpool.nn.modules.torch.torch_module import TorchModule
    import torch

    class TorchFFwdStack(FFwdStackMixin, TorchModule):
        _dot = staticmethod(torch.matmul)
        pass

else:
    TorchFFwdStack = missing_backend_shim("TorchFFwdStack", "torch")

    class TorchModule:
        pass


def FFwdStack(*args, **kwargs):
    """
    Assemble modules into a feed-forward stack, with linear weights in between

    `.FFwdStack` accepts any number of modules as positional arguments, along with the required keyword argument `weight_init_func`.

    The weights placed in between each module will map the :py:attr:`~.Module.size_out` of one module with the :py:attr:`~.Module.size_in` of the following module. Weights are not placed on the input or output of the stack.

    Examples:

        >>> FFwdStack(mod0, mod1, weight_init_func = lambda s: np.random.normal(size = s))

        A stack with two modules and one set of linear weights is generated. The weights will have shape ``(mod0.size_out, mod1.size_in)``.

    Args:
        *mods (Module): Any number of modules
        weight_init_func (Callable): A function that accepts a tuple defining the shape of a matrix, and returns a matrix of that shape to be used as a set of weights
    """
    # - Check for Jax submodules
    use_jax = False
    for item in args:
        if isinstance(item, JaxModule):
            use_jax = True
            break

    # - Check for Torch submodultes
    use_torch = False
    for item in args:
        if isinstance(item, TorchModule):
            use_torch = True
            break

    # - Use either the JaxFFwdStack or ModFFwdStack classes
    if use_jax:
        if "weight_init_func" not in kwargs:
            kwargs.update({"weight_init_func": jnp.zeros})
        return JaxFFwdStack(*args, **kwargs)
    elif use_torch:
        if "weight_init_func" not in kwargs:
            kwargs.update({"weight_init_func": torch.zeros})
        return TorchFFwdStack(*args, **kwargs)
    else:
        if "weight_init_func" not in kwargs:
            kwargs.update({"weight_init_func": onp.zeros})
        return ModFFwdStack(*args, **kwargs)
