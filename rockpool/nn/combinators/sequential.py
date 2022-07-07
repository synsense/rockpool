"""
Implement the :py:class:`.Sequential` combinator, with helper classes for Jax and Torch backends
"""

from rockpool.nn.modules.module import Module, ModuleBase
from rockpool import TSContinuous, TSEvent

from copy import copy
from typing import Tuple, Any
from abc import ABC

import rockpool.graph as rg

__all__ = ["Sequential"]


class SequentialMixin(ABC):
    """
    Base class for `Sequential` modules
    """

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
            if isinstance(item, ModuleBase):
                # - Collect the module and define a name
                submods.append(item)
                submod_names.append(f"{mod_index}_{item.class_name}")
                mod_index += 1

            else:
                other_args.append(item)

        # - Work out shape of each submodule
        shape_in = [mod.size_in for mod in submods]
        shape_out = [mod.size_out for mod in submods]

        # - Check that shapes are compatible
        for mod_index in range(len(submods) - 1):
            if (
                shape_out[mod_index] is not None
                and shape_in[mod_index + 1] is not None
                and shape_out[mod_index] != shape_in[mod_index + 1]
            ):
                raise ValueError(
                    f"The output of submodule {mod_index} "
                    + f"({type(submods[mod_index]).__name__}) "
                    + f"does not match the input shape of submodule "
                    + f"{mod_index+1} ({type(submods[mod_index+1]).__name__}): "
                    + f"{shape_out[mod_index]} ≠ {shape_in[mod_index+1]}"
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

        # - Record module list
        self._submodule_names = submod_names

    def evolve(self, input_data, record: bool = False) -> Tuple[Any, Any, Any]:
        # - Initialise state and record dictionaries
        new_state_dict = {}
        record_dict = {}

        x = input_data

        # - Loop through submodules
        for submod_name in self._submodule_names:
            # - Get this submodule and weight
            mod = getattr(self, submod_name)

            # - Push data through submodule
            x, substate, subrec = mod(x, record=record)
            new_state_dict.update({submod_name: substate})
            record_dict.update(
                {
                    submod_name: subrec,
                    f"{submod_name}_output": copy(x),
                }
            )

        # - Return output, state and record
        return x, new_state_dict, record_dict

    def __getitem__(self, item: int) -> Module:
        """
        Permit indexing into the sequence of modules

        Args:
            item (int): The index of the module to return

        Returns:
            Module: The ``item``th module in the sequence
        """
        return self.modules()[self._submodule_names[item]]

    def as_graph(self):
        mod_graphs = []

        for mod in self:
            mod_graphs.append(mod.as_graph())

        for source, dest in zip(mod_graphs[:-1], mod_graphs[1:]):
            rg.connect_modules(source, dest)

        return rg.GraphHolder(
            mod_graphs[0].input_nodes,
            mod_graphs[-1].output_nodes,
            f"{type(self).__name__}_{self.name}_{id(self)}",
            self,
        )

    def _wrap_recorded_state(self, state_dict: dict, t_start: float = 0.0) -> dict:
        # - Wrap each sub-dictionary in turn
        for mod_name in self._submodule_names:
            mod = self.modules()[mod_name]
            state_dict[mod_name].update(
                mod._wrap_recorded_state(state_dict[mod_name], t_start)
            )

            # - Wrap recorded output for this module
            output_key = f"{mod_name}_output"
            dt = mod.dt if hasattr(mod, "dt") else self.dt
            if mod.spiking_output:
                ts_output = TSEvent.from_raster(
                    state_dict[output_key][0],
                    dt=dt,
                    name=output_key,
                    t_start=t_start,
                )
            else:
                ts_output = TSContinuous.from_clocked(
                    state_dict[output_key][0],
                    dt=dt,
                    name=output_key,
                    t_start=t_start,
                )

            state_dict.update({output_key: ts_output})

        # - Return wrapped dictionary
        return state_dict


class ModSequential(SequentialMixin, Module):
    """
    The :py:class:`.Sequential` combinator for native modules
    """

    pass


try:
    from rockpool.nn.modules.jax.jax_module import JaxModule
    from jax import numpy as jnp

    class JaxSequential(SequentialMixin, JaxModule):
        """
        The :py:class:`.Sequential` combinator for Jax modules
        """

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            """Unflatten a tree of modules from Jax to Rockpool"""
            params, sim_params, state, modules, init_params = children
            _name, _shape, _submodulenames = aux_data
            modules = tuple(modules.values())
            obj = cls(*modules)
            obj._name = _name

            # - Restore configuration
            obj = obj.set_attributes(params)
            obj = obj.set_attributes(state)
            obj = obj.set_attributes(sim_params)

            return obj

except:

    class JaxModule:
        pass

    class JaxSequential:
        """
        The :py:class:`.Sequential` combinator for Jax modules
        """

        def __init__(self):
            raise ImportError(
                "'Jax' and 'Jaxlib' backend not found. Modules relying on Jax will not be available."
            )


try:
    from rockpool.nn.modules.torch.torch_module import TorchModule
    import torch
    from torch.nn import Module as torch_nn_module

    class TorchSequential(SequentialMixin, TorchModule):
        """
        The :py:class:`.Sequential` combinator for torch modules
        """

        def __init__(
            self,
            *args,
            **kwargs,
        ):
            # - Convert torch modules to Rockpool TorchModules
            for item in args:
                if isinstance(item, torch_nn_module) and not isinstance(
                    item, TorchModule
                ):
                    TorchModule.from_torch(item, retain_torch_api=False)

            # - Call super-class constructor
            super().__init__(*args, **kwargs)

        def forward(self, *args, **kwargs):
            # - By default, record state
            record = kwargs.get("record", True)
            kwargs["record"] = record

            # - Return output
            return self.evolve(*args, **kwargs)[0]

except:

    class TorchModule:
        pass

    class torch_nn_module:
        pass

    class TorchSequential:
        """
        The :py:class:`.Sequential` combinator for torch modules
        """

        def __init__(self):
            raise ImportError(
                "'Torch' backend not found. Modules relying on PyTorch will not be available."
            )


def Sequential(*args, **kwargs) -> ModuleBase:
    """
    Build a sequential stack of modules by connecting them end-to-end

    :py:class:`.Sequential` accepts any number of modules. The shapes of the modules must be compatible -- the output size :py:attr:`~.Module.size_out` of each module must match the input size :py:attr:`~.Module.size_in` of the following module.

    Examples:

        Build a :py:class:`.Sequential` stack will be returned a :py:class:`.Module`, containing ``mod0``, ``mod1`` and ``mod2``. When evolving this stack, signals will be passed through ``mod0``, then ``mod1``, then ``mod2``:

        >>> Sequential(mod0, mod1, mod2)

        Index into a :py:class:`.Sequential` stack using Python indexing:

        >>> mod = Sequential(mod0, mod1, mod2)
        >>> mod[0]
        A module with shape (xx, xx)

    Args:
        *mods: Any number of modules to connect. The :py:attr:`~.Module.size_out` attribute of one module must match the :py:attr:`~.Module.size_in` attribute of the following module.

    Returns:
        A :py:class:`.Module` subclass object that encapsulates the provided modules
    """
    # - Check for Jax and Torch submodules
    for item in args:
        if isinstance(item, JaxModule):
            return JaxSequential(*args, **kwargs)
        if isinstance(item, (TorchModule, torch_nn_module)):
            return TorchSequential(*args, **kwargs)

    # - Use ModSequential if no JaxModule or TorchModule is in the submodules
    return ModSequential(*args, **kwargs)
