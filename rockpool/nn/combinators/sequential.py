"""
Implement the :py:class:`.Sequential` combinator, with helper classes for Jax and Torch backends
"""

from rockpool.nn.modules.module import Module, ModuleBase
from rockpool import TSContinuous, TSEvent

from copy import copy
from typing import Tuple, Any, Optional, Union
from abc import ABC

from collections import OrderedDict

import rockpool.graph as rg

__all__ = ["Sequential"]


class SequentialMixin(ABC):
    """
    Base class for :py:class:`.Sequential` modules
    """

    def __init__(self, *args, **kwargs):
        """
        Initialise a :py:class:`.Sequential` module
        """
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

        # - Extract OrderedDict modules from arguments list
        if len(args) > 0 and isinstance(args[0], OrderedDict):
            submods = args[0]
            args = args[1:]
            other_args = []
            mod_index = 1
        else:
            submods = OrderedDict()
            other_args = []
            mod_index = 0

        # - Extract additional modules from arguments list
        for item in args:
            if isinstance(item, ModuleBase):
                # - Collect the module and define a name
                name = f"{mod_index}_{item.class_name}"
                submods[name] = item
                mod_index += 1
            else:
                other_args.append(item)

        # - Call super-class initialisation
        super().__init__(shape=(0, 0), *other_args, **kwargs)

        # - Call `append` for each module
        [self.append(mod, name) for name, mod in submods.items()]

    def append(self, mod: ModuleBase, name: Optional[str] = None) -> ModuleBase:
        """
        Append a module to the :py:class:`.Sequential` network stack

        Args:
            mod (Module): A rockpool :py:class:`.Module` to append to this network stack. The input size of `mod` must match the output size of the existing network.
            name (str): An optional name to assign to the new module. If ``None``, a name will automatically be generated.
        """
        # - Get a name and index for this module
        mod_index = len(self._submodulenames)

        if name is None:
            name = f"{mod_index}_{mod.class_name}"

        if name in self._submodulenames:
            raise ValueError(
                f'Submodule "{name}" already exists in Sequential network. Cannot append a module with the same name.'
            )

        # - Check if the shapes are compatible
        if len(self._submodulenames) == 0:
            self._shape = mod.shape
            self._spiking_input = mod._spiking_input
            self._spiking_output = mod._spiking_output
        elif (
            self.size_out != mod.size_in
            and self.size_out is not None
            and mod.size_in is not None
        ):
            raise ValueError(
                f"The output of submodule {mod_index-1} "
                + f"({type(self[-1]).__name__}) "
                + f"does not match the input shape of submodule "
                + f"{mod_index} ({type(mod).__name__}): "
                + f"{self[-1].size_out} ≠ {mod.size_in}"
            )

        # - Assign module
        setattr(self, name, mod)

        # - Fix shape and output type
        self._shape = (self.size_in, mod.size_out)
        self._spiking_output = mod.spiking_output

    def evolve(self, input_data, record: bool = False) -> Tuple[Any, Any, Any]:
        # - Initialise state and record dictionaries
        new_state_dict = {}
        record_dict = {}

        x = input_data

        # - Loop through submodules
        for submod_name in self._submodulenames:
            # - Get this submodule
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

    def __getitem__(self, item: Union[int, str]) -> Module:
        """
        Permit indexing into the sequence of modules

        Args:
            item (Union[int, str]): The index of the module to return, or the name of the module to access

        Returns:
            Module: The ``item``th module in the sequence
        """
        if isinstance(item, str):
            return Module.modules(self)[item]
        else:
            return Module.modules(self)[self._submodulenames[item]]

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
        for mod_name in self._submodulenames:
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

    When provided with a list of modules, :py:class:`.Sequential` will assign module names automatically to each module. If you would like more control over module names, you can provide an `OrderedDict` to construct the network. In that case, dictionary keys will be used as module names.

    You can also append additional modules to a network with the :py:meth:`.Sequential.append` method. Module names can optionally be provided in this case as well.

    Examples:

        Build a :py:class:`.Sequential` stack will be returned a :py:class:`.Module`, containing ``mod0``, ``mod1`` and ``mod2``. When evolving this stack, signals will be passed through ``mod0``, then ``mod1``, then ``mod2``:

        >>> Sequential(mod0, mod1, mod2)

        Index into a :py:class:`.Sequential` stack using Python indexing:

        >>> mod = Sequential(mod0, mod1, mod2)
        >>> mod[0]
        A module with shape (xx, xx)

        Build a :py:class:`.Sequential` stack from an `OrderedDict`:

        >>> od = OrderedDict([('mod0', mod0), ('mod1', mod1)])
        >>> seq = Sequential(od)

        Build an empty :py:class:`.Sequential`, and use :py:meth:`.Sequential.append`:

        >>> seq = Sequential()
        >>> seq.append(mod0)
        >>> seq.append(mod1, 'mod1)

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
