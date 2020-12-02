from rockpool.nn.modules.module import Module

# from .parameters import Parameter, State, SimulationParameter

import torch
from torch import nn

import rockpool.parameters as rp

import functools

from abc import ABC

from typing import Iterable, Tuple, Any, Callable


class TorchModule(Module, nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evolve(self, input_data, record: bool = False) -> Tuple[Any, Any, Any]:

        self.record = record

        # - Call nn.Module.__call__() method to get output data
        output_data = nn.Module.__call__(self, input_data)

        # - Build a new state dictionary
        new_states = self.state()

        # - No general solution as yet to access recorded states
        record_dict = {}

        return output_data, new_states, record_dict

    def __setattr__(self, key, value):
        if isinstance(value, nn.Parameter):
            # - Also register as a rockpool parameter
            print("Setting a new Torch Parameter", value)
            self._register_attribute(key, rp.Parameter(value, None, None, value.shape))

        if isinstance(value, (rp.Parameter, rp.State)):
            # - register as a torch buffer
            super().register_buffer(key, value.data)

            # - Register as a Rockpool attribute
            self._register_attribute(key, value)
        else:
            # - Call __setattr__
            super().__setattr__(key, value)

        if isinstance(value, nn.Module) and not isinstance(value, TorchModule):
            # - Convert torch module to a Rockpool Module and assign
            TorchModule.from_torch(value, retain_torch_api=True)
            super().__setattr__(key, value)

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        self._register_attribute(name, rp.State(tensor, None, None, tensor.shape))
        super().register_buffer(name, tensor)

    def register_parameter(self, name: str, param: nn.Parameter) -> None:
        self._register_attribute(name, rp.Parameter(param, None, None, param.shape))
        super().register_parameter(name, param)

    @classmethod
    def from_torch(cls: type, obj: nn.Module, retain_torch_api: bool = False):
        if not isinstance(obj, nn.Module):
            raise TypeError("`from_torch` can only patch torch.nn.Module objects.")

        # - Patch a torch nn.Module to be a Rockpool TorchModule
        orig_call = obj.__call__

        class MonkeyPatch(obj.__class__, TorchModule):
            def __call__(self, *args, **kwargs):
                if retain_torch_api:
                    return orig_call(*args, **kwargs)
                else:
                    return super().__call__(*args, **kwargs)

        obj.__class__ = MonkeyPatch

        assert isinstance(obj, TorchModule)

        # - Ensure attribute registry is initialised
        obj._get_attribute_registry()

        # - Ensure other base-class attributes are set
        obj._shape = (None,)
        obj._spiking_input = False
        obj._spiking_output = False
        obj._name = obj._get_name()

        # - Identify torch buffers and parameters, and register them
        for name, param in obj.named_parameters(recurse=False):
            obj._register_attribute(name, rp.Parameter(param, None, None, param.shape))

        for name, buffer in obj.named_buffers(recurse=False):
            obj._register_attribute(name, rp.State(buffer, None, None, buffer.shape))

        # - Convert and register submodules
        for name, mod in obj.named_children():
            TorchModule.from_torch(mod, retain_torch_api=True)
