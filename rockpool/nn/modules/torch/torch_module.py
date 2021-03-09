"""
Provide a base class for build Torch-compatible modules
"""
from rockpool.nn.modules.module import Module

# from .parameters import Parameter, State, SimulationParameter

import torch
from torch import nn

import rockpool.parameters as rp

from copy import deepcopy

from typing import Iterable, Tuple, Any, Callable


class TorchModule(Module, nn.Module):
    """
    Base class for modules that are compatible with both Torch and Rockpool
    
    Use this base class to build Rockpool modules that use Torch as a backend. You can also use this class to convert a ``torch.nn.module`` to a Rockpool :py:class:`.Module` in one line.
    
    See Also:
        See :ref:`/in-depth/torch-api.ipynb` for details of using the Torch API.
        
    To implement a module from scratch using the Torch low-level API, simply inherit from :py:class:`.TorchModule` instead of ``torch.nn.Module``. You must implement the Torch API in the form of :py:meth:`.forward` etc. :py:class:`.TorchModule` will convert the API for you, and provides its own :py:meth:`.evolve` method. You should not implement the :py:meth:`.evolve` method yourself.  
    
    In your :py:meth:`.forward` method you should use the Torch API and semantics as usual. Sub-modules of a Rockpool :py:class:`.TorchModule` are expected to be Torch ``nn.Module`` s. Only the top-level module needs to be wrapped as a Rockpool :py:class:`.TorchModule`.
    
    :py:class:`.TorchModule` automatically converts Torch parameters to Rockpool :py:class:`.Parameter` s, and Torch named buffers to Rockpool :py:class:`.State` s. In this way calls to :py:meth:`.parameters` and :py:meth:`.state` function as expected. 
        
    Examples:
        
        Convert a ``torch`` module to a Rockpool :py:class:`.TorchModule`:
        
        >>> mod = TorchModule.from_torch(torch_mod)
    
    """

    def __init__(self, *args, **kwargs):
        """
        Initialise this module
        
        You must override this method to initialise your module.
        
        Args:
            *args: 
            **kwargs: 
        """
        # - Ensure super-class initialisation ocurs
        super().__init__(*args, **kwargs)

    def evolve(self, input_data, record: bool = False) -> Tuple[Any, Any, Any]:
        """
        Implement the Rockpool low-level evolution API
        
        :py:meth:`.evolve` is provided by :py:class:`.TorchModule` to connect the Rockpool low-level API to the Torch API (i.e. :py:meth:`.forward` etc.). You should *not* override :py:meth:`.evolve` if using :py:class:`.TorchModule` directly, but should implement the Torch API to perform evaluation of the module.  
        
        Args:
            input_data: This might be a numpy array or Torch tensor, containing the input data to evolve over
            record (bool): Iff ``True``, return a dictionary of state variables as ``record_dict``, containing the time series of those state variables over evolution. Default: ``False``, do not record state during evolution 

        Returns:
            (array, dict, dict): (output_data, new_states, record_dict)
                ``output_data`` is the output from the :py:class:`.TorchModule`, probably as a torch ``Tensor``.
                ``new_states`` is a dictionary containing the updated state for this module, post evolution.
                If the ``record`` argument is ``True``, ``record_dict`` is a dictionary containing the recorded state variables for this and all submodules, recorded over evolution.
        """

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
    def from_torch(cls: type, obj: nn.Module, retain_torch_api: bool = False) -> None:
        """
        Convert a torch module into a Rockpool :py:class:`.TorchModule` in-place
        
        Args:
            obj (torch.nn.Module): Torch module to convert to a Rockpool   
            retain_torch_api (bool): If ``True``, calling the resulting module will use the Torch API. Default: ``False``, convert the module to the Rockpool low-level API for :py:meth:`__call__`.
        """
        # - Check that we have a Torch ``nn.Module``
        if not isinstance(obj, nn.Module):
            raise TypeError("`from_torch` can only patch torch.nn.Module objects.")

        # - Patch a torch nn.Module to be a Rockpool TorchModule
        orig_call = obj.__call__
        old_class_name = obj.__class__.__name__

        class TorchModulePatch(obj.__class__, TorchModule):
            def __call__(self, *args, **kwargs):
                if retain_torch_api:
                    return orig_call(*args, **kwargs)
                else:
                    return super().__call__(*args, **kwargs)

            @property
            def class_name(self) -> str:
                return old_class_name

        obj.__class__ = TorchModulePatch
        obj.__old_class_name = old_class_name

        assert isinstance(obj, TorchModule)

        # - Ensure attribute registry is initialised
        _, __modules = obj._get_attribute_registry()

        # - Ensure other base-class attributes are set
        obj._shape = (None,)
        obj._spiking_input = False
        obj._spiking_output = False
        obj._name = obj._get_name()
        obj._submodulenames = []

        # - Identify torch buffers and parameters, and register them
        for name, param in obj.named_parameters(recurse=False):
            obj._register_attribute(name, rp.Parameter(param, None, None, param.shape))

        for name, buffer in obj.named_buffers(recurse=False):
            obj._register_attribute(name, rp.State(buffer, None, None, buffer.shape))

        # - Convert and register submodules
        for name, mod in obj.named_children():
            # - Convert submodule
            TorchModule.from_torch(mod, retain_torch_api=True)

            # - Assign submodule to Rockpool module dictionary
            __modules[name] = [mod, type(mod).__name__]
            obj._submodulenames.append(name)
