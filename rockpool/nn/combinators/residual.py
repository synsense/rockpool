"""
Implement the :py:class:`.Residual` combinator, with helper classes for Jax and Torch backends
"""

from rockpool.nn.modules.module import Module, ModuleBase
from rockpool.nn.combinators.sequential import SequentialMixin, JaxSequential
from rockpool.graph import AliasConnection, as_GraphHolder

from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)

from typing import Tuple, Any


class ResidualMixin(SequentialMixin):
    """
    The base class for the :py:class:`.Residual` combinator
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.size_in != self.size_out:
            raise ValueError(
                "`size_in` and `size_out` must be identical for a residual block."
            )

    def evolve(self, input_data, record: bool = False) -> Tuple[Any, Any, Any]:
        out, new_state_dict, record_dict = super().evolve(input_data, record)
        return out + input_data, new_state_dict, record_dict

    def as_graph(self):
        # - Use the Sequential `as_graph()` method
        graph = super().as_graph()

        # - Wrap it with an AliasConnection
        return as_GraphHolder(
            AliasConnection(
                graph.input_nodes,
                graph.output_nodes,
                f"Residual_{self.name}_aliases",
                None,
            )
        )


class ModResidual(ResidualMixin, Module):
    """
    The :py:class:`.Residual` combinator for native modules
    """

    pass


if backend_available("jax"):
    from rockpool.nn.modules.jax.jax_module import JaxModule
    from jax import numpy as jnp

    class JaxResidual(JaxSequential, ResidualMixin):
        """
        The :py:class:`.Residual` combinator for jax modules
        """

        pass

else:
    JaxResidual = missing_backend_shim("JaxResidual", "jax")

    class JaxModule:
        pass


if backend_available("torch"):
    from rockpool.nn.modules.torch.torch_module import TorchModule

    class TorchResidual(ResidualMixin, TorchModule):
        """
        The :py:class:`.Residual` combinator for torch modules
        """

        pass

else:
    TorchResidual = missing_backend_shim("TorchResidual", "torch")

    class TorchModule:
        pass


def Residual(*args, **kwargs) -> ModuleBase:
    """
    Build a residual block over a sequential stack of modules

    :py:class:`.Residual` accepts any number of modules. The shapes of the modules must be compatible -- the output size :py:attr:`~.Module.size_out` of each module must match the input size :py:attr:`~.Module.size_in` of the following module.

    Examples:

        Build a :py:class:`.Residual` stack will be returned a :py:class:`.Module`, containing ``mod0``, ``mod1`` and ``mod2``. When evolving this stack, signals will be passed through ``mod0``, then ``mod1``, then ``mod2``:

        >>> Residual(mod0, mod1, mod2)

        Index into a :py:class:`.Residual` stack using Python indexing:

        >>> mod = Residual(mod0, mod1, mod2)
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
            return JaxResidual(*args, **kwargs)
        if isinstance(item, TorchModule):
            return TorchResidual(*args, **kwargs)

    # - Use ModResidual if no JaxModule or TorchModule is in the submodules
    return ModResidual(*args, **kwargs)
