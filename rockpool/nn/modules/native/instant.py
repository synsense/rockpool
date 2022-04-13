"""
Encapsulate a simple instantaneous function as a jax module
"""

from rockpool.nn.modules import Module
from rockpool.parameters import SimulationParameter
from rockpool.typehints import P_Callable

from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)

from warnings import warn

from typing import Callable, Union

__all__ = ["Instant", "InstantJax", "InstantTorch"]


class InstantMixin:
    """
    Wrap a callable function as an instantaneous Rockpool module
    """

    def __init__(
        self,
        shape: Union[int, tuple] = None,
        function: Callable = lambda x: x,
        *args,
        **kwargs,
    ):
        """
        Wrap a callable function as an instantaneous Rockpool module

        Args:
            shape (Optional[tuple]):
            function (Callable): A scalar function of its arguments, with a single output. Default: identity function
        """
        # - Check that a shape was provided
        if shape is None:
            raise ValueError("The `shape` argument to `Instant` may not be `None`.")

        # - Call superclass init
        super().__init__(shape=shape, *args, **kwargs)

        # - Store the function
        self.function: P_Callable = SimulationParameter(function)

    def evolve(
        self,
        input,
        record: bool = False,
    ) -> (tuple, tuple, tuple):
        return self.function(input), {}, {}


class Instant(InstantMixin, Module):
    """
    Wrap a callable function as an instantaneous Rockpool module
    """

    pass


if backend_available("jax"):
    from rockpool.nn.modules.jax.jax_module import JaxModule
    from jax.tree_util import Partial

    class InstantJax(InstantMixin, JaxModule):
        """
        Wrap a callable function as an instantaneous Rockpool module, with a Jax backend
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.function = Partial(self.function)

else:
    InstantJax = missing_backend_shim("InstantJax", "jax")


if backend_available("torch"):
    from rockpool.nn.modules.torch.torch_module import TorchModule

    class InstantTorch(InstantMixin, TorchModule):
        """
        Wrap a callable function as an instantaneous Rockpool module, with a Torch backend
        """

else:
    InstantTorch = missing_backend_shim("InstantTorch", "torch")
