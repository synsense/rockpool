"""
Encapsulate a simple instantaneous function as a jax module
"""

from rockpool.nn.modules import Module
from rockpool.parameters import SimulationParameter

from warnings import warn

from typing import Callable, Union

__all__ = ["Instant", "InstantJax", "InstantTorch"]


class InstantMixin:
    """
    Wrap a callable function as an instantaneous Rockpool module
    """

    def __init__(
        self,
        shape: tuple = None,
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
        self.function: Union[Callable, SimulationParameter] = SimulationParameter(
            function
        )

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


try:
    from rockpool.nn.modules.jax.jax_module import JaxModule
    from jax.tree_util import Partial

    class InstantJax(InstantMixin, JaxModule):
        """
        Wrap a callable function as an instantaneous Rockpool module, with a Jax backend
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.function = Partial(self.function)


except (ImportError, ModuleNotFoundError) as err:
    warn(
        "'Jax' and 'Jaxlib' backend not found. Modules that rely on Jax will not be available.\n{err}"
    )

    class InstantJax:
        def __init__(self, *_, **__):
            raise ImportError(
                "'Jax' and 'Jaxlib' backend not found. Modules that rely on Jax will not be available."
            )


try:
    from rockpool.nn.modules.torch.torch_module import TorchModule

    class InstantTorch(InstantMixin, TorchModule):
        """
        Wrap a callable function as an instantaneous Rockpool module, with a Torch backend
        """


except (ImportError, ModuleNotFoundError) as err:
    warn("'torch' backend not found. Modules that rely on Torch will not be available.")

    class InstantTorch:
        def __init__(self, *_, **__):
            raise ImportError(
                "'torch' backend not found. Modules that rely on Torch will not be available."
            )
