"""
Implements linear weight matrix modules for numpy and Jax
"""


from rockpool.nn.modules.module import Module
from rockpool.parameters import Parameter

import numpy as onp
from warnings import warn

from typing import Tuple, Any

from abc import ABC


class LinearMixin(ABC):
    """
    Encapsulate a linear weight matrix
    """

    _dot = None

    def __init__(
        self,
        shape,
        weight_init_func=lambda s: onp.random.uniform(
            -onp.sqrt(1 / s[0]), onp.sqrt(1 / s[0]), s
        ),
        weight=None,
        *args,
        **kwargs
    ):
        """
        Encapsulate a linear weight matrix

        `.Linear` essentially wraps a single weight matrix, and passes data through by using the matrix as a set of weights. The shape of the matrix must be specified as a tuple ``(Nin, Nout)``.

        A weight initialisation function may be specified. By default the weights will be Normally distributed around zero, and normalised by N_in.

        Examples:

            Build a linear weight matrix with shape ``(3, 4)``:

            >>> Linear((3, 4))
            Linear  with shape (3, 4)

            Build a linear weight matrix with shape ``(2, 5)``, which will be initialised with zeros:

            >>> Linear((2, 5), weight_init_func = lambda s: np.zeros(s))
            Linear  with shape (2, 5)

            Provide a concrete initialisation for the linear weights:

            >>> Linear((2, 2), weight = np.array([[1, 2], [3, 4]]))
            Linear  with shape (2, 2)

        Args:
            shape (tuple): The desired shape of the weight matrix. Must have two entries ``(Nin, Nout)``
            weight_init_func (Callable): The initialisation function to use for the weights. Default: Uniform on the range :math:`(\\sqrt(1/Nin), \\sqrt(1/Nin))`
            weight (Optional[np.array]): A concrete weight matrix to assign to the weights on initialisation. ``weight.shape`` must match the ``shape`` argument
        """
        # - Base class must be `Module`
        if not isinstance(self, Module):
            raise TypeError(
                "`LinearMixin` mix-in class may only be used with `Module` classes."
            )

        # - Call superclass init
        super().__init__(shape=shape, *args, **kwargs)

        if len(self.shape) != 2:
            raise ValueError("`shape` must specify input and output sizes for Linear.")

        # - Specify weight parameter
        self.weight = Parameter(
            weight, shape=self.shape, init_func=weight_init_func, family="weights"
        )

    def evolve(self, input_data, record: bool = False) -> Tuple[Any, Any, Any]:
        return self._dot(input_data, self.weight), {}, {}


class Linear(LinearMixin, Module):
    """
    Encapsulates a linear weight matrix
    """

    _dot = staticmethod(onp.dot)
    pass


# - Graceful failure if Jax is not available
try:
    from rockpool.nn.modules.jax.jax_module import JaxModule
    import jax.numpy as jnp

    class LinearJax(LinearMixin, JaxModule):
        """
        Encapsulates a linear weight matrix, with a Jax backend
        """

        _dot = staticmethod(jnp.dot)
        pass


except:
    warn(
        "'Jax' and 'Jaxlib' backend not found. Modules that rely on Jax will not be available."
    )

    class LinearJax:
        def __init__(self):
            raise ImportError(
                "'Jax' and 'Jaxlib' backend not found. Modules that rely on Jax will not be available."
            )
