"""
Linear module with a Jax backend
"""

from rockpool.nn.modules.native.linear import LinearMixin
from rockpool.nn.modules.jax.jax_module import JaxModule
import jax.numpy as jnp


class LinearJax(LinearMixin, JaxModule):
    """
    Encapsulates a linear weight matrix, with a Jax backend
    """

    _dot = staticmethod(jnp.dot)
    pass
