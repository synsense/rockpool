from rockpool.nn.modules.module import Module
from rockpool.nn.modules.jax.jax_module import JaxModule
from rockpool.parameters import Parameter

import numpy as onp
import jax.numpy as jnp

from typing import Tuple, Any

from abc import ABC


class LinearMixin(ABC):
    _dot = None

    def __init__(
        self,
        shape,
        weight_init_func=lambda s: onp.random.standard_normal(s) / s[-1],
        weight=None,
        *args,
        **kwargs
    ):
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
        self.weight = Parameter(weight, shape=self.shape, init_func=weight_init_func, family='weights')

    def evolve(self, input_data, record: bool = False) -> Tuple[Any, Any, Any]:
        return self._dot(input_data, self.weight), {}, {}


class Linear(LinearMixin, Module):
    _dot = staticmethod(onp.dot)
    pass


class JaxLinear(LinearMixin, JaxModule):
    _dot = staticmethod(jnp.dot)
    pass
