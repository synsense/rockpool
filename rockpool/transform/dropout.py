"""
Provide a Dropout parameter transformation
"""

from rockpool.nn.modules.module import Module

from rockpool.transform.param_transformer import (
    ParameterTransformerMixin,
    JaxParameterTransformerMixin,
)

import jax.random as rand
import jax.numpy as jnp
import numpy as onp

__all__ = ["Dropout"]

try:
    from rockpool.nn.modules.jax.jax_module import JaxModule

    class JaxDropout(JaxParameterTransformerMixin, JaxModule):
        def _transform(
            self, param: jnp.ndarray, prob_dropout: float = 0.5, *args, **kwargs
        ) -> jnp.ndarray:
            return param * rand.bernoulli(
                self.rng_key, p=jnp.array(1.0 - prob_dropout), shape=param.shape
            )

except (ImportError, ModuleNotFoundError) as err:
    from rockpool.utilities.backend_management import missing_backend_shim

    JaxDropout = missing_backend_shim("JaxDropout", "jax")

    class JaxModule:
        pass


class ModDropout(ParameterTransformerMixin, Module):
    def _transform(
        self, param: onp.ndarray, prob_dropout: float = 0.5, *args, **kwargs
    ) -> onp.ndarray:
        return param * (onp.random.random(param.shape) <= prob_dropout)


def Dropout(mod: Module, *args, **kwargs):
    if isinstance(mod, JaxModule):
        return JaxDropout(mod, *args, **kwargs)
    else:
        return ModDropout(mod, *args, **kwargs)
