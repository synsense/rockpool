"""
Provide a stochastic quantization parameter transformation module
"""

from rockpool.nn.modules.module import Module

from rockpool.transform.param_transformer import (
    ParameterTransformerMixin,
    JaxParameterTransformerMixin,
)

import numpy as onp
import warnings

__all__ = ["StochasticQuantize"]

try:
    from rockpool.nn.modules.jax.jax_module import JaxModule
    import jax.random as rand
    import jax.numpy as jnp

    class JaxStochasticQuantize(JaxParameterTransformerMixin, JaxModule):
        def _transform(self, param: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
            a = jnp.floor(param)
            return a + ((param - a) > rand.uniform(self.rng_key, shape=param.shape))


except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(f"Could not import module: {err}")

    class JaxStochasticQuantize:
        def __init__(self, *_, **__):
            raise ImportError("Jax backend not found.")


class ModStochasticQuantize(ParameterTransformerMixin, Module):
    def _transform(self, param: onp.ndarray, *args) -> onp.ndarray:
        a = onp.floor(param)
        return a + ((param - a) > onp.random.random(param.shape))


def StochasticQuantize(mod: Module, *args, **kwargs):
    if isinstance(mod, JaxModule):
        return JaxStochasticQuantize(mod, *args, **kwargs)
    else:
        return ModStochasticQuantize(mod, *args, **kwargs)
