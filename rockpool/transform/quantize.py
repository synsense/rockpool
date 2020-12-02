from rockpool.nn.modules.module import Module
from rockpool.nn.modules.jax.jax_module import JaxModule

from rockpool.transform.param_transformer import (
    ParameterTransformerMixin,
    JaxParameterTransformerMixin,
)

import jax.random as rand
import jax.numpy as jnp
import numpy as onp


class JaxStochasticQuantize(JaxParameterTransformerMixin, JaxModule):
    def _transform(self, param: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        a = jnp.floor(param)
        return a + ((param - a) > rand.uniform(self.rng_key, shape=param.shape))


class ModStochasticQuantize(ParameterTransformerMixin, Module):
    def _transform(self, param: onp.ndarray, *args) -> onp.ndarray:
        a = onp.floor(param)
        return a + ((param - a) > onp.random.random(param.shape))


def StochasticQuantize(mod: Module, *args, **kwargs):
    if isinstance(mod, JaxModule):
        return JaxStochasticQuantize(mod, *args, **kwargs)
    else:
        return ModStochasticQuantize(mod, *args, **kwargs)
