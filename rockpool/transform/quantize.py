"""
Provide a stochastic quantization parameter transformation module
"""

from rockpool.nn.modules.module import Module

from rockpool.transform.param_transformer import (
    ParameterTransformerMixin,
    JaxParameterTransformerMixin,
)

import numpy as onp

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
    from rockpool.utilities.backend_management import missing_backend_shim

    JaxStochasticQuantize = missing_backend_shim("JaxStochasticQuantize", "jax")

    class JaxModule:
        pass


class ModStochasticQuantize(ParameterTransformerMixin, Module):
    def _transform(self, param: onp.ndarray, *args) -> onp.ndarray:
        a = onp.floor(param)
        return a + ((param - a) > onp.random.random(param.shape))


def StochasticQuantize(mod: Module, *args, **kwargs):
    if isinstance(mod, JaxModule):
        return JaxStochasticQuantize(mod, *args, **kwargs)
    else:
        return ModStochasticQuantize(mod, *args, **kwargs)
