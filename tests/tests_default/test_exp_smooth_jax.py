import pytest


def test_imports():
    from rockpool.nn.modules.jax.exp_smooth_jax import ExpSmoothJax


def test_ExpSmoothJax():
    from rockpool.nn.modules.jax.exp_smooth_jax import ExpSmoothJax
    import numpy as np
    import jax.numpy as jnp
    import jax

    N = 5
    T = 10
    mod = ExpSmoothJax(N)
    out, _, _ = mod(np.random.rand(T, N))

    mod = ExpSmoothJax(shape=(N,), tau=10e-3)
    out, _, _ = mod(np.random.rand(T, N))

    mod = ExpSmoothJax(N, activation_fun=jnp.tanh)
    out, _, _ = mod(np.random.rand(T, N))

    out, _, _ = jax.jit(mod)(np.random.rand(T, N))

    def loss(params, mod, inputs):
        mod = mod.set_attributes(params)
        out, _, _ = mod(inputs)
        return jnp.sum(out)

    loss_vgf = jax.value_and_grad(loss)
    l, g = loss_vgf(mod.parameters(), mod, np.random.rand(T, N))
    l, g = jax.jit(loss_vgf)(mod.parameters(), mod, np.random.rand(T, N))
