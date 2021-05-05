import pytest


def test_imports():
    from rockpool.nn.modules import SoftmaxJax, LogSoftmaxJax


def test_SoftmaxJax():
    from rockpool.nn.modules import SoftmaxJax

    import numpy as np
    import jax
    import jax.numpy as jnp

    Nin = 3
    Nout = 5
    T = 10

    # - Test module instantiation
    mod = SoftmaxJax((Nin, Nout))

    # - Test evolution and compilation
    out, _, _ = mod(np.random.rand(T, Nin))
    out, _, _ = jax.jit(mod)(np.random.rand(T, Nin))

    # - Test module shape inference from concrete weights
    mod = SoftmaxJax(weight=np.zeros((Nin, Nout)))
    assert mod.shape == (Nin, Nout)

    # - Test gradient
    def loss(params, mod, inputs):
        mod = mod.set_attributes(params)
        out, _, _ = mod(inputs)
        return jnp.sum(out)

    loss_vgf = jax.grad(loss)
    l, g = loss_vgf(mod.parameters(), mod, np.random.rand(T, Nin))
    l, g = jax.jit(loss_vgf)(mod.parameters(), mod, np.random.rand(T, Nin))


def test_LogSoftmaxJax():
    from rockpool.nn.modules import LogSoftmaxJax

    import numpy as np
    import jax
    import jax.numpy as jnp

    Nin = 3
    Nout = 5
    T = 10

    # - Test module instantiation
    mod = LogSoftmaxJax((Nin, Nout))

    # - Test evolution and compilation
    out, _, _ = mod(np.random.rand(T, Nin))
    out, _, _ = jax.jit(mod)(np.random.rand(T, Nin))

    # - Test module shape inference from concrete weights
    mod = LogSoftmaxJax(weight=np.zeros((Nin, Nout)))
    assert mod.shape == (Nin, Nout)

    # - Test gradient
    def loss(params, mod, inputs):
        mod = mod.set_attributes(params)
        out, _, _ = mod(inputs)
        return jnp.sum(out)

    loss_vgf = jax.grad(loss)
    l, g = loss_vgf(mod.parameters(), mod, np.random.rand(T, Nin))
    l, g = jax.jit(loss_vgf)(mod.parameters(), mod, np.random.rand(T, Nin))
