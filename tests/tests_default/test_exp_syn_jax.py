import pytest

pytest.importorskip("jax")


def test_ExpSynJax():
    from rockpool.nn.modules import ExpSynJax
    import numpy as np
    import jax
    from jax import config

    config.update("jax_debug_nans", True)

    T = 1000
    dt = 1e-3
    N = 4
    p = 0.1

    esMod = ExpSynJax(shape=N, tau=[10e-3, 50e-3, 100e-3, 200e-3], dt=dt)

    sp_rand = np.random.rand(T, N) < p

    out, ns, rs = esMod(sp_rand)
    print(out)

    jesMod = jax.jit(esMod)
    out, ns, rs = jesMod(sp_rand)
    print(out)

    def loss(params, net, input):
        net = net.set_attributes(params)
        output, _, _ = net(input)
        return np.sum(output)

    vgf = jax.jit(jax.value_and_grad(loss))
    l, g = vgf(esMod.parameters(), esMod, sp_rand)
    print(l, g)


def test_ExpSynJax_default():
    from rockpool.nn.modules import ExpSynJax
    import numpy as np
    import jax
    from jax import config

    config.update("jax_debug_nans", True)

    T = 1000
    dt = 1e-3
    N = 4
    p = 0.1

    esMod = ExpSynJax(shape=N, dt=dt)

    sp_rand = np.random.rand(T, N) < p

    out, ns, rs = esMod(sp_rand)
    print(out)

    jesMod = jax.jit(esMod)
    out, ns, rs = jesMod(sp_rand)
    print(out)

    def loss(params, net, input):
        net = net.set_attributes(params)
        output, _, _ = net(input)
        return np.sum(output)

    vgf = jax.jit(jax.value_and_grad(loss))
    l, g = vgf(esMod.parameters(), esMod, sp_rand)
    print(l, g)


def test_ExpSynJax_single_tau():
    from rockpool.nn.modules import ExpSynJax
    import numpy as np
    import jax
    from jax import config

    config.update("jax_debug_nans", True)

    T = 1000
    dt = 1e-3
    N = 4
    p = 0.1

    esMod = ExpSynJax(shape=N, tau=20e-3, dt=dt)

    sp_rand = np.random.rand(T, N) < p

    out, ns, rs = esMod(sp_rand)
    print(out)

    jesMod = jax.jit(esMod)
    out, ns, rs = jesMod(sp_rand)
    print(out)

    def loss(params, net, input):
        net = net.set_attributes(params)
        output, _, _ = net(input)
        return np.sum(output)

    vgf = jax.jit(jax.value_and_grad(loss))
    l, g = vgf(esMod.parameters(), esMod, sp_rand)
    print(l, g)


def test_ExpSynJax_constant():
    from rockpool.nn.modules import ExpSynJax
    from rockpool.parameters import Constant
    import numpy as np
    import jax
    from jax import config

    config.update("jax_debug_nans", True)

    T = 1000
    dt = 1e-3
    N = 4
    p = 0.1

    esMod = ExpSynJax(shape=N, tau=Constant(20e-3), dt=dt)

    sp_rand = np.random.rand(T, N) < p

    out, ns, rs = esMod(sp_rand)
    print(out)

    jesMod = jax.jit(esMod)
    out, ns, rs = jesMod(sp_rand)
    print(out)

    def loss(params, net, input):
        net = net.set_attributes(params)
        output, _, _ = net(input)
        return np.sum(output)

    vgf = jax.jit(jax.value_and_grad(loss))
    l, g = vgf(esMod.parameters(), esMod, sp_rand)
    print(l, g)
