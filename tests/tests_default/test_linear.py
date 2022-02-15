import pytest


def test_linear():
    from rockpool.nn.modules.native.linear import Linear

    import numpy as np

    mod = Linear((2, 10))

    input = np.random.rand(10, 2)

    output, ns, r_d = mod(input)
    mod.set_attributes(ns)

    assert output.shape == (10, 10)

    with pytest.raises(ValueError):
        mod = Linear(10)

    with pytest.raises(ValueError):
        mod = Linear((10,))

    with pytest.raises(ValueError):
        mod = Linear((10, 10, 10))


def test_linear_nobias():
    from rockpool.nn.modules.native.linear import Linear

    import numpy as np

    mod = Linear((2, 10), has_bias=False)

    input = np.random.rand(10, 2)

    output, ns, r_d = mod(input)
    mod.set_attributes(ns)

    assert output.shape == (10, 10)

    with pytest.raises(ValueError):
        mod = Linear(10, has_bias=False)

    with pytest.raises(ValueError):
        mod = Linear((10,), has_bias=False)

    with pytest.raises(ValueError):
        mod = Linear((10, 10, 10), has_bias=False)


def test_jaxlinear():
    pytest.importorskip("jax")

    from rockpool.nn.modules import LinearJax
    import jax
    import jax.numpy as jnp

    import numpy as np

    mod = LinearJax((2, 10))

    input = np.random.rand(10, 2)

    output, ns, r_d = mod(input)
    mod.set_attributes(ns)

    assert output.shape == (10, 10)

    with pytest.raises(ValueError):
        mod = LinearJax(10)

    with pytest.raises(ValueError):
        mod = LinearJax((10,))

    with pytest.raises(ValueError):
        mod = LinearJax((10, 10, 10))

    # - Test compiled
    je = jax.jit(mod)
    output, ns, r_d = je(input)
    mod.set_attributes(ns)

    # - Test gradient
    def loss_mse(params, net, input, target):
        net = net.set_attributes(params)
        output, _, _ = net(input)
        return jnp.sum((target - output) ** 2)

    loss_vgf = jax.jit(jax.value_and_grad(loss_mse))
    loss, grad = loss_vgf(mod.parameters(), mod, input, 0.0)
    loss, grad = loss_vgf(mod.parameters(), mod, input, 0.0)

    print("loss: ", loss)
    print("grad: ", grad)


def test_jaxlinear_nobias():
    pytest.importorskip("jax")

    from rockpool.nn.modules import LinearJax
    import jax
    import jax.numpy as jnp

    import numpy as np

    mod = LinearJax((2, 10), has_bias=False)

    input = np.random.rand(10, 2)

    output, ns, r_d = mod(input)
    mod.set_attributes(ns)

    assert output.shape == (10, 10)

    with pytest.raises(ValueError):
        mod = LinearJax(10, has_bias=False)

    with pytest.raises(ValueError):
        mod = LinearJax((10,), has_bias=False)

    with pytest.raises(ValueError):
        mod = LinearJax((10, 10, 10), has_bias=False)

    # - Test compiled
    je = jax.jit(mod)
    output, ns, r_d = je(input)
    mod.set_attributes(ns)

    # - Test gradient
    def loss_mse(params, net, input, target):
        net = net.set_attributes(params)
        output, _, _ = net(input)
        return jnp.sum((target - output) ** 2)

    loss_vgf = jax.jit(jax.value_and_grad(loss_mse))
    loss, grad = loss_vgf(mod.parameters(), mod, input, 0.0)
    loss, grad = loss_vgf(mod.parameters(), mod, input, 0.0)

    print("loss: ", loss)
    print("grad: ", grad)
