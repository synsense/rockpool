from pytest import raises


def test_linear():
    from rockpool.linear import Linear

    import numpy as np

    mod = Linear((2, 10))

    input = np.random.rand(10, 2)

    output, ns, r_d = mod(input)
    mod.set_attributes(ns)

    assert output.shape == (10, 10)

    with raises(ValueError):
        mod = Linear(10)

    with raises(ValueError):
        mod = Linear((10,))

    with raises(ValueError):
        mod = Linear((10, 10, 10))


def test_jaxlinear():
    from rockpool.linear import JaxLinear
    from jax import jit

    import numpy as np

    mod = JaxLinear((2, 10))

    input = np.random.rand(10, 2)

    output, ns, r_d = mod(input)
    mod.set_attributes(ns)

    assert output.shape == (10, 10)

    with raises(ValueError):
        mod = JaxLinear(10)

    with raises(ValueError):
        mod = JaxLinear((10,))

    with raises(ValueError):
        mod = JaxLinear((10, 10, 10))

    # - Test compiled
    je = jit(mod)
    output, ns, r_d = je(input)
    mod.set_attributes(ns)
