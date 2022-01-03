import pytest


def test_imports():
    from rockpool.nn.combinators import Residual


def test_JaxResidual():
    from rockpool.nn.combinators import Residual
    from rockpool.nn.modules import LinearJax, LIFJax
    from rockpool.training import jax_loss as jl

    import numpy as np
    import jax
    import jax.tree_util as jtu

    N = 10
    T = 20

    mod = Residual(LinearJax((N, N)), LIFJax((N,)))

    input = np.random.rand(T, N)

    # - Test evolution
    out, state_dict, rec_dict = mod(input, record=True)

    # - test jit evolution
    jmod = jax.jit(mod)
    jout, state_dict, rec_dict = jmod(input, record=True)

    assert np.allclose(out, jout), "JIT output does not match non-compiled output"

    # - Test gradients
    def loss(params, net, input, target):
        net = net.set_attributes(params)
        out, _, _ = net(input)
        return jl.mse(out, input)

    grad_fn = jax.value_and_grad(loss)

    l_val = loss(mod.parameters(), mod, input, 10.0)
    gl_val, g = grad_fn(mod.parameters(), mod, input, 10.0)

    assert np.allclose(
        l_val, gl_val
    ), "Loss value after gradient calculation does not match loss before gradient"

    check_grads, _ = jtu.tree_flatten(
        jtu.tree_map(lambda p: np.all(np.abs(p) < 1e-10), g)
    )

    assert not any(check_grads), "All gradients for one parameter are zero"

    print(g)


def test_TorchResidual():
    from rockpool.nn.combinators import Residual
    from rockpool.nn.modules import LinearTorch, LIFTorch

    import torch as t

    N = 10
    T = 20

    mod = Residual(LinearTorch((N, N)), LIFTorch((N,)))

    input = t.rand((1, T, N))

    # - Test evolution
    out, state_dict, rec_dict = mod(input, record=True)


def test_NativeResidual():
    from rockpool.nn.combinators import Residual
    from rockpool.nn.modules import Linear

    N = 10
    T = 30

    mod = Residual(
        Linear((N, 20)),
        Linear((20, N)),
    )

    import numpy as np

    input = np.random.rand(T, N)

    # - Test evolution
    out, state_dict, rec_dict = mod(input, record=True)
