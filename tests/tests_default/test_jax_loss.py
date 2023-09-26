"""
Unit tests for jax training utilities, loss functions
"""


def test_imports():
    import pytest

    pytest.importorskip("jax")

    from rockpool.training.jax_loss import (
        mse,
        bounds_cost,
        mse,
        sse,
        l0_norm_approx,
        l2sqr_norm,
        softmax,
        logsoftmax,
    )

    # - Ensure that NaNs in compiled functions are errors
    from jax.config import config


def test_mse():
    import pytest

    pytest.importorskip("jax")

    from rockpool.nn.modules import RateJax
    from jax.example_libraries.optimizers import adam

    import jax
    from jax import jit
    import numpy as np

    from rockpool.training.jax_loss import mse

    # - Ensure that NaNs in compiled functions are errors
    from jax.config import config

    config.update("jax_debug_nans", True)

    # - Seed for reproducibility
    np.random.seed(1)

    mod = RateJax(2)
    params0 = mod.parameters()

    init_fun, update_fun, get_params = adam(1e-2)

    update_fun = jit(update_fun)

    opt_state = init_fun(params0)
    inputs = np.random.rand(10, 2)
    target = np.random.rand(10, 2)

    def loss(grad_params, net, input, target):
        net = net.set_attributes(grad_params)
        output, ns, _ = net(input)

        return mse(output, target)

    loss_t = []
    vgf = jit(jax.value_and_grad(loss))

    from tqdm.autonotebook import tqdm

    with tqdm(range(5)) as t:
        for i in t:
            loss, grads = vgf(get_params(opt_state), mod, inputs, target)
            opt_state = update_fun(i, grads, opt_state)
            loss_t.append(loss)
            t.set_postfix({"loss": loss})

    print(f"Losses: [0] {loss_t[0]} .. [-1] {loss_t[-1]}")


def test_bounds_cost():
    import pytest

    pytest.importorskip("jax")

    from rockpool.nn.modules.jax.rate_jax import RateEulerJax
    from rockpool.training.jax_loss import bounds_cost, make_bounds
    from rockpool.training.jax_debug import flatten

    from jax.example_libraries.optimizers import adam

    from copy import deepcopy

    import jax
    from jax import jit, numpy as jnp
    import numpy as np

    # - Ensure that NaNs in compiled functions are errors
    from jax.config import config

    config.update("jax_debug_nans", True)

    mod = RateEulerJax((2, 2))
    params0 = mod.parameters()

    # - Get bounds
    min_bounds, max_bounds = make_bounds(params0)

    min_bounds["tau"] = 1.0
    min_bounds["tau"] = 1.0

    c = bounds_cost(params0, min_bounds, max_bounds)
    print(c)

    c = jit(bounds_cost)(params0, min_bounds, max_bounds)
    print(c)

    bounds_vgf = jax.jit(jax.value_and_grad(bounds_cost))
    c, grads = bounds_vgf(params0, min_bounds, max_bounds)
    for g in flatten(grads).values():
        assert ~np.any(np.isnan(g)), "NaN gradients found"
    print(grads)

    max_bounds["bias"] = -1.0
    max_bounds["w_rec"] = -1.0
    c, grads = bounds_vgf(params0, min_bounds, max_bounds)
    print(c, grads)

    def cost(p):
        return jnp.nanmean(p["tau"]) + bounds_cost(p, min_bounds, max_bounds)

    g = jax.jit(jax.grad(cost))(params0)
    print(g)


def test_l2sqr_norm():
    import pytest

    pytest.importorskip("jax")

    import jax
    from rockpool.nn.modules.jax.rate_jax import RateJax
    from rockpool.training.jax_loss import l2sqr_norm

    # - Ensure that NaNs in compiled functions are errors
    from jax.config import config

    config.update("jax_debug_nans", True)

    mod = RateJax(2, has_rec=True)

    c = l2sqr_norm(mod.parameters("weights"))

    # - Test compiled version and gradients
    c = jax.jit(l2sqr_norm)(mod.parameters("weights"))
    print(c)
    g = jax.jit(jax.grad(l2sqr_norm))(mod.parameters("weights"))
    print(g)


def test_l0norm():
    import pytest

    pytest.importorskip("jax")

    from rockpool.nn.modules.jax.rate_jax import RateJax
    from rockpool.training.jax_loss import l0_norm_approx
    from rockpool.nn.combinators.ffwd_stack import FFwdStack
    import jax

    mod = RateJax(2, has_rec=True)

    c = l0_norm_approx(mod.parameters("weights"))

    mod = FFwdStack(RateJax(10), RateJax(5, has_rec=True), RateJax(1))

    c = l0_norm_approx(mod.parameters("weights"))

    # - Test compiled version and gradients
    c = jax.jit(l0_norm_approx)(mod.parameters("weights"))
    g = jax.jit(jax.grad(l0_norm_approx))(mod.parameters("weights"))


def test_softmax():
    import pytest

    pytest.importorskip("jax")

    from rockpool.training.jax_loss import softmax
    import numpy as np
    import jax

    # - Ensure that NaNs in compiled functions are errors
    from jax.config import config

    config.update("jax_debug_nans", True)

    N = 3
    temp = 0.01
    softmax(np.random.rand(N))
    jax.jit(softmax)(np.random.rand(N))

    softmax(np.random.rand(N), temp)
    jax.jit(softmax)(np.random.rand(N), temp)


def test_logsoftmax():
    import pytest

    pytest.importorskip("jax")

    from rockpool.training.jax_loss import logsoftmax
    import numpy as np
    import jax

    # - Ensure that NaNs in compiled functions are errors
    from jax.config import config

    config.update("jax_debug_nans", True)

    N = 3
    temp = 0.01
    logsoftmax(np.random.rand(N))
    jax.jit(logsoftmax)(np.random.rand(N))

    logsoftmax(np.random.rand(N), temp)
    jax.jit(logsoftmax)(np.random.rand(N), temp)
