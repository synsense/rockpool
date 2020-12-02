def test_imports():
    from rockpool.jax_loss import mse, bounds_cost


def test_mse():
    from rockpool.rate_jax import RateEulerJax
    from jax.experimental.optimizers import adam

    import jax
    from jax import jit, numpy as jnp
    import numpy as np

    from rockpool.jax_loss import mse

    mod = RateEulerJax(2)
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

    with tqdm(range(100)) as t:
        for i in t:
            loss, grads = vgf(get_params(opt_state), mod, inputs, target)
            opt_state = update_fun(i, grads, opt_state)
            loss_t.append(loss)
            t.set_postfix({"loss": loss})

    print(f"Losses: [0] {loss_t[0]} .. [-1] {loss_t[-1]}")


def test_bounds_cost():
    from rockpool.rate_jax import RateEulerJax
    from jax.experimental.optimizers import adam

    from copy import deepcopy

    import jax
    from jax import jit, numpy as jnp
    import numpy as np

    from rockpool.jax_loss import bounds_cost, make_bounds

    mod = RateEulerJax((2, 2))
    params0 = mod.parameters()

    # - Get bounds
    min_bounds, max_bounds = make_bounds(params0)

    min_bounds["tau"] = 1e-3
    max_bounds["tau"] = 100e-3
    min_bounds["tau"] = 1e-3
    max_bounds["tau"] = 100e-3

    c = bounds_cost(params0, min_bounds, max_bounds)


def test_l2norm():
    from rockpool.rate_jax import RateEulerJax
    from rockpool.jax_loss import l2_norm

    mod = RateEulerJax((2, 2))

    c = l2_norm(mod.parameters("weights"))


def test_l0norm():
    from rockpool.rate_jax import RateEulerJax
    from rockpool.jax_loss import l0_norm_approx
    from rockpool.ffwd_stack import FFwdStack

    mod = RateEulerJax((2, 2))

    c = l0_norm_approx(mod.parameters("weights"))

    mod = FFwdStack(RateEulerJax(10), RateEulerJax((5, 5)), RateEulerJax(1))

    c = l0_norm_approx(mod.parameters("weights"))
