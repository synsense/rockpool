"""
Unit tests for jax training utilities, loss functions
"""
# - Ensure that NaNs in compiled functions are errors
from jax.config import config

config.update("jax_debug_nans", True)


def test_imports():
    from rockpool.training.jax_loss import (
        mse,
        bounds_cost,
        mse,
        sse,
        l0_norm_approx,
        l2sqr_norm,
        softmax,
        logsoftmax,
        adversarial_loss,
        split_and_sample
    )


def test_mse():
    from rockpool.nn.modules.jax.rate_jax import RateEulerJax
    from jax.experimental.optimizers import adam

    import jax
    from jax import jit, numpy as jnp
    import numpy as np

    from rockpool.training.jax_loss import mse

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
    from rockpool.nn.modules.jax.rate_jax import RateEulerJax
    from rockpool.training.jax_loss import bounds_cost, make_bounds
    from rockpool.training.jax_debug import flatten

    from jax.experimental.optimizers import adam

    from copy import deepcopy

    import jax
    from jax import jit, numpy as jnp
    import numpy as np

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
    import jax
    from rockpool.nn.modules.jax.rate_jax import RateEulerJax
    from rockpool.training.jax_loss import l2sqr_norm

    mod = RateEulerJax((2, 2))

    c = l2sqr_norm(mod.parameters("weights"))

    # - Test compiled version and gradients
    c = jax.jit(l2sqr_norm)(mod.parameters("weights"))
    print(c)
    g = jax.jit(jax.grad(l2sqr_norm))(mod.parameters("weights"))
    print(g)


def test_l0norm():
    from rockpool.nn.modules.jax.rate_jax import RateEulerJax
    from rockpool.training.jax_loss import l0_norm_approx
    from rockpool.nn.combinators.ffwd_stack import FFwdStack
    import jax

    mod = RateEulerJax((2, 2))

    c = l0_norm_approx(mod.parameters("weights"))

    mod = FFwdStack(RateEulerJax(10), RateEulerJax((5, 5)), RateEulerJax(1))

    c = l0_norm_approx(mod.parameters("weights"))

    # - Test compiled version and gradients
    c = jax.jit(l0_norm_approx)(mod.parameters("weights"))
    g = jax.jit(jax.grad(l0_norm_approx))(mod.parameters("weights"))


def test_softmax():
    from rockpool.training.jax_loss import softmax
    import numpy as np
    import jax

    N = 3
    temp = 0.01
    softmax(np.random.rand(N))
    jax.jit(softmax)(np.random.rand(N))

    softmax(np.random.rand(N), temp)
    jax.jit(softmax)(np.random.rand(N), temp)


def test_logsoftmax():
    from rockpool.training.jax_loss import logsoftmax
    import numpy as np
    import jax

    N = 3
    temp = 0.01
    logsoftmax(np.random.rand(N))
    jax.jit(logsoftmax)(np.random.rand(N))

    logsoftmax(np.random.rand(N), temp)
    jax.jit(logsoftmax)(np.random.rand(N), temp)

def test_adversarial_loss():
    PLOT = False
    from rockpool.training.jax_loss import (
        robustness_loss,
        pga_attack,
        adversarial_loss,
        mse)
    from jax import jacfwd
    from rockpool.nn.combinators import Sequential
    from rockpool.nn.modules import LinearJax, InstantJax
    import jax.numpy as jnp
    import jax.random as random
    import jax.tree_util as tu
    import numpy as np
    if PLOT:
        import matplotlib.pyplot as plt

    np.random.seed(0)

    Nin = 2
    Nhidden = 10
    T = 100
    Nout = 1

    net = Sequential(
            LinearJax((Nin, Nhidden)),
            InstantJax(Nhidden, jnp.tanh),
            LinearJax((Nhidden, Nout)),
        )
    
    parameters = net.parameters()
    parameters_flattened, tree_def_params = tu.tree_flatten(parameters)
    attack_steps = 10
    mismatch_level = 0.2
    initial_std = 0.001
    inputs = np.random.normal(0,1,(T,Nin))

    net = net.reset_state()
    net = net.set_attributes(parameters)
    output_theta, _, _ = net(inputs)
    rand_key = random.PRNGKey(0)

    def f(parameters_flattened):
        theta_star, _ =  pga_attack(params_flattened=parameters_flattened,
                                    net=net,
                                    rand_key=rand_key,
                                    attack_steps=attack_steps,
                                    mismatch_level=mismatch_level,
                                    initial_std=initial_std,
                                    inputs=inputs,
                                    output_theta=output_theta,
                                    tree_def_params=tree_def_params,
                                    boundary_loss=mse)
        return theta_star

    theta_star, verbose = pga_attack(params_flattened=parameters_flattened,
                                    net=net,
                                    rand_key=rand_key,
                                    attack_steps=attack_steps,
                                    mismatch_level=mismatch_level,
                                    initial_std=initial_std,
                                    inputs=inputs,
                                    output_theta=output_theta,
                                    tree_def_params=tree_def_params,
                                    boundary_loss=mse)

    diagonals = []
    for idx,p in enumerate(parameters_flattened):
        diagonal_tmp = jnp.zeros_like(p)
        for step in range(attack_steps):
            diagonal_tmp += (mismatch_level * jnp.sign(p)) / attack_steps * jnp.sign(verbose["grads"][step][idx])
        diagonals.append(1 + jnp.reshape(diagonal_tmp, newshape=(-1,)))

    J = jacfwd(f)(parameters_flattened)
    W = []
    for idx,p in enumerate(parameters_flattened):
        J_sub = J[idx][idx]
        W.append(np.reshape(J_sub, newshape=(np.prod(p.shape),np.prod(p.shape))))
    
    for idx in range(len(parameters_flattened)):
        jacobian_diag = np.diagonal(W[idx])
        diag = diagonals[idx]
        assert np.allclose(jacobian_diag, diag, rtol=1e-2), "Not close"
        if PLOT:
            plt.plot(jacobian_diag, linestyle="--", color="b")
            plt.plot(diag, linestyle="solid", color="r")
    if PLOT:
        plt.show()

    loss = adversarial_loss(parameters=parameters,
                        net=net,
                        inputs=inputs,
                        target=random.normal(rand_key, shape=(Nout,T)),
                        training_loss=mse,
                        boundary_loss=mse,
                        rand_key=rand_key,
                        noisy_forward_std=0.2,
                        initial_std=0.001,
                        mismatch_level=0.01,
                        beta_robustness=0.1,
                        attack_steps=5)

    assert not (loss in [np.inf,np.nan]), "Loss is NaN/Inf"