import pytest

pytest.importorskip("jax")


def test_imports():
    from rockpool.training.adversarial_jax import (
        pga_attack,
        adversarial_loss,
    )


def test_adversarial_loss():
    PLOT = False
    from rockpool.training.jax_loss import mse
    from rockpool.training.adversarial_jax import (
        pga_attack,
        adversarial_loss,
    )
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
    inputs = np.random.normal(0, 1, (T, Nin))

    net = net.reset_state()
    net = net.set_attributes(parameters)
    output_nominal, _, _ = net(inputs)
    rng_key = random.PRNGKey(0)

    def f(parameters_flattened):
        theta_star, _ = pga_attack(
            params_flattened=parameters_flattened,
            net=net,
            rng_key=rng_key,
            attack_steps=attack_steps,
            mismatch_level=mismatch_level,
            initial_std=initial_std,
            inputs=inputs,
            net_out_original=output_nominal,
            tree_def_params=tree_def_params,
            mismatch_loss=tu.Partial(mse),
        )
        return theta_star

    theta_star, verbose = pga_attack(
        params_flattened=parameters_flattened,
        net=net,
        rng_key=rng_key,
        attack_steps=attack_steps,
        mismatch_level=mismatch_level,
        initial_std=initial_std,
        inputs=inputs,
        net_out_original=output_nominal,
        tree_def_params=tree_def_params,
        mismatch_loss=tu.Partial(mse),
    )

    diagonals = []
    for idx, p in enumerate(parameters_flattened):
        diagonal_tmp = jnp.zeros_like(p)
        for step in range(attack_steps):
            diagonal_tmp += (
                (mismatch_level * jnp.sign(p))
                / attack_steps
                * jnp.sign(verbose["grads"][step][idx])
            )
        diagonals.append(1 + jnp.reshape(diagonal_tmp, newshape=(-1,)))

    J = jacfwd(f)(parameters_flattened)
    W = []
    for idx, p in enumerate(parameters_flattened):
        J_sub = J[idx][idx]
        W.append(np.reshape(J_sub, newshape=(np.prod(p.shape), np.prod(p.shape))))

    for idx in range(len(parameters_flattened)):
        jacobian_diag = np.diagonal(W[idx])
        diag = diagonals[idx]
        assert np.allclose(jacobian_diag, diag, rtol=1e-2), "Not close"
        if PLOT:
            plt.plot(jacobian_diag, linestyle="--", color="b")
            plt.plot(diag, linestyle="solid", color="r")
    if PLOT:
        plt.show()

    loss = adversarial_loss(
        parameters=parameters,
        net=net,
        inputs=inputs,
        target=random.normal(rng_key, shape=(Nout, T)),
        task_loss=tu.Partial(mse),
        mismatch_loss=tu.Partial(mse),
        rng_key=rng_key,
        noisy_forward_std=0.2,
        initial_std=0.001,
        mismatch_level=0.01,
        beta_robustness=0.1,
        attack_steps=5,
    )

    assert not (np.array(loss).item() in [np.inf, np.nan]), "Loss is NaN/Inf"
