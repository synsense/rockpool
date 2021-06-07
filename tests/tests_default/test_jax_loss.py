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

if __name__ == "__main__":

    test_adversarial_loss()

    from jax import config
    config.FLAGS.jax_log_compiles=False
    config.update('jax_disable_jit', False)

    from rockpool.training import jax_loss as l
    from rockpool.training.jax_loss import adversarial_loss, split_and_sample
    from rockpool.nn.modules import LinearJax, InstantJax
    from rockpool.nn.combinators import Sequential

    import jax
    import jax.numpy as jnp
    import jax.tree_util as tu
    import jax.random as random
    from jax.experimental.optimizers import adam, sgd
    
    from tqdm.autonotebook import tqdm
    from copy import deepcopy
    from itertools import count
    import numpy as np

    np.random.seed(0)

    # - Import and configure matplotlib for plotting
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [7, 4]
    plt.rcParams['figure.dpi'] = 300

    # - Define a dataset class implementing the indexing interface
    class MultiClassRandomSinMapping:
        def __init__(self,
                    num_classes: int = 2,
                    sample_length: int = 100,
                    input_channels: int = 50,
                    target_channels: int = 2):
            # - Record task parameters
            self._num_classes = num_classes
            self._sample_length = sample_length

            # - Draw random input signals
            self._inputs = np.random.randn(num_classes, sample_length, input_channels) + 1.

            # - Draw random sinusoidal target parameters
            self._target_phase = np.random.rand(num_classes, 1, target_channels) * 2 * np.pi
            self._target_omega = np.random.rand(num_classes, 1, target_channels) * sample_length / 50

            # - Generate target output signals
            time_base = np.atleast_2d(np.arange(sample_length) / sample_length).T
            self._targets = np.sin(2 * np.pi * self._target_omega * time_base + self._target_phase)

        def __len__(self):
            # - Return the total size of this dataset
            return self._num_classes

        def __getitem__(self, i):
            # - Return the indexed dataset sample
            return self._inputs[i], self._targets[i]

    def loss_mse(parameters, net, inputs, target):
        net = net.reset_state()
        net = net.set_attributes(parameters)
        output, _, _ = net(inputs)
        return l.mse(output, target)

    # - Instantiate a dataset
    Nin = 2000
    Nout = 2
    num_classes = 3
    T = 100
    ds = MultiClassRandomSinMapping(num_classes = num_classes,
                                    input_channels = Nin,
                                    target_channels = Nout,
                                    sample_length = T)

    Nhidden = 8
    N_train = 100
    N_test = 50

    data = {
        "train": [el for el in  [sample for sample in ds] for _ in range(N_train)],
        "test": [el for el in  [sample for sample in ds] for _ in range(N_test)],
    }

    def train_net(net,
                loss_vgf,
                data,
                num_epochs=1000,
                training_loss=None,
                boundary_loss=None, 
                noisy_forward_std=None,
                initial_std=None,
                mismatch_level=None,
                beta_robustness=None,
                attack_steps=None,
    ):

        # - Define initial seed
        rand_key = random.PRNGKey(0)

        # - Get the optimiser functions
        init_fun, update_fun, get_params = adam(1e-4)

        # - Initialise the optimiser with the initial parameters
        params0 = deepcopy(net.parameters())
        opt_state = init_fun(params0)

        # - Compile the optimiser update function
        update_fun = jax.jit(update_fun)

        # - Record the loss values over training iterations
        loss_t = []
        grad_t = []

        # - Loop over iterations
        i_trial = count()
        for _ in tqdm(range(num_epochs)):
            for sample in data:
                # - Get an input / target sample
                input, target = sample[0], sample[1]

                # - Get parameters for this iteration
                params = get_params(opt_state)

                # - Split the random key
                rand_key, _ = random.split(rand_key)

                # - Get the loss value and gradients for this iteration
                if boundary_loss is None:
                    # - Normal training
                    loss_val, grads = loss_vgf(params, net, input, target)
                else:
                    loss_val, grads = loss_vgf(params, net, input, target, training_loss, boundary_loss, rand_key, noisy_forward_std, initial_std, mismatch_level, beta_robustness, attack_steps)

                # - Update the optimiser
                opt_state = update_fun(next(i_trial), grads, opt_state)

                # - Keep track of the loss
                loss_t.append(loss_val)

        return net, loss_t, params

    def eval_loss(inputs, target, net):
        output, _, _ = net(inputs)
        return l.mse(output,target)

    def get_average_loss_mismatch(data, mm_level, N, net, params, rand_key):
        params_flattened, tree_def_params = tu.tree_flatten(params)

        loss = []
        for _ in range(N):
            params_gaussian_flattened = []
            for p in params_flattened:
                rand_key, random_normal_var = split_and_sample(rand_key, p.shape)
                params_gaussian_flattened.append(p + jnp.abs(p)*mm_level*random_normal_var)

            params_gaussian = tu.tree_unflatten(tree_def_params, params_gaussian_flattened)
            net = net.set_attributes(params_gaussian)
            loss_tmp = []
            for sample in data:
                # - Get an input / target sample
                inputs, target = sample[0], sample[1]
                net = net.reset_state()
                loss_tmp.append(eval_loss(inputs, target, net))
            loss.append(np.mean(loss_tmp))
        return np.mean(loss), np.std(loss)

    num_epochs = 300

    # - Create network
    net = Sequential(
            LinearJax((Nin, Nhidden)),
            InstantJax(Nhidden, jnp.tanh),
            LinearJax((Nhidden, Nout)),
        )

    # - Train robust network
    loss_vgf = jax.jit(jax.value_and_grad(adversarial_loss), static_argnums=(4,5,11))
    net_robust, loss_t_robust, params_robust = train_net(net=deepcopy(net),
                                                        loss_vgf=loss_vgf,
                                                        data=data["train"],
                                                        num_epochs=num_epochs,
                                                        training_loss=l.mse,
                                                        boundary_loss=l.mse,
                                                        noisy_forward_std=0.0,
                                                        initial_std=0.001,
                                                        mismatch_level=0.025,
                                                        beta_robustness=0.25,
                                                        attack_steps=10)

    # - Train a standard network
    loss_vgf = jax.jit(jax.value_and_grad(loss_mse))
    net_standard, loss_t_standard, params_standard = train_net(net=deepcopy(net),
                                                            loss_vgf=loss_vgf,
                                                            data=data["train"],
                                                            num_epochs=num_epochs)

    mismatch_levels = [0.0,0.1,0.2,0.3,0.4,0.5,0.6]
    results = {
        "rob": {"mean":np.empty(len(mismatch_levels)), "std":np.empty(len(mismatch_levels))},
        "standard": {"mean":np.empty(len(mismatch_levels)), "std":np.empty(len(mismatch_levels))}
    }
    rand_key = random.PRNGKey(0)
    N_rep = 20
    for i,mm_level in enumerate(mismatch_levels):
        rob_mean, rob_std = get_average_loss_mismatch(data=data["test"], mm_level=mm_level, N=N_rep, net=net_robust, params=params_robust, rand_key=rand_key)
        standard_mean, standard_std = get_average_loss_mismatch(data=data["test"], mm_level=mm_level, N=N_rep, net=net_standard, params=params_standard, rand_key=rand_key)
        rand_key, _ = random.split(rand_key)
        results["rob"]["mean"][i] = rob_mean
        results["rob"]["std"][i] = rob_std
        results["standard"]["mean"][i] = standard_mean
        results["standard"]["std"][i] = standard_std

        print(f"ROBUST Mismatch level {mm_level} Loss {rob_mean}+-{rob_std}")
        print(f"STANDARD Mismatch level {mm_level} Loss {standard_mean}+-{standard_std} \n")

    x = np.arange(0,len(mismatch_levels),1)
    plt.plot(x, results["rob"]["mean"], color="r", label="Robust")
    plt.fill_between(x, results["rob"]["mean"]-results["rob"]["std"],results["rob"]["mean"]+results["rob"]["std"], alpha=0.1, color="r")

    plt.plot(x, results["standard"]["mean"], color="b", label="Standard")
    plt.fill_between(x, results["standard"]["mean"]-results["standard"]["std"],results["standard"]["mean"]+results["standard"]["std"], alpha=0.1, color="b")
    
    plt.gca().set_xticks(x)
    plt.gca().set_xticklabels([str(s) for s in mismatch_levels])
    plt.gca().set_xlabel(r"$\zeta_{attack}$")
    plt.legend()
    plt.show()