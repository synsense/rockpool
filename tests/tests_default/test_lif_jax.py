def test_imports():
    import pytest

    pytest.importorskip("jax")

    from rockpool.nn.modules.jax.lif_jax import LIFJax
    import jax
    from jax import config


def test_lif_jax():
    import pytest

    pytest.importorskip("jax")

    from rockpool.nn.modules.jax.lif_jax import LIFJax
    from jax import jit
    import numpy as np
    import jax
    from jax import config

    config.update("jax_enable_x64", True)
    config.update("jax_log_compiles", True)
    config.update("jax_debug_nans", True)
    config.update("jax_check_tracer_leaks", True)

    np.random.seed(1)

    Nin = 4
    Nout = 2
    T = 20
    lyr = LIFJax((Nin, Nout))

    # - Test getting and setting
    p = lyr.parameters()
    lyr.set_attributes(p)

    s = lyr.state()
    lyr.set_attributes(s)

    sp = lyr.simulation_parameters()
    lyr.set_attributes(sp)

    print("evolve func")
    _, new_state, _ = lyr.evolve(np.random.rand(T, Nin))
    lyr = lyr.set_attributes(new_state)

    print("evolving with call")
    _, new_state, _ = lyr(np.random.rand(T, Nin))
    lyr = lyr.set_attributes(new_state)

    _, new_state, _ = lyr(np.random.rand(T, Nin))
    lyr = lyr.set_attributes(new_state)

    lyr.vmem = np.array([1.0] * Nout)

    def grad_check(params, mod, input):
        mod = mod.set_attributes(params)
        out, _, _ = mod(input)
        return np.sum(out**2)

    @jit
    def evolve(state, mod, input):
        mod = mod.set_attributes(state)
        return mod(input)

    print("evolving with jit")
    _, new_state, _ = evolve(lyr.state(), lyr, np.random.rand(T, Nin))
    _, new_state, _ = evolve(new_state, lyr, np.random.rand(T, Nin))

    lyr = LIFJax((Nin, Nout))
    gf = jit(jax.grad(grad_check))
    grads = gf(lyr.parameters(), lyr, np.random.rand(T, Nin))

    assert not np.allclose(
        grads["tau_syn"], np.zeros_like(grads["tau_syn"])
    ), "`tau_syn` gradients are zero in FFwd mode."
    assert not np.allclose(
        grads["tau_mem"], np.zeros_like(grads["tau_mem"])
    ), "`tau_mem` gradients are zero in FFwd mode."
    assert not np.allclose(
        grads["bias"], np.zeros_like(grads["bias"])
    ), "`bias` gradients are zero in FFwd mode."
    assert not np.allclose(
        grads["threshold"], np.zeros_like(grads["threshold"])
    ), "`threshold` gradients are zero in FFwd mode."

    ## - Test recurrent mode
    lyr = LIFJax((Nin, Nout), has_rec=True)

    print("evolving recurrent")
    o, ns, r_d = lyr(np.random.rand(T, Nin))
    lyr = lyr.set_attributes(ns)

    print("evolving recurrent with jit")
    o, n_s, r_d = evolve(lyr.state(), lyr, np.random.rand(T, Nin))
    o, n_s, r_d = evolve(n_s, lyr, np.random.rand(T, Nin))

    lyr = LIFJax((Nin, Nout), has_rec=True)
    gf = jit(jax.grad(grad_check))
    grads = gf(lyr.parameters(), lyr, np.random.rand(T, Nin))

    assert not np.allclose(
        grads["tau_syn"], np.zeros_like(grads["tau_syn"])
    ), "`tau_syn` gradients are zero in recurrent mode."
    assert not np.allclose(
        grads["tau_mem"], np.zeros_like(grads["tau_mem"])
    ), "`tau_mem` gradients are zero in recurrent mode."
    assert not np.allclose(
        grads["bias"], np.zeros_like(grads["bias"])
    ), "`bias` gradients are zero in recurrent mode."
    assert not np.allclose(
        grads["threshold"], np.zeros_like(grads["threshold"])
    ), "`threshold` gradients are zero in recurrent mode."
    assert not np.allclose(
        grads["w_rec"], np.zeros_like(grads["w_rec"])
    ), "`w_rec` gradients are zero in recurrent mode."


def test_ffwd_net():
    import pytest

    pytest.importorskip("jax")

    from rockpool.nn.modules.jax.lif_jax import LIFJax
    from rockpool.nn.modules.jax.jax_module import JaxModule
    from rockpool.parameters import Parameter

    import numpy as np
    import jax.numpy as jnp

    np.random.seed(1)

    from jax import config

    config.update("jax_enable_x64", True)
    config.update("jax_log_compiles", True)
    config.update("jax_debug_nans", True)
    config.update("jax_check_tracer_leaks", True)

    class my_ffwd_net(JaxModule):
        def __init__(self, shape, *args, **kwargs):
            super().__init__(shape, *args, **kwargs)

            for index, (N_in, N_out) in enumerate(zip(shape[:-1], shape[1:])):
                setattr(
                    self,
                    f"weight_{index}",
                    Parameter(
                        shape=(N_in, N_out),
                        init_func=np.random.standard_normal,
                        family="weights",
                    ),
                )

                setattr(self, f"lif_{index}", LIFJax(N_out))

        def evolve(self, input, record: bool = False):
            new_state = {}
            record_dict = {}
            for layer in range(len(self._shape) - 1):
                w = getattr(self, f"weight_{layer}")
                mod_name = f"lif_{layer}"
                lif = getattr(self, mod_name)

                outputs, substate, subrec = lif(jnp.dot(input, w), record=record)
                new_state.update({mod_name: substate})
                record_dict.update({mod_name: subrec})

                input = outputs[0]

            return input, new_state, record_dict

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            params, sim_params, state, modules = children
            _name, _shape, _submodulenames = aux_data

            obj = my_ffwd_net(_shape)
            obj._name = _name

            # - Restore parameters and configuration
            obj.set_attributes(params)
            obj.set_attributes(sim_params)
            obj.set_attributes(state)

            return obj

    net = my_ffwd_net([2, 3, 2])
    print(net(np.random.rand(1, 2)))
    print(net.parameters())

    print(net.state())
    _, ns, _ = net(np.random.rand(10, 2))
    net = net.set_attributes(ns)
    _, ns, _ = net(np.random.rand(10, 2))
    net = net.set_attributes(ns)
    _, ns, _ = net(np.random.rand(10, 2))
    net = net.set_attributes(ns)
    print(ns)
    print(net.state())

    print(net.parameters("weights"))

    print(np.sum([np.sum(v**2) for v in net.parameters("weights").values()]))


def test_sgd():
    import pytest

    pytest.importorskip("jax")

    from rockpool.nn.modules import LIFJax, LinearJax
    from rockpool.nn.combinators import Sequential
    from rockpool.training.jax_loss import mse, l0_norm_approx

    from jax import jit
    import jax
    from jax import config
    import numpy as np

    np.random.seed(1)

    config.update("jax_enable_x64", True)
    config.update("jax_log_compiles", True)
    config.update("jax_debug_nans", True)
    config.update("jax_check_tracer_leaks", True)

    print("Instantiating sequential net")
    net = Sequential(LinearJax((2, 3)), LIFJax(3), LinearJax((3, 1)), LIFJax(1))
    print("Testing sequential net jit")
    jnet = jit(net)
    jnet(np.random.rand(10, 2))

    def mse_loss(grad_params, net, input, target):
        net = net.reset_state()
        net = net.set_attributes(grad_params)
        outputs, _, _ = net(input)

        return mse(outputs, target) + l0_norm_approx(net.parameters("weights"))

    # net = my_ffwd_net([2, 3, 2])
    params0 = net.parameters()

    print("Testing loss function")
    mse_loss(params0, net, np.random.rand(10, 2), np.random.rand(10, 1))

    vgf = jax.value_and_grad(mse_loss)

    print("Testing loss grad jit")
    loss, grads = vgf(params0, net, np.random.rand(10, 2), np.random.rand(10, 1))
    print(loss, grads)

    from jax.example_libraries.optimizers import adam

    init_fun, update_fun, get_params = adam(1e-2)

    update_fun = jit(update_fun)

    opt_state = init_fun(params0)
    inputs = np.random.rand(10, 2)
    target = np.random.rand(10, 1)

    loss_t = []
    vgf = jax.value_and_grad(mse_loss)

    from tqdm.autonotebook import tqdm

    print("Testing training loop")
    with tqdm(range(2)) as t:
        for i in t:
            loss, grads = vgf(get_params(opt_state), net, inputs, target)
            opt_state = update_fun(i, grads, opt_state)
            loss_t.append(loss)
            t.set_postfix({"loss": loss})

    print(f"Losses: [0] {loss_t[0]} .. [-1] {loss_t[-1]}")


def test_lif_jax_batches():
    import pytest

    pytest.importorskip("jax")

    from rockpool.nn.modules.jax.lif_jax import LIFJax
    from jax import jit
    import numpy as np

    import jax

    jax.config.update("jax_check_tracer_leaks", True)

    batches = 5
    N = 10
    T = 20
    lyr = LIFJax(N)

    # - Test getting and setting
    p = lyr.parameters()
    lyr.set_attributes(p)

    s = lyr.state()
    lyr.set_attributes(s)

    sp = lyr.simulation_parameters()
    lyr.set_attributes(sp)

    print("evolve func")
    _, new_state, _ = lyr.evolve(np.random.rand(batches, T, N))
    lyr = lyr.set_attributes(new_state)

    print("evolving with call")
    _, new_state, _ = lyr(np.random.rand(batches, T, N))
    lyr = lyr.set_attributes(new_state)

    _, new_state, _ = lyr(np.random.rand(batches, T, N))
    lyr = lyr.set_attributes(new_state)

    lyr.vmem = np.array([1.0] * N)

    print("evolving with jit")
    je = jit(lyr)
    _, new_state, _ = je(np.random.rand(batches, T, N))
    lyr = lyr.set_attributes(new_state)

    _, new_state, _ = je(np.random.rand(batches, T, N))
    lyr = lyr.set_attributes(new_state)

    ## - Test recurrent mode
    lyr = LIFJax((N, N), has_rec=True)

    print("evolving recurrent")
    o, ns, r_d = lyr(np.random.rand(batches, T, N))
    lyr = lyr.set_attributes(ns)

    print("evolving recurrent with jit")
    je = jit(lyr)
    o, n_s, r_d = je(np.random.rand(batches, T, N))
    lyr = lyr.set_attributes(n_s)


def test_linear_lif():
    import pytest

    pytest.importorskip("jax")
    import jax

    jax.config.update("jax_check_tracer_leaks", True)

    import numpy as np

    from rockpool.nn.combinators import Sequential
    from rockpool.nn.modules import LIFJax, LinearJax

    import numpy as np

    # - Generate a network using the sequential combinator
    Nin = 200
    N = 50
    Nout = 1
    dt = 1e-3

    np.random.seed(1)

    mod = Sequential(
        LinearJax((Nin, N), has_bias=False, spiking_input=True),
        LIFJax(N, dt=dt, tau_syn=100e-3, tau_mem=200e-3, has_rec=True),
        LinearJax((N, Nout), has_bias=False),
    )

    import rockpool.training.jax_loss as l

    # - Define a loss function
    def loss_mse(params, net, input, target):
        # - Reset the network state
        net = net.reset_state()

        # - Apply the parameters
        net = net.set_attributes(params)

        # - Evolve the network
        output, _, states = net(input, record=True)

        # - Return the loss
        return l.mse(output, target)

    import jax

    jax.jit(jax.grad(loss_mse))(mod.parameters(), mod, 1.0, 0.0)
