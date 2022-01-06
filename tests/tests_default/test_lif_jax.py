from jax.config import config

config.update("jax_log_compiles", True)
config.update("jax_debug_nans", True)


def test_imports():
    from rockpool.nn.modules.jax.lif_jax import LIFJax


def test_lif_jax():
    from rockpool.nn.modules.jax.lif_jax import LIFJax

    from jax import jit

    import numpy as np

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
    _, new_state, _ = lyr.evolve(np.random.rand(T, N))
    lyr = lyr.set_attributes(new_state)

    print("evolving with call")
    _, new_state, _ = lyr(np.random.rand(T, N))
    lyr = lyr.set_attributes(new_state)

    _, new_state, _ = lyr(np.random.rand(T, N))
    lyr = lyr.set_attributes(new_state)

    lyr.Vmem = np.array([1.0] * N)

    print("evolving with jit")
    je = jit(lyr)
    _, new_state, _ = je(np.random.rand(T, N))
    lyr = lyr.set_attributes(new_state)

    _, new_state, _ = je(np.random.rand(T, N))
    lyr = lyr.set_attributes(new_state)

    ## - Test recurrent mode
    lyr = LIFJax((N, N), has_rec=True)

    print("evolving recurrent")
    o, ns, r_d = lyr(np.random.rand(T, N))
    lyr = lyr.set_attributes(ns)

    print("evolving recurrent with jit")
    je = jit(lyr)
    o, n_s, r_d = je(np.random.rand(T, N))
    lyr = lyr.set_attributes(n_s)


def test_ffwd_net():
    from rockpool.nn.modules.jax.lif_jax import LIFJax
    from rockpool.nn.modules.jax.jax_module import JaxModule
    from rockpool.parameters import Parameter

    import numpy as np
    import jax.numpy as jnp

    class my_ffwd_net(JaxModule):
        def __init__(self, shape, *args, **kwargs):
            super().__init__(shape, *args, **kwargs)

            for (index, (N_in, N_out)) in enumerate(zip(shape[:-1], shape[1:])):
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

    print(np.sum([np.sum(v ** 2) for v in net.parameters("weights").values()]))


def test_sgd():
    from rockpool.nn.modules import LIFJax, LinearJax
    from rockpool.nn.combinators import Sequential
    from rockpool.parameters import Parameter
    from rockpool.training.jax_loss import mse, l0_norm_approx

    import jax
    from jax import jit

    import numpy as np
    import jax.numpy as jnp

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

    from jax.experimental.optimizers import adam

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
