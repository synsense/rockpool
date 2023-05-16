import pytest

pytest.importorskip("jax")


def test_imports():
    from rockpool.nn.modules import RateJax
    from rockpool.nn.modules import JaxModule
    from jax.config import config


def test_assignment_jax():
    from rockpool.nn.modules import RateJax
    import numpy as np

    # - Construct a module
    mod = RateJax(5)

    # - Modify attributes and test assignment
    p = mod.parameters()
    p["tau"] = np.array(p["tau"])
    p["tau"][:] = 3.0

    mod = mod.set_attributes(p)
    assert np.all(mod.tau == 3.0)

    # - Test direct assignment from top level
    tau = np.array(mod.tau)
    tau[:] = 4.0
    mod.tau = tau
    assert np.all(mod.tau == 4.0)


def test_rate_jax():
    from rockpool.nn.modules import RateJax

    import jax
    from jax import jit
    import jax.numpy as jnp

    import numpy as np
    from jax.config import config

    config.update("jax_log_compiles", True)

    # - Generate module
    lyr = RateJax(
        shape=2,
        tau=np.random.rand(2) * 10,
        bias=np.random.rand(2),
        activation_func="relu",
    )
    lyr = lyr.reset_state()

    # - evolve using evolve method
    _, new_state, _ = lyr.evolve(np.random.rand(10, 2))
    lyr = lyr.set_attributes(new_state)

    # - evolve using direct call
    input_rand = np.random.rand(10, 2)
    lyr = lyr.reset_state()
    output, new_state, _ = lyr(input_rand)
    lyr = lyr.set_attributes(new_state)

    # - Set attributes method
    _, new_state, _ = lyr(input_rand)
    lyr = lyr.set_attributes(new_state)

    # - Direct set state
    lyr.activation = np.array([100.0, 100.0])

    # - Reset state and evolve
    lyr = lyr.reset_state()
    je = jit(lyr)
    output_jit, new_state, _ = je(input_rand)
    lyr = lyr.set_attributes(new_state)

    _, new_state, _ = je(input_rand)
    lyr = lyr.set_attributes(new_state)

    # - Compare non-jit and jitted output
    assert np.allclose(
        output, output_jit
    ), "Compiled evolution does not match non-compiled evolution"

    # - Define loss function to test gradients
    def loss_fn(grad_params, net, input, target):
        net = net.set_attributes(grad_params)
        output, _, _ = net(input)
        return jnp.sum((output - target) ** 2)

    loss_vgf = jit(jax.value_and_grad(loss_fn))
    params = lyr.parameters()
    loss, grad = loss_vgf(params, lyr, np.random.rand(10, 2), 0.0)
    loss, grad = loss_vgf(params, lyr, np.random.rand(10, 2), 0.0)

    assert not np.allclose(
        grad["bias"], np.zeros_like(grad["bias"])
    ), "Bias gradients are zero in FFwd module"
    assert not np.allclose(
        grad["tau"], np.zeros_like(grad["tau"])
    ), "Tau gradients are zero in FFwd module"
    assert not np.allclose(
        grad["threshold"], np.zeros_like(grad["threshold"])
    ), "Threshold gradients are zero in FFwd module"

    # - Test recurrent mode evolution and gradients
    lyr = RateJax(2, has_rec=True)

    o, ns, r_d = lyr(np.random.rand(20, 2))
    lyr = lyr.set_attributes(ns)

    je = jit(lyr)
    o, n_s, r_d = je(np.random.rand(20, 2))
    lyr = lyr.set_attributes(n_s)

    loss_vgf = jit(jax.value_and_grad(loss_fn))
    params = lyr.parameters()
    loss, grad = loss_vgf(params, lyr, np.random.rand(20, 2), 0.0)
    loss, grad = loss_vgf(params, lyr, np.random.rand(20, 2), 0.0)

    assert not np.allclose(
        grad["bias"], np.zeros_like(grad["bias"])
    ), "Bias gradients are zero in recurrent module"
    assert not np.allclose(
        grad["tau"], np.zeros_like(grad["tau"])
    ), "Tau gradients are zero in recurrent module"
    assert not np.allclose(
        grad["w_rec"], np.zeros_like(grad["w_rec"])
    ), "Recurrent weight gradients are zero in recurrent module"
    assert not np.allclose(
        grad["threshold"], np.zeros_like(grad["threshold"])
    ), "Threshold gradients are zero in recurrent module"


def test_rate_jax_tree():
    from rockpool.nn.modules import RateJax
    from jax.tree_util import tree_flatten, tree_unflatten
    from jax.config import config

    config.update("jax_log_compiles", True)

    mod = RateJax(2)
    tree, treedef = tree_flatten(mod)
    tree_unflatten(treedef, tree)

    mod = RateJax(2, has_rec=True)
    tree, treedef = tree_flatten(mod)
    tree_unflatten(treedef, tree)


def test_ffwd_net():
    from rockpool.nn.modules import RateJax
    from rockpool.nn.modules import JaxModule
    from rockpool.parameters import Parameter

    import numpy as np
    import jax.numpy as jnp
    from jax.config import config

    config.update("jax_log_compiles", True)

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

                tau = np.random.rand(N_out) * 10
                bias = np.random.rand(N_out)
                setattr(self, f"iaf_{index}", RateJax(N_out, tau=tau, bias=bias))

        def evolve(self, input, record: bool = False):
            new_state = {}
            record_dict = {}
            for layer in range(len(self._shape) - 1):
                w = getattr(self, f"weight_{layer}")
                mod_name = f"iaf_{layer}"
                iaf = getattr(self, mod_name)

                outputs, substate, subrec = iaf(jnp.dot(input, w), record=record)
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

            # # - Assign sub-modules
            # for name, mod in modules.items():
            #     setattr(obj, name, mod)

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
    from rockpool.nn.modules import RateJax
    from rockpool.nn.modules import JaxModule
    from rockpool.parameters import Parameter

    import jax
    from jax import jit

    import numpy as np
    import jax.numpy as jnp
    from jax.config import config

    config.update("jax_log_compiles", True)

    class my_ffwd_net(JaxModule):
        def __init__(self, shape, *args, **kwargs):
            super().__init__(shape, *args, **kwargs)

            for index, (N_in, N_out) in enumerate(zip(shape[:-1], shape[1:])):
                setattr(
                    self,
                    f"weight_{index}",
                    Parameter(np.random.rand(N_in, N_out), "weights"),
                )

                tau = (
                    np.random.rand(
                        N_out,
                    )
                    * 10
                )
                bias = np.random.rand(
                    N_out,
                )
                setattr(self, f"iaf_{index}", RateJax(N_out, tau=tau, bias=bias))

        def evolve(self, input, record: bool = False):
            new_state = {}
            record_dict = {}
            for layer in range(len(self._shape) - 1):
                w = getattr(self, f"weight_{layer}")
                mod_name = f"iaf_{layer}"
                iaf = getattr(self, mod_name)

                outputs, substate, subrec = iaf(jnp.dot(input, w))
                new_state.update({mod_name: substate})
                record_dict.update({mod_name: subrec})

                input = outputs[0]

            return input, new_state, record_dict

        # @classmethod
        # def tree_unflatten(cls, aux_data, children):
        #     params, sim_params, state, modules = children
        #     _name, _shape, _submodulenames = aux_data
        #
        #     obj = my_ffwd_net(_shape)
        #     obj._name = _name
        #
        #     # - Restore parameters and configuration
        #     obj.set_attributes(params)
        #     obj.set_attributes(sim_params)
        #     obj.set_attributes(state)
        #
        #     return obj

    def mse_loss(grad_params, net, input, target):
        net = net.reset_state()
        net = net.set_attributes(grad_params)
        outputs, _, _ = net(input)

        mse = np.sum((outputs[0] - target) ** 2)

        return mse

    net = my_ffwd_net([2, 3, 2])
    net = net.reset_state()
    params0 = net.parameters()

    mse_loss(params0, net, np.random.rand(10, 2), np.random.rand(10, 1))

    vgf = jax.value_and_grad(mse_loss)

    loss, grads = vgf(params0, net, np.random.rand(10, 2), np.random.rand(10, 1))
    print(loss, grads)

    from jax.example_libraries.optimizers import adam

    init_fun, update_fun, get_params = adam(1e-2)

    update_fun = jit(update_fun)

    opt_state = init_fun(params0)
    inputs = np.random.rand(10, 2)
    target = np.random.rand(10, 2)

    loss_t = []
    vgf = jit(jax.value_and_grad(mse_loss))

    from tqdm.autonotebook import tqdm

    with tqdm(range(100)) as t:
        for i in t:
            loss, grads = vgf(get_params(opt_state), net, inputs, target)
            opt_state = update_fun(i, grads, opt_state)
            loss_t.append(loss)
            t.set_postfix({"loss": loss})

    print(f"Losses: [0] {loss_t[0]} .. [-1] {loss_t[-1]}")


def test_nonjax_submodule():
    from rockpool.nn.modules import Module
    from rockpool.nn.modules import JaxModule

    class nonjax_mod(Module):
        def evolve(self, input, record: bool = False):
            return None, None, None

    class jax_mod(JaxModule):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, *kwargs)

            self.mod = nonjax_mod()

        def evolve(self, input, record: bool = False):
            return None, None, None

    with pytest.raises(ValueError):
        jax_mod()
