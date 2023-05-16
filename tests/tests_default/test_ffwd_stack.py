def test_imports():
    from rockpool.nn.combinators.ffwd_stack import FFwdStackMixin, FFwdStack


def test_FFwdStack_nojax():
    from rockpool.nn.combinators.ffwd_stack import FFwdStack
    from rockpool.nn.modules.module import Module
    from rockpool.parameters import State, Parameter

    import numpy as np

    # - Define a simple module
    class Mod(Module):
        def __init__(self, shape=None, *args, **kwargs):
            super().__init__(shape=shape, *args, *kwargs)
            self.activation = State(shape=self.size_out, init_func=np.zeros)
            self.bias = Parameter(
                shape=self._shape[-1], init_func=np.random.standard_normal
            )

        def evolve(self, input_data, weights_recurrent=None, record: bool = False):
            return input_data + self.bias, {}, {}

    seq = FFwdStack(
        Mod(10),
        Mod(20),
        Mod(30),
        Mod(1),
    )
    print(seq)

    input_data = np.random.rand(100, 10)

    # - Test evolve
    (
        output,
        _,
        _,
    ) = seq(input_data)
    print(output.shape)

    # - Test parameters
    print(seq.parameters())
    print(seq.state())


def test_FFwdStack_jax():
    import pytest

    pytest.importorskip("jax")

    from rockpool.nn.combinators.ffwd_stack import FFwdStack
    from rockpool.nn.modules.jax.jax_module import JaxModule
    from rockpool.parameters import State, Parameter

    import numpy as np
    import jax.numpy as jnp
    from jax import jit, value_and_grad

    # - Define a simple module
    class Mod(JaxModule):
        def __init__(self, shape=None, *args, **kwargs):
            super().__init__(shape=shape, *args, *kwargs)
            self.activation = State(shape=self.size_out, init_func=np.zeros)
            self.bias = Parameter(
                shape=self._shape[-1], init_func=np.random.standard_normal
            )

        def evolve(self, input_data, weights_recurrent=None, record: bool = False):
            return input_data + self.bias, {}, {}

    seq = FFwdStack(
        Mod(10),
        Mod(20),
        Mod(30),
        Mod(1),
    )
    print(seq)

    input_data = np.random.rand(100, 10)

    # - Test evolve
    (
        output,
        new_state,
        _,
    ) = seq(input_data)
    seq = seq.set_attributes(new_state)
    print(output.shape)

    # - Test parameters
    print(seq.parameters())
    print(seq.state())

    # - Test jit evolve
    je = jit(seq)
    (
        output,
        new_state,
        _,
    ) = je(input_data)
    seq = seq.set_attributes(new_state)
    print(output.shape)

    # - Test jit grad
    def loss(params, seq, input_data):
        seq = seq.set_attributes(params)
        output, new_state, record_dict = seq(input_data)
        return jnp.sum(output)

    vg = value_and_grad(loss)
    (
        loss,
        grads,
    ) = vg(seq.parameters(), seq, input_data)
    print(loss, grads)
