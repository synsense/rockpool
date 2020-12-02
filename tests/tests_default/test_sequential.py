def test_imports():
    from rockpool.sequential import (
        SequentialMixin,
        Sequential,
        JaxSequential,
        ModSequential,
    )


def test_Sequential_nojax():
    from rockpool.ffwd_stack import FFwdStack
    from rockpool.linear import Linear
    from rockpool.module import Module
    from rockpool.parameters import State, Parameter

    import numpy as np

    # - Define a simple module
    class Mod(Module):
        def __init__(self, shape=None, *args, **kwargs):
            super().__init__(shape=shape, *args, *kwargs)
            self.activation = State(shape=self._shape[-1], init_func=np.zeros)
            self.bias = Parameter(
                shape=self._shape[-1], init_func=np.random.standard_normal
            )

        def evolve(self, input_data, weights_recurrent=None, record: bool = False):
            return input_data + self.bias, {}, {}

    seq = FFwdStack(
        Mod(10),
        Linear((10, 20)),
        Mod(20),
        Linear((20, 30)),
        Mod(30),
        Linear((30, 1)),
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
    from rockpool.ffwd_stack import FFwdStack
    from rockpool.linear import JaxLinear
    from rockpool.jax_module import JaxModule
    from rockpool.parameters import State, Parameter

    import numpy as np
    from jax import jit

    # - Define a simple module
    class Mod(JaxModule):
        def __init__(self, shape=None, *args, **kwargs):
            super().__init__(shape=shape, *args, *kwargs)
            self.activation = State(shape=self._shape[-1], init_func=np.zeros)
            self.bias = Parameter(
                shape=self._shape[-1], init_func=np.random.standard_normal
            )

        def evolve(self, input_data, weights_recurrent=None, record: bool = False):
            return input_data + self.bias, {}, {}

    seq = FFwdStack(
        Mod(10),
        JaxLinear((10, 20)),
        Mod(20),
        JaxLinear((20, 30)),
        Mod(30),
        JaxLinear((30, 1)),
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

    # - Test compilation
    je = jit(seq)
    (
        output,
        _,
        _,
    ) = seq(input_data)
