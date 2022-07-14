import pytest

pytest.importorskip("torch")


def test_import():
    from rockpool.parameters import (
        ParameterBase,
        Parameter,
        SimulationParameter,
        State,
        RP_Constant,
    )


def test_parameters():
    from rockpool.parameters import (
        ParameterBase,
        Parameter,
        SimulationParameter,
        State,
        Constant,
        RP_Constant,
    )
    from rockpool.nn.modules import Module

    import numpy as np
    import torch as t

    Parameter(4)
    Parameter(4.0)
    Parameter([3])
    Parameter(np.ones(4))
    Parameter(t.tensor(3))

    with pytest.raises(ValueError):
        Parameter()
        Parameter(shape=3)
        Parameter(3, shape=(1,))
        Parameter([3], shape=())

    Parameter(shape=3, init_func=np.random.standard_normal)
    Parameter(3, shape=())
    Parameter(3, shape=[(), (1,)])

    assert isinstance(Parameter(Constant(3)), SimulationParameter)

    Parameter(Constant(3))
    Parameter(Constant([3]))
    Parameter(Constant(np.array(3)))
    Parameter(Constant(t.tensor(3)))
    assert isinstance(Constant(np.array(3)), np.ndarray)
    assert isinstance(Constant(np.array(3)), RP_Constant)
    assert isinstance(Constant(t.tensor(4)), t.Tensor)
    assert isinstance(Constant(t.tensor(4)), RP_Constant)

    class TestMod(Module):
        def __init__(self, shape, param):
            super().__init__(shape=shape)
            self.param = param

        def evolve(self, *args, **kwargs):
            pass

    mod = TestMod(
        None, Parameter(shape=[(), (1,)], init_func=np.random.standard_normal)
    )
    assert np.shape(mod.param) == ()
    assert "param" in mod.parameters()

    mod = TestMod(None, Parameter(Constant(3)))
    assert "param" not in mod.parameters()
    assert "param" in mod.simulation_parameters()
