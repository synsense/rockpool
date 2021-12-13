import pytest


def test_imports():
    from rockpool.graph.graph_modules import (
        LinearWeights,
        GenericNeurons,
        AliasConnection,
        LIFNeuronWithSynsRealValue,
        RateNeuronWithSynsRealValue,
    )


def test_LinearWeights():
    from rockpool.graph import LinearWeights
    import numpy as np

    lw = LinearWeights._factory(2, 3, "test", None, np.empty((2, 3)))

    with pytest.raises(ValueError):
        # - Weight size must match I/O size
        lw = LinearWeights._factory(2, 3, "test", None, np.empty((1, 1)))

    with pytest.raises(TypeError):
        # - Weights are compulsory
        lw = LinearWeights._factory(2, 3, "test", None)


def test_GenericNeurons():
    from rockpool.graph import GenericNeurons


def test_AliasConnection():
    from rockpool.graph import AliasConnection


def test_LIFNeuronWithSynsRealValue():
    from rockpool.graph import LIFNeuronWithSynsRealValue

    gmod = LIFNeuronWithSynsRealValue._factory(
        size_in=2,
        size_out=3,
        name="test",
        tau_mem=[100e-3] * 3,
        tau_syn=[100e-3] * 2,
        threshold=[0.0] * 3,
        bias=[0.0] * 3,
        dt=10e-3,
    )


def test_RateNeuronWithSynsRealValue():
    from rockpool.graph import RateNeuronWithSynsRealValue

    gmod = RateNeuronWithSynsRealValue._factory(
        size_in=2,
        size_out=3,
        name="test",
        tau=[100e-3] * 3,
        bias=[0.1] * 3,
        dt=10e-3,
    )
