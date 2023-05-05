import pytest

pytest.importorskip("brian2")


def test_imports():
    from rockpool.nn.layers.exp_synapses_brian import FFExpSynBrian


def test_FFExpSynBrian():
    from rockpool.nn.layers.exp_synapses_brian import FFExpSynBrian
    from rockpool.timeseries import TSContinuous, TSEvent
    import numpy as np

    N_in = 10
    N = 5
    T = 1000
    f_rate = 0.1
    dt = 1e-3

    # - Generate layer
    lyr = FFExpSynBrian(
        weights=np.random.rand(N_in, N),
        dt=dt,
    )

    # - Generate an input time trace
    ts_input = TSEvent.from_raster(
        np.random.rand(T, N_in) < f_rate, dt=dt, name="Input events"
    )

    # - Evolve
    ts_output, new_state, _ = lyr.evolve(ts_input)
