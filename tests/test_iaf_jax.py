"""
Test rate-based Euler models in iaf_jax.py
"""

import numpy as np
import pytest


def test_imports():
    from Rockpool.layers import RecIAFExpJax
    from Rockpool.layers import RecIAFExpSpikeOutJax
    from Rockpool.layers import RecIAFExpWithIOJax


def test_RecIAFExpJax():
    """ Test RecIAFExpJax """
    from Rockpool import TSContinuous
    from Rockpool.layers import RecIAFExpJax

    # - Generic parameters
    net_size = 2
    dt = 10e-3

    w_recurrent = 2 * np.random.rand(net_size, net_size) - 1
    bias = 2 * np.random.rand(net_size) - 1
    tau_m = 20e-3 * np.ones(net_size)
    tau_s = 20e-3 * np.ones(net_size)

    # - Layer generation
    fl0 = RecIAFExpJax(
        w_recurrent=w_recurrent,
        bias=bias,
        noise_std=0.1,
        tau_mem=tau_m,
        tau_syn=tau_s,
        dt=dt,
    )

    # - Input signal
    tsInCont = TSContinuous(times=np.arange(15) * dt, samples=np.ones((15, net_size)))

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.state)
    ts_output = fl0.evolve(tsInCont, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state).all()

    # - Test that some errors are caught
    with pytest.raises(AssertionError):
        fl1 = RecIAFExpJax(
            w_recurrent=np.zeros((3, 2)),
            tau_mem=np.zeros(3),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = RecIAFExpJax(
            w_recurrent=np.zeros((2, 2)),
            tau_mem=np.zeros(3),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = RecIAFExpJax(
            w_recurrent=np.zeros((2, 2)),
            tau_mem=np.zeros(2),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )


def test_RecIAFExpSpikeOutJax():
    """ Test RecIAFExpJax """
    from Rockpool import TSContinuous
    from Rockpool.layers import RecIAFExpSpikeOutJax

    # - Generic parameters
    net_size = 2
    dt = 10e-3

    w_recurrent = 2 * np.random.rand(net_size, net_size) - 1
    bias = 2 * np.random.rand(net_size) - 1
    tau_m = 20e-3 * np.ones(net_size)
    tau_s = 20e-3 * np.ones(net_size)

    # - Layer generation
    fl0 = RecIAFExpSpikeOutJax(
        w_recurrent=w_recurrent,
        bias=bias,
        noise_std=0.1,
        tau_mem=tau_m,
        tau_syn=tau_s,
        dt=dt,
    )

    # - Input signal
    tsInCont = TSContinuous(times=np.arange(15) * dt, samples=np.ones((15, net_size)))

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.state)
    ts_output = fl0.evolve(tsInCont, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state).all()

    # - Test that some errors are caught
    with pytest.raises(AssertionError):
        fl1 = RecIAFExpSpikeOutJax(
            w_recurrent=np.zeros((3, 2)),
            tau_mem=np.zeros(3),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = RecIAFExpSpikeOutJax(
            w_recurrent=np.zeros((2, 2)),
            tau_mem=np.zeros(3),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = RecIAFExpSpikeOutJax(
            w_recurrent=np.zeros((2, 2)),
            tau_mem=np.zeros(2),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )


def test_RecIAFExpWithIOJax():
    """ Test RecIAFExpJax """
    from Rockpool import TSContinuous
    from Rockpool.layers import RecIAFExpWithIOJax

    # - Generic parameters
    input_size = 1
    net_size = 2
    output_size = 1
    dt = 10e-3

    w_in = 2 * np.random.rand(input_size, net_size) - 1
    w_recurrent = 2 * np.random.rand(net_size, net_size) - 1
    w_out = 2 * np.random.rand(net_size, output_size) - 1
    bias = 2 * np.random.rand(net_size) - 1
    tau_m = 20e-3 * np.ones(net_size)
    tau_s = 20e-3 * np.ones(net_size)

    # - Layer generation
    fl0 = RecIAFExpWithIOJax(
        w_in=w_in,
        w_recurrent=w_recurrent,
        w_out=w_out,
        bias=bias,
        noise_std=0.1,
        tau_mem=tau_m,
        tau_syn=tau_s,
        dt=dt,
    )

    # - Input signal
    tsInCont = TSContinuous(times=np.arange(15) * dt, samples=np.ones((15, input_size)))

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.state)
    ts_output = fl0.evolve(tsInCont, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state).all()

    # - Test that some errors are caught
    with pytest.raises(AssertionError):
        fl1 = RecIAFExpWithIOJax(
            w_in=np.zeros((1, 3)),
            w_recurrent=np.zeros((3, 2)),
            w_out=np.zeros((3, 1)),
            tau_mem=np.zeros(3),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = RecIAFExpWithIOJax(
            w_in=np.zeros((1, 2)),
            w_recurrent=np.zeros((2, 2)),
            w_out=np.zeros((2, 1)),
            tau_mem=np.zeros(3),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = RecIAFExpWithIOJax(
            w_in=np.zeros((1, 2)),
            w_recurrent=np.zeros((2, 2)),
            w_out=np.zeros((2, 1)),
            tau_mem=np.zeros(2),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )
