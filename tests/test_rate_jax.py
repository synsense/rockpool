"""
Test rate-based Euler models in rate-jax.py
"""

import numpy as np
import pytest


def test_imports():
    from rockpool.layers import RecRateEulerJax


def test_RecRateEulerJax():
    """ Test RecRateEulerJax """
    from rockpool import TSContinuous
    from rockpool.layers import RecRateEulerJax

    # - Generic parameters
    w_in = 2 * np.random.rand(1, 2) - 1
    w_recurrent = 2 * np.random.rand(2, 2) - 1
    w_out = 2 * np.random.rand(2, 1) - 1
    bias = 2 * np.random.rand(2) - 1
    tau = 20e-3 * np.ones(2)

    # - Layer generation
    fl0 = RecRateEulerJax(
        w_in=w_in,
        w_recurrent=w_recurrent,
        w_out=w_out,
        bias=bias,
        noise_std=0.1,
        tau=tau,
        dt=0.01,
    )

    # - Input signal
    tsInCont = TSContinuous(times=np.arange(15) * 0.01, samples=np.ones((15, 1)))

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
        fl1 = RecRateEulerJax(
            w_in=np.zeros((1, 2)),
            w_recurrent=np.zeros((3, 2)),
            w_out=np.zeros((3, 1)),
            tau=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = RecRateEulerJax(
            w_in=np.zeros((1, 2)),
            w_recurrent=np.zeros((2, 2)),
            w_out=np.zeros((3, 1)),
            tau=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = RecRateEulerJax(
            w_in=np.zeros((1, 2)),
            w_recurrent=np.zeros((2, 2)),
            w_out=np.zeros((2, 1)),
            tau=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = RecRateEulerJax(
            w_in=np.zeros((1, 2)),
            w_recurrent=np.zeros((2, 2)),
            w_out=np.zeros((2, 1)),
            tau=np.zeros(2),
            bias=np.zeros(3),
        )


def test_ForceRateEulerJax():
    """ Test ForceRateEulerJax """
    from rockpool import TSContinuous
    from rockpool.layers import ForceRateEulerJax

    # - Generic parameters
    w_in = 2 * np.random.rand(1, 2) - 1
    w_out = 2 * np.random.rand(2, 1) - 1
    bias = 2 * np.random.rand(2) - 1
    tau = 20e-3 * np.ones(2)

    # - Layer generation
    fl0 = ForceRateEulerJax(
        w_in=w_in, w_out=w_out, bias=bias, noise_std=0.1, tau=tau, dt=0.01
    )

    # - Input signal
    tsInCont = TSContinuous(times=np.arange(15) * 0.01, samples=np.ones((15, 1)))

    tsForceCont = TSContinuous(times=np.arange(15) * 0.01, samples=np.ones((15, 2)))

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.state)
    ts_output = fl0.evolve(tsInCont, tsForceCont, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state).all()

    # - Test that some errors are caught
    with pytest.raises(AssertionError):
        fl1 = ForceRateEulerJax(
            w_in=np.zeros((1, 2)),
            w_out=np.zeros((3, 1)),
            tau=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = ForceRateEulerJax(
            w_in=np.zeros((1, 2)),
            w_out=np.zeros((2, 1)),
            tau=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = ForceRateEulerJax(
            w_in=np.zeros((1, 2)),
            w_out=np.zeros((2, 1)),
            tau=np.zeros(2),
            bias=np.zeros(3),
        )
