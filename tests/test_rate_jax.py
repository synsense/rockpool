"""
Test rate-based Euler models in rate-jax.py
"""

import numpy as np
import pytest

def test_imports():
   from NetworksPython.layers import RecRateEulerJax


def test_RecRateEulerJax():
    """ Test RecRateEuler """
    from NetworksPython import TSContinuous
    from NetworksPython.layers import RecRateEulerJax

    # - Generic parameters
    w_in = 2 * np.random.rand(1, 2) - 1
    w_recurrent = 2 * np.random.rand(2, 2) - 1
    w_out = 2 * np.random.rand(2, 1) - 1
    bias = 2 * np.random.rand(2) - 1
    tau = 20e-3 * np.ones(2)

    # - Layer generation
    fl0 = RecRateEulerJax(
        weights=w_in,
        w_recurrent = w_recurrent,
        w_out = w_out,
        bias=bias,
        noise_std=0.1,
        tau = tau,
        dt=0.01,
    )

    # - Input signal
    tsInCont = TSContinuous(
        times=np.arange(15) * 0.01, samples=np.ones((15, 1))
    )

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.state)
    fl0.evolve(tsInCont, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state).all()

    # - Test that some errors are caught
    with pytest.raises(AssertionError):
        fl1 = RecRateEuler(weights = np.zeros((1, 2)))

    with pytest.raises(AssertionError):
        RecRateEuler(weights = np.zeros((2, 2)), tau = None)

    with pytest.raises(AssertionError):
        RecRateEuler(weights = np.zeros((2, 2)), noise_std = None)
