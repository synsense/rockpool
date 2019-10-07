"""
Test rate-based Euler models in rate.py
"""

import numpy as np
import pytest

def test_imports():
   from rockpool.layers import FFRateEuler, RecRateEuler

def test_FFRateEuler():
    """ Test FFRateEuler """
    from rockpool import TSContinuous
    from rockpool.layers import FFRateEuler

    # - Generic parameters
    weights = 2 * np.random.rand(2, 3) - 1
    bias = 2 * np.random.rand(3) - 1

    # - Layer generation
    fl0 = FFRateEuler(
        weights=weights,
        bias=bias,
        noise_std=0.1,
        dt=0.01,
    )

    # - Input signal
    tsInCont = TSContinuous(
        times=np.arange(15) * 0.01, samples=np.ones((15, 2))
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
        fl1 = FFRateEuler(weights = None)

    with pytest.raises(AssertionError):
        fl1 = FFRateEuler(weights = 1, bias = [1, 1])

    with pytest.raises(AssertionError):
        fl1 = FFRateEuler(weights = 1, tau = [1, 1])

    with pytest.raises(AssertionError):
        fl1 = FFRateEuler(weights = 1, gain = [1, 1])

def test_RecRateEuler():
    """ Test RecRateEuler """
    from rockpool import TSContinuous
    from rockpool.layers import RecRateEuler

    # - Generic parameters
    weights = 2 * np.random.rand(2, 2) - 1
    bias = 2 * np.random.rand(2) - 1

    # - Layer generation
    fl0 = RecRateEuler(
        weights=weights,
        bias=bias,
        noise_std=0.1,
        dt=0.01,
    )

    # - Input signal
    tsInCont = TSContinuous(
        times=np.arange(15) * 0.01, samples=np.ones((15, 2))
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
