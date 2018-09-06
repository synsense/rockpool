"""
Test rate-based Euler models in rate.py
"""

import numpy as np
import pytest

def test_imports():
   from NetworksPython.layers import FFRateEuler, RecRateEuler

def test_FFRateEuler():
    """ Test FFRateEuler """
    from NetworksPython import TSContinuous
    from NetworksPython.layers import FFRateEuler

    # - Generic parameters
    mfW = 2 * np.random.rand(2, 3) - 1
    vfBias = 2 * np.random.rand(3) - 1

    # - Layer generation
    fl0 = FFRateEuler(
        mfW=mfW,
        vfBias=vfBias,
        fNoiseStd=0.1,
        tDt=0.01,
    )

    # - Input signal
    tsInCont = TSContinuous(
        vtTimeTrace=np.arange(15) * 0.01, mfSamples=np.ones((15, 2))
    )

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.vState)
    fl0.evolve(tsInCont, tDuration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.vState).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.vState).all()

    # - Test that some errors are caught
    with pytest.raises(AssertionError):
        fl1 = FFRateEuler(mfW = None)

    with pytest.raises(AssertionError):
        fl1 = FFRateEuler(mfW = 1, vfBias = [1, 1])

    with pytest.raises(AssertionError):
        fl1 = FFRateEuler(mfW = 1, vtTau = [1, 1])

    with pytest.raises(AssertionError):
        fl1 = FFRateEuler(mfW = 1, vfGain = [1, 1])

def test_RecRateEuler():
    """ Test RecRateEuler """
    from NetworksPython import TSContinuous
    from NetworksPython.layers import RecRateEuler

    # - Generic parameters
    mfW = 2 * np.random.rand(2, 2) - 1
    vfBias = 2 * np.random.rand(2) - 1

    # - Layer generation
    fl0 = RecRateEuler(
        mfW=mfW,
        vfBias=vfBias,
        fNoiseStd=0.1,
        tDt=0.01,
    )

    # - Input signal
    tsInCont = TSContinuous(
        vtTimeTrace=np.arange(15) * 0.01, mfSamples=np.ones((15, 2))
    )

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.vState)
    fl0.evolve(tsInCont, tDuration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.vState).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.vState).all()

    # - Test that some errors are caught
    with pytest.raises(AssertionError):
        fl1 = RecRateEuler(mfW = np.zeros((1, 2)))

