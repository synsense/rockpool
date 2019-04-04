"""
Test FFUpDown layer in updown.py
"""

import numpy as np


def test_imports():
    from NetworksPython.layers import FFUpDown


def test_updown():
    """ Test FFUpDown """
    from NetworksPython import TSContinuous
    from NetworksPython.layers import FFUpDown

    # - Generic parameters
    mfW = np.random.rand(2, 4)

    # - Layer generation
    fl0 = FFUpDown(mfW=mfW, tDt=0.01, vfThrDown=0.02, vfThrUp=0.01, vtTauDecay=0.1)

    # - Check layer properties
    assert fl0.nSize == 4, "Problem with nSize"
    assert fl0.nSizeIn == 2, "Problem with nSizeIn"
    assert (fl0.vfThrDown == np.array([0.02, 0.02])).all(), "Problem with vfThrDown"
    assert (fl0.vfThrUp == np.array([0.01, 0.01])).all(), "Problem with vfThrUp"

    # - Input signal
    tsInCont = TSContinuous(
        vtTimeTrace=np.arange(15) * 0.01,
        mfSamples=np.vstack(
            (np.sin(np.linspace(0, 1, 15)), np.cos(np.linspace(0, 1, 15)))
        ).T,
    )

    # - Compare states and time before and after evolution
    vStateBefore = np.copy(fl0.vState)
    fl0.evolve(tsInCont, tDuration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.vState).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.vState).all()


def test_updown_in_net():
    """ Test RecRateEuler """
    from NetworksPython import TSContinuous
    from NetworksPython.networks import Network
    from NetworksPython.layers import FFUpDown
    from NetworksPython.layers import RecDIAF

    # - Generic parameters
    mfW = np.random.rand(2, 4)

    # - Layer generation
    fl0 = FFUpDown(mfW=mfW)
    fl1 = RecDIAF(np.zeros((4, 2)), np.zeros((2, 2)), tDt=0.002)
    # - Generate network
    net = Network(fl0, fl1)

    # - Input signal
    tsInCont = TSContinuous(
        vtTimeTrace=np.arange(15) * 0.01,
        mfSamples=np.vstack(
            (np.sin(np.linspace(0, 1, 15)), np.cos(np.linspace(0, 1, 15)))
        ).T,
    )

    # - Compare states and time before and after evolution
    vStateBefore = np.copy(fl1.vState)
    net.evolve(tsInCont, tDuration=0.1)
    assert net.t == 0.1
    assert (vStateBefore != fl1.vState).any()
