"""
Test Brian-based spiking layers from layers.internal.iaf_brian and layers.recurrent.iaf_brian
"""

import numpy as np
import sys
import os.path


# def test_imports():
#    from NetworksPython.layers import iaf_brian
#
#    # from NetworksPython.layers import iaf_brian


def test_ffiaf():
    """ Test FFIAFBrian """
    from brian2 import second
    from NetworksPython import timeseries as ts
    from NetworksPython.layers import FFIAFBrian

    # - Generic parameters
    weights = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])
    vfBias = 0.01
    vtTauN = [0.02, 0.05, 0.1]

    # - Layer generation
    fl0 = FFIAFBrian(
        weights=weights,
        dt=0.01,
        vfBias=vfBias,
        vtTauN=vtTauN,
        noise_std=0.1,
        tRefractoryTime=0.001 * second,
    )

    # - Input signal
    tsInCont = ts.TSContinuous(times=np.arange(15) * 0.01, samples=np.ones((15, 2)))

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.state)
    fl0.evolve(tsInCont, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state).all()


def test_ffiaf_spkin():
    """ Test FFIAFSpkInBrian """
    from NetworksPython import timeseries as ts
    from NetworksPython.layers import FFIAFSpkInBrian

    # - Generic parameters
    weights = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])
    vfBias = 0.01
    vtTauN = [0.02, 0.05, 0.1]

    # - Layer generation
    fl1 = FFIAFSpkInBrian(
        weights=weights,
        vfBias=vfBias,
        vtTauN=vtTauN,
        dt=0.01,
        noise_std=0.1,
        tRefractoryTime=0.001,
    )

    # - Input signal
    tsInEvt = ts.TSEvent(times=[0.02, 0.04, 0.04, 0.06, 0.12], channels=[1, 0, 1, 1, 0])

    # - Compare states and time before and after
    vStateBefore = np.copy(fl1.state)
    fl1.evolve(tsInEvt, duration=0.1)
    assert fl1.t == 0.1
    assert (vStateBefore != fl1.state).any()

    fl1.reset_all()
    assert fl1.t == 0
    assert np.allclose(vStateBefore, fl1.state)


def test_reciaf():
    """ Test RecIAFBrian """
    from brian2 import second
    from NetworksPython import timeseries as ts
    from NetworksPython.layers import RecIAFBrian

    # - Generic parameters
    np.random.seed(1)
    weights = 2 * np.random.rand(3, 3) - 1
    vfBias = 2 * np.random.rand(3) - 1
    vtTauN, vtTauSynR = np.clip(np.random.rand(2, 3), 0.01, None)

    # - Layer generation
    rl0 = RecIAFBrian(
        weights=weights,
        vfBias=vfBias,
        vtTauN=vtTauN,
        vtTauSynR=vtTauSynR,
        dt=0.01,
        noise_std=0.1,
        tRefractoryTime=0.001 * second,
    )

    # - Input signal
    tsInCont = ts.TSContinuous(times=np.arange(15) * 0.01, samples=np.ones((15, 3)))

    # - Compare states and time before and after
    vStateBefore = np.copy(rl0.state)
    rl0.evolve(tsInCont, duration=0.1)
    assert rl0.t == 0.1
    assert (vStateBefore != rl0.state).any()

    rl0.reset_all()
    assert rl0.t == 0
    assert (vStateBefore == rl0.state).all()


def test_reciaf_spkin():
    """ Test RecIAFSpkInBrian """
    from NetworksPython import timeseries as ts
    from NetworksPython.layers import RecIAFSpkInBrian

    # - Negative weights, so that layer doesn't spike and gets reset
    np.random.seed(1)
    weights_in = np.random.rand(2, 3) - 1
    weights_rec = 2 * np.random.rand(3, 3) - 1
    vfBias = 2 * np.random.rand(3) - 1
    vtTauN, vtTauSInp, vtTauSRec = np.clip(np.random.rand(3, 3), 0.01, None)

    # - Layer generation
    rl1 = RecIAFSpkInBrian(
        weights_in=weights_in,
        weights_rec=weights_rec,
        vfBias=vfBias,
        vtTauN=vtTauN,
        vtTauSInp=vtTauSInp,
        vtTauSRec=vtTauSRec,
        dt=0.01,
        noise_std=0.1,
        tRefractoryTime=0.001,
    )

    # - Input signal
    tsInEvt = ts.TSEvent(times=[0.02, 0.04, 0.04, 0.06, 0.12], channels=[1, 0, 1, 1, 0])

    # - Compare states and time before and after
    vStateBefore = np.copy(rl1.state)
    rl1.evolve(tsInEvt, duration=0.1)
    assert rl1.t == 0.1
    assert (vStateBefore != rl1.state).any()

    rl1.reset_all()
    assert rl1.t == 0
    assert (vStateBefore == rl1.state).all()
