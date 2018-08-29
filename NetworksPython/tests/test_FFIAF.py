'''
Test Brian-based spiking layers from layers.feedforward.iaf_brian and layers.recurrent.iaf_brian
'''

import numpy as np
import sys
import os.path
strNetworkPath = os.path.abspath(sys.path[0] + "/../..")
sys.path.insert(1, strNetworkPath)


def test_imports():
    from NetworksPython.layers.feedforward import iaf_brian
    from NetworksPython.layers.recurrent import iaf_brian

def test_ffiaf():
    """ Test FFIAFBrian """
    from brian2 import second
    from NetworksPython import timeseries as ts
    from NetworksPython.layers.feedforward.iaf_brian import FFIAFBrian

    # - Generic parameters
    mfW = 2*np.random.rand(2,3)-1
    vfBias = 2*np.random.rand(3)-1
    vtTauN = np.random.rand(3)

    # - Layer generation
    fl0 = FFIAFBrian(
        mfW = mfW,
        vfBias = vfBias,
        vtTauN = vtTauN,
        fNoiseStd =0.1,
        tRefractoryTime = 0.001 * second,
    )

    # - Input signal
    tsInCont = ts.TSContinuous(
        vtTimeTrace = np.arange(15) * 0.01,
        mfSamples = np.ones((15,2))
    )

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.vState)
    fl0.evolve(tsInCont, tDuration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.vState).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.vState).all()

def test_ffiaf_spkin():
    """ Test FFIAFSpkInBrian """
    from brian2 import second
    from NetworksPython import timeseries as ts
    from NetworksPython.layers.feedforward.iaf_brian import FFIAFSpkInBrian

    # - Generic parameters
    mfW = 2*np.random.rand(2,3)-1
    vfBias = 2*np.random.rand(3)-1
    vtTauN = np.random.rand(3)

    # - Layer generation
    fl1 = FFIAFSpkInBrian(
        mfW = mfW,
        vfBias = vfBias,
        vtTauN = vtTauN,
        fNoiseStd =0.1,
        tRefractoryTime = 0.001
    )

    # - Input signal
    tsInEvt = ts.TSEvent(
        vtTimeTrace = [0.02, 0.04, 0.04, 0.06, 0.12],
        vnChannels = [1, 0, 1, 1, 0]
    )

    # - Compare states and time before and after
    vStateBefore = np.copy(fl1.vState)
    fl1.evolve(tsInEvt, tDuration=0.1)
    assert fl1.t == 0.1
    assert (vStateBefore != fl1.vState).any()

    fl1.reset_all()
    assert fl1.t == 0
    assert (vStateBefore == fl1.vState).all()


def test_reciaf():
    """ Test RecIAFBrian """
    from brian2 import second
    from NetworksPython import timeseries as ts
    from NetworksPython.layers.recurrent.iaf_brian import RecIAFBrian

    # - Generic parameters
    mfW = 2*np.random.rand(3,3)-1
    vfBias = 2*np.random.rand(3)-1
    vtTauN, vtTauSynR = np.random.rand(2,3)

    # - Layer generation
    rl0 = RecIAFBrian(
        mfW = mfW,
        vfBias = vfBias,
        vtTauN = vtTauN,
        vtTauSynR = vtTauSynR,
        fNoiseStd =0.1,
        tRefractoryTime = 0.001 * second,
    )

    # - Input signal
    tsInCont = ts.TSContinuous(
        vtTimeTrace = np.arange(15) * 0.01,
        mfSamples = np.ones((15,3))
    )

    # - Compare states and time before and after
    vStateBefore = np.copy(rl0.vState)
    rl0.evolve(tsInCont, tDuration=0.1)
    assert rl0.t == 0.1
    assert (vStateBefore != rl0.vState).any()

    rl0.reset_all()
    assert rl0.t == 0
    assert (vStateBefore == rl0.vState).all()

def test_reciaf_spkin():
    """ Test RecIAFSpkInBrian """
    from brian2 import second
    from NetworksPython import timeseries as ts
    from NetworksPython.layers.recurrent.iaf_brian import RecIAFSpkInBrian

    # - Negative weights, so that layer doesn't spike and gets reset
    mfWIn = np.random.rand(2,3)-1
    mfWRec = 2*np.random.rand(3,3)-1
    vfBias = 2*np.random.rand(3)-1
    vtTauN, vtTauSInp, vtTauSRec = np.random.rand(3, 3)

    # - Layer generation
    rl1 = RecIAFSpkInBrian(
        mfWIn = mfWIn,
        mfWRec = mfWRec,
        vfBias = vfBias,
        vtTauN = vtTauN,
        vtTauSInp = vtTauSInp,
        vtTauSRec = vtTauSRec,
        fNoiseStd =0.1,
        tRefractoryTime = 0.001
    )

    # - Input signal
    tsInEvt = ts.TSEvent(
        vtTimeTrace = [0.02, 0.04, 0.04, 0.06, 0.12],
        vnChannels = [1, 0, 1, 1, 0]
    )

    # - Compare states and time before and after
    vStateBefore = np.copy(rl1.vState)
    rl1.evolve(tsInEvt, tDuration=0.1)
    assert rl1.t == 0.1
    assert (vStateBefore != rl1.vState).any()

    rl1.reset_all()
    assert rl1.t == 0
    assert (vStateBefore == rl1.vState).all()