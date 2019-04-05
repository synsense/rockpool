"""
Test torch-based IAF layers
"""

fTol = 1e-7  # - Tolerance for numerical comparisons


def test_ffiaf_torch():
    # - Test FFIAFTorch
    from NetworksPython.layers import FFIAFTorch
    from NetworksPython.timeseries import TSContinuous
    import numpy as np

    nSizeIn = 384
    nSize = 512

    tDur = 0.01
    tDt = 0.001

    mfW = np.random.randn(nSizeIn, nSize)
    fl = FFIAFTorch(mfW, tDt=tDt, bRecord=False)

    mfIn = (
        0.005
        * np.array(
            [
                np.sin(np.linspace(0, 10 * tDur, int(tDur / tDt)) + fPhase)
                for fPhase in np.linspace(0, 3, nSizeIn)
            ]
        ).T
    )
    vtIn = np.arange(mfIn.shape[0]) * tDt
    tsIn = TSContinuous(vtIn, mfIn)

    # - Compare states and time before and after
    vStateBefore = np.copy(fl.vState)
    fl.evolve(tsIn, tDuration=0.08)
    assert fl.t == 0.08
    assert (vStateBefore != fl.vState).any()

    fl.reset_all()
    assert fl.t == 0
    assert (vStateBefore == fl.vState).all()

    # - Make sure item assignment works
    mfWTarget = mfW.copy()
    mfWTarget[0] = np.random.rand(*(mfWTarget[0].shape))
    fl.mfW[0] = mfWTarget[0]
    assert np.allclose(fl._mfW.cpu().numpy(), mfWTarget)


def test_ffiaf_spkin_torch():
    # - Test FFIAFSpkInTorch
    from NetworksPython.layers import FFIAFSpkInTorch
    from NetworksPython.timeseries import TSEvent
    import numpy as np

    nSizeIn = 384
    nSize = 512

    tDur = 0.01
    tDt = 0.001
    nSpikesIn = 50

    mfW = np.random.randn(nSizeIn, nSize)
    fl = FFIAFSpkInTorch(mfW, tDt=tDt, bRecord=False)

    vtIn = np.sort(np.random.rand(nSpikesIn)) * tDur
    vnChIn = np.random.randint(nSizeIn, size=nSpikesIn)
    tsIn = TSEvent(vtIn, vnChIn, nNumChannels=nSizeIn)

    # - Compare states and time before and after
    vStateBefore = np.copy(fl.vState)
    fl.evolve(tsIn, tDuration=0.08)
    assert fl.t == 0.08
    assert (vStateBefore != fl.vState).any()

    fl.reset_all()
    assert fl.t == 0
    assert (vStateBefore == fl.vState).all()

    # - Make sure item assignment works
    mfWTarget = mfW.copy()
    mfWTarget[0] = np.random.rand(*(mfWTarget[0].shape))
    fl.mfW[0] = mfWTarget[0]
    assert np.allclose(fl._mfW.cpu().numpy(), mfWTarget)


def test_reciaf_torch():
    # - Test RecIAFTorch

    from NetworksPython.layers import RecIAFTorch
    from NetworksPython.timeseries import TSContinuous
    import numpy as np

    nSize = 4

    tDur = 0.01
    tDt = 0.001

    mfW = 0.001 * np.random.randn(nSize, nSize)
    rl = RecIAFTorch(mfW, tDt=tDt, vfBias=0.0101, bRecord=False)

    mfIn = (
        0.0001
        * np.array(
            [
                np.sin(np.linspace(0, 10 * tDur, int(tDur / tDt)) + fPhase)
                for fPhase in np.linspace(0, 3, nSize)
            ]
        ).T
    )
    vtIn = np.arange(mfIn.shape[0]) * tDt
    tsIn = TSContinuous(vtIn, mfIn)

    # - Compare states and time before and after
    vStateBefore = np.copy(rl.vState)
    rl.evolve(tsIn, tDuration=0.4)
    assert rl.t == 0.4
    assert (vStateBefore != rl.vState).any()

    rl.reset_all()
    assert rl.t == 0
    assert (vStateBefore == rl.vState).all()

    # - Make sure item assignment works
    mfWTarget = rl.mfW.copy()
    mfWTarget[0] = np.random.rand(*(mfWTarget[0].shape))
    rl.mfW[0] = mfWTarget[0]
    assert np.allclose(rl._mfW.cpu().numpy(), mfWTarget)


def test_reciaf_spkin_torch():
    # - Test RecIAFSpkInTorch
    from NetworksPython.layers import RecIAFSpkInTorch
    from NetworksPython.timeseries import TSEvent
    import numpy as np

    nSizeIn = 384
    nSize = 512

    tDur = 0.01
    tDt = 0.001
    nSpikesIn = 50

    vtIn = np.sort(np.random.rand(nSpikesIn)) * tDur
    vnChIn = np.random.randint(nSizeIn, size=nSpikesIn)
    tsIn = TSEvent(vtIn, vnChIn, nNumChannels=nSizeIn)

    mfWIn = 0.1 * np.random.randn(nSizeIn, nSize)
    mfWRec = 0.001 * np.random.randn(nSize, nSize)
    rl = RecIAFSpkInTorch(mfWIn, mfWRec, tDt=tDt, bRecord=False)

    # - Compare states and time before and after
    vStateBefore = np.copy(rl.vState)
    rl.evolve(tsIn, tDuration=0.08)
    assert rl.t == 0.08
    assert (vStateBefore != rl.vState).any()

    rl.reset_all()
    assert rl.t == 0
    assert (vStateBefore == rl.vState).all()

    # - Make sure item assignment works
    mfWTarget = rl.mfW.copy()
    mfWTarget[0] = np.random.rand(*(mfWTarget[0].shape))
    rl.mfW[0] = mfWTarget[0]
    assert np.allclose(rl._mfW.cpu().numpy(), mfWTarget)


def test_ffiaf_refr_torch():
    # - Test FFIAFTorch
    from NetworksPython.layers import FFIAFTorch
    from NetworksPython.layers import FFIAFRefrTorch
    from NetworksPython.timeseries import TSContinuous
    import numpy as np

    nSizeIn = 384
    nSize = 512

    tDur = 0.01
    tDt = 0.001

    mfW = np.random.randn(nSizeIn, nSize)
    fl = FFIAFTorch(mfW, tDt=tDt, bRecord=False)
    flr = FFIAFRefrTorch(mfW, tDt=tDt, bRecord=False)

    mfIn = (
        0.005
        * np.array(
            [
                np.sin(np.linspace(0, 10 * tDur, int(tDur / tDt)) + fPhase)
                for fPhase in np.linspace(0, 3, nSizeIn)
            ]
        ).T
    )
    vtIn = np.arange(mfIn.shape[0]) * tDt
    tsIn = TSContinuous(vtIn, mfIn)

    # - Compare states and time before and after
    vStateBefore = np.copy(flr.vState)
    tsOut = fl.evolve(tsIn, tDuration=0.08)
    tsOutR = flr.evolve(tsIn, tDuration=0.08)

    assert flr.t == 0.08
    assert (vStateBefore != flr.vState).any()
    assert (tsOut.vtTimeTrace == tsOutR.vtTimeTrace).all()
    assert (tsOut.vnChannels == tsOutR.vnChannels).all()

    flr.reset_all()
    assert flr.t == 0
    assert (vStateBefore == flr.vState).all()

    flr.tRefractoryTime = 0.01

    tsOutR2 = flr.evolve(tsIn, tDuration=0.08)
    assert tsOutR2.vtTimeTrace.size < tsOutR.vtTimeTrace.size
    for iChannel in range(flr.nSize):
        assert (
            np.diff(tsOutR2.vtTimeTrace[tsOutR2.vnChannels == iChannel]) >= 0.01 - fTol
        ).all()

    flr2 = FFIAFRefrTorch(mfW, tDt=tDt, bRecord=False, tRefractoryTime=0.01)
    tsOutR3 = flr2.evolve(tsIn, tDuration=0.08)

    assert (tsOutR2.vtTimeTrace == tsOutR3.vtTimeTrace).all()
    assert (tsOutR2.vnChannels == tsOutR3.vnChannels).all()

    # - Make sure item assignment works
    mfWTarget = flr.mfW.copy()
    mfWTarget[0] = np.random.rand(*(mfWTarget[0].shape))
    flr.mfW[0] = mfWTarget[0]
    assert np.allclose(flr._mfW.cpu().numpy(), mfWTarget)


def test_ffiaf_spkin_refr_torch():
    # - Test FFIAFSpkInTorch
    from NetworksPython.layers import FFIAFSpkInTorch
    from NetworksPython.layers import FFIAFSpkInRefrTorch
    from NetworksPython.timeseries import TSEvent
    import numpy as np

    nSizeIn = 384
    nSize = 512

    tDur = 0.01
    tDt = 0.001
    nSpikesIn = 50

    mfW = np.random.randn(nSizeIn, nSize)
    fl = FFIAFSpkInTorch(mfW, tDt=tDt, bRecord=False)
    flr = FFIAFSpkInRefrTorch(mfW, tDt=tDt, bRecord=False)

    vtIn = np.sort(np.random.rand(nSpikesIn)) * tDur
    vnChIn = np.random.randint(nSizeIn, size=nSpikesIn)
    tsIn = TSEvent(vtIn, vnChIn, nNumChannels=nSizeIn)

    # - Compare states and time before and after
    vStateBefore = np.copy(flr.vState)
    tsOut = fl.evolve(tsIn, tDuration=0.08)
    tsOutR = flr.evolve(tsIn, tDuration=0.08)
    assert flr.t == 0.08
    assert (vStateBefore != flr.vState).any()
    assert (tsOut.vtTimeTrace == tsOutR.vtTimeTrace).all()
    assert (tsOut.vnChannels == tsOutR.vnChannels).all()

    flr.reset_all()
    assert flr.t == 0
    assert (vStateBefore == flr.vState).all()

    flr.tRefractoryTime = 0.01

    tsOutR2 = flr.evolve(tsIn, tDuration=0.08)
    assert tsOutR2.vtTimeTrace.size < tsOutR.vtTimeTrace.size
    for iChannel in range(flr.nSize):
        assert (
            np.diff(tsOutR2.vtTimeTrace[tsOutR2.vnChannels == iChannel]) >= 0.01 - fTol
        ).all()

    flr2 = FFIAFSpkInRefrTorch(mfW, tDt=tDt, bRecord=False, tRefractoryTime=0.01)
    tsOutR3 = flr2.evolve(tsIn, tDuration=0.08)

    assert (tsOutR2.vtTimeTrace == tsOutR3.vtTimeTrace).all()
    assert (tsOutR2.vnChannels == tsOutR3.vnChannels).all()

    # - Make sure item assignment works
    mfWTarget = flr.mfW.copy()
    mfWTarget[0] = np.random.rand(*(mfWTarget[0].shape))
    flr.mfW[0] = mfWTarget[0]
    assert np.allclose(flr._mfW.cpu().numpy(), mfWTarget)


def test_reciaf_refr_torch():
    # - Test RecIAFTorch

    from NetworksPython.layers import RecIAFTorch
    from NetworksPython.layers import RecIAFRefrTorch
    from NetworksPython.timeseries import TSContinuous
    import numpy as np

    np.random.seed(0)

    nSize = 4

    tDur = 0.01
    tDt = 0.001

    mfW = 0.001 * np.random.randn(nSize, nSize)
    rl = RecIAFTorch(mfW, tDt=tDt, vfBias=0.0101, bRecord=False)
    rlr = RecIAFRefrTorch(mfW, tDt=tDt, vfBias=0.0101, bRecord=False)

    mfIn = (
        0.001
        * np.array(
            [
                np.sin(np.linspace(0, 10 * tDur, int(tDur / tDt)) + fPhase)
                for fPhase in np.linspace(0, 3, nSize)
            ]
        ).T
    )
    vtIn = np.arange(mfIn.shape[0]) * tDt
    tsIn = TSContinuous(vtIn, mfIn)

    # - Compare states and time before and after
    vStateBefore = np.copy(rlr.vState)
    tsOut = rl.evolve(tsIn, tDuration=0.4)
    tsOutR = rlr.evolve(tsIn, tDuration=0.4)
    assert rlr.t == 0.4
    assert (vStateBefore != rlr.vState).any()
    assert (tsOut.vtTimeTrace == tsOutR.vtTimeTrace).all()
    assert (tsOut.vnChannels == tsOutR.vnChannels).all()

    rlr.reset_all()
    assert rlr.t == 0
    assert (vStateBefore == rlr.vState).all()

    rlr.tRefractoryTime = 0.01

    tsOutR2 = rlr.evolve(tsIn, tDuration=0.4)
    assert tsOutR2.vtTimeTrace.size < tsOutR.vtTimeTrace.size
    for iChannel in range(rlr.nSize):
        assert (
            np.diff(tsOutR2.vtTimeTrace[tsOutR2.vnChannels == iChannel]) >= 0.01 - fTol
        ).all()

    rlr2 = RecIAFRefrTorch(
        mfW, tDt=tDt, vfBias=0.0101, bRecord=False, tRefractoryTime=0.01
    )
    tsOutR3 = rlr2.evolve(tsIn, tDuration=0.4)

    assert (tsOutR2.vtTimeTrace == tsOutR3.vtTimeTrace).all()
    assert (tsOutR2.vnChannels == tsOutR3.vnChannels).all()

    # - Make sure item assignment works
    mfWTarget = rl.mfW.copy()
    mfWTarget[0] = np.random.rand(*(mfWTarget[0].shape))
    rl.mfW[0] = mfWTarget[0]
    assert np.allclose(rl._mfW.cpu().numpy(), mfWTarget)


def test_reciaf_spkin_refr_torch():
    # - Test RecIAFSpkInTorch
    from NetworksPython.layers import RecIAFSpkInTorch
    from NetworksPython.layers import RecIAFSpkInRefrTorch
    from NetworksPython.timeseries import TSEvent
    import numpy as np

    np.random.seed(1)

    nSizeIn = 4
    nSize = 5

    tDur = 0.01
    tDt = 0.001
    nSpikesIn = 50

    vtIn = np.sort(np.random.rand(nSpikesIn)) * tDur
    vnChIn = np.random.randint(nSizeIn, size=nSpikesIn)
    tsIn = TSEvent(vtIn, vnChIn, nNumChannels=nSizeIn)

    mfWIn = 0.1 * np.random.randn(nSizeIn, nSize)
    mfWRec = 0.001 * np.random.randn(nSize, nSize)
    rl = RecIAFSpkInTorch(mfWIn, mfWRec, tDt=tDt, bRecord=False)
    rlr = RecIAFSpkInRefrTorch(mfWIn, mfWRec, tDt=tDt, bRecord=False)

    # - Compare states and time before and after
    vStateBefore = np.copy(rl.vState)
    tsOut = rl.evolve(tsIn, tDuration=0.08)
    tsOutR = rlr.evolve(tsIn, tDuration=0.08)
    assert rlr.t == 0.08
    assert (vStateBefore != rlr.vState).any()
    assert (tsOut.vtTimeTrace == tsOutR.vtTimeTrace).all()
    assert (tsOut.vnChannels == tsOutR.vnChannels).all()

    rlr.reset_all()
    assert rlr.t == 0
    assert (vStateBefore == rlr.vState).all()

    rlr.tRefractoryTime = 0.01

    tsOutR2 = rlr.evolve(tsIn, tDuration=0.08)
    assert tsOutR2.vtTimeTrace.size < tsOutR.vtTimeTrace.size
    for iChannel in range(rlr.nSize):
        assert (
            np.diff(tsOutR2.vtTimeTrace[tsOutR2.vnChannels == iChannel]) >= 0.01 - fTol
        ).all()

    rlr2 = RecIAFSpkInRefrTorch(
        mfWIn, mfWRec, tDt=tDt, bRecord=False, tRefractoryTime=0.01
    )
    tsOutR3 = rlr2.evolve(tsIn, tDuration=0.08)

    assert (tsOutR2.vtTimeTrace == tsOutR3.vtTimeTrace).all()
    assert (tsOutR2.vnChannels == tsOutR3.vnChannels).all()

    # - Make sure item assignment works
    mfWTarget = rl.mfW.copy()
    mfWTarget[0] = np.random.rand(*(mfWTarget[0].shape))
    rl.mfW[0] = mfWTarget[0]
    assert np.allclose(rl._mfW.cpu().numpy(), mfWTarget)


def test_reciaf_spkin_refr_cl_torch():
    # - Test RecIAFSpkInTorch
    from NetworksPython.layers import RecIAFSpkInRefrCLTorch
    from NetworksPython.timeseries import TSEvent
    import numpy as np

    np.random.seed(1)

    nSizeIn = 4
    nSize = 5

    tDur = 0.01
    tDt = 0.001
    nSpikesIn = 50

    vtIn = np.sort(np.random.rand(nSpikesIn)) * tDur
    vnChIn = np.random.randint(nSizeIn, size=nSpikesIn)
    tsIn = TSEvent(vtIn, vnChIn, nNumChannels=nSizeIn)

    mfWIn = 0.1 * np.random.randn(nSizeIn, nSize)
    mfWRec = 0.001 * np.random.randn(nSize, nSize)
    rl = RecIAFSpkInRefrCLTorch(mfWIn, mfWRec, tDt=tDt, bRecord=False)

    # - Compare states and time before and after
    vStateBefore = np.copy(rl.vState)
    rl.evolve(tsIn, tDuration=0.08)
    assert rl.t == 0.08
    assert (vStateBefore != rl.vState).any()

    rl.reset_all()
    assert rl.t == 0
    assert (vStateBefore == rl.vState).all()

    # - Make sure item assignment works
    mfWTarget = rl.mfW.copy()
    mfWTarget[0] = np.random.rand(*(mfWTarget[0].shape))
    rl.mfW[0] = mfWTarget[0]
    assert np.allclose(rl._mfW.cpu().numpy(), mfWTarget)
