def test_ffexpsyn():
    # - Test FFExpSyn

    # from NetworksPython.layers import FFExpSynTorch
    from NetworksPython.layers import FFExpSyn
    from NetworksPython.timeseries import TSEvent, TSContinuous
    import numpy as np

    # - Layers

    nSizeIn = 512
    nSize = 3
    tDt = 0.001

    mfW = np.linspace(-1, 1, nSizeIn*nSize).reshape(nSizeIn, nSize)
    vfBias = np.linspace(-1,1, nSize)
    tTauSyn = 0.15
    # flT = FFExpSynTorch(mfW, tDt=tDt, vfBias=vfBias, tTauSyn=tTauSyn)
    flM = FFExpSyn(mfW, tDt=tDt, vfBias=vfBias, tTauSyn=tTauSyn)

    # - Input signal

    tDur = 0.01
    nSpikes = 5

    vnC = np.tile(np.arange(nSizeIn), int(np.ceil(1./nSpikes*nSize)))[:nSpikes]
    vtT = np.linspace(0, tDur, nSpikes, endpoint=False)
    tsIn = TSEvent(vtT, vnC, nNumChannels=nSizeIn)

    # - Evolve
    # tsT = flT.evolve(tsIn)
    tsM = flM.evolve(tsIn)
    # flT.reset_all()
    flM.reset_all()

    # assert(
    #         np.isclose(tsT.mfSamples, tsM.mfSamples, rtol=1e-4, atol=1e-5).all()
    #     # and np.isclose(tsT.vtTimeTrace, tsM.vtTimeTrace).all()
    # ), "Layer outputs are not the same."

    # - Training (only FFExpSyn and FFExpSynTorch)
    mfTgt = np.array([
        np.sin(np.linspace(0,10*tDur,int(tDur/tDt)) + fPhase) for fPhase in np.linspace(0,3, nSize)
    ]).T
    tsTgt = TSContinuous(np.arange(int(tDur/tDt)) * tDt, mfTgt)

    # flT.train_rr(tsTgt, tsIn, fRegularize=0.1, bFirst=True, bFinal=True)
    flM.train_rr(tsTgt, tsIn, fRegularize=0.1, bFirst=True, bFinal=True)

    # assert(
    #             np.isclose(flT.mfW, flM.mfW, rtol=1e-4, atol=1e-2).all()
    #         and np.isclose(flT.vfBias, flM.vfBias, rtol=1e-4, atol=1e-2).all()
    # ), "Training led to different results"

def test_ffexpsyntorch():
    # - Test FFIAFTorch

    # from NetworksPython.layers import FFExpSynTorch
    from NetworksPython.layers import FFExpSyn
    from NetworksPython.layers import FFExpSynTorch
    from NetworksPython.timeseries import TSEvent, TSContinuous
    import numpy as np

    # - Layers

    nSizeIn = 512
    nSize = 3
    tDt = 0.001

    mfW = np.linspace(-1, 1, nSizeIn*nSize).reshape(nSizeIn, nSize)
    vfBias = np.linspace(-1,1, nSize)
    tTauSyn = 0.15
    flT = FFExpSynTorch(mfW, tDt=tDt, vfBias=vfBias, tTauSyn=tTauSyn)
    flM = FFExpSyn(mfW, tDt=tDt, vfBias=vfBias, tTauSyn=tTauSyn)

    # - Input signal

    tDur = 0.01
    nSpikes = 5

    vnC = np.tile(np.arange(nSizeIn), int(np.ceil(1./nSpikes*nSize)))[:nSpikes]
    vtT = np.linspace(0, tDur, nSpikes, endpoint=False)
    tsIn = TSEvent(vtT, vnC, nNumChannels=nSizeIn)

    # - Evolve
    tsT = flT.evolve(tsIn)
    tsM = flM.evolve(tsIn)
    flT.reset_all()
    flM.reset_all()

    assert(
            np.isclose(tsT.mfSamples, tsM.mfSamples, rtol=1e-4, atol=1e-5).all()
        # and np.isclose(tsT.vtTimeTrace, tsM.vtTimeTrace).all()
    ), "Layer outputs are not the same."

    # - Training
    mfTgt = np.array([
        np.sin(np.linspace(0,10*tDur,int(tDur/tDt)) + fPhase) for fPhase in np.linspace(0,3, nSize)
    ]).T
    tsTgt = TSContinuous(np.arange(int(tDur/tDt)) * tDt, mfTgt)

    flT.train_rr(tsTgt, tsIn, fRegularize=0.1, bFirst=True, bFinal=True)
    flM.train_rr(tsTgt, tsIn, fRegularize=0.1, bFirst=True, bFinal=True)

    assert(
                np.isclose(flT.mfW, flM.mfW, rtol=1e-4, atol=1e-2).all()
            and np.isclose(flT.vfBias, flM.vfBias, rtol=1e-4, atol=1e-2).all()
    ), "Training led to different results"
