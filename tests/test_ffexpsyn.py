def test_ffexpsyn():
    # - Test FFExpSyn

    # from rockpool.layers import FFExpSynTorch
    from rockpool.layers import FFExpSyn
    from rockpool.timeseries import TSEvent, TSContinuous
    import numpy as np

    # - Layers

    size_in = 512
    size = 3
    dt = 0.001

    weights = np.linspace(-1, 1, size_in * size).reshape(size_in, size)
    bias = np.linspace(-1, 1, size)
    tau_syn = 0.15
    # flT = FFExpSynTorch(weights, dt=dt, bias=bias, tau_syn=tau_syn)
    flM = FFExpSyn(weights, dt=dt, bias=bias, tau_syn=tau_syn)

    # - Input signal

    tDur = 0.01
    nSpikes = 5

    vnC = np.tile(np.arange(size_in), int(np.ceil(1.0 / nSpikes * size)))[:nSpikes]
    vtT = np.linspace(0, tDur, nSpikes, endpoint=False)
    tsIn = TSEvent(vtT, vnC, num_channels=size_in)

    # - Evolve
    # tsT = flT.evolve(tsIn)
    tsM = flM.evolve(tsIn)
    # flT.reset_all()
    flM.reset_all()

    # assert(
    #         np.isclose(tsT.samples, tsM.samples, rtol=1e-4, atol=1e-5).all()
    #     # and np.isclose(tsT.times, tsM.times).all()
    # ), "Layer outputs are not the same."

    # - Training (only FFExpSyn and FFExpSynTorch)
    mfTgt = np.array(
        [
            np.sin(np.linspace(0, 10 * tDur, int(tDur / dt)) + fPhase)
            for fPhase in np.linspace(0, 3, size)
        ]
    ).T
    tsTgt = TSContinuous(np.arange(int(tDur / dt)) * dt, mfTgt)

    # flT.train_rr(tsTgt, tsIn, regularize=0.1, is_first=True, is_last=True)
    flM.train_rr(
        tsTgt,
        tsIn,
        regularize=0.1,
        is_first=True,
        is_last=True,
        return_trained_output=True,
    )

    # assert(
    #             np.isclose(flT.weights, flM.weights, rtol=1e-4, atol=1e-2).all()
    #         and np.isclose(flT.bias, flM.bias, rtol=1e-4, atol=1e-2).all()
    # ), "Training led to different results"


def test_ffexpsyntorch():
    # - Test FFIAFTorch

    # from rockpool.layers import FFExpSynTorch
    from rockpool.layers import FFExpSyn
    from rockpool.layers import FFExpSynTorch
    from rockpool.timeseries import TSEvent, TSContinuous
    import numpy as np

    # - Layers

    size_in = 512
    size = 3
    dt = 0.001

    weights = np.linspace(-1, 1, size_in * size).reshape(size_in, size)
    bias = np.linspace(-1, 1, size)
    tau_syn = 0.15
    flT = FFExpSynTorch(weights, dt=dt, bias=bias, tau_syn=tau_syn)
    flM = FFExpSyn(weights, dt=dt, bias=bias, tau_syn=tau_syn)

    # - Input signal

    tDur = 0.01
    nSpikes = 5

    vnC = np.tile(np.arange(size_in), int(np.ceil(1.0 / nSpikes * size)))[:nSpikes]
    vtT = np.linspace(0, tDur, nSpikes, endpoint=False)
    tsIn = TSEvent(vtT, vnC, num_channels=size_in)

    # - Evolve
    try:
        tsT = flT.evolve(tsIn)
    # - Catch runtime error ("code is too big") that occurs on the gitlab server
    except RuntimeError:
        return
    else:
        tsM = flM.evolve(tsIn)
        flT.reset_all()
        flM.reset_all()

        assert (
            np.isclose(tsT.samples, tsM.samples, rtol=1e-4, atol=1e-5).all()
            # and np.isclose(tsT.times, tsM.times).all()
        ), "Layer outputs are not the same."

        # - Training
        mfTgt = np.array(
            [
                np.sin(np.linspace(0, 10 * tDur, int(tDur / dt)) + fPhase)
                for fPhase in np.linspace(0, 3, size)
            ]
        ).T
        tsTgt = TSContinuous(np.arange(int(tDur / dt)) * dt, mfTgt)

        flT.train_rr(tsTgt, tsIn, regularize=0.1, is_first=True, is_last=True)
        flM.train_rr(tsTgt, tsIn, regularize=0.1, is_first=True, is_last=True)

        assert (
            np.isclose(flT.weights, flM.weights, rtol=1e-4, atol=1e-2).all()
            and np.isclose(flT.bias, flM.bias, rtol=1e-4, atol=1e-2).all()
        ), "Training led to different results"
