"""
Test torch-based IAF layers
"""

fTol = 1e-7  # - Tolerance for numerical comparisons


def test_ffiaf_torch():
    # - Test FFIAFTorch
    from NetworksPython.layers import FFIAFTorch
    from NetworksPython.timeseries import TSContinuous
    import numpy as np

    size_in = 384
    size = 512

    tDur = 0.01
    dt = 0.001

    weights = np.random.randn(size_in, size)
    fl = FFIAFTorch(weights, dt=dt, bRecord=False)

    mfIn = (
        0.005
        * np.array(
            [
                np.sin(np.linspace(0, 10 * tDur, int(tDur / dt)) + fPhase)
                for fPhase in np.linspace(0, 3, size_in)
            ]
        ).T
    )
    vtIn = np.arange(mfIn.shape[0]) * dt
    tsIn = TSContinuous(vtIn, mfIn)

    # - Compare states and time before and after
    vStateBefore = np.copy(fl.state)
    fl.evolve(tsIn, duration=0.08)
    assert fl.t == 0.08
    assert (vStateBefore != fl.state).any()

    fl.reset_all()
    assert fl.t == 0
    assert (vStateBefore == fl.state).all()

    # - Make sure item assignment works
    mfWTarget = weights.copy()
    mfWTarget[0] = np.random.rand(*(mfWTarget[0].shape))
    fl.weights[0] = mfWTarget[0]
    assert np.allclose(fl._mfW.cpu().numpy(), mfWTarget)


def test_ffiaf_spkin_torch():
    # - Test FFIAFSpkInTorch
    from NetworksPython.layers import FFIAFSpkInTorch
    from NetworksPython.timeseries import TSEvent
    import numpy as np

    size_in = 384
    size = 512

    tDur = 0.01
    dt = 0.001
    nSpikesIn = 50

    weights = np.random.randn(size_in, size)
    fl = FFIAFSpkInTorch(weights, dt=dt, bRecord=False)

    vtIn = np.sort(np.random.rand(nSpikesIn)) * tDur
    vnChIn = np.random.randint(size_in, size=nSpikesIn)
    tsIn = TSEvent(vtIn, vnChIn, num_channels=size_in)

    # - Compare states and time before and after
    vStateBefore = np.copy(fl.state)
    fl.evolve(tsIn, duration=0.08)
    assert fl.t == 0.08
    assert (vStateBefore != fl.state).any()

    fl.reset_all()
    assert fl.t == 0
    assert (vStateBefore == fl.state).all()

    # - Make sure item assignment works
    mfWTarget = weights.copy()
    mfWTarget[0] = np.random.rand(*(mfWTarget[0].shape))
    fl.weights[0] = mfWTarget[0]
    assert np.allclose(fl._mfW.cpu().numpy(), mfWTarget)


def test_reciaf_torch():
    # - Test RecIAFTorch

    from NetworksPython.layers import RecIAFTorch
    from NetworksPython.timeseries import TSContinuous
    import numpy as np

    size = 4

    tDur = 0.01
    dt = 0.001

    weights = 0.001 * np.random.randn(size, size)
    rl = RecIAFTorch(weights, dt=dt, vfBias=0.0101, bRecord=False)

    mfIn = (
        0.0001
        * np.array(
            [
                np.sin(np.linspace(0, 10 * tDur, int(tDur / dt)) + fPhase)
                for fPhase in np.linspace(0, 3, size)
            ]
        ).T
    )
    vtIn = np.arange(mfIn.shape[0]) * dt
    tsIn = TSContinuous(vtIn, mfIn)

    # - Compare states and time before and after
    vStateBefore = np.copy(rl.state)
    rl.evolve(tsIn, duration=0.4)
    assert rl.t == 0.4
    assert (vStateBefore != rl.state).any()

    rl.reset_all()
    assert rl.t == 0
    assert (vStateBefore == rl.state).all()

    # - Make sure item assignment works
    mfWTarget = rl.weights.copy()
    mfWTarget[0] = np.random.rand(*(mfWTarget[0].shape))
    rl.weights[0] = mfWTarget[0]
    assert np.allclose(rl._mfW.cpu().numpy(), mfWTarget)


def test_reciaf_spkin_torch():
    # - Test RecIAFSpkInTorch
    from NetworksPython.layers import RecIAFSpkInTorch
    from NetworksPython.timeseries import TSEvent
    import numpy as np

    size_in = 384
    size = 512

    tDur = 0.01
    dt = 0.001
    nSpikesIn = 50

    vtIn = np.sort(np.random.rand(nSpikesIn)) * tDur
    vnChIn = np.random.randint(size_in, size=nSpikesIn)
    tsIn = TSEvent(vtIn, vnChIn, num_channels=size_in)

    weights_in = 0.1 * np.random.randn(size_in, size)
    weights_rec = 0.001 * np.random.randn(size, size)
    rl = RecIAFSpkInTorch(weights_in, weights_rec, dt=dt, bRecord=False)

    # - Compare states and time before and after
    vStateBefore = np.copy(rl.state)
    rl.evolve(tsIn, duration=0.08)
    assert rl.t == 0.08
    assert (vStateBefore != rl.state).any()

    rl.reset_all()
    assert rl.t == 0
    assert (vStateBefore == rl.state).all()

    # - Make sure item assignment works
    mfWTarget = rl.weights.copy()
    mfWTarget[0] = np.random.rand(*(mfWTarget[0].shape))
    rl.weights[0] = mfWTarget[0]
    assert np.allclose(rl._mfW.cpu().numpy(), mfWTarget)


def test_ffiaf_refr_torch():
    # - Test FFIAFTorch
    from NetworksPython.layers import FFIAFTorch
    from NetworksPython.layers import FFIAFRefrTorch
    from NetworksPython.timeseries import TSContinuous
    import numpy as np

    size_in = 384
    size = 512

    tDur = 0.01
    dt = 0.001

    weights = np.random.randn(size_in, size)
    fl = FFIAFTorch(weights, dt=dt, bRecord=False)
    flr = FFIAFRefrTorch(weights, dt=dt, bRecord=False)

    mfIn = (
        0.005
        * np.array(
            [
                np.sin(np.linspace(0, 10 * tDur, int(tDur / dt)) + fPhase)
                for fPhase in np.linspace(0, 3, size_in)
            ]
        ).T
    )
    vtIn = np.arange(mfIn.shape[0]) * dt
    tsIn = TSContinuous(vtIn, mfIn)

    # - Compare states and time before and after
    vStateBefore = np.copy(flr.state)
    tsOut = fl.evolve(tsIn, duration=0.08)
    tsOutR = flr.evolve(tsIn, duration=0.08)

    assert flr.t == 0.08
    assert (vStateBefore != flr.state).any()
    assert (tsOut.times == tsOutR.times).all()
    assert (tsOut.channels == tsOutR.channels).all()

    flr.reset_all()
    assert flr.t == 0
    assert (vStateBefore == flr.state).all()

    flr.tRefractoryTime = 0.01

    tsOutR2 = flr.evolve(tsIn, duration=0.08)
    assert tsOutR2.times.size < tsOutR.times.size
    for iChannel in range(flr.size):
        assert (
            np.diff(tsOutR2.times[tsOutR2.channels == iChannel]) >= 0.01 - fTol
        ).all()

    flr2 = FFIAFRefrTorch(weights, dt=dt, bRecord=False, tRefractoryTime=0.01)
    tsOutR3 = flr2.evolve(tsIn, duration=0.08)

    assert (tsOutR2.times == tsOutR3.times).all()
    assert (tsOutR2.channels == tsOutR3.channels).all()

    # - Make sure item assignment works
    mfWTarget = flr.weights.copy()
    mfWTarget[0] = np.random.rand(*(mfWTarget[0].shape))
    flr.weights[0] = mfWTarget[0]
    assert np.allclose(flr._mfW.cpu().numpy(), mfWTarget)


def test_ffiaf_spkin_refr_torch():
    # - Test FFIAFSpkInTorch
    from NetworksPython.layers import FFIAFSpkInTorch
    from NetworksPython.layers import FFIAFSpkInRefrTorch
    from NetworksPython.timeseries import TSEvent
    import numpy as np

    size_in = 384
    size = 512

    tDur = 0.01
    dt = 0.001
    nSpikesIn = 50

    weights = np.random.randn(size_in, size)
    fl = FFIAFSpkInTorch(weights, dt=dt, bRecord=False)
    flr = FFIAFSpkInRefrTorch(weights, dt=dt, bRecord=False)

    vtIn = np.sort(np.random.rand(nSpikesIn)) * tDur
    vnChIn = np.random.randint(size_in, size=nSpikesIn)
    tsIn = TSEvent(vtIn, vnChIn, num_channels=size_in)

    # - Compare states and time before and after
    vStateBefore = np.copy(flr.state)
    tsOut = fl.evolve(tsIn, duration=0.08)
    tsOutR = flr.evolve(tsIn, duration=0.08)
    assert flr.t == 0.08
    assert (vStateBefore != flr.state).any()
    assert (tsOut.times == tsOutR.times).all()
    assert (tsOut.channels == tsOutR.channels).all()

    flr.reset_all()
    assert flr.t == 0
    assert (vStateBefore == flr.state).all()

    flr.tRefractoryTime = 0.01

    tsOutR2 = flr.evolve(tsIn, duration=0.08)
    assert tsOutR2.times.size < tsOutR.times.size
    for iChannel in range(flr.size):
        assert (
            np.diff(tsOutR2.times[tsOutR2.channels == iChannel]) >= 0.01 - fTol
        ).all()

    flr2 = FFIAFSpkInRefrTorch(weights, dt=dt, bRecord=False, tRefractoryTime=0.01)
    tsOutR3 = flr2.evolve(tsIn, duration=0.08)

    assert (tsOutR2.times == tsOutR3.times).all()
    assert (tsOutR2.channels == tsOutR3.channels).all()

    # - Make sure item assignment works
    mfWTarget = flr.weights.copy()
    mfWTarget[0] = np.random.rand(*(mfWTarget[0].shape))
    flr.weights[0] = mfWTarget[0]
    assert np.allclose(flr._mfW.cpu().numpy(), mfWTarget)


def test_reciaf_refr_torch():
    # - Test RecIAFTorch

    from NetworksPython.layers import RecIAFTorch
    from NetworksPython.layers import RecIAFRefrTorch
    from NetworksPython.timeseries import TSContinuous
    import numpy as np

    np.random.seed(0)

    size = 4

    tDur = 0.01
    dt = 0.001

    weights = 0.001 * np.random.randn(size, size)
    rl = RecIAFTorch(weights, dt=dt, vfBias=0.0101, bRecord=False)
    rlr = RecIAFRefrTorch(weights, dt=dt, vfBias=0.0101, bRecord=False)

    mfIn = (
        0.001
        * np.array(
            [
                np.sin(np.linspace(0, 10 * tDur, int(tDur / dt)) + fPhase)
                for fPhase in np.linspace(0, 3, size)
            ]
        ).T
    )
    vtIn = np.arange(mfIn.shape[0]) * dt
    tsIn = TSContinuous(vtIn, mfIn)

    # - Compare states and time before and after
    vStateBefore = np.copy(rlr.state)
    tsOut = rl.evolve(tsIn, duration=0.4)
    tsOutR = rlr.evolve(tsIn, duration=0.4)
    assert rlr.t == 0.4
    assert (vStateBefore != rlr.state).any()
    assert (tsOut.times == tsOutR.times).all()
    assert (tsOut.channels == tsOutR.channels).all()

    rlr.reset_all()
    assert rlr.t == 0
    assert (vStateBefore == rlr.state).all()

    rlr.tRefractoryTime = 0.01

    tsOutR2 = rlr.evolve(tsIn, duration=0.4)
    assert tsOutR2.times.size < tsOutR.times.size
    for iChannel in range(rlr.size):
        assert (
            np.diff(tsOutR2.times[tsOutR2.channels == iChannel]) >= 0.01 - fTol
        ).all()

    rlr2 = RecIAFRefrTorch(
        weights, dt=dt, vfBias=0.0101, bRecord=False, tRefractoryTime=0.01
    )
    tsOutR3 = rlr2.evolve(tsIn, duration=0.4)

    assert (tsOutR2.times == tsOutR3.times).all()
    assert (tsOutR2.channels == tsOutR3.channels).all()

    # - Make sure item assignment works
    mfWTarget = rl.weights.copy()
    mfWTarget[0] = np.random.rand(*(mfWTarget[0].shape))
    rl.weights[0] = mfWTarget[0]
    assert np.allclose(rl._mfW.cpu().numpy(), mfWTarget)


def test_reciaf_spkin_refr_torch():
    # - Test RecIAFSpkInTorch
    from NetworksPython.layers import RecIAFSpkInTorch
    from NetworksPython.layers import RecIAFSpkInRefrTorch
    from NetworksPython.timeseries import TSEvent
    import numpy as np

    np.random.seed(1)

    size_in = 4
    size = 5

    tDur = 0.01
    dt = 0.001
    nSpikesIn = 50

    vtIn = np.sort(np.random.rand(nSpikesIn)) * tDur
    vnChIn = np.random.randint(size_in, size=nSpikesIn)
    tsIn = TSEvent(vtIn, vnChIn, num_channels=size_in)

    weights_in = 0.1 * np.random.randn(size_in, size)
    weights_rec = 0.001 * np.random.randn(size, size)
    rl = RecIAFSpkInTorch(weights_in, weights_rec, dt=dt, bRecord=False)
    rlr = RecIAFSpkInRefrTorch(weights_in, weights_rec, dt=dt, bRecord=False)

    # - Compare states and time before and after
    vStateBefore = np.copy(rl.state)
    tsOut = rl.evolve(tsIn, duration=0.08)
    tsOutR = rlr.evolve(tsIn, duration=0.08)
    assert rlr.t == 0.08
    assert (vStateBefore != rlr.state).any()
    assert (tsOut.times == tsOutR.times).all()
    assert (tsOut.channels == tsOutR.channels).all()

    rlr.reset_all()
    assert rlr.t == 0
    assert (vStateBefore == rlr.state).all()

    rlr.tRefractoryTime = 0.01

    tsOutR2 = rlr.evolve(tsIn, duration=0.08)
    assert tsOutR2.times.size < tsOutR.times.size
    for iChannel in range(rlr.size):
        assert (
            np.diff(tsOutR2.times[tsOutR2.channels == iChannel]) >= 0.01 - fTol
        ).all()

    rlr2 = RecIAFSpkInRefrTorch(
        weights_in, weights_rec, dt=dt, bRecord=False, tRefractoryTime=0.01
    )
    tsOutR3 = rlr2.evolve(tsIn, duration=0.08)

    assert (tsOutR2.times == tsOutR3.times).all()
    assert (tsOutR2.channels == tsOutR3.channels).all()

    # - Make sure item assignment works
    mfWTarget = rl.weights.copy()
    mfWTarget[0] = np.random.rand(*(mfWTarget[0].shape))
    rl.weights[0] = mfWTarget[0]
    assert np.allclose(rl._mfW.cpu().numpy(), mfWTarget)


def test_reciaf_spkin_refr_cl_torch():
    # - Test RecIAFSpkInTorch
    from NetworksPython.layers import RecIAFSpkInRefrCLTorch
    from NetworksPython.timeseries import TSEvent
    import numpy as np

    np.random.seed(1)

    size_in = 4
    size = 5

    tDur = 0.01
    dt = 0.001
    nSpikesIn = 50

    vtIn = np.sort(np.random.rand(nSpikesIn)) * tDur
    vnChIn = np.random.randint(size_in, size=nSpikesIn)
    tsIn = TSEvent(vtIn, vnChIn, num_channels=size_in)

    weights_in = 0.1 * np.random.randn(size_in, size)
    weights_rec = 0.001 * np.random.randn(size, size)
    rl = RecIAFSpkInRefrCLTorch(weights_in, weights_rec, dt=dt, bRecord=False)

    # - Compare states and time before and after
    vStateBefore = np.copy(rl.state)
    rl.evolve(tsIn, duration=0.08)
    assert rl.t == 0.08
    assert (vStateBefore != rl.state).any()

    rl.reset_all()
    assert rl.t == 0
    assert (vStateBefore == rl.state).all()

    # - Make sure item assignment works
    mfWTarget = rl.weights.copy()
    mfWTarget[0] = np.random.rand(*(mfWTarget[0].shape))
    rl.weights[0] = mfWTarget[0]
    assert np.allclose(rl._mfW.cpu().numpy(), mfWTarget)
