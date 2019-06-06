"""
Test Nest-based spiking layers from layers.internal.iaf_nest
"""

import numpy as np

from NetworksPython.timeseries import set_plotting_backend

set_plotting_backend("plt")


def test_chargeSingleNeuron():
    """
    single neuron test
    charge neuron exactly to threshold without crossing using the bias
    """

    from NetworksPython.layers import FFIAFNest

    weights = [[1.0]]
    vfBias = [0.375]
    vtTauN = [0.01]
    vReset = -0.07
    vRest = -0.07
    vTh = -0.055
    fC = 250.0
    dt = 0.001
    tRef = 0.002

    fl0 = FFIAFNest(
        weights=weights,
        dt=dt,
        vfBias=vfBias,
        vtTauN=vtTauN,
        vfVReset=vReset,
        vfVRest=vRest,
        vfVThresh=vTh,
        vfCapacity=fC,
        tRefractoryTime=tRef,
        bRecord=True,
        name="test",
    )

    dFl0 = fl0.evolve(duration=1.0)

    assert fl0.state[0] > vTh - 0.00001
    assert fl0.state[0] <= vTh
    assert dFl0.isempty()


def test_chargeAndSpikeSingleNeuron():
    """
    single neuron test
    charge neuron exactly to threshold without crossing using the bias
    add small input to make it spike
    """

    from NetworksPython import timeseries as ts
    from NetworksPython.layers import FFIAFNest

    weights = [[1.0]]
    vfBias = [0.375]
    vtTauN = [0.01]
    vReset = -0.07
    vRest = -0.07
    vTh = -0.055
    fC = 250.0
    dt = 0.001
    tRef = 0.002

    fl0 = FFIAFNest(
        weights=weights,
        dt=dt,
        vfBias=vfBias,
        vtTauN=vtTauN,
        vfVReset=vReset,
        vfVRest=vRest,
        vfVThresh=vTh,
        vfCapacity=fC,
        tRefractoryTime=tRef,
        bRecord=True,
        name="test",
    )

    # - Input signal
    vTime = np.arange(0, 1, dt)
    vVal = np.zeros([len(vTime), 1])
    vVal[500] = 0.001

    tsInCont = ts.TSContinuous(vTime, vVal)

    dFl0 = fl0.evolve(tsInCont, duration=1.0)

    assert not dFl0.isempty()


def test_FFNestLayer():
    """ Test FFIAFNest"""
    from NetworksPython import timeseries as ts
    from NetworksPython.layers import FFIAFNest

    # - Generic parameters
    weights = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])
    vfBias = 0.01
    vtTauN = [0.02, 0.05, 0.1]

    # - Layer generation
    fl0 = FFIAFNest(
        weights=weights,
        dt=0.001,
        vfBias=vfBias,
        vtTauN=vtTauN,
        tRefractoryTime=0.001,
        bRecord=True,
    )

    # - Input signal
    tsInCont = ts.TSContinuous(times=np.arange(15) * 0.01, samples=np.ones((15, 2)))

    # - Compare states before and after
    vStateBefore = np.copy(fl0.state)
    dFl0 = fl0.evolve(tsInCont, duration=0.1)

    dFl0 = fl0.evolve(tsInCont, duration=0.1)

    assert fl0.t == 0.2
    assert (vStateBefore != fl0.state).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state).all()


def test_RecNestLayer():
    """ Test RecIAFNest"""
    from NetworksPython.layers import RecIAFSpkInNest
    import numpy as np

    # - Generic parameters
    weights_in = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])
    weights_rec = np.random.rand(3, 3) * 0.01
    vfBias = 0.01
    vtTauN = [0.02, 0.05, 0.1]
    vtTauS = [0.2, 0.01, 0.01]

    # - Layer generation
    fl0 = RecIAFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=0.001,
        vfBias=vfBias,
        vtTauN=vtTauN,
        vtTauS=vtTauS,
        tRefractoryTime=0.001,
        bRecord=True,
    )

    # - Compare states before and after
    vStateBefore = np.copy(fl0.state)
    dFl0 = fl0.evolve(duration=0.2)

    assert fl0.t == 0.2
    assert (vStateBefore != fl0.state).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state).all()


def test_FFToRecLayer():
    """ Test FFToRecNest"""

    from NetworksPython import timeseries as ts
    from NetworksPython.layers import FFIAFNest, RecIAFSpkInNest
    from NetworksPython.networks import network as nw
    import numpy as np

    weights = [[0.0, 0.001, 0.0]]
    vfBias = 0.375
    vtTauN = 0.01
    vReset = -0.07
    vRest = -0.07
    vTh = -0.055
    fC = 250.0
    dt = 0.001
    tRef = 0.002

    fl0 = FFIAFNest(
        weights=weights,
        dt=dt,
        vfBias=vfBias,
        vtTauN=vtTauN,
        vfVReset=vReset,
        vfVRest=vRest,
        vfVThresh=vTh,
        vfCapacity=fC,
        tRefractoryTime=tRef,
        bRecord=True,
        name="FF",
    )

    weights_in = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.6], [0.0, 0.0, 0.0]]
    weights_rec = np.random.rand(3, 3) * 0.001
    vfBiasRec = 0.0
    vtTauNRec = [0.02, 0.05, 0.1]
    vtTauSRec = [0.2, 0.01, 0.01]

    # - Layer generation
    fl1 = RecIAFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=0.001,
        vfBias=vfBiasRec,
        vtTauN=vtTauNRec,
        vtTauS=vtTauSRec,
        tRefractoryTime=0.001,
        bRecord=True,
        name="Rec",
    )

    net = nw.Network(fl0, fl1)

    # - Input signal
    vTime = np.arange(0, 1, dt)
    vVal = np.zeros([len(vTime), 1])
    vVal[500] = 0.01

    tsInCont = ts.TSContinuous(vTime, vVal)

    # - Compare states before and after
    vStateBefore = np.copy(fl1.state)

    dAct = net.evolve(tsInCont, duration=1.0)

    assert fl0.t == 1.0
    assert fl1.t == 1.0

    assert (vStateBefore != fl1.state).any()

    net.reset_all()
    assert fl0.t == 0
    assert fl1.t == 0
    assert (vStateBefore == fl1.state).all()


def test_randomizeStateRec():
    """ test Randomize State """

    """ Test RecIAFNest"""
    from NetworksPython.layers import RecIAFSpkInNest
    import numpy as np

    # - Generic parameters
    weights_in = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])
    weights_rec = np.random.rand(3, 3) * 0.01
    vfBias = 0.01
    vtTauN = [0.02, 0.05, 0.1]
    vtTauS = [0.2, 0.01, 0.01]

    # - Layer generation
    fl0 = RecIAFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=0.0001,
        vfBias=vfBias,
        vtTauN=vtTauN,
        vtTauS=vtTauS,
        tRefractoryTime=0.001,
        bRecord=True,
    )

    # - Compare states before and after
    vStateBefore = np.copy(fl0.state)

    fl0.randomize_state()

    assert (vStateBefore != fl0.state).any()


def test_randomizeStateFF():
    """ Test FFIAFNest"""
    from NetworksPython import timeseries as ts
    from NetworksPython.layers import FFIAFNest

    # - Generic parameters
    weights = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])
    vfBias = 0.01
    vtTauN = [0.02, 0.05, 0.1]

    # - Layer generation
    fl0 = FFIAFNest(
        weights=weights,
        dt=0.001,
        vfBias=vfBias,
        vtTauN=vtTauN,
        tRefractoryTime=0.001,
        bRecord=True,
    )

    # - Compare states before and after
    vStateBefore = np.copy(fl0.state)

    fl0.randomize_state()

    assert (vStateBefore != fl0.state).any()


def test_recording():
    """ tests if the shape of recording is correct """

    import numpy as np
    from NetworksPython import timeseries as ts
    from NetworksPython.layers import FFIAFNest

    # - Generic parameters
    weights = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])
    vfBias = 0.01
    vtTauN = [0.02, 0.05, 0.1]

    # - Layer generation
    fl0 = FFIAFNest(
        weights=weights,
        dt=0.0001,
        vfBias=vfBias,
        vtTauN=vtTauN,
        tRefractoryTime=0.001,
        bRecord=True,
    )

    # - Input signal
    tsInCont = ts.TSContinuous(times=np.arange(15) * 0.01, samples=np.ones((15, 2)))

    # - Compare states before and after
    vStateBefore = np.copy(fl0.state)
    dFl0 = fl0.evolve(tsInCont, duration=0.1)

    assert np.shape(fl0.mfRecordStates) == (3, 100)


def test_FFToRecLayerRepeat():
    """ Test FFToRecNest"""

    from NetworksPython import timeseries as ts
    from NetworksPython.layers import FFIAFNest, RecIAFSpkInNest
    from NetworksPython.networks import network as nw
    import numpy as np

    weights = [[0.0, 0.001, 0.0]]
    vfBias = 0.375
    vtTauN = 0.01
    vReset = -0.07
    vRest = -0.07
    vTh = -0.055
    fC = 250.0
    dt = 0.001
    tRef = 0.002

    fl0 = FFIAFNest(
        weights=weights,
        dt=dt,
        vfBias=vfBias,
        vtTauN=vtTauN,
        vfVReset=vReset,
        vfVRest=vRest,
        vfVThresh=vTh,
        vfCapacity=fC,
        tRefractoryTime=tRef,
        bRecord=True,
        name="FF",
    )

    weights_in = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.6], [0.0, 0.0, 0.0]]
    weights_rec = np.random.rand(3, 3) * 0.001
    vfBiasRec = 0.0
    vtTauNRec = [0.02, 0.05, 0.1]
    vtTauSRec = [0.2, 0.01, 0.01]

    # - Layer generation
    fl1 = RecIAFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=0.001,
        vfBias=vfBiasRec,
        vtTauN=vtTauNRec,
        vtTauS=vtTauSRec,
        tRefractoryTime=0.001,
        bRecord=True,
        name="Rec",
    )

    net = nw.Network(fl0, fl1)

    # - Input signal
    vTime = np.arange(0, 1, dt)
    vVal = np.zeros([len(vTime), 1])
    vVal[500] = 0.01

    tsInCont = ts.TSContinuous(vTime, vVal)

    # - Compare states before and after
    vStateBefore = np.copy(fl1.state)

    for _ in range(10):
        dAct = net.evolve(tsInCont, duration=1.0 / 10)

    assert fl0.t == 1.0
    assert fl1.t == 1.0

    assert (vStateBefore != fl1.state).any()

    net.reset_all()
    assert fl0.t == 0
    assert fl1.t == 0
    assert (vStateBefore == fl1.state).all()


def test_DefaultParams():
    """ Test RecIAFNest"""
    from NetworksPython.layers import RecIAFSpkInNest, FFIAFNest
    from NetworksPython import timeseries as ts
    from NetworksPython.networks import network as nw
    import numpy as np

    # - Generic parameters
    weights = np.ones([1, 2]) * 0.01
    weights_in = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])
    weights_rec = np.random.rand(3, 3) * 0.01
    vfBias = 0.0
    dt = 0.0001
    vtTauN = 0.02
    vtTauS = 0.05
    vfVThresh = -0.055
    vfVReset = -0.065
    vfVRest = -0.065
    vfCapacity = 100.0
    tRef = 0.001

    fl0 = FFIAFNest(
        weights=weights,
        dt=dt,
        vfBias=vfBias,
        vtTauN=vtTauN,
        vfVReset=vfVReset,
        vfVRest=vfVRest,
        vfVThresh=vfVThresh,
        vfCapacity=vfCapacity,
        tRefractoryTime=tRef,
        bRecord=True,
        name="FF",
    )

    fl1 = RecIAFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=dt,
        vfBias=vfBias,
        vtTauN=vtTauN,
        vtTauS=vtTauS,
        vfVThresh=vfVThresh,
        vfVReset=vfVReset,
        vfVRest=vfVRest,
        vfCapacity=vfCapacity,
        tRefractoryTime=tRef,
        bRecord=True,
        name="Rec",
    )

    net0 = nw.Network(fl0, fl1)

    fl2 = FFIAFNest(weights=weights, bRecord=True, name="FF")

    fl3 = RecIAFSpkInNest(
        weights_in=weights_in, weights_rec=weights_rec, bRecord=True, name="Rec"
    )

    net1 = nw.Network(fl2, fl3)

    # - Input signal
    vTime = np.arange(0, 1, dt)
    vVal = np.zeros([len(vTime), 1])
    vVal[2000:7000] = 0.01

    tsInCont = ts.TSContinuous(vTime, vVal)

    assert (fl0.state == fl2.state).all()
    assert (fl1.state == fl3.state).all()

    # - Compare states before and after
    dAct0 = net0.evolve(tsInCont, duration=1.0)
    dAct1 = net1.evolve(tsInCont, duration=1.0)

    assert (fl0.state == fl2.state).all()
    assert (fl1.state == fl3.state).all()


def test_Multithreading():
    """ Test RecIAFNest"""
    from NetworksPython.layers import RecIAFSpkInNest, FFIAFNest
    from NetworksPython import timeseries as ts
    from NetworksPython.networks import network as nw
    import time
    import numpy as np

    # - Generic parameters
    weights = np.ones([1, 200]) * 0.01
    weights_in = np.random.rand(200, 300) * 0.001
    weights_rec = np.random.rand(300, 300) * 0.001
    vfBias = 0.01
    dt = 0.0001
    vtTauN = 0.02
    vtTauS = 0.05
    vfVThresh = -0.055
    vfVReset = -0.065
    vfVRest = -0.065
    vfCapacity = 100.0
    tRef = 0.001

    np.random.seed(0)

    fl0 = FFIAFNest(
        weights=weights,
        dt=dt,
        vfBias=vfBias,
        vtTauN=vtTauN,
        vfVReset=vfVReset,
        vfVRest=vfVRest,
        vfVThresh=vfVThresh,
        vfCapacity=vfCapacity,
        tRefractoryTime=tRef,
        num_cores=1,
        bRecord=True,
        name="FF",
    )

    fl1 = RecIAFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=dt,
        vfBias=vfBias,
        vtTauN=vtTauN,
        vtTauS=vtTauS,
        vfVThresh=vfVThresh,
        vfVReset=vfVReset,
        vfVRest=vfVRest,
        vfCapacity=vfCapacity,
        tRefractoryTime=tRef,
        num_cores=1,
        bRecord=True,
        name="Rec",
    )

    net0 = nw.Network(fl0, fl1)

    np.random.seed(0)

    fl2 = FFIAFNest(
        weights=weights,
        dt=dt,
        vfBias=vfBias,
        vtTauN=vtTauN,
        vfVReset=vfVReset,
        vfVRest=vfVRest,
        vfVThresh=vfVThresh,
        vfCapacity=vfCapacity,
        tRefractoryTime=tRef,
        bRecord=True,
        num_cores=4,
        name="FF",
    )

    fl3 = RecIAFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=dt,
        vfBias=vfBias,
        vtTauN=vtTauN,
        vtTauS=vtTauS,
        vfVThresh=vfVThresh,
        vfVReset=vfVReset,
        vfVRest=vfVRest,
        vfCapacity=vfCapacity,
        tRefractoryTime=tRef,
        bRecord=True,
        num_cores=4,
        name="Rec",
    )

    net1 = nw.Network(fl2, fl3)

    # - Input signal
    vTime = np.arange(0, 1, dt)
    vVal = np.zeros([len(vTime), 1])
    vVal[2000:7000] = 0.01

    tsInCont = ts.TSContinuous(vTime, vVal)

    epsilon = 0.0000001

    assert (np.abs(fl0.state - fl2.state) < epsilon).all()
    assert (np.abs(fl1.state - fl3.state) < epsilon).all()

    # - Compare states before and after
    tStart0 = time.time()
    np.random.seed(0)
    dAct0 = net0.evolve(tsInCont, duration=10.0)
    tStop0 = time.time()

    tStart1 = time.time()
    np.random.seed(0)
    dAct1 = net1.evolve(tsInCont, duration=10.0)
    tStop1 = time.time()

    # assert (tStop1 - tStart1 < tStop0 - tStart0)  # multithreading is faster
    assert (np.abs(fl0.state - fl2.state) < epsilon).all()
    assert (np.abs(fl1.state - fl3.state) < epsilon).all()


def test_delays():
    """ test delays """
    from NetworksPython import timeseries as ts
    from NetworksPython.layers import FFIAFNest, RecIAFSpkInNest
    from NetworksPython.networks import network as nw
    import numpy as np

    weights = [[0.001, 0.0, 0.0, 0.0]]
    vfBias = 0.375
    vtTauN = 0.01
    vReset = -0.07
    vRest = -0.07
    vTh = -0.055
    fC = 250.0
    dt = 0.001
    tRef = 0.002

    fl0 = FFIAFNest(
        weights=weights,
        dt=dt,
        vfBias=vfBias,
        vtTauN=vtTauN,
        vfVReset=vReset,
        vfVRest=vRest,
        vfVThresh=vTh,
        vfCapacity=fC,
        tRefractoryTime=tRef,
        bRecord=True,
        name="FF",
    )

    weights_in = [
        [0.015, 0.015, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    weights_rec = [
        [0.0, 0.0, 0.015, 0.015],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    delay_in = [
        [0.001, 0.011, 0.001, 0.001],
        [0.001, 0.001, 0.001, 0.001],
        [0.001, 0.001, 0.001, 0.001],
    ]
    delay_rec = [
        [0.001, 0.001, 0.001, 0.011],
        [0.001, 0.001, 0.001, 0.001],
        [0.001, 0.001, 0.001, 0.001],
    ]
    vfBiasRec = 0.0
    vtTauNRec = [0.2, 0.2, 0.2, 0.2]
    vtTauSRec = [0.2, 0.2, 0.2, 0.2]

    # - Layer generation
    fl1 = RecIAFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        delay_in=delay_in,
        delay_rec=delay_rec,
        dt=0.001,
        vfBias=vfBiasRec,
        vtTauN=vtTauNRec,
        vtTauS=vtTauSRec,
        tRefractoryTime=0.001,
        bRecord=True,
        name="Rec",
    )

    net = nw.Network(fl0, fl1)

    # - Input signal
    vTime = np.arange(0, 1, dt)
    vVal = np.zeros([len(vTime), 1])
    vVal[500] = 0.01

    tsInCont = ts.TSContinuous(vTime, vVal)
    dAct = net.evolve(tsInCont, duration=1.0)

    times = dAct["Rec"].times

    eps = 0.000001

    assert times[1] - times[0] - 0.01 < eps
    assert times[3] - times[2] - 0.01 < eps


def test_IAF2AEIFNest():
    """ Test RecIAFNest to RecAEIFNest """
    from NetworksPython.layers import RecIAFSpkInNest
    from NetworksPython.layers import RecAEIFSpkInNest
    from NetworksPython import timeseries as ts
    import numpy as np

    # - Generic parameters
    weights_in = np.array([[-0.001, 0.002, 0.004], [0.03, -0.003, -0.0015]])
    weights_rec = np.random.rand(3, 3) * 0.001
    vfBias = 0.0
    vtTauN = [0.02, 0.05, 0.1]
    vtTauS = [0.2, 0.01, 0.01]
    dt = 0.001
    vThresh = -0.055
    vRest = -0.065

    # - Layer generation
    fl0 = RecIAFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=dt,
        vfBias=vfBias,
        vtTauN=vtTauN,
        vtTauS=vtTauS,
        vfVThresh=vThresh,
        vfVReset=vRest,
        vfVRest=vRest,
        tRefractoryTime=0.001,
        bRecord=True,
    )

    # - Layer generation
    fl1 = RecAEIFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=dt,
        vfBias=vfBias,
        vtTauN=vtTauN,
        vtTauS=vtTauS,
        vfVThresh=vThresh,
        vfVReset=vRest,
        vfVRest=vRest,
        a=0.0,
        b=0.0,
        delta_t=0.0,
        tRefractoryTime=0.001,
        bRecord=True,
    )

    # - Input signal
    spikeTimes = np.arange(0, 1, dt)
    spikeTrain0 = np.sort(np.random.choice(spikeTimes, size=20, replace=False))
    channels = np.random.choice([0, 1], 20, replace=True).astype(int)

    tsInEvent = ts.TSEvent(spikeTrain0, channels)

    # - Compare states before and after

    assert (np.abs(fl0.state - fl1.state) < 0.00001).all()

    dFl0 = fl0.evolve(tsInEvent, duration=1.0)
    dFl1 = fl1.evolve(tsInEvent, duration=1.0)

    assert (np.abs(fl0.state - fl1.state) < 0.00001).all()


def test_SaveLoad():
    """ Test save and load RecAEIFNest """
    from NetworksPython.layers import RecIAFSpkInNest
    from NetworksPython.layers import RecAEIFSpkInNest
    from NetworksPython import timeseries as ts
    from NetworksPython.networks import network as nw
    import numpy as np

    # - Generic parameters
    weights_in = np.array([[-0.001, 0.002, 0.004], [0.03, -0.003, -0.0015]])
    weights_rec = np.random.rand(3, 3) * 0.001
    vfBias = 0.0
    vtTauN = [0.02, 0.05, 0.1]
    vtTauS = [0.2, 0.01, 0.01]
    dt = 0.001
    vThresh = -0.055
    vRest = -0.065

    # - Layer generation
    fl0 = RecAEIFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=dt,
        vfBias=vfBias,
        vtTauN=vtTauN,
        vtTauS=vtTauS,
        vfVThresh=vThresh,
        vfVReset=vRest,
        vfVRest=vRest,
        a=0.,
        b=1.,
        delta_t=0.,
        tRefractoryTime=0.001,
        bRecord=True,
    )

    net0 = nw.Network(fl0)
    net0.save("tmp.model")

    net1 = nw.Network.load("tmp.model")

    # - Input signal
    spikeTimes = np.arange(0, 1, dt)
    spikeTrain0 = np.sort(np.random.choice(spikeTimes, size=20, replace=False))
    channels = np.random.choice([0, 1], 20, replace=True).astype(int)

    tsInEvent = ts.TSEvent(spikeTrain0, channels)

    # - Compare states before and after

    assert (np.abs(net0.inputlayer.state - net1.inputlayer.state) < 0.00001).all()

    dFl0 = net0.evolve(tsInEvent, duration=1.0)
    dFl1 = net1.evolve(tsInEvent, duration=1.0)

    assert (np.abs(net0.inputlayer.state - net1.inputlayer.state) < 0.00001).all()
