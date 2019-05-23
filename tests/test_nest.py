"""
Test Nest-based spiking layers from layers.internal.iaf_nest
"""

import numpy as np
import time
import pylab as plt

from hyperopt import hp
from hyperopt import fmin, tpe, space_eval, Trials
from NetworksPython.timeseries import set_plotting_backend
set_plotting_backend("plt")


def test_chargeSingleNeuron():
    """
    single neuron test
    charge neuron exactly to threshold without crossing using the bias
    """

    from NetworksPython.layers import FFIAFNest

    mfW = [[1.]]
    vfBias = [0.375]
    vtTauN = [0.01]
    vReset = -0.07
    vRest = -0.07
    vTh = -0.055
    fC = 250.
    tDt = 0.001
    tRef = 0.002

    fl0 = FFIAFNest(mfW=mfW,
                    tDt=tDt,
                    vfBias=vfBias,
                    vtTauN=vtTauN,
                    vfVReset=vReset,
                    vfVRest=vRest,
                    vfVThresh=vTh,
                    vfCapacity=fC,
                    tRefractoryTime=tRef,
                    bRecord=True,
                    strName="test")

    dFl0 = fl0.evolve(tDuration=1.)

    assert(fl0.vState[0] > vTh - 0.00001)
    assert(fl0.vState[0] <= vTh)
    assert(dFl0.isempty())


def test_chargeAndSpikeSingleNeuron():
    """
    single neuron test
    charge neuron exactly to threshold without crossing using the bias
    add small input to make it spike
    """

    from NetworksPython import timeseries as ts
    from NetworksPython.layers import FFIAFNest

    mfW = [[1.]]
    vfBias = [0.375]
    vtTauN = [0.01]
    vReset = -0.07
    vRest = -0.07
    vTh = -0.055
    fC = 250.
    tDt = 0.001
    tRef = 0.002

    fl0 = FFIAFNest(mfW=mfW,
                    tDt=tDt,
                    vfBias=vfBias,
                    vtTauN=vtTauN,
                    vfVReset=vReset,
                    vfVRest=vRest,
                    vfVThresh=vTh,
                    vfCapacity=fC,
                    tRefractoryTime=tRef,
                    bRecord=True,
                    strName="test")

    # - Input signal
    vTime = np.arange(0, 1, tDt)
    vVal = np.zeros([len(vTime), 1])
    vVal[500] = 0.001

    tsInCont = ts.TSContinuous(vTime, vVal)

    dFl0 = fl0.evolve(tsInCont, tDuration=1.)

    assert(not dFl0.isempty())


def test_FFNestLayer():
    """ Test FFIAFNest"""
    from NetworksPython import timeseries as ts
    from NetworksPython.layers import FFIAFNest

    # - Generic parameters
    mfW = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])
    vfBias = 0.01
    vtTauN = [0.02, 0.05, 0.1]

    # - Layer generation
    fl0 = FFIAFNest(
        mfW=mfW,
        tDt=0.001,
        vfBias=vfBias,
        vtTauN=vtTauN,
        tRefractoryTime=0.001,
        bRecord=True,
    )

    # - Input signal
    tsInCont = ts.TSContinuous(
        times=np.arange(15) * 0.01, samples=np.ones((15, 2))
    )

    # - Compare states before and after
    vStateBefore = np.copy(fl0.vState)
    dFl0 = fl0.evolve(tsInCont, tDuration=0.1)

    dFl0 = fl0.evolve(tsInCont, tDuration=0.1)

    assert fl0.t == 0.2
    assert (vStateBefore != fl0.vState).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.vState).all()


def test_RecNestLayer():
    """ Test RecIAFNest"""
    from NetworksPython.layers import RecIAFSpkInNest
    import numpy as np

    # - Generic parameters
    mfWIn = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])
    mfWRec = np.random.rand(3, 3) * 0.01
    vfBias = 0.01
    vtTauN = [0.02, 0.05, 0.1]
    vtTauS = [0.2, 0.01, 0.01]

    # - Layer generation
    fl0 = RecIAFSpkInNest(
        mfWIn=mfWIn,
        mfWRec=mfWRec,
        tDt=0.001,
        vfBias=vfBias,
        vtTauN=vtTauN,
        vtTauS=vtTauS,
        tRefractoryTime=0.001,
        bRecord=True,
    )

    # - Compare states before and after
    vStateBefore = np.copy(fl0.vState)
    dFl0 = fl0.evolve(tDuration=0.2)

    assert fl0.t == 0.2
    assert (vStateBefore != fl0.vState).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.vState).all()


def test_FFToRecLayer():
    """ Test FFToRecNest"""

    from NetworksPython import timeseries as ts
    from NetworksPython.layers import FFIAFNest, RecIAFSpkInNest
    from NetworksPython.networks import network as nw
    import numpy as np

    mfW = [[0., 0.001, 0.]]
    vfBias = 0.375
    vtTauN = 0.01
    vReset = -0.07
    vRest = -0.07
    vTh = -.055
    fC = 250.
    tDt = 0.001
    tRef = 0.002

    fl0 = FFIAFNest(mfW=mfW,
                    tDt=tDt,
                    vfBias=vfBias,
                    vtTauN=vtTauN,
                    vfVReset=vReset,
                    vfVRest=vRest,
                    vfVThresh=vTh,
                    vfCapacity=fC,
                    tRefractoryTime=tRef,
                    bRecord=True,
                    strName="FF")

    mfWIn = [[0., 0., 0.], [0., 0., 0.6], [0., 0., 0.]]
    mfWRec = np.random.rand(3, 3) * 0.001
    vfBiasRec = 0.0
    vtTauNRec = [0.02, 0.05, 0.1]
    vtTauSRec = [0.2, 0.01, 0.01]

    # - Layer generation
    fl1 = RecIAFSpkInNest(
        mfWIn=mfWIn,
        mfWRec=mfWRec,
        tDt=0.001,
        vfBias=vfBiasRec,
        vtTauN=vtTauNRec,
        vtTauS=vtTauSRec,
        tRefractoryTime=0.001,
        bRecord=True,
        strName="Rec")

    net = nw.Network(fl0, fl1)

    # - Input signal
    vTime = np.arange(0, 1, tDt)
    vVal = np.zeros([len(vTime), 1])
    vVal[500] = 0.01

    tsInCont = ts.TSContinuous(vTime, vVal)

    # - Compare states before and after
    vStateBefore = np.copy(fl1.vState)

    dAct = net.evolve(tsInCont, tDuration=1.0)

    assert fl0.t == 1.0
    assert fl1.t == 1.0

    assert (vStateBefore != fl1.vState).any()

    net.reset_all()
    assert fl0.t == 0
    assert fl1.t == 0
    assert (vStateBefore == fl1.vState).all()


def test_randomizeStateRec():
    """ test Randomize State """

    """ Test RecIAFNest"""
    from NetworksPython.layers import RecIAFSpkInNest
    import numpy as np

    # - Generic parameters
    mfWIn = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])
    mfWRec = np.random.rand(3, 3) * 0.01
    vfBias = 0.01
    vtTauN = [0.02, 0.05, 0.1]
    vtTauS = [0.2, 0.01, 0.01]

    # - Layer generation
    fl0 = RecIAFSpkInNest(
        mfWIn=mfWIn,
        mfWRec=mfWRec,
        tDt=0.0001,
        vfBias=vfBias,
        vtTauN=vtTauN,
        vtTauS=vtTauS,
        tRefractoryTime=0.001,
        bRecord=True,
    )

    # - Compare states before and after
    vStateBefore = np.copy(fl0.vState)

    fl0.randomize_state()

    assert (vStateBefore != fl0.vState).any()


def test_randomizeStateFF():
    """ Test FFIAFNest"""
    from NetworksPython import timeseries as ts
    from NetworksPython.layers import FFIAFNest

    # - Generic parameters
    mfW = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])
    vfBias = 0.01
    vtTauN = [0.02, 0.05, 0.1]

    # - Layer generation
    fl0 = FFIAFNest(
        mfW=mfW,
        tDt=0.001,
        vfBias=vfBias,
        vtTauN=vtTauN,
        tRefractoryTime=0.001,
        bRecord=True,
    )

    # - Compare states before and after
    vStateBefore = np.copy(fl0.vState)

    fl0.randomize_state()

    assert (vStateBefore != fl0.vState).any()


def test_recording():
    """ tests if the shape of recording is correct """

    import numpy as np
    from NetworksPython import timeseries as ts
    from NetworksPython.layers import FFIAFNest

    # - Generic parameters
    mfW = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])
    vfBias = 0.01
    vtTauN = [0.02, 0.05, 0.1]

    # - Layer generation
    fl0 = FFIAFNest(
        mfW=mfW,
        tDt=0.0001,
        vfBias=vfBias,
        vtTauN=vtTauN,
        tRefractoryTime=0.001,
        bRecord=True,
    )

    # - Input signal
    tsInCont = ts.TSContinuous(
        times=np.arange(15) * 0.01, samples=np.ones((15, 2))
    )

    # - Compare states before and after
    vStateBefore = np.copy(fl0.vState)
    dFl0 = fl0.evolve(tsInCont, tDuration=0.1)

    assert(np.shape(fl0.mfRecordStates) == (3, 100))


def test_FFToRecLayerRepeat():
    """ Test FFToRecNest"""

    from NetworksPython import timeseries as ts
    from NetworksPython.layers import FFIAFNest, RecIAFSpkInNest
    from NetworksPython.networks import network as nw
    import numpy as np

    mfW = [[0., 0.001, 0.]]
    vfBias = 0.375
    vtTauN = 0.01
    vReset = -0.07
    vRest = -0.07
    vTh = -0.055
    fC = 250.
    tDt = 0.001
    tRef = 0.002

    fl0 = FFIAFNest(mfW=mfW,
                    tDt=tDt,
                    vfBias=vfBias,
                    vtTauN=vtTauN,
                    vfVReset=vReset,
                    vfVRest=vRest,
                    vfVThresh=vTh,
                    vfCapacity=fC,
                    tRefractoryTime=tRef,
                    bRecord=True,
                    strName="FF")

    mfWIn = [[0., 0., 0.], [0., 0., 0.6], [0., 0., 0.]]
    mfWRec = np.random.rand(3, 3) * 0.001
    vfBiasRec = 0.0
    vtTauNRec = [0.02, 0.05, 0.1]
    vtTauSRec = [0.2, 0.01, 0.01]

    # - Layer generation
    fl1 = RecIAFSpkInNest(
        mfWIn=mfWIn,
        mfWRec=mfWRec,
        tDt=0.001,
        vfBias=vfBiasRec,
        vtTauN=vtTauNRec,
        vtTauS=vtTauSRec,
        tRefractoryTime=0.001,
        bRecord=True,
        strName="Rec")

    net = nw.Network(fl0, fl1)

    # - Input signal
    vTime = np.arange(0, 1, tDt)
    vVal = np.zeros([len(vTime), 1])
    vVal[500] = 0.01

    tsInCont = ts.TSContinuous(vTime, vVal)

    # - Compare states before and after
    vStateBefore = np.copy(fl1.vState)

    for _ in range(10):
        dAct = net.evolve(tsInCont, tDuration=1.0 / 10)

    assert fl0.t == 1.0
    assert fl1.t == 1.0

    assert (vStateBefore != fl1.vState).any()

    net.reset_all()
    assert fl0.t == 0
    assert fl1.t == 0
    assert (vStateBefore == fl1.vState).all()


def test_DefaultParams():
    """ Test RecIAFNest"""
    from NetworksPython.layers import RecIAFSpkInNest, FFIAFNest
    from NetworksPython import timeseries as ts
    from NetworksPython.networks import network as nw
    import numpy as np

    # - Generic parameters
    mfW = np.ones([1, 2]) * 0.01
    mfWIn = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])
    mfWRec = np.random.rand(3, 3) * 0.01
    vfBias = 0.0
    tDt = 0.0001
    vtTauN = 0.02
    vtTauS = 0.05
    vfVThresh = -0.055
    vfVReset = -0.065
    vfVRest = -0.065
    vfCapacity = 100.
    tRef = 0.001

    fl0 = FFIAFNest(mfW=mfW,
                    tDt=tDt,
                    vfBias=vfBias,
                    vtTauN=vtTauN,
                    vfVReset=vfVReset,
                    vfVRest=vfVRest,
                    vfVThresh=vfVThresh,
                    vfCapacity=vfCapacity,
                    tRefractoryTime=tRef,
                    bRecord=True,
                    strName="FF")

    fl1 = RecIAFSpkInNest(mfWIn=mfWIn,
                          mfWRec=mfWRec,
                          tDt=tDt,
                          vfBias=vfBias,
                          vtTauN=vtTauN,
                          vtTauS=vtTauS,
                          vfVThresh=vfVThresh,
                          vfVReset=vfVReset,
                          vfVRest=vfVRest,
                          vfCapacity=vfCapacity,
                          tRefractoryTime=tRef,
                          bRecord=True,
                          strName="Rec")

    net0 = nw.Network(fl0, fl1)

    fl2 = FFIAFNest(mfW=mfW,
                    bRecord=True,
                    strName="FF")

    fl3 = RecIAFSpkInNest(mfWIn=mfWIn,
                          mfWRec=mfWRec,
                          bRecord=True,
                          strName="Rec")

    net1 = nw.Network(fl2, fl3)

    # - Input signal
    vTime = np.arange(0, 1, tDt)
    vVal = np.zeros([len(vTime), 1])
    vVal[2000:7000] = 0.01

    tsInCont = ts.TSContinuous(vTime, vVal)

    assert (fl0.vState == fl2.vState).all()
    assert (fl1.vState == fl3.vState).all()

    # - Compare states before and after
    dAct0 = net0.evolve(tsInCont, tDuration=1.0)
    dAct1 = net1.evolve(tsInCont, tDuration=1.0)

    assert (fl0.vState == fl2.vState).all()
    assert (fl1.vState == fl3.vState).all()


def test_Multithreading():
    """ Test RecIAFNest"""
    from NetworksPython.layers import RecIAFSpkInNest, FFIAFNest
    from NetworksPython import timeseries as ts
    from NetworksPython.networks import network as nw
    import numpy as np

    # - Generic parameters
    mfW = np.ones([1, 200]) * 0.01
    mfWIn = np.random.rand(200, 300) * 0.001
    mfWRec = np.random.rand(300, 300) * 0.001
    vfBias = 0.01
    tDt = 0.0001
    vtTauN = 0.02
    vtTauS = 0.05
    vfVThresh = -0.055
    vfVReset = -0.065
    vfVRest = -0.065
    vfCapacity = 100.
    tRef = 0.001

    np.random.seed(0)

    fl0 = FFIAFNest(mfW=mfW,
                    tDt=tDt,
                    vfBias=vfBias,
                    vtTauN=vtTauN,
                    vfVReset=vfVReset,
                    vfVRest=vfVRest,
                    vfVThresh=vfVThresh,
                    vfCapacity=vfCapacity,
                    tRefractoryTime=tRef,
                    nNumCores=1,
                    bRecord=True,
                    strName="FF")

    fl1 = RecIAFSpkInNest(mfWIn=mfWIn,
                          mfWRec=mfWRec,
                          tDt=tDt,
                          vfBias=vfBias,
                          vtTauN=vtTauN,
                          vtTauS=vtTauS,
                          vfVThresh=vfVThresh,
                          vfVReset=vfVReset,
                          vfVRest=vfVRest,
                          vfCapacity=vfCapacity,
                          tRefractoryTime=tRef,
                          nNumCores=1,
                          bRecord=True,
                          strName="Rec")

    net0 = nw.Network(fl0, fl1)

    np.random.seed(0)

    fl2 = FFIAFNest(mfW=mfW,
                    tDt=tDt,
                    vfBias=vfBias,
                    vtTauN=vtTauN,
                    vfVReset=vfVReset,
                    vfVRest=vfVRest,
                    vfVThresh=vfVThresh,
                    vfCapacity=vfCapacity,
                    tRefractoryTime=tRef,
                    bRecord=True,
                    nNumCores=4,
                    strName="FF")

    fl3 = RecIAFSpkInNest(mfWIn=mfWIn,
                          mfWRec=mfWRec,
                          tDt=tDt,
                          vfBias=vfBias,
                          vtTauN=vtTauN,
                          vtTauS=vtTauS,
                          vfVThresh=vfVThresh,
                          vfVReset=vfVReset,
                          vfVRest=vfVRest,
                          vfCapacity=vfCapacity,
                          tRefractoryTime=tRef,
                          bRecord=True,
                          nNumCores=4,
                          strName="Rec")

    net1 = nw.Network(fl2, fl3)

    # - Input signal
    vTime = np.arange(0, 1, tDt)
    vVal = np.zeros([len(vTime), 1])
    vVal[2000:7000] = 0.01

    tsInCont = ts.TSContinuous(vTime, vVal)

    epsilon = 0.0000001

    assert (np.abs(fl0.vState - fl2.vState) < epsilon).all()
    assert (np.abs(fl1.vState - fl3.vState) < epsilon).all()

    # - Compare states before and after
    tStart0 = time.time()
    np.random.seed(0)
    dAct0 = net0.evolve(tsInCont, tDuration=10.0)
    tStop0 = time.time()

    tStart1 = time.time()
    np.random.seed(0)
    dAct1 = net1.evolve(tsInCont, tDuration=10.0)
    tStop1 = time.time()

    assert (tStop1 - tStart1 < tStop0 - tStart0)  # multithreading is faster
    assert (np.abs(fl0.vState - fl2.vState) < epsilon).all()
    assert (np.abs(fl1.vState - fl3.vState) < epsilon).all()


def test_delays():
    """ test delays """
    from NetworksPython import timeseries as ts
    from NetworksPython.layers import FFIAFNest, RecIAFSpkInNest
    from NetworksPython.networks import network as nw
    import numpy as np
    import pylab as plt

    mfW = [[0.001, 0., 0., 0.]]
    vfBias = 0.375
    vtTauN = 0.01
    vReset = -0.07
    vRest = -0.07
    vTh = -.055
    fC = 250.
    tDt = 0.001
    tRef = 0.002

    fl0 = FFIAFNest(mfW=mfW,
                    tDt=tDt,
                    vfBias=vfBias,
                    vtTauN=vtTauN,
                    vfVReset=vReset,
                    vfVRest=vRest,
                    vfVThresh=vTh,
                    vfCapacity=fC,
                    tRefractoryTime=tRef,
                    bRecord=True,
                    strName="FF")

    mfWIn = [[0.015, 0.015, 0.0, 0.], [0., 0., 0.0, 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]
    mfWRec = [[0.0, 0.0, 0.015, 0.015], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.], [0., 0., 0., 0.]]
    mfDelayIn = [[0.001, 0.011, 0.001, 0.001], [0.001, 0.001, 0.001, 0.001], [0.001, 0.001, 0.001, 0.001]]
    mfDelayRec = [[0.001, 0.001, 0.001, 0.011], [0.001, 0.001, 0.001, 0.001], [0.001, 0.001, 0.001, 0.001]]
    vfBiasRec = 0.0
    vtTauNRec = [0.2, 0.2, 0.2, 0.2]
    vtTauSRec = [0.2, 0.2, 0.2, 0.2]

    # - Layer generation
    fl1 = RecIAFSpkInNest(
        mfWIn=mfWIn,
        mfWRec=mfWRec,
        mfDelayIn=mfDelayIn,
        mfDelayRec=mfDelayRec,
        tDt=0.001,
        vfBias=vfBiasRec,
        vtTauN=vtTauNRec,
        vtTauS=vtTauSRec,
        tRefractoryTime=0.001,
        bRecord=True,
        strName="Rec")

    net = nw.Network(fl0, fl1)

    # - Input signal
    vTime = np.arange(0, 1, tDt)
    vVal = np.zeros([len(vTime), 1])
    vVal[500] = 0.01

    tsInCont = ts.TSContinuous(vTime, vVal)
    dAct = net.evolve(tsInCont, tDuration=1.0)

    times = dAct['Rec'].times

    eps = 0.000001

    assert (times[1] - times[0] - 0.01 < eps)
    assert (times[3] - times[2] - 0.01 < eps)




