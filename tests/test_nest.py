"""
Test Nest-based spiking layers from layers.internal.iaf_nest
"""

import numpy as np


def test_chargeSingleNeuron():

    """
    single neuron test
    charge neuron exactly to threshold without crossing using the bias
    """

    from NetworksPython.layers import FFIAFNest

    mfW = [[1.]]
    vfBias = [0.375]
    vtTauN = [0.01]
    vReset = -70.
    vRest = -70.
    vTh = -55.
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
    vReset = -70.
    vRest = -70.
    vTh = -55.
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
        vtTimeTrace=np.arange(15) * 0.01, mfSamples=np.ones((15, 2))
    )

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.vState)
    dFl0 = fl0.evolve(tsInCont, tDuration=0.1)

    dFl0 = fl0.evolve(tsInCont, tDuration=0.1)

    assert fl0.t == 0.2
    assert (vStateBefore != fl0.vState).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.vState).all()




def test_Multithreading():
    """ Test RecIAFNest"""
    from NetworksPython.layers import RecIAFSpkInNest
    import numpy as np

    # - Generic parameters
    mfWIn = np.random.rand(1, 1024) * 0.01
    mfWRec = np.random.rand(1024, 1024) * 0.01
    vfBias = 0.2
    vtTauN = 0.01
    vtTauS = 0.01

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
        nNumCores=10,
    )

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.vState)
    dFl0 = fl0.evolve(tDuration=10.0)

    assert fl0.t == 10.
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

    # - Compare states and time before and after
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

    mfW = [[0., 1., 0.]]
    vfBias = 0.375
    vtTauN = 0.01
    vReset = -70.
    vRest = -70.
    vTh = -55.
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


    mfWIn = [[0., 0., 0.], [0., 0., 600.], [0., 0., 0.]]
    mfWRec = np.random.rand(3, 3)
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

    # - Compare states and time before and after
    vStateBefore = np.copy(fl1.vState)

    dAct = net.evolve(tsInCont, tDuration=1.0)

    assert fl0.t == 1.0
    assert fl1.t == 1.0

    assert (vStateBefore != fl1.vState).any()

    net.reset_all()
    assert fl0.t == 0
    assert fl1.t == 0
    assert (vStateBefore == fl1.vState).all()


