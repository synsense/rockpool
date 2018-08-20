'''
Test recurrent iaf layers with constant leak
'''
import sys
import pytest
import numpy as np

strNetworkPath = sys.path[0] + "/../.."
sys.path.insert(1, strNetworkPath)

### --- Test iaf_cl.RecCLIAF

def test_cliaf_evolve_subtracting():
    """
    Test initialization and evolution of RecCLIAF layer using subtraction after spikes.
    """
    from NetworksPython.layers.recurrent.iaf_cl import RecCLIAF
    from NetworksPython.timeseries import TSEvent

    # - Input weight matrix
    mfWIn = np.array([[12, 0, 0], [0, 0, 10]])
    # - Recurrent weight matrix
    mfWRec = np.array([[0, 3, 0], [0, 0, 0], [0, 0, 0]])
    
    # - Generate layer
    rl = RecCLIAF(
        mfWIn = mfWIn,
        mfWRec = mfWRec,
        vfVBias = -0.05,
        vfVThresh = 5,
        tDt = 0.1,
        vnIdMonitor = True,
        vfVSubtract = 5,
    )

    # - Input spike
    tsInput = TSEvent(vtTimeTrace = [0.55, 0.8], vnChannels = [0, 1])

    # - Evolution
    tsOutput = rl.evolve(tsInput, tDuration = 0.75)

    # - Expectation: Input spike will cause neuron 0 to spike 2 times at t=0.6
    #                These spikes will cause neuron 1 to spike once at t=0.7
    #                Last input spike will not have effect because evolution
    #                stops beforehand
    print(tsOutput.vtTimeTrace)
    assert np.allclose(tsOutput.vtTimeTrace, np.array([0.6, 0.6, 0.7])), \
    "Output spike times not as expected"
    assert (tsOutput.vnChannels == np.array([0, 0, 1])).all(), \
    "Output spike channels not as expected"

    # - Reset
    rl.reset_all()
    assert rl.t == 0, "Time has not been reset correctly"
    assert (rl.vState == 0).all(), "State has not been reset correctly"

def test_cliaf_evolve_resetting():
    """
    Test initialization and evolution of RecCLIAF layer using reset after spikes.
    """
    from NetworksPython.layers.recurrent.iaf_cl import RecCLIAF
    from NetworksPython.timeseries import TSEvent

    # - Input weight matrix
    mfWIn = np.array([[12, 0, 0], [0, 0, 10]])
    # - Recurrent weight matrix
    mfWRec = np.array([[0, 3, 0], [0, 0, 0], [0, 0, 0]])
    
    # - Generate layer
    rl = RecCLIAF(
        mfWIn = mfWIn,
        mfWRec = mfWRec,
        vfVBias = -0.05,
        vfVThresh = 5,
        tDt = 0.1,
        vnIdMonitor = True,
        vfVSubtract = None,
    )

    # - Input spike
    tsInput = TSEvent(vtTimeTrace = [0.55, 0.8], vnChannels = [0, 1])

    # - Evolution
    tsOutput = rl.evolve(tsInput, tDuration = 0.7)

    # - Expectation: Input spike will cause neuron 0 to spike once at t=0.6
    #                This spike will not be enough to make other neuron spike.
    #                Last input spike will not have any effect do anything 
    #                either because evolution stops beforehand
    assert np.allclose(tsOutput.vtTimeTrace, np.array([0.6])), \
    "Output spike times not as expected"
    assert (tsOutput.vnChannels == np.array([0])).all(), \
    "Output spike channels not as expected"

    # - Reset
    rl.reset_all()
    assert rl.t == 0, "Time has not been reset correctly"
    assert (rl.vState == 0).all(), "State has not been reset correctly"


### --- Test iaf_digital.RecDIAF

def test_diaf_evolve_subtracting():
    """
    Test initialization and evolution of RecDIAF layer using subtraction after spikes.
    """
    from NetworksPython.layers.recurrent.iaf_digital import RecDIAF
    from NetworksPython.timeseries import TSEvent

    # - Input weight matrix
    mfWIn = np.array([[16, 0, 0], [0, 0, 10]])
    # - Recurrent weight matrix
    mfWRec = np.array([[0, 5, 0], [0, 0, 0], [0, 0, 0]])
    
    # - Generate layer
    rl = RecDIAF(
        mfWIn = mfWIn,
        mfWRec = mfWRec,
        vfVThresh = 5,
        vfVReset = 0,
        vfVSubtract = 5,
        tSpikeDelay = 0.04,
        vtRefractoryTime = 0.01,
        tTauLeak = 0.2,   # - Subtract vfCleak every 0.2 seconds
        vfCleak = 1,
    )

    # - Input spike
    tsInput = TSEvent(vtTimeTrace = [0.55, 0.8], vnChannels = [0, 1])

    # - Evolution
    tsOutput = rl.evolve(tsInput, tDuration = 0.7)

    # - Expectation: Input spike will cause neuron 0 to spike 2 times
    #                (not three times because of the leak that reduced state before),
    #                once at t=0.55, then at t=0.55 + vtRefractoryTime = 0.56.
    #                These spikes will cause neuron 1 to spike once at
    #                t = 0.56 + tSpikeDelay = 0.6.
    #                Last input spike will not have effect because evolution
    #                stops beforehand
    print(tsOutput.vtTimeTrace)
    assert np.allclose(tsOutput.vtTimeTrace, np.array([0.55, 0.56, 0.6])), \
    "Output spike times not as expected"
    assert (tsOutput.vnChannels == np.array([0, 0, 1])).all(), \
    "Output spike channels not as expected"

    # - Reset
    rl.reset_all()
    assert rl.t == 0, "Time has not been reset correctly"
    assert (rl.vState == 0).all(), "State has not been reset correctly"


def test_diaf_evolve_resetting():
    """
    Test initialization and evolution of RecDIAF layer using reset after spikes.
    """

    from NetworksPython.layers.recurrent.iaf_digital import RecDIAF
    from NetworksPython.timeseries import TSEvent

    # - Input weight matrix
    mfWIn = np.array([[16, 0, 0], [0, 0, 10]])
    # - Recurrent weight matrix
    mfWRec = np.array([[0, 5, 0], [0, 0, 0], [0, 0, 0]])
    
    # - Generate layer
    rl = RecDIAF(
        mfWIn = mfWIn,
        mfWRec = mfWRec,
        vfVThresh = 5,
        vfVReset = 0,
        vfVSubtract = None,  # Reset instead of subtracting
        tSpikeDelay = 0.04,
        vtRefractoryTime = 0.01,
        tTauLeak = 0.2,   # - Subtract vfCleak every 0.2 seconds
        vfCleak = 1,
    )

    # - Input spike
    tsInput = TSEvent(vtTimeTrace = [0.55, 0.8], vnChannels = [0, 1])

    # - Evolution
    tsOutput = rl.evolve(tsInput, tDuration = 0.7)

    # - Expectation: Input spike will cause neuron 0 to spike once at t=0.55
    #                This spikes will not be enough to make other neuron spike.
    #                Last input spike will not have any effect do anything 
    #                either because evolution stops beforehand
    assert np.allclose(tsOutput.vtTimeTrace, np.array([0.55])), \
    "Output spike times not as expected"
    assert (tsOutput.vnChannels == np.array([0])).all(), \
    "Output spike channels not as expected"

    # - Reset
    rl.reset_all()
    assert rl.t == 0, "Time has not been reset correctly"
    assert (rl.vState == 0).all(), "State has not been reset correctly"