"""
Test weigh access and indexing for CNNWeight class
"""
import sys
import pytest
import numpy as np


def test_import():
    """
    Test import of the class
    """
    from NetworksPython.layers import FFCLIAF
    from NetworksPython.layers import PassThroughEvents

def test_pt_events():
    """ Test PassThroughEvents layer"""
    from NetworksPython.layers import PassThroughEvents
    from NetworksPython import TSEvent

    # - Input signal
    vtTimeTrace = [0.1, 0.2, 0.7, 0.8, 0.9]
    vnChannels = [1, 2, 0, 1, 1]
    tsInput = TSEvent(vtTimeTrace, vnChannels)

    # - Layer
    lpt = PassThroughEvents(np.array([[0,2],[1,1],[0,0]]), tDt=0.4)

    # - Evolution
    tsOut = lpt.evolve(tsInput)

    assert (tsOut.channels == np.array([0, 1, 1, 1, 0, 1, 0, 1])).all(), (
        "Output channels incorrect"
    )


def test_cnn_initialization():
    """
    Test initialization of the layer
    """
    from NetworksPython.layers import FFCLIAF
    from NetworksPython.weights import CNNWeight

    # Initialize weights
    cnnW = CNNWeight(inShape=(20, 20))

    # Initialize a CNN layer with CN weights
    lyrCNN = FFCLIAF(mfW=cnnW, strName="CNN")


def test_cnn_evolve():
    """
    Test initialization of the layer
    """
    from NetworksPython import TSEvent
    from NetworksPython.layers import FFCLIAF
    from NetworksPython.weights import CNNWeight

    # Initialize weights
    cnnW = CNNWeight(inShape=(20, 20))

    # Initialize a CNN layer with CN weights
    lyrCNN = FFCLIAF(mfW=cnnW, vfVThresh=0.5, strName="CNN")

    # Generate time series input
    evInput = TSEvent(None, name="Input")
    for nId in range(lyrCNN.nSize):
        vSpk = poisson_generator(40.0, t_stop=100)
        evInput.merge(TSEvent(vSpk, nId), inplace=True)
    # Evolve
    evOut = lyrCNN.evolve(tsInput=evInput, tDuration=100)
    print(evOut())


def test_cnn_evolve_empty():
    """
    Test initialization of the layer
    """
    from NetworksPython import TSEvent
    from NetworksPython.layers import FFCLIAF
    from NetworksPython.weights import CNNWeight

    # Initialize weights
    cnnW = CNNWeight(inShape=(20, 20))

    # Initialize a CNN layer with CN weights
    lyrCNN = FFCLIAF(mfW=cnnW, vfVThresh=0.5, strName="CNN")

    # Generate time series input
    evInput = TSEvent(None, name="Input", num_channels=lyrCNN.nSize)
    # Evolve
    evOut = lyrCNN.evolve(tsInput=evInput, tDuration=100)
    print(evOut())



def test_cnn_multilayer():
    """
    Test initialization of the layer
    """
    from NetworksPython import TSEvent, Network
    from NetworksPython.layers import FFCLIAF
    from NetworksPython.weights import CNNWeight

    # Parameters
    imageShape = (10, 10)

    # Initialize weights
    cnnW1 = CNNWeight(inShape=imageShape, nKernels=2, kernel_size=(3, 3))
    cnnW2 = CNNWeight(inShape=(2, *imageShape), nKernels=2, kernel_size=(3, 3))

    # Initialize a CNN layer with CN weights
    lyrCnn1 = FFCLIAF(mfW=cnnW1, vfVThresh=0.5, strName="CNN1")
    lyrCnn2 = FFCLIAF(mfW=cnnW2, vfVThresh=0.5, strName="CNN2")

    net = Network(*[lyrCnn1, lyrCnn2])

    # Generate time series input
    evInput = TSEvent(None, name="Input")
    for nId in range(imageShape[0] * imageShape[1]):
        vSpk = poisson_generator(40.0, t_stop=100)
        evInput.merge(TSEvent(vSpk, nId), inplace=True)
    # Evolve
    evOut = net.evolve(tsInput=evInput, tDuration=100)
    print(evOut)


def test_ffcliaf_none_attributes():
    """
    Make sure an exception is thrown if FFCLIAF is to be assigned None
    as weight, bias or state.
    """
    from NetworksPython.layers import FFCLIAF

    # - Input weight matrix
    mfWIn = np.array([[12, 0, 5], [0, 0, 0.4]])

    # - Generate layer
    lyrFF = FFCLIAF(
        mfW=mfWIn, vfVBias=-0.05, vfVThresh=5, tDt=0.1, vnIdMonitor=True, vfVSubtract=5
    )

    for strVarName in ("mfW", "vfVBias", "vState"):
        try:
            setattr(lyrFF, strVarName, None)
        except AssertionError:
            pass
        else:
            raise AssertionError("Should not accept None as {}".format(strVarName))


def test_ffcliaf_evolve_subtracting():
    """
    Test initialization and evolution of FFCLIAF layer using subtraction after spikes.
    """
    from NetworksPython.layers import FFCLIAF
    from NetworksPython.timeseries import TSEvent

    # - Input weight matrix
    mfWIn = np.array([[12, 0, 5], [0, 0, 0.4]])

    # - Generate layer
    lyrFF = FFCLIAF(
        mfW=mfWIn, vfVBias=-0.05, vfVThresh=5, tDt=0.1, vnIdMonitor=True, vfVSubtract=5
    )

    # - Input spike
    tsInput = TSEvent(times=[0.55, 0.7, 0.8], channels=[0, 1, 1])

    # - Evolution
    tsOutput = lyrFF.evolve(tsInput, tDuration=0.75)

    print(lyrFF._mfStateTimeSeries)

    # - Expectation: First input spike will cause neuron 0 to spike 2 times at
    #                t=0.6 but not neuron 2 because of negative vfVBias.
    #                Second input spike will cause neuron 2 to spike at t=0.7
    #                Last input spike will not have effect because evolution
    #                stops beforehand
    print(tsOutput.times)
    assert np.allclose(
        tsOutput.times, np.array([0.6, 0.6, 0.7])
    ), "Output spike times not as expected"
    assert (
        tsOutput.channels == np.array([0, 0, 2])
    ).all(), "Output spike channels not as expected"

    # - Reset
    lyrFF.reset_all()
    assert lyrFF.t == 0, "Time has not been reset correctly"
    assert (lyrFF.vState == 0).all(), "State has not been reset correctly"


def test_cliaf_evolve_resetting():
    """
    Test initialization and evolution of RecCLIAF layer using reset after spikes.
    """
    from NetworksPython.layers import FFCLIAF
    from NetworksPython.timeseries import TSEvent

    # - Input weight matrix
    mfWIn = np.array([[12, 0, 5], [0, 0, 0.4]])

    # - Generate layer
    lyrFF = FFCLIAF(
        mfW=mfWIn,
        vfVBias=-0.05,
        vfVThresh=5,
        tDt=0.1,
        vnIdMonitor=True,
        vfVSubtract=None,
    )

    # - Input spike
    tsInput = TSEvent(times=[0.55, 0.7, 0.8], channels=[0, 1, 1])

    # - Evolution
    tsOutput = lyrFF.evolve(tsInput, tDuration=0.8)

    print(lyrFF._mfStateTimeSeries)

    # - Expectation: First input spike will cause neuron 0 to spike once at
    #                t=0.6 but not neuron 2 because of negative vfVBias.
    #                Second input spike will cause neuron 2 to spike at t=0.7
    #                Last input spike will not have effect because evolution
    #                stops beforehand
    assert np.allclose(
        tsOutput.times, np.array([0.6, 0.7])
    ), "Output spike times not as expected"
    assert (
        tsOutput.channels == np.array([0, 2])
    ).all(), "Output spike channels not as expected"

    # - Reset
    lyrFF.reset_all()
    assert lyrFF.t == 0, "Time has not been reset correctly"
    assert (lyrFF.vState == 0).all(), "State has not been reset correctly"


# Place holder
# def test_raise_exception_on_incorrect_shape():
#    '''
#    Test exception on size incompatibility
#    '''
#    from NetworksPython.weights import CNNWeight
#
#    W = CNNWeight(inShape=(200,200))
#
#    # Create an image
#    myImg = np.random.rand(400, 400) > 0.999
#
#    # Test indexing with entire image
#    with pytest.raises(ValueError):
#        outConv = W[myImg]
#


# This is a convenience function, not a test function
def poisson_generator(rate, t_start=0.0, t_stop=1000.0, refractory=0):
    """
    Returns a SpikeTrain whose spikes are a realization of a Poisson process
    with the given rate (Hz) and stopping time t_stop (milliseconds).
    Note: t_start is always 0.0, thus all realizations are as if
    they spiked at t=0.0, though this spike is not included in the SpikeList.
    Inputs:
        rate    - the rate of the discharge (in Hz)
        t_start - the beginning of the SpikeTrain (in ms)
        t_stop  - the end of the SpikeTrain (in ms)
    """
    n = (t_stop - t_start) / 1000.0 * rate
    number = int(np.ceil(n + 3 * np.sqrt(n)))
    if number < 100:
        number = min(5 + int(np.ceil(2 * n)), 100)
    if number > 0:
        isi = np.random.exponential(1.0 / rate, number) * 1000.0

        if number > 1:
            spikes = np.add.accumulate(isi)
        else:
            spikes = isi
    else:
        spikes = np.array([])
    spikes += t_start
    i = np.searchsorted(spikes, t_stop)
    extra_spikes = []
    if i == len(spikes):
        # ISI buf overrun
        t_last = spikes[-1] + np.random.exponential(1.0 / rate, 1)[0] * 1000.0
        while t_last < t_stop:
            extra_spikes.append(t_last)
            t_last += np.random.exponential(1.0 / rate, 1)[0] * 1000.0

        spikes = np.concatenate((spikes, extra_spikes))
    else:
        spikes = np.resize(spikes, (i,))

    return spikes
