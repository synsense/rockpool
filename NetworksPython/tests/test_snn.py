'''
Test weigh access and indexing for CNNWeight class
'''
import sys
import pytest
import numpy as np

strNetworkPath = sys.path[0] + "/../.."
sys.path.insert(1, strNetworkPath)


def test_import():
    '''
    Test import of the class
    '''
    from NetworksPython.layers.feedforward import SpikingLayer


def test_cnn_initialization():
    '''
    Test initialization of the layer
    '''
    from NetworksPython.layers.feedforward.evSpikeLayer import SpikingLayer
    from NetworksPython.layers.cnnweights import CNNWeight

    # Initialize weights
    cnnW = CNNWeight(inShape=(20, 20))

    # Initialize a CNN layer with CN weights
    lyrCNN = SpikingLayer(mfW=cnnW, strName='CNN')


def test_cnn_evolve():
    '''
    Test initialization of the layer
    '''
    from NetworksPython import TSEvent
    from NetworksPython.layers.feedforward import SpikingLayer
    from NetworksPython.layers.cnnweights import CNNWeight

    # Initialize weights
    cnnW = CNNWeight(inShape=(20, 20))

    # Initialize a CNN layer with CN weights
    lyrCNN = SpikingLayer(mfW=cnnW, fVth=0.5, strName='CNN')

    # Generate time series input
    evInput = TSEvent(None, strName='Input')
    for nId in range(lyrCNN.nSize):
        vSpk = poisson_generator(40.0, t_stop=100)
        evInput.merge(TSEvent(vSpk, nId))
    # Evolve
    evOut = lyrCNN.evolve(tsInput=evInput, tDuration=100)
    print(evOut.find())


def test_cnn_multilayer():
    '''
    Test initialization of the layer
    '''
    from NetworksPython import TSEvent, Network
    from NetworksPython.layers.feedforward.evSpikeLayer import SpikingLayer
    from NetworksPython.layers.cnnweights import CNNWeight

    # Parameters
    imageShape = (10, 10)

    # Initialize weights
    cnnW1 = CNNWeight(inShape=imageShape, nKernels=2, kernel_size=(3, 3))
    cnnW2 = CNNWeight(inShape=(2, *imageShape), nKernels=2, kernel_size=(3, 3))

    # Initialize a CNN layer with CN weights
    lyrCnn1 = SpikingLayer(mfW=cnnW1, fVth=0.5, strName='CNN1')
    lyrCnn2 = SpikingLayer(mfW=cnnW2, fVth=0.5, strName='CNN2')

    net = Network(*[lyrCnn1, lyrCnn2])

    # Generate time series input
    evInput = TSEvent(None, strName='Input')
    for nId in range(imageShape[0]*imageShape[1]):
        vSpk = poisson_generator(40.0, t_stop=100)
        evInput.merge(TSEvent(vSpk, nId))
    # Evolve
    evOut = net.evolve(tsExternalInput=evInput, tDuration=100)
    print(evOut)

# Place holder
#def test_raise_exception_on_incorrect_shape():
#    '''
#    Test exception on size incompatibility
#    '''
#    from NetworksPython.layers.cnnweights import CNNWeight
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
        t_last = spikes[-1] + np.random.exponential(1.0 /
                                                    rate, 1)[0] * 1000.0
        while (t_last < t_stop):
            extra_spikes.append(t_last)
            t_last += np.random.exponential(1.0 / rate, 1)[0] * 1000.0

        spikes = np.concatenate((spikes, extra_spikes))
    else:
        spikes = np.resize(spikes, (i,))

    return spikes
