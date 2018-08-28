"""
Test pooling layer
"""
import sys
import numpy as np

strNetworkPath = sys.path[0] + "/../.."
sys.path.insert(1, strNetworkPath)


def test_import():
    """
    Test import of the class
    """
    from NetworksPython.layers.feedforward.averagepooling import AveragePooling


def test_averagepooling():
    """
    Test import of the class
    """
    from NetworksPython.layers.feedforward.averagepooling import AveragePooling
    from NetworksPython.timeseries import TSEvent

    lyrAverage = AveragePooling(inShape=(10, 10), pool_size=(2, 5))
    # since the output image dimension should be (5,2)
    assert lyrAverage.nSize == 10

    # Process some input
    tseInput = TSEvent([0, 1, 2, 3], [1, 2, 19, 20], nNumChannels=(10 * 10))

    tseOutput = lyrAverage.evolve(tseInput, tDuration=100)

    # Spike times are still the same
    assert tseOutput.vtTimeTrace == np.array([0, 1, 2, 3])

    # Neuron indices are updated
    assert tseOutput.vnChannels == np.array([0, 0, 1, 1])
