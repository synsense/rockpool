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
    from NetworksPython.layers.internal import AveragePooling2D


def test_averagepooling():
    """
    Test import of the class
    """
    from NetworksPython.layers.internal import AveragePooling2D
    from NetworksPython.timeseries import TSEvent

    lyrAverage = AveragePooling2D(
        inShape=(10, 10, 1), pool_size=(2, 5), img_data_format="channels_last"
    )
    # since the output image dimension should be (5,2)
    assert lyrAverage.nSize == 10

    # Process some input
    tseInput = TSEvent([0, 1, 2, 3], [1, 2, 19, 18], nNumChannels=(10 * 10))

    tseOutput = lyrAverage.evolve(tseInput, tDuration=100)

    # Spike times are still the same
    assert np.array_equal(tseOutput.vtTimeTrace, np.array([0, 1, 2, 3]))

    # Neuron indices are updated
    assert np.array_equal(tseOutput.vnChannels, np.array([0, 0, 1, 1]))
