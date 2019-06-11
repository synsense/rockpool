"""
Test pooling layer
"""
import sys
import numpy as np


def test_import():
    """
    Test import of the class
    """
    from NetworksPython.layers import AveragePooling2D


def test_averagepooling():
    """
    Test averagepooling implementation
    """
    from NetworksPython.layers import AveragePooling2D
    from NetworksPython.timeseries import TSEvent

    lyrAverage = AveragePooling2D(
        inp_shape=(10, 10, 1), pool_size=(2, 5), img_data_format="channels_last"
    )
    # since the output image dimension should be (5,2)
    assert lyrAverage.size == 10

    # Process some input
    tseInput = TSEvent([0, 1, 2, 3], [1, 2, 19, 18], num_channels=(10 * 10))

    tseOutput = lyrAverage.evolve(tseInput, duration=100)

    # Spike times are still the same
    assert np.array_equal(tseOutput.times, np.array([0, 1, 2, 3]))

    # Neuron indices are updated
    assert np.array_equal(tseOutput.channels, np.array([0, 0, 1, 1]))
