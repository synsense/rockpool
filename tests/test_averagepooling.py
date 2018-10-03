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


def test_torch_sumpooling():
    """
    Perform sumpooling with torch implementation
    This is pure torch code and has very little to do with library layers
    """
    import torch
    from NetworksPython.layers import TorchSumPooling2dLayer

    lyrSumPool = TorchSumPooling2dLayer(kernel_size=(2, 5))

    # Generate some random input
    tsrIn = (torch.rand((100, 2, 10, 20)) > 0.99).float()

    # Process some input
    tsrOutput = lyrSumPool(tsrIn)

    # Verify output dimensions
    assert tsrOutput.shape == (100, 2, 5, 4)

    # Asset no. of spikes is retained
    assert tsrOutput.sum() == tsrIn.sum()

    # Spiketimes are still the same
    tSpkOut, _, _, _ = np.where(tsrOutput.numpy())
    tSpkIn, _, _, _ = np.where(tsrIn.numpy())

    # Verify that as many
    assert len(tSpkOut) <= len(tSpkIn)
