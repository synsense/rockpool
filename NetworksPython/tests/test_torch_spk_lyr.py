"""
Test weigh access and indexing for CNNWeight class
"""
import sys
import pytest
import numpy as np

strNetworkPath = sys.path[0] + "/../.."
sys.path.insert(1, strNetworkPath)


def test_torch_lyr_prepare_input_empty():
    """
    Test basic layer evolution of this layer
    """
    from NetworksPython import TSEvent
    from NetworksPython.layers import CNNWeightTorch
    from NetworksPython.layers import FFCLIAFTorch

    # Create weights
    W = CNNWeightTorch(
        inShape=(1, 400, 400),
        nKernels=3,
        kernel_size=(1, 1),
        mode="same",
        img_data_format="channels_first",
    )

    # Create an image
    myImg = np.zeros((1, 400, 400))
    myImg[0, 5, 0] = 1  # One pixel in image active

    # Create an empty TSEvent
    evInput = TSEvent(None, strName="Input")

    # Create a FFIAFTorch layer
    lyrConv2d = FFCLIAFTorch(mfW=W, strName="TorchConv2d")

    lyrConv2d.evolve(evInput, tDuration=10)


def test_torch_lyr_prepare_input_partial():
    """
    Test basic layer evolution of this layer
    """
    from NetworksPython import TSEvent
    from NetworksPython.layers import CNNWeightTorch
    from NetworksPython.layers import FFCLIAFTorch

    # Create weights
    W = CNNWeightTorch(
        inShape=(1, 400, 400),
        nKernels=3,
        kernel_size=(1, 1),
        mode="same",
        img_data_format="channels_first",
    )

    # Create an image
    myImg = np.zeros((1, 400, 400))
    myImg[0, 5, 0] = 1  # One pixel in image active

    # - Input spike on only two neurons
    tsInput = TSEvent(
        vtTimeTrace=[0.55, 0.7, 1.8],
        vnChannels=[0, 1, 1],
        # nNumChannels=400 * 400,
        strName="input",
    )

    # Create a FFIAFTorch layer
    lyrConv2d = FFCLIAFTorch(mfW=W, strName="TorchConv2d")

    lyrConv2d.evolve(tsInput, tDuration=10)
