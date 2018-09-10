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
    cnnWTorch = CNNWeightTorch(
        inShape=(1, 20, 20),
        nKernels=3,
        kernel_size=(1, 1),
        mode="same",
        img_data_format="channels_first",
    )

    # Create a FFIAFTorch layer
    lyrCNNTorch = FFCLIAFTorch(mfW=cnnWTorch, fVThresh=0.5, strName="TorchCNN")

    # Generate time series input
    evInput = TSEvent(None, strName="Input")
    for nId in range(20):
        vSpk = poisson_generator(40.0, t_stop=10)
        evInput.merge(TSEvent(vSpk, nId))

    # Check that a warning is raised on expanding the input
    with pytest.warns(UserWarning, match="Expanding input"):
        # Evolve
        evOut = lyrCNNTorch.evolve(tsInput=evInput, tDuration=100)

    assert evOut.nNumChannels == lyrCNNTorch.nSize


def test_toch_activity_comparison_to_skimage_default_params():
    """
    Test basic layer evolution of this layer
    """
    from NetworksPython import TSEvent
    from NetworksPython.layers import CNNWeight
    from NetworksPython.layers import CNNWeightTorch
    from NetworksPython.layers import FFCLIAFTorch
    from NetworksPython.layers import FFCLIAF

    # Initialize weights
    cnnW = CNNWeight(
        inShape=(1, 20, 20),
        nKernels=3,
        kernel_size=(1, 1),
        mode="same",
        img_data_format="channels_first",
    )

    # Create weights
    cnnWTorch = CNNWeightTorch(
        inShape=(1, 20, 20),
        nKernels=3,
        kernel_size=(1, 1),
        mode="same",
        img_data_format="channels_first",
    )
    cnnWTorch.data = np.copy(cnnW.data)

    # Initialize a CNN layer with CN weights
    lyrCNN = FFCLIAF(mfW=cnnW, vfVThresh=0.5, strName="CNN")
    # Create a FFIAFTorch layer
    lyrCNNTorch = FFCLIAFTorch(mfW=cnnWTorch, fVThresh=0.5, strName="TorchCNN")

    # Generate time series input
    evInput = TSEvent(None, strName="Input")
    for nId in range(20 * 20):
        vSpk = poisson_generator(40.0, t_stop=100)
        evInput.merge(TSEvent(vSpk, nId))

    # Create a copy of the input
    evInputTorch = TSEvent(
        evInput.vtTimeTrace.copy(), evInput.vnChannels.copy(), strName="Input copy"
    )

    # Evolve
    evOut = lyrCNN.evolve(tsInput=evInput, tDuration=100)

    evOutTorch = lyrCNNTorch.evolve(tsInput=evInputTorch, tDuration=100)

    # Check that the outputs are identical
    assert evOut.nNumChannels == evOutTorch.nNumChannels
    assert (evOut.vtTimeTrace == evOutTorch.vtTimeTrace).all()


def test_toch_activity_comparison_to_skimage():
    """
    Test basic layer evolution of this layer
    """
    from NetworksPython import TSEvent
    from NetworksPython.layers import CNNWeight
    from NetworksPython.layers import CNNWeightTorch
    from NetworksPython.layers import FFCLIAFTorch
    from NetworksPython.layers import FFCLIAF

    # Initialize weights
    cnnW = CNNWeight(
        inShape=(1, 20, 20),
        nKernels=3,
        kernel_size=(1, 1),
        mode="same",
        img_data_format="channels_first",
    )

    # Create weights
    cnnWTorch = CNNWeightTorch(
        inShape=(1, 20, 20),
        nKernels=3,
        kernel_size=(1, 1),
        mode="same",
        img_data_format="channels_first",
    )
    cnnWTorch.data = np.copy(cnnW.data)

    # Initialize a CNN layer with CN weights
    lyrCNN = FFCLIAF(mfW=cnnW, vfVThresh=0.5, vfVSubtract=None, strName="CNN")
    # Create a FFIAFTorch layer
    lyrCNNTorch = FFCLIAFTorch(
        mfW=cnnWTorch, fVThresh=0.5, fVSubtract=None, strName="TorchCNN"
    )

    # Generate time series input
    evInput = TSEvent(None, strName="Input")
    for nId in range(20 * 20):
        vSpk = poisson_generator(40.0, t_stop=100)
        evInput.merge(TSEvent(vSpk, nId))

    # Create a copy of the input
    evInputTorch = TSEvent(
        evInput.vtTimeTrace.copy(), evInput.vnChannels.copy(), strName="Input copy"
    )

    # Evolve
    evOut = lyrCNN.evolve(tsInput=evInput, tDuration=100)

    evOutTorch = lyrCNNTorch.evolve(tsInput=evInputTorch, tDuration=100)

    # Check that the outputs are identical
    assert evOut.nNumChannels == evOutTorch.nNumChannels
    assert (evOut.vtTimeTrace == evOutTorch.vtTimeTrace).all()


def test_TorchSpikingConv2dLayer():
    from NetworksPython.layers.internal.torch_cnn_layer import TorchSpikingConv2dLayer
    import torch

    # Create a torch layer
    lyrTorchPure = TorchSpikingConv2dLayer(
        nInChannels=2,
        nOutChannels=4,
        kernel_size=(3, 3),
        strides=(3, 3),
        padding=(0, 0),
        fVThresh=0.5,
        fVSubtract=None,
        fVReset=0.0,
    )

    # Create an input
    tsrInp = torch.from_numpy((np.random.rand(10, 2, 10, 10) > 0.7).astype(int)).float()

    lyrTorchPure(tsrInp)


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
