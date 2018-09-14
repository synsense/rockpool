"""
Test weigh access and indexing for CNNWeightTorch class
"""
import sys
import pytest
import numpy as np

strNetworkPath = sys.path[0] + "/../.."
sys.path.insert(1, strNetworkPath)


def test_import():
    """
    Test import of the class
    """
    from NetworksPython.layers import CNNWeightTorch


def test_raise_exception_on_incorrect_shape():
    """
    Test exception on size incompatibility
    """
    from NetworksPython.layers import CNNWeightTorch

    W = CNNWeightTorch(inShape=(1, 200, 200), img_data_format="channels_first")

    # Create an image
    myImg = np.random.rand(1, 400, 400) > 0.999

    # Test indexing with entire image
    with pytest.raises(ValueError):
        W[myImg]


def test_raise_exception_on_undefined_shape():
    """
    Test exception on size incompatibility
    """
    from NetworksPython.layers import CNNWeightTorch

    W = CNNWeightTorch()

    # Create an image
    myImg = np.random.rand(400, 400) > 0.999
    myImgIndex = myImg.flatten().nonzero()[0]

    # Test indexing with entire image
    with pytest.raises(IndexError):
        W[myImgIndex]


def test_convolution_full_image():
    """
    Test convolution of full image
    """
    from NetworksPython.layers import CNNWeightTorch

    W = CNNWeightTorch(inShape=(1, 400, 400), img_data_format="channels_first")

    # Create an image
    myImg = np.random.rand(1, 400, 400) > 0.999

    # Test indexing with entire image
    outConv = W[myImg]
    assert myImg.size == outConv.size


def test_convolutionl_nonzero_index():
    """
    Test convolution when the indexing is done by non-zero pixels
    """
    from NetworksPython.layers import CNNWeightTorch

    W = CNNWeightTorch(inShape=(400, 400, 1))

    # Create an image
    myImg = np.random.rand(400, 400) > 0.999
    myImgIndex = myImg.flatten().nonzero()[0]

    # Test indexing with entire image
    outConv = W[myImgIndex]
    assert myImg.size == outConv.size


def test_data_format_channels_last():
    """
    Test indexing and output dimensions with channels last data format
    """
    from NetworksPython.layers import CNNWeightTorch

    W = CNNWeightTorch(
        inShape=(400, 400, 1),
        nKernels=3,
        kernel_size=(1, 1),
        mode="same",
        img_data_format="channels_last",
    )

    # Create an image
    myImg = np.zeros((400, 400, 1))
    myImg[0, 5, 0] = 1  # One pixel in image active
    myImgIndex = myImg.flatten().nonzero()[0]

    # Test indexing with entire image
    outConv = W[myImgIndex]
    # Ensure size of output is as expected
    assert myImg.size * 3 == outConv.size
    # Ensure image dimensions are understood and maintained
    assert myImg.shape[:2] == W.outShape[:2]
    # Ensure convolution data is accurate
    outConv = outConv.reshape((400, 400, 3))
    assert outConv[0, 5, 0] != 0
    assert outConv[5, 0, 0] == 0


def test_data_format_channels_first():
    """
    Test indexing and output dimensions with channels last data format
    """
    from NetworksPython.layers import CNNWeightTorch

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
    myImgIndex = myImg.flatten().nonzero()[0]

    # Test indexing with entire image
    outConv = W[myImgIndex]
    # Ensure image dimensions are understood and maintained
    assert myImg.shape[-2:] == W.outShape[-2:]
    # Ensure size of output is as expected
    assert myImg.size * 3 == outConv.size
    # Ensure convolution data is accurate
    outConv = outConv.reshape((3, 400, 400))
    assert outConv[0, 5, 0] != 0
    assert outConv[0, 0, 5] == 0


def test_strides_on_convolution():
    """
    Test the convolution upon a custom stride specified by user
    """
    from NetworksPython.layers import CNNWeightTorch

    W = CNNWeightTorch(
        inShape=(1, 10, 10),
        nKernels=3,
        kernel_size=(2, 2),
        strides=(2, 5),
        mode="valid",
        img_data_format="channels_first",
    )

    assert W.outShape == (3, 5, 2)


def test_strides_on_convolution_channels_last():
    """
    Test the convolution upon a custom stride specified by user
    """
    from NetworksPython.layers import CNNWeightTorch

    W = CNNWeightTorch(
        inShape=(10, 10, 1),
        nKernels=3,
        kernel_size=(2, 2),
        strides=(2, 5),
        mode="valid",
        img_data_format="channels_last",
    )

    assert W.outShape == (5, 2, 3)


def test_compare_skimage_torch_channels_first():
    """
    Compare convolution results for torch vs skimage with channels first format
    """
    from NetworksPython.layers import CNNWeightTorch
    from NetworksPython.layers import CNNWeight

    Wtorch = CNNWeightTorch(
        inShape=(1, 60, 60),
        nKernels=3,
        kernel_size=(1, 1),
        mode="same",
        img_data_format="channels_first",
    )

    Wskimage = CNNWeight(
        inShape=(1, 60, 60),
        nKernels=3,
        kernel_size=(1, 1),
        mode="same",
        img_data_format="channels_first",
    )

    # Copy weights in both
    Wtorch.data = Wskimage.data.copy()

    # Create an image
    myImg = np.random.rand(1, 60, 60) > 0.98
    myImgIndex = myImg.flatten().nonzero()[0]

    # Test indexing with entire image
    outConvTorch = Wtorch[myImgIndex]
    outConvSkimage = Wskimage[myImgIndex]

    # Convert both to the same type
    outConvSkimage = outConvSkimage.astype(np.float32)

    # Compare the output shapes
    assert outConvTorch.shape == outConvSkimage.shape

    # Compare output values
    assert np.all(outConvTorch == outConvSkimage)


def test_compare_skimage_torch_channels_last():
    """
    Compare convolution results for torch vs skimage with channels first format
    """
    from NetworksPython.layers import CNNWeightTorch
    from NetworksPython.layers import CNNWeight

    Wtorch = CNNWeightTorch(
        inShape=(60, 60, 1),
        nKernels=3,
        kernel_size=(1, 1),
        mode="same",
        img_data_format="channels_last",
    )

    Wskimage = CNNWeight(
        inShape=(60, 60, 1),
        nKernels=3,
        kernel_size=(1, 1),
        mode="same",
        img_data_format="channels_last",
    )

    # Copy weights in both
    Wtorch.data = Wskimage.data.copy()

    # Create an image
    myImg = np.random.rand(60, 60, 1) > 0.98
    myImgIndex = myImg.flatten().nonzero()[0]

    # Test indexing with entire image
    outConvTorch = Wtorch[myImgIndex]
    outConvSkimage = Wskimage[myImgIndex]

    # Convert both to the same type
    outConvSkimage = outConvSkimage.astype(np.float32)

    # Compare the output shapes
    assert outConvTorch.shape == outConvSkimage.shape

    # Compare output values
    assert np.all(outConvTorch == outConvSkimage)
