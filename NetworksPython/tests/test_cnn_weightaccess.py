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
    from NetworksPython.layers.cnnweights import CNNWeight


def test_raise_exception_on_incorrect_shape():
    '''
    Test exception on size incompatibility
    '''
    from NetworksPython.layers.cnnweights import CNNWeight

    W = CNNWeight(inShape=(200,200))

    # Create an image
    myImg = np.random.rand(400, 400) > 0.999

    # Test indexing with entire image
    with pytest.raises(ValueError):
        outConv = W[myImg]


def test_raise_exception_on_undefined_shape():
    '''
    Test exception on size incompatibility
    '''
    from NetworksPython.layers.cnnweights import CNNWeight

    W = CNNWeight()

    # Create an image
    myImg = np.random.rand(400, 400) > 0.999
    myImgIndex = myImg.flatten().nonzero()[0]

    # Test indexing with entire image
    with pytest.raises(IndexError):
        outConv = W[myImgIndex]

def test_convolution_full_image():
    '''
    Test convolution of full image
    '''
    from NetworksPython.layers.cnnweights import CNNWeight

    W = CNNWeight(inShape=(400,400))

    # Create an image
    myImg = np.random.rand(400, 400) > 0.999

    # Test indexing with entire image
    outConv = W[myImg]
    assert myImg.size == outConv.size


def test_convolutionl_nonzero_index():
    '''
    Test convolution when the indexing is done by non-zero pixels
    '''
    from NetworksPython.layers.cnnweights import CNNWeight

    W = CNNWeight(inShape=(400,400))

    # Create an image
    myImg = np.random.rand(400, 400) > 0.999
    myImgIndex = myImg.flatten().nonzero()[0]

    # Test indexing with entire image
    outConv = W[myImgIndex]
    assert myImg.size == outConv.size



