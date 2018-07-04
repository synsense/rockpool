'''
Test weigh access and indexing for CNNWeight class
'''
import sys
import pytest
import numpy as np
sys.path.insert(0, "../")

def test_import():
    '''
    Test import of the class
    '''
    from NetworksPython.layers.feedforward.evCNNLayer import EventCNNLayer

def test_cnn_initialization():
    '''
    Test initialization of the layer
    '''
    from NetworksPython.layers.feedforward.evCNNLayer import EventCNNLayer
    from NetworksPython.layers.cnnweights import CNNWeight

    # Initialize weights
    cnnW = CNNWeight(inShape=(20,20))

    # Initialize a CNN layer with CN weights
    lyrCNN = EventCNNLayer(mfW=cnnW, strName='CNN')

def test_cnn_evolve():
    '''
    Test initialization of the layer
    '''
    from NetworksPython import TSEvent
    from NetworksPython.layers.feedforward.evCNNLayer import EventCNNLayer
    from NetworksPython.layers.cnnweights import CNNWeight

    # Initialize weights
    cnnW = CNNWeight(inShape=(20,20))

    # Initialize a CNN layer with CN weights
    lyrCNN = EventCNNLayer(mfW=cnnW, strName='CNN')

    # Generate time series input
    evInput = TSEvent(None, strName='Input')
    for nId in range(lyrCNN.nSize):
        pass


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
