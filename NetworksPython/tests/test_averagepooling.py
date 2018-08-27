"""
Test pooling layer
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
    from NetworksPython.layers.feedforward.averagepooling import AveragePooling


def test_averagepooling():
    """
    Test import of the class
    """
    from NetworksPython.layers.feedforward.averagepooling import AveragePooling

    lyrAverage = AveragePooling(inShape=(10, 10), pool_size=(2, 5))
    print(lyrAverage.nSize, lyrAverage.mfW.outShape)
    raise Exception
