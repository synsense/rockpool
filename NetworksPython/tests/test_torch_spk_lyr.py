"""
Test weigh access and indexing for CNNWeight class
"""
import sys
import pytest
import numpy as np

strNetworkPath = sys.path[0] + "/../.."
sys.path.insert(1, strNetworkPath)


def test_torch_lyr_evolve():
    """
    Test basic layer evolution of this layer
    """
    from NetworksPython.layers import CNNWeightTorch
    from NetworksPython.layers import FFCLIAFTorch
