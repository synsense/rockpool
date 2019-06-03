"""
Test rate-based Euler reservoir in net_rate_reservoir.py
"""

import numpy as np
import pytest

# - Test imports
def test_imports():
    from NetworksPython.networks import BuildRateReservoir


# - Test building a reservoir
def test_build():
    from NetworksPython.networks import BuildRateReservoir

    # - Generate weight matrices
    mfWI = np.random.rand(10)
    mfWR = np.random.rand(10, 10)
    mfWO = np.random.rand(10, 5)

    # - Build the network
    netRes = BuildRateReservoir(mfWI, mfWR, mfWO)
    return netRes


def test_evolve():
    from NetworksPython import TSContinuous

    # - Get a network
    netRes = test_build()

    # - Generate an input
    vtTimeTrace = np.linspace(0, 10, 100)
    ts_input = TSContinuous(vtTimeTrace, np.random.rand(100))

    # - Evolve the network
    dResp = netRes.evolve(ts_input)
