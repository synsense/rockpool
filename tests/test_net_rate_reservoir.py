"""
Test rate-based Euler reservoir in net_rate_reservoir.py
"""

import numpy as np
import pytest

# - Test imports
def test_imports():
    from rockpool.networks import build_rate_reservoir


# - Test building a reservoir
def test_build():
    from rockpool.networks import build_rate_reservoir

    # - Generate weight matrices
    mfWI = np.random.rand(10)
    mfWR = np.random.rand(10, 10)
    mfWO = np.random.rand(10, 5)

    # - Build the network
    netRes = build_rate_reservoir(mfWI, mfWR, mfWO)
    return netRes


def test_evolve():
    from rockpool import TSContinuous

    # - Get a network
    netRes = test_build()

    # - Generate an input
    time_trace = np.linspace(0, 10, 100)
    ts_input = TSContinuous(time_trace, np.random.rand(100))

    # - Evolve the network
    resp = netRes.evolve(ts_input)
