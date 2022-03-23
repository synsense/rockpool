import pytest


def test_imports():
    from rockpool.nn.modules import LIF


def test_lif():
    from rockpool.nn.modules import LIF

    import numpy as np

    batches = 2
    N = 10
    T = 20
    lyr = LIF(N)

    # - Test parameters
    p = lyr.parameters()
    s = lyr.state()
    sp = lyr.simulation_parameters()

    print("evolve func")
    _, new_state, _ = lyr.evolve(np.random.rand(T, N))

    print("evolving with call")
    _, new_state, _ = lyr(np.random.rand(T, N))
    _, new_state, _ = lyr(np.random.rand(batches, T, N))

    ## - Test recurrent mode
    lyr = LIF(N, has_rec=True)

    print("evolving recurrent")
    o, ns, r_d = lyr(np.random.rand(T, N))
    o, ns, r_d = lyr(np.random.rand(batches, T, N))
