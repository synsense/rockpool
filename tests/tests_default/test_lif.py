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


def test_lif_batch_state_warns_and_keeps_sample_0():
    """A batched evolve() call simulates each batch index independently,
    but Module's shape-locked state attributes can only ever persist ONE
    sample's final state (batch index 0) -- the rest is discarded. This
    used to happen completely silently. It must now raise a UserWarning,
    and the retained state must genuinely be sample 0's real final value
    (not some other silently-wrong combination)."""
    import numpy as np
    import warnings
    from rockpool.nn.modules import LIF

    N = 4
    T = 20
    lyr = LIF(N)

    sample0 = np.zeros((T, N))
    sample1 = np.ones((T, N)) * 5.0
    batch = np.stack([sample0, sample1], axis=0)

    with pytest.warns(UserWarning, match="only batch index 0"):
        _, new_state, record = lyr.evolve(batch, record=True)

    assert np.allclose(new_state["vmem"], record["vmem"][0, -1, :])

    # batch size 1 must NOT warn -- nothing is actually discarded
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lyr2 = LIF(N)
        lyr2.evolve(np.random.rand(1, T, N))
