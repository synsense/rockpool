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


def test_lif_batch_state_warns_and_keeps_only_sample_0():
    import numpy as np
    import warnings
    from rockpool.nn.modules import LIF

    N = 4
    T = 20
    rng = np.random.default_rng(0)
    sample0 = rng.random((T, N)) * 3.0
    sample1 = rng.random((T, N)) * 3.0 + 2.0

    lyr = LIF(N)

    # Final state of each sample, from separate single-sample calls
    finals = []
    for sample in (sample0, sample1):
        lyr.reset_state()
        lyr.evolve(sample[None])
        finals.append(
            {k: np.array(getattr(lyr, k)) for k in ("vmem", "isyn", "spikes")}
        )

    # The samples must differ, otherwise the test cannot tell them apart
    assert not np.allclose(finals[0]["vmem"], finals[1]["vmem"])
    assert not np.allclose(finals[0]["isyn"], finals[1]["isyn"])

    lyr.reset_state()
    with pytest.warns(UserWarning, match="only batch index 0"):
        lyr.evolve(np.stack([sample0, sample1]))

    for name in ("vmem", "isyn", "spikes"):
        assert np.allclose(getattr(lyr, name), finals[0][name])
    assert not np.allclose(lyr.vmem, finals[1]["vmem"])
    assert not np.allclose(lyr.isyn, finals[1]["isyn"])

    # A batch of one discards nothing, so it must not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        LIF(N).evolve(rng.random((1, T, N)))
