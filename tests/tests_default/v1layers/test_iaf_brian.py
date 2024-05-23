"""
Test Brian-based spiking layers from layers.internal.iaf_brian and layers.recurrent.iaf_brian
"""

import pytest

pytest.skip("DEPRECATED Brian2 layers are deprecated, no testing performed")


def test_ffiaf():
    import pytest

    pytest.importorskip("brian2")

    """Test FFIAFBrian"""
    from brian2 import second
    from rockpool import timeseries as ts
    from rockpool.nn.layers.iaf_brian import FFIAFBrian
    import numpy as np

    # - Generic parameters
    weights = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])
    bias = 0.01
    tau_mem = [0.02, 0.05, 0.1]

    # - Layer generation
    fl0 = FFIAFBrian(
        weights=weights,
        dt=0.01,
        bias=bias,
        tau_mem=tau_mem,
        noise_std=0.1,
        refractory=0.001 * second,
    )

    # - Input signal
    tsInCont = ts.TSContinuous(times=np.arange(15) * 0.01, samples=np.ones((15, 2)))

    # - Compare states and time before and after
    state_before = np.copy(fl0.state)
    out_ts, new_state, rs = fl0.evolve(tsInCont, duration=0.1)
    assert fl0.t == 0.1
    assert (state_before != new_state).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (state_before == fl0.state).all()


def test_ffiaf_spkin():
    """Test FFIAFSpkInBrian"""
    import pytest

    pytest.importorskip("brian2")

    from rockpool import timeseries as ts
    from rockpool.nn.layers.iaf_brian import FFIAFSpkInBrian
    import numpy as np

    # - Generic parameters
    weights = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])
    bias = 0.01
    tau_mem = [0.02, 0.05, 0.1]

    # - Layer generation
    fl1 = FFIAFSpkInBrian(
        weights=weights,
        bias=bias,
        tau_mem=tau_mem,
        dt=0.01,
        noise_std=0.1,
        refractory=0.001,
    )

    # - Input signal
    tsInEvt = ts.TSEvent(
        times=[0.02, 0.04, 0.04, 0.06, 0.12], channels=[1, 0, 1, 1, 0], t_stop=0.13
    )

    # - Compare states and time before and after
    state_before = np.copy(fl1.state)
    out_ts, new_state, rs = fl1.evolve(tsInEvt, duration=0.1)
    assert fl1.t == 0.1
    assert (state_before != new_state).any()

    fl1.reset_all()
    assert fl1.t == 0
    assert (state_before == fl1.state).all()


def test_reciaf():
    """Test RecIAFBrian"""
    import pytest

    pytest.importorskip("brian2")

    from brian2 import second
    from rockpool import timeseries as ts
    from rockpool.nn.layers.iaf_brian import RecIAFBrian
    import numpy as np

    # - Generic parameters
    np.random.seed(1)
    weights = 2 * np.random.rand(3, 3) - 1
    bias = 2 * np.random.rand(3) - 1
    tau_mem, tau_syn_r = np.clip(np.random.rand(2, 3), 0.01, None)

    # - Layer generation
    rl0 = RecIAFBrian(
        weights=weights,
        bias=bias,
        tau_mem=tau_mem,
        tau_syn_r=tau_syn_r,
        dt=0.01,
        noise_std=0.1,
        refractory=0.001 * second,
    )

    # - Input signal
    tsInCont = ts.TSContinuous(times=np.arange(15) * 0.01, samples=np.ones((15, 3)))

    # - Compare states and time before and after
    state_before = np.copy(rl0.state)
    out_ts, new_state, rs = rl0.evolve(tsInCont, duration=0.1)
    assert rl0.t == 0.1
    assert (state_before != new_state).any()

    rl0.reset_all()
    assert rl0.t == 0
    assert (state_before == rl0.state).all()


def test_reciaf_spkin():
    """Test RecIAFSpkInBrian"""
    import pytest

    pytest.importorskip("brian2")

    from rockpool import timeseries as ts
    from rockpool.nn.layers.iaf_brian import RecIAFSpkInBrian
    import numpy as np

    # - Negative weights, so that layer doesn't spike and gets reset
    np.random.seed(1)
    weights_in = np.random.rand(2, 3) - 1
    weights_rec = 2 * np.random.rand(3, 3) - 1
    bias = 2 * np.random.rand(3) - 1
    tau_mem, tau_syn_inp, tau_syn_rec = np.clip(np.random.rand(3, 3), 0.01, None)

    # - Layer generation
    rl1 = RecIAFSpkInBrian(
        weights_in=weights_in,
        weights_rec=weights_rec,
        bias=bias,
        tau_mem=tau_mem,
        tau_syn_inp=tau_syn_inp,
        tau_syn_rec=tau_syn_rec,
        dt=0.01,
        noise_std=0.1,
        refractory=0.001,
    )

    # - Input signal
    tsInEvt = ts.TSEvent(
        times=[0.02, 0.04, 0.04, 0.06, 0.12], channels=[1, 0, 1, 1, 0], t_stop=0.13
    )

    # - Compare states and time before and after
    state_before = np.copy(rl1.state)
    out_ts, new_state, rs = rl1.evolve(tsInEvt, duration=0.1)
    assert rl1.t == 0.1
    assert (state_before != new_state).any()

    rl1.reset_all()
    assert rl1.t == 0
    assert (state_before == rl1.state).all()
