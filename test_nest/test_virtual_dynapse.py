import numpy as np

from NetworksPython.layers import VirtualDynapse
from NetworksPython import TSEvent


def test_change_baseweight():
    np.random.seed(1)
    neuron_ids = [3, 14, 130, 2050, 2222, 2223, 2800, 3200]
    input_ids = np.arange(3) + 5
    ids_row, ids_col = np.meshgrid(neuron_ids, neuron_ids, indexing="ij")
    connections_ext = np.random.randint(2, size=(3, 8))
    connections_rec = 2 * np.random.randint(2, size=(8, 8))
    connections_rec_full = np.zeros((3300, 3250))
    connections_rec_full[ids_row, ids_col] = connections_rec
    connections_ext_full = np.zeros((100, 3300))
    connections_ext_full[
        np.repeat(input_ids[:, None], axis=1, repeats=8), ids_col[:3]
    ] = connections_ext
    bias = 0  # np.random.rand(16) * 1
    v_thresh = np.random.rand(16) * 0.02 + 0.01
    refractory = np.random.rand(16) * 0.01
    tau_mem_1 = np.random.rand(16) * 0.05
    tau_mem_2 = np.random.rand(16) * 0.05
    has_tau_mem_2 = np.random.randint(2, size=4096).astype(bool)
    tau_syn_exc = np.random.rand(16) * 0.1
    tau_syn_inh = np.random.rand(16) * 0.1

    # - Layer generation
    vd = VirtualDynapse(
        connections_ext=connections_ext_full,
        connections_rec=connections_rec_full,
        bias=bias,
        tau_mem_1=tau_mem_1,
        tau_mem_2=tau_mem_2,
        has_tau_mem_2=has_tau_mem_2,
        v_thresh=v_thresh,
        refractory=refractory,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        dt=0.01,
        baseweight_e=0.2,
        baseweight_i=0.25,
        mismatch=False,
        record=False,
    )

    # - Layer generation
    vd0 = VirtualDynapse(
        connections_ext=connections_ext_full,
        connections_rec=connections_rec_full,
        bias=bias,
        tau_mem_1=tau_mem_1,
        tau_mem_2=tau_mem_2,
        has_tau_mem_2=has_tau_mem_2,
        v_thresh=v_thresh,
        refractory=refractory,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        dt=0.01,
        baseweight_e=0.1,
        baseweight_i=0.1,
        mismatch=False,
        record=False,
    )

    # - Test get_weights method
    weights_lyr_ext = vd0.get_weights(input_ids, neuron_ids, external=True)
    weights_lyr_rec = vd0.get_weights(neuron_ids, neuron_ids, external=False)
    assert (weights_lyr_ext == connections_ext * 0.1).all()
    assert (weights_lyr_rec == connections_rec * 0.1).all()

    # - Change baseweight
    vd0.baseweight_e = 0.2
    vd0.baseweight_i[:] = 0.25

    # - Input signal
    ts_input = TSEvent(times=[0.02, 0.04, 0.04, 0.06, 0.12], channels=[1, 0, 2, 1, 0])

    # - Compare states and time before and after
    vd.evolve(ts_input, duration=0.1)
    vd0.evolve(ts_input, duration=0.1)
    assert (vd0.state == vd.state).all()


def test_item_assignment():
    vd = VirtualDynapse(mismatch=False)
    v_thresh_orig = vd.v_thresh.copy()
    v_thresh_orig_ = vd.v_thresh_.copy()
    # - Try wrong assignment of neuron parameters
    try:
        vd.v_thresh_[0] = 0.0
    except AttributeError:
        pass
    else:
        raise AssertionError("Did not detect wrong assignment of neuron parameter")
    assert (v_thresh_orig_ == vd.v_thresh_).all()
    assert (v_thresh_orig == vd.v_thresh).all()
    # - Correct assignment: Core-wise
    # Single core
    vd.v_thresh[0] = 0.02
    assert vd.v_thresh[0] == 0.02
    assert vd._v_thresh[0] == 0.02
    assert (vd.v_thresh_[:256] == 0.02).all()
    assert (vd._simulator.v_thresh[:256] == 0.02).all()
    # All cores
    vd.v_thresh = 0.03
    assert (vd._v_thresh == 0.03).all()
    assert (vd.v_thresh == 0.03).all()
    assert (vd.v_thresh_ == 0.03).all()
    assert (vd._simulator.v_thresh == 0.03).all()

    # - Assigning different time constants
    tau_mem_correct = vd.tau_mem_.copy()
    vd.tau_mem_2 = 0.001
    vd.has_tau_mem_2[2500:3000] = True
    tau_mem_correct[2500:3000] = 0.001
    assert (vd.tau_mem_ == tau_mem_correct).all()
    vd.tau_mem_2[11:] = 0.0015
    tau_mem_correct[11 * 256 : 3000] = 0.0015
    assert (vd.tau_mem_ == tau_mem_correct).all()

    # - Make sure parameters match those of `_simulator`
    param_names = [
        "bias",
        "tau_syn_exc",
        "tau_syn_inh",
        "v_thresh",
        "refractory",
        "tau_adapt",
        "spike_adapt",
        "delta_t",
    ]
    for param in param_names:
        assert (getattr(vd, param + "_") == getattr(vd._simulator, param)).all()
    assert (vd.weights_rec == vd._simulator.weights_rec).all()
    assert (vd.weights_ext == vd._simulator.weights_in).all()


def test_evolve():
    np.random.seed(1)
    neuron_ids = [3, 14, 130, 2050, 2222, 2223, 2800, 3200]
    input_ids = np.arange(3) + 5
    ids_row, ids_col = np.meshgrid(neuron_ids, neuron_ids, indexing="ij")
    connections_ext = np.random.randint(2, size=(3, 8))
    connections_rec = 2 * np.random.randint(2, size=(8, 8))
    connections_rec_full = np.zeros((3300, 3250))
    connections_rec_full[ids_row, ids_col] = connections_rec
    connections_ext_full = np.zeros((100, 3300))
    connections_ext_full[
        np.repeat(input_ids[:, None], axis=1, repeats=8), ids_col[:3]
    ] = connections_ext
    bias = 0  # np.random.rand(16) * 1
    v_thresh = np.random.rand(16) * 0.02 + 0.01
    refractory = np.random.rand(16) * 0.01
    tau_mem_1 = np.random.rand(16) * 0.05
    tau_mem_2 = np.random.rand(16) * 0.05
    has_tau_mem_2 = np.random.randint(2, size=4096).astype(bool)
    tau_syn_exc = np.random.rand(16) * 0.1
    tau_syn_inh = np.random.rand(16) * 0.1
    baseweight = 0.01

    # - Layer generation
    vd = VirtualDynapse(
        connections_ext=connections_ext_full,
        connections_rec=connections_rec_full,
        bias=bias,
        tau_mem_1=tau_mem_1,
        tau_mem_2=tau_mem_2,
        has_tau_mem_2=has_tau_mem_2,
        v_thresh=v_thresh,
        refractory=refractory,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        baseweight_e=baseweight,
        baseweight_i=baseweight,
        dt=0.001,
        record=False,
        delta_t=0,
    )

    # - Input signal
    ts_input = TSEvent(times=[0.02, 0.04, 0.04, 0.06, 0.12], channels=[1, 0, 2, 1, 0])

    # - Evolve with `wrong` input channels
    state_before = np.copy(vd.state)
    vd.evolve(ts_input, duration=0.1)
    assert vd.t == 0.1
    assert (state_before == vd.state).all()
    vd.reset_all()

    # - Evolve with correctly mapped input channels
    vd.evolve(ts_input, duration=0.1, ids_in=input_ids)
    assert vd.t == 0.1
    assert (state_before != vd.state).any()
    vd.reset_all()
    assert vd.t == 0
    assert (state_before == vd.state).all()


def test_multiprocessing():
    np.random.seed(1)
    neuron_ids = [3, 14, 130, 2050, 2222, 2223, 2800, 3200]
    input_ids = np.arange(3) + 5
    ids_row, ids_col = np.meshgrid(neuron_ids, neuron_ids, indexing="ij")
    connections_ext = np.random.randint(2, size=(3, 8))
    connections_rec = 2 * np.random.randint(2, size=(8, 8))
    connections_rec_full = np.zeros((3300, 3250))
    connections_rec_full[ids_row, ids_col] = connections_rec
    connections_ext_full = np.zeros((100, 3300))
    connections_ext_full[
        np.repeat(input_ids[:, None], axis=1, repeats=8), ids_col[:3]
    ] = connections_ext
    bias = np.random.rand(16) * 0.001
    v_thresh = np.random.rand(16) * 0.02 + 0.01
    refractory = np.random.rand(16) * 0.01
    tau_mem_1 = np.random.rand(16) * 0.05
    tau_mem_2 = np.random.rand(16) * 0.05
    has_tau_mem_2 = np.random.randint(2, size=4096).astype(bool)
    tau_syn_exc = np.random.rand(16) * 0.1
    tau_syn_inh = np.random.rand(16) * 0.1
    baseweight = 0.01

    # - Layer generation
    vd = VirtualDynapse(
        connections_ext=connections_ext_full,
        connections_rec=connections_rec_full,
        bias=bias,
        tau_mem_1=tau_mem_1,
        tau_mem_2=tau_mem_2,
        has_tau_mem_2=has_tau_mem_2,
        v_thresh=v_thresh,
        refractory=refractory,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        baseweight_e=baseweight,
        baseweight_i=baseweight,
        dt=0.001,
        record=False,
        num_threads=1,
        delta_t=0.002,
        mismatch=False,
    )

    # - Layer generation
    vd_multi = VirtualDynapse(
        connections_ext=connections_ext_full,
        connections_rec=connections_rec_full,
        bias=bias,
        tau_mem_1=tau_mem_1,
        tau_mem_2=tau_mem_2,
        has_tau_mem_2=has_tau_mem_2,
        v_thresh=v_thresh,
        refractory=refractory,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        baseweight_e=baseweight,
        baseweight_i=baseweight,
        dt=0.001,
        record=False,
        num_threads=4,
        delta_t=0.002,
        mismatch=False,
    )

    # - Input signal
    ts_input = TSEvent(times=[0.02, 0.04, 0.04, 0.06, 0.12], channels=[1, 0, 2, 1, 0])

    # - Evolve
    state_before = np.copy(vd_multi.state)
    vd.evolve(ts_input, duration=0.1, ids_in=input_ids)
    vd_multi.evolve(ts_input, duration=0.1, ids_in=input_ids)
    assert vd_multi.t == 0.1
    assert (vd_multi.state == vd.state).all()
    vd_multi.reset_all()
    assert vd_multi.t == 0
    assert (state_before == vd_multi.state).all()


def test_adaptation():
    np.random.seed(1)
    neuron_ids = [3, 14, 130, 2050, 2222, 2223, 2800, 3200]
    input_ids = np.arange(3) + 5
    ids_row, ids_col = np.meshgrid(neuron_ids, neuron_ids, indexing="ij")
    connections_ext = np.random.randint(2, size=(3, 8))
    connections_rec = 2 * np.random.randint(2, size=(8, 8))
    connections_rec_full = np.zeros((3300, 3250))
    connections_rec_full[ids_row, ids_col] = connections_rec
    connections_ext_full = np.zeros((100, 3300))
    connections_ext_full[
        np.repeat(input_ids[:, None], axis=1, repeats=8), ids_col[:3]
    ] = connections_ext
    bias = np.random.rand(16) * 0.001
    v_thresh = np.random.rand(16) * 0.02 + 0.01
    refractory = np.random.rand(16) * 0.01
    tau_mem_1 = np.random.rand(16) * 0.05
    tau_mem_2 = np.random.rand(16) * 0.05
    has_tau_mem_2 = np.random.randint(2, size=4096).astype(bool)
    tau_syn_exc = np.random.rand(16) * 0.1
    tau_syn_inh = np.random.rand(16) * 0.1
    baseweight = 0.01

    # - Layer generation
    vd = VirtualDynapse(
        connections_ext=connections_ext_full,
        connections_rec=connections_rec_full,
        bias=bias,
        tau_mem_1=tau_mem_1,
        tau_mem_2=tau_mem_2,
        has_tau_mem_2=has_tau_mem_2,
        v_thresh=v_thresh,
        refractory=refractory,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        baseweight_e=baseweight,
        baseweight_i=baseweight,
        dt=0.001,
        record=False,
        tau_adapt=0.1,
        spike_adapt=0.0,
        delta_t=0.002,
    )

    # - Layer generation
    vd_adapt = VirtualDynapse(
        connections_ext=connections_ext_full,
        connections_rec=connections_rec_full,
        bias=bias,
        tau_mem_1=tau_mem_1,
        tau_mem_2=tau_mem_2,
        has_tau_mem_2=has_tau_mem_2,
        v_thresh=v_thresh,
        refractory=refractory,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        baseweight_e=baseweight,
        baseweight_i=baseweight,
        dt=0.001,
        record=False,
        tau_adapt=0.1,
        spike_adapt=0.005,
        delta_t=0.002,
    )

    # - Input signal
    ts_input = TSEvent(times=[0.02, 0.04, 0.04, 0.06, 0.12], channels=[1, 0, 2, 1, 0])

    # - Evolve
    vd.evolve(ts_input, duration=0.1, ids_in=input_ids)
    vd_adapt.evolve(ts_input, duration=0.1, ids_in=input_ids)
    assert (vd_adapt.state != vd.state).any()


def test_conn_validation():
    # - Instantiation
    # TODO
    vd = VirtualDynapse()

    ## -- Weight validation

    # - Fan-in
    # Fan-in ok
    weights_ok_fanin = np.zeros((4096, 4096))
    weights_ok_fanin[:30, 0] = 1
    weights_ok_fanin[30:64, 0] = -1
    assert vd.validate_connections(weights_ok_fanin) == 0
    # Fan-in too large
    weights_high_fanin = weights_ok_fanin.copy()
    weights_high_fanin[65, 0] = -1
    assert vd.validate_connections(weights_high_fanin) == 1
    # Fan-in too large with external inputs
    weights_ext = np.zeros((1024, 4096))
    weights_ext[100, 0] = 1
    assert vd.validate_connections(weights_ok_fanin, weights_ext) == 1

    # - Fan-out
    # Fan-out ok
    weights_ok_fanout = np.zeros((4096, 4096))
    weights_ok_fanout[0, [2, 1050, 2100]] = [1, 2, -1]
    assert vd.validate_connections(weights_ok_fanout) == 0
    # Fan-out ok
    weights_ok_fanout_I = np.zeros((4096, 4096))
    weights_ok_fanout_I[0, :1024] = 10
    assert vd.validate_connections(weights_ok_fanout_I) == 0
    # Fan-out not compatible
    weights_wrong_fanout = weights_ok_fanout.copy()
    weights_wrong_fanout[0, 4000] = 1
    assert vd.validate_connections(weights_wrong_fanout) == 2

    # - Connection aliasing
    weights_no_aliasing = np.zeros((4096, 4096))
    weights_no_aliasing[1024, 1] = 1
    # Different presyn. ID and same presyn. chip -> no aliasing
    weights_no_aliasing_I = weights_no_aliasing.copy()
    weights_no_aliasing_I[1025, 4] = 1
    assert vd.validate_connections(weights_no_aliasing_I) == 0
    # Different presyn. ID and different presyn. chip -> no aliasing
    weights_no_aliasing_II = weights_no_aliasing.copy()
    weights_no_aliasing_II[2049, 4] = 1
    assert vd.validate_connections(weights_no_aliasing_II) == 0
    # Same presyn. ID, different presyn. chip but different postsyn. core -> no aliasing
    weights_no_aliasing_II[2048, 257] = 1
    assert vd.validate_connections(weights_no_aliasing_II) == 0
    # Same presyn. ID, same postsyn. core -> aliasing
    weights_aliasing = weights_no_aliasing.copy()
    weights_aliasing[2048, 1] = 1
    assert vd.validate_connections(weights_aliasing) == 4
    # Aliasing with input weights
    weights_external = np.zeros((1024, 4096))
    weights_external[0, 1] = 1
    assert vd.validate_connections(weights_no_aliasing, weights_external) == 4

    ## -- Same test using neuron_ids and smaller weight matrices
    # - Fan-in
    # Fan-in ok
    neurons_pre = range(66)
    neurons_post = [0]
    weights_ok_fanin = np.zeros((66, 1))
    weights_ok_fanin[:30, 0] = 1
    weights_ok_fanin[30:64, 0] = -1
    assert (
        vd.validate_connections(
            weights_ok_fanin, neurons_pre=neurons_pre, neurons_post=neurons_post
        )
        == 0
    )
    # Fan-in too large
    weights_high_fanin = weights_ok_fanin.copy()
    weights_high_fanin[65, 0] = -1
    assert (
        vd.validate_connections(
            weights_high_fanin, neurons_pre=neurons_pre, neurons_post=neurons_post
        )
        == 1
    )
    # Fan-in too large with external inputs
    channels_ext = [100]
    weights_ext = [[1]]
    assert (
        vd.validate_connections(
            weights_ok_fanin,
            weights_ext,
            neurons_pre=neurons_pre,
            neurons_post=neurons_post,
            channels_ext=channels_ext,
        )
        == 1
    )

    # - Fan-out
    # Fan-out ok
    neurons_pre = [0]
    neurons_post = [2, 1050, 2100]
    weights_ok_fanout = [[1, 2, -1]]
    assert (
        vd.validate_connections(
            weights_ok_fanout, neurons_pre=neurons_pre, neurons_post=neurons_post
        )
        == 0
    )
    # Fan-out ok
    neurons_post = range(3 * 1024)
    weights_ok_fanout_I = np.ones((1, 3 * 1024)) * 10
    assert (
        vd.validate_connections(
            weights_ok_fanout_I, neurons_pre=neurons_pre, neurons_post=neurons_post
        )
        == 0
    )
    # Fan-out not compatible
    neurons_post = [2, 1050, 2100, 4000]
    weights_wrong_fanout = weights_ok_fanout
    weights_wrong_fanout[0].append(1)
    assert (
        vd.validate_connections(
            weights_wrong_fanout, neurons_pre=neurons_pre, neurons_post=neurons_post
        )
        == 2
    )

    # - Connection aliasing
    neurons_pre = [1024, 1025]
    neurons_post = [1, 4]
    # Different presyn. ID and same presyn. chip -> no aliasing
    weights_no_aliasing_I = np.eye(2)
    assert (
        vd.validate_connections(
            weights_no_aliasing_I, neurons_pre=neurons_pre, neurons_post=neurons_post
        )
        == 0
    )
    # Different presyn. ID and different presyn. chip -> no aliasing
    neurons_pre = [1024, 2049]
    neurons_post = [1, 4]
    assert (
        vd.validate_connections(
            weights_no_aliasing_I, neurons_pre=neurons_pre, neurons_post=neurons_post
        )
        == 0
    )
    # Same presyn. ID, different presyn. chip but different postsyn. core -> no aliasing
    neurons_pre += [2048]
    neurons_post += [257]
    weights_no_aliasing_II = np.eye(3)
    assert (
        vd.validate_connections(
            weights_no_aliasing_II, neurons_pre=neurons_pre, neurons_post=neurons_post
        )
        == 0
    )
    # Same presyn. ID, same postsyn. core -> aliasing
    neurons_pre = [1024, 2048]
    neurons_post = [1, 4]
    weights_aliasing = np.eye(2)
    assert (
        vd.validate_connections(
            weights_aliasing, neurons_pre=neurons_pre, neurons_post=neurons_post
        )
        == 4
    )
    # Aliasing with input weights
    neurons_pre = [1024]
    neurons_post = [1]
    channels_ext = [0]
    weights_ext = [[1]]
    weights_no_aliasing = [[1]]
    assert (
        vd.validate_connections(
            weights_no_aliasing,
            weights_ext,
            neurons_pre=neurons_pre,
            neurons_post=neurons_post,
            channels_ext=channels_ext,
        )
        == 4
    )


def test_saveload():
    np.random.seed(1)
    neuron_ids = [3, 14, 130, 2050, 2222, 2223, 2800, 3200]
    input_ids = np.arange(3) + 5
    ids_row, ids_col = np.meshgrid(neuron_ids, neuron_ids, indexing="ij")
    connections_ext = np.random.randint(2, size=(3, 8))
    connections_rec = 2 * np.random.randint(2, size=(8, 8))
    connections_ext_full = np.zeros((100, 3300))
    connections_ext_full[
        np.repeat(input_ids[:, None], axis=1, repeats=8), ids_col[:3]
    ] = connections_ext
    bias = np.random.rand(16) * 0.001
    v_thresh = np.random.rand(16) * 0.02 + 0.01
    refractory = np.random.rand(16) * 0.01
    tau_mem_1 = np.random.rand(16) * 0.05
    tau_mem_2 = np.random.rand(16) * 0.05
    has_tau_mem_2 = np.random.randint(2, size=4096).astype(bool)
    tau_syn_exc = np.random.rand(16) * 0.1
    tau_syn_inh = np.random.rand(16) * 0.1
    baseweight = 0.01

    # - Layer generation
    vd = VirtualDynapse(
        connections_ext=connections_ext_full,
        bias=bias,
        tau_mem_1=tau_mem_1,
        tau_mem_2=tau_mem_2,
        has_tau_mem_2=has_tau_mem_2,
        v_thresh=v_thresh,
        refractory=refractory,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        baseweight_e=baseweight,
        baseweight_i=baseweight,
        dt=0.001,
        tau_adapt=0.1,
        spike_adapt=0.0,
        delta_t=0.002,
        record=True,
    )
    vd.set_connections(
        connections=connections_rec,
        ids_pre=neuron_ids,
        ids_post=neuron_ids,
        external=False,
    )
    vd.refractory = 0.015

    vd.save_layer("temp")
    vd_new = VirtualDynapse.load_from_file("temp")

    param_names = [
        "connections_rec",
        "connections_ext",
        "bias",
        "tau_mem_1",
        "tau_mem_2",
        "has_tau_mem_2",
        "tau_syn_exc",
        "tau_syn_inh",
        "baseweight_e",
        "baseweight_i",
        "v_thresh",
        "refractory",
        "tau_adapt",
        "spike_adapt",
        "delta_t",
        "weights_rec",
        "weights_ext",
        "bias_",
        "tau_mem_",
        "tau_syn_exc_",
        "tau_syn_inh_",
        "v_thresh_",
        "refractory_",
        "tau_adapt_",
        "spike_adapt_",
        "delta_t_",
    ]
    for param in param_names:
        assert (getattr(vd, param) == getattr(vd_new, param)).all()
    assert vd_new.record
