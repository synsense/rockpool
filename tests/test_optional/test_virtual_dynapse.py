import numpy as np


def test_change_baseweight():
    from rockpool.devices.dynapse import VirtualDynapse
    from rockpool import TSEvent

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
        record=True,
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
        record=True,
    )

    # - Test get_weights method
    # weights_lyr_ext = vd0.get_weights(input_ids, neuron_ids, external=True)
    # weights_lyr_rec = vd0.get_weights(neuron_ids, neuron_ids, external=False)
    np.testing.assert_allclose(vd0.weights_ext, connections_ext * 0.1)
    np.testing.assert_allclose(vd0.weights_rec, connections_rec * 0.1)

    # - Change baseweight
    vd0.baseweight_e = 0.2
    vd0.baseweight_i[:] = 0.25

    # - Input signal
    ts_input = TSEvent(
        times=[0.02, 0.04, 0.04, 0.06, 0.12], channels=[1, 0, 2, 1, 0], t_stop=0.15
    )

    # - Compare states and time before and after
    output, new_state, rec_state = vd.evolve(ts_input, duration=0.1)
    output0, new_state0, rec_state0 = vd0.evolve(ts_input, duration=0.1)
    np.testing.assert_allclose(new_state["Vmem"], new_state0["Vmem"])


def test_item_assignment():
    from rockpool.devices.dynapse import VirtualDynapse
    from rockpool import TSEvent

    vd = VirtualDynapse(mismatch=False)
    v_thresh_orig = vd.v_thresh.copy()
    np.testing.assert_allclose(v_thresh_orig, vd.v_thresh)
    # - Correct assignment: Core-wise
    # Single core
    vd.v_thresh[0] = 0.02
    assert vd.v_thresh[0] == 0.02
    # assert vd._v_thresh[0] == 0.02
    # np.testing.assert_allclose(vd._simulator.v_thresh[:256], 0.02)
    # All cores
    vd.v_thresh[:] = 0.03
    np.testing.assert_allclose(vd.v_thresh, 0.03)

    # - Assigning different time constants
    tau_mem_correct = vd.tau_mem_2.copy()
    vd.tau_mem_2[:] = 0.001
    np.testing.assert_allclose(vd.tau_mem_2, 0.001)
    vd.tau_mem_2[11:] = 0.0015


def test_evolve():
    from rockpool.devices.dynapse import VirtualDynapse
    from rockpool import TSEvent

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
        record=True,
        delta_t=0,
    )

    # - Input signal
    ts_input = TSEvent(
        times=[0.02, 0.04, 0.04, 0.06, 0.12],
        channels=[i for i in [1, 0, 2, 1, 0]],
        t_stop=0.15,
    )

    # - Evolve with `wrong` input channels
    state_before = np.copy(vd.Vmem)
    vd.evolve(ts_input, duration=0.1)
    assert vd.t == 0.1
    np.testing.assert_allclose(state_before, vd.Vmem)
    vd.reset_all()

    # - Evolve with correctly mapped input channels
    ts_input = ts_input.remap_channels(input_ids)
    vd.evolve(ts_input, duration=0.1)
    assert vd.t == 0.1
    assert (state_before != vd.Vmem).any()
    vd.reset_all()
    assert vd.t == 0
    np.testing.assert_allclose(state_before, vd.Vmem)


def test_multiprocessing():
    from rockpool.devices.dynapse import VirtualDynapse
    from rockpool import TSEvent

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
    ts_input = TSEvent(
        times=[0.02, 0.04, 0.04, 0.06, 0.12], channels=[1, 0, 2, 1, 0], t_stop=0.15
    )

    # - Evolve
    state_before = np.copy(vd_multi.Vmem)
    vd.evolve(ts_input, duration=0.1, ids_in=input_ids)
    vd_multi.evolve(ts_input, duration=0.1, ids_in=input_ids)
    assert vd_multi.t == 0.1
    np.testing.assert_allclose(vd_multi.Vmem, vd.Vmem)
    vd_multi.reset_all()
    assert vd_multi.t == 0
    np.testing.assert_allclose(state_before, vd_multi.Vmem)


def test_adaptation():
    from rockpool.devices.dynapse import VirtualDynapse
    from rockpool import TSEvent

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
    ts_input = TSEvent(
        times=[0.02, 0.04, 0.04, 0.06, 0.12], channels=[1, 0, 2, 1, 0], t_stop=0.15
    )

    # - Evolve
    vd.evolve(ts_input, duration=0.1, ids_in=input_ids)
    vd_adapt.evolve(ts_input, duration=0.1, ids_in=input_ids)
    assert (vd_adapt.Vmem != vd.Vmem).any()


def test_conn_validation():
    from rockpool.devices.dynapse import VirtualDynapse
    from rockpool import TSEvent

    # - Instantiation
    # TODO
    vd = VirtualDynapse()

    ## -- Weight validation

    # - Fan-in
    # Fan-in ok
    weights_ok_fanin = np.zeros((4096, 4096))
    weights_ok_fanin[:30, 0] = 1
    weights_ok_fanin[30:64, 0] = -1
    assert vd._module.validate_connections(weights_ok_fanin) == 0
    # Fan-in too large
    weights_high_fanin = weights_ok_fanin.copy()
    weights_high_fanin[65, 0] = -1
    assert vd._module.validate_connections(weights_high_fanin) == 1
    # Fan-in too large with external inputs
    weights_ext = np.zeros((1024, 4096))
    weights_ext[100, 0] = 1
    assert vd._module.validate_connections(weights_ok_fanin, weights_ext) == 1

    # - Fan-out
    # Fan-out ok
    weights_ok_fanout = np.zeros((4096, 4096))
    weights_ok_fanout[0, [2, 1050, 2100]] = [1, 2, -1]
    assert vd._module.validate_connections(weights_ok_fanout) == 0
    # Fan-out ok
    weights_ok_fanout_I = np.zeros((4096, 4096))
    weights_ok_fanout_I[0, :1024] = 10
    assert vd._module.validate_connections(weights_ok_fanout_I) == 0
    # Fan-out not compatible
    weights_wrong_fanout = weights_ok_fanout.copy()
    weights_wrong_fanout[0, 4000] = 1
    assert vd._module.validate_connections(weights_wrong_fanout) == 2

    # - Connection aliasing
    weights_no_aliasing = np.zeros((4096, 4096))
    weights_no_aliasing[1024, 1] = 1
    # Different presyn. ID and same presyn. chip -> no aliasing
    weights_no_aliasing_I = weights_no_aliasing.copy()
    weights_no_aliasing_I[1025, 4] = 1
    assert vd._module.validate_connections(weights_no_aliasing_I) == 0
    # Different presyn. ID and different presyn. chip -> no aliasing
    weights_no_aliasing_II = weights_no_aliasing.copy()
    weights_no_aliasing_II[2049, 4] = 1
    assert vd._module.validate_connections(weights_no_aliasing_II) == 0
    # Same presyn. ID, different presyn. chip but different postsyn. core -> no aliasing
    weights_no_aliasing_II[2048, 257] = 1
    assert vd._module.validate_connections(weights_no_aliasing_II) == 0
    # Same presyn. ID, same postsyn. core -> aliasing
    weights_aliasing = weights_no_aliasing.copy()
    weights_aliasing[2048, 1] = 1
    assert vd._module.validate_connections(weights_aliasing) == 4
    # Aliasing with input weights
    weights_external = np.zeros((1024, 4096))
    weights_external[0, 1] = 1
    assert vd._module.validate_connections(weights_no_aliasing, weights_external) == 4

    ## -- Same test using neuron_ids and smaller weight matrices
    # - Fan-in
    # Fan-in ok
    neurons_pre = range(66)
    neurons_post = [0]
    weights_ok_fanin = np.zeros((66, 1))
    weights_ok_fanin[:30, 0] = 1
    weights_ok_fanin[30:64, 0] = -1
    assert (
        vd._module.validate_connections(
            weights_ok_fanin, neurons_pre=neurons_pre, neurons_post=neurons_post
        )
        == 0
    )
    # Fan-in too large
    weights_high_fanin = weights_ok_fanin.copy()
    weights_high_fanin[65, 0] = -1
    assert (
        vd._module.validate_connections(
            weights_high_fanin, neurons_pre=neurons_pre, neurons_post=neurons_post
        )
        == 1
    )
    # Fan-in too large with external inputs
    channels_ext = [100]
    weights_ext = [[1]]
    assert (
        vd._module.validate_connections(
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
        vd._module.validate_connections(
            weights_ok_fanout, neurons_pre=neurons_pre, neurons_post=neurons_post
        )
        == 0
    )
    # Fan-out ok
    neurons_post = range(3 * 1024)
    weights_ok_fanout_I = np.ones((1, 3 * 1024)) * 10
    assert (
        vd._module.validate_connections(
            weights_ok_fanout_I, neurons_pre=neurons_pre, neurons_post=neurons_post
        )
        == 0
    )
    # Fan-out not compatible
    neurons_post = [2, 1050, 2100, 4000]
    weights_wrong_fanout = weights_ok_fanout
    weights_wrong_fanout[0].append(1)
    assert (
        vd._module.validate_connections(
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
        vd._module.validate_connections(
            weights_no_aliasing_I, neurons_pre=neurons_pre, neurons_post=neurons_post
        )
        == 0
    )
    # Different presyn. ID and different presyn. chip -> no aliasing
    neurons_pre = [1024, 2049]
    neurons_post = [1, 4]
    assert (
        vd._module.validate_connections(
            weights_no_aliasing_I, neurons_pre=neurons_pre, neurons_post=neurons_post
        )
        == 0
    )
    # Same presyn. ID, different presyn. chip but different postsyn. core -> no aliasing
    neurons_pre += [2048]
    neurons_post += [257]
    weights_no_aliasing_II = np.eye(3)
    assert (
        vd._module.validate_connections(
            weights_no_aliasing_II, neurons_pre=neurons_pre, neurons_post=neurons_post
        )
        == 0
    )
    # Same presyn. ID, same postsyn. core -> aliasing
    neurons_pre = [1024, 2048]
    neurons_post = [1, 4]
    weights_aliasing = np.eye(2)
    assert (
        vd._module.validate_connections(
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
        vd._module.validate_connections(
            weights_no_aliasing,
            weights_ext,
            neurons_pre=neurons_pre,
            neurons_post=neurons_post,
            channels_ext=channels_ext,
        )
        == 4
    )
