import numpy as np

from NetworksPython.layers import VirtualDynapse
from NetworksPython import TSEvent


def test_virtualdynapse_evolve():
    np.random.seed(1)
    neuron_ids = [3, 14, 130, 2050, 2222, 2223, 2800, 3200]
    ids_row, ids_col = np.meshgrid(neuron_ids, neuron_ids, indexing="ij")
    connections_ext = np.random.randint(2, size=(3, 8))
    connections_rec = 2 * np.random.randint(2, size=(8, 8))
    connections_rec_full = np.zeros((3300, 3250))
    connections_rec_full[ids_row, ids_col] = connections_rec
    connections_ext_full = np.zeros((100, 3300))
    connections_ext_full[
        np.repeat(np.arange(3)[:, None], axis=1, repeats=8), ids_col[:3]
    ] = connections_ext
    bias = np.random.rand(16) * 1
    v_thresh = np.random.rand(16) * 0.05
    refractory = np.random.rand(16) * 0.01
    tau_mem_1 = np.random.rand(16) * 0.1
    tau_mem_2 = np.random.rand(16) * 0.1
    has_tau2 = np.random.randint(2, size=4096).astype(bool)
    tau_syn_exc = np.random.rand(16) * 0.1
    tau_syn_inh = np.random.rand(16) * 0.1

    # - Layer generation
    vd = VirtualDynapse(
        connections_ext=connections_ext_full,
        connections_rec=connections_rec_full,
        bias=bias,
        tau_mem_1=tau_mem_1,
        tau_mem_2=tau_mem_2,
        has_tau2=has_tau2,
        v_thresh=v_thresh,
        refractory=refractory,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        dt=0.01,
    )

    # - Input signal
    ts_input = TSEvent(times=[0.02, 0.04, 0.04, 0.06, 0.12], channels=[1, 0, 2, 1, 0])

    # - Compare states and time before and after
    state_before = np.copy(vd.state)
    vd.evolve(ts_input, duration=0.1)
    assert vd.t == 0.1
    assert (state_before != vd.state).any()

    vd.reset_all()
    assert vd.t == 0
    assert (state_before == vd.state).all()


def test_virtualdynapse_conn_validation():
    # - Instantiation
    # TODO
    vd = VirtualDynapse()

    # - Weight validation
    weights_ok_fanin = np.zeros((4096, 4096))
    weights_ok_fanin[:30, 0] = 1
    weights_ok_fanin[30:64, 0] = -1
    assert vd.validate_connections(weights_ok_fanin) == 0
    weights_high_fanin = weights_ok_fanin.copy()
    weights_high_fanin[65, 0] = -1
    assert vd.validate_connections(weights_high_fanin) == 1
    weights_ok_fanout = np.zeros((4096, 4096))
    weights_ok_fanout[0, [2, 1050, 2100]] = [1, 2, -1]
    assert vd.validate_connections(weights_ok_fanout) == 0
    weights_wrong_fanout = weights_ok_fanout.copy()
    weights_wrong_fanout[0, 4000] = 1
    assert vd.validate_connections(weights_wrong_fanout) == 2
    weights_no_aliasing = np.zeros((4096, 4096))
    weights_no_aliasing[1024, 1] = 1
    weights_no_aliasing_I = weights_no_aliasing.copy()
    weights_no_aliasing_I[1025, 4] = 1
    # Different presyn. ID and same presyn. chip -> no aliasing
    assert vd.validate_connections(weights_no_aliasing_I) == 0
    weights_no_aliasing_II = weights_no_aliasing.copy()
    weights_no_aliasing_II[2049, 4] = 1
    # Different presyn. ID and different presyn. chip -> no aliasing
    assert vd.validate_connections(weights_no_aliasing_II) == 0
    weights_no_aliasing_II[2048, 257] = 1
    # Same presyn. ID, different presyn. chip but different postsyn. core -> no aliasing
    assert vd.validate_connections(weights_no_aliasing_II) == 0
    weights_aliasing = weights_no_aliasing.copy()
    weights_aliasing[2048, 1] = 1  # Same presyn. ID, same postsyn. core -> aliasing
    assert vd.validate_connections(weights_aliasing) == 4
