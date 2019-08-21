from pathlib import Path

import numpy as np

from NetworksPython import TSContinuous
from NetworksPython.devices import dynapse_control_extd as dce
from NetworksPython.layers import FFUpDown
from NetworksPython.layers.internal.devices.dynap_hw import RecDynapSEDemo
import ECG.recordings as rec

### --- Test connections
con = dce.DynapseControlExtd(
    init_chips=[0, 1], clearcores_list=range(8), rpyc_connection="1301"
)
# - Set connections
ids_pre = [1, 1025, 1026]
ids_post = [2, 258, 3]
connections = np.eye(3)
connections[2, 2] = -1
connections[0, 0] = 2
con.set_connections_from_weights(
    connections, ids_pre, ids_post, con.syn_exc_slow, con.syn_inh_fast
)
camtype, conn_pre, conn_post = np.nonzero(con.connections)
assert (camtype == np.array([1, 1, 2])).all()
assert (conn_pre == np.array([1, 1025, 1026])).all()
assert (conn_post == np.array([2, 258, 3])).all()
assert (con.connections[con.connections != 0] == np.array([2, 1, 1])).all()
# - Virtual connection
con.add_connections_from_weights([1], [3], [3], con.syn_exc_fast, virtual_pre=True)
camtype, conn_pre, conn_post = np.nonzero(con.connections)
assert (camtype == np.array([1, 1, 2])).all()
assert (conn_pre == np.array([1, 1025, 1026])).all()
assert (conn_post == np.array([2, 258, 3])).all()
assert (con.connections[con.connections != 0] == np.array([2, 1, 1])).all()
# connections_virtual
camtype, conn_pre, conn_post = np.nonzero(con.connections_virtual)
# by default, cams are set to preneur 0, slow_inh
camtype_tgt = np.array([0, 1, 1, 2] + 2048 * [3])
conn_pre_tgt = np.array([3, 1, 1, 2] + 2048 * [0])
conn_post_tgt = np.array([3, 2, 258, 3] + list(range(2048)))
weight_tgt = np.array([1, 2, 1, 1] + 2048 * [64])
weight_tgt[[6, 7, 262]] = [62, 62, 63]
assert (camtype == camtype_tgt).all()
assert (conn_pre == conn_pre_tgt).all()
assert (conn_post == conn_post_tgt).all()
assert (con.connections_virtual[con.connections_virtual != 0] == weight_tgt).all()

# - Provoke aliasing with virtual weights
try:
    con.add_connections_from_weights(
        [-1], [1], [3], syn_inh=con.syn_inh_slow, virtual_pre=True
    )
except ValueError:
    pass
else:
    raise AssertionError(
        "Should have raised exception because of aliasing with virtual weights."
    )

# - Provoke aliasing
try:
    con.add_connections_from_weights([-1], [1025], [4], syn_inh=con.syn_inh_slow)
except ValueError:
    pass
else:
    raise AssertionError("Should have raised exception because of aliasing.")
# - Ignore aliasing
con.add_connections_from_weights(
    [-1], [1025], [4], syn_inh=con.syn_inh_slow, prevent_aliasing=False
)
camtype, conn_pre, conn_post = np.nonzero(con.connections)
assert (camtype == np.array([1, 1, 1, 2, 3, 3])).all()
assert (conn_pre == np.array([1, 1025, 1025, 1026, 1, 1025])).all()
assert (conn_post == np.array([2, 2, 258, 3, 4, 4])).all()
assert (con.connections[con.connections != 0] == np.array([2, 2, 1, 1, 1, 1])).all()

# - Setting weights instead of adding
con.set_connections_from_weights([2], [5], [4], con.syn_exc_fast)
camtype, conn_pre, conn_post = np.nonzero(con.connections)
assert (camtype == np.array([0, 1, 1, 1, 2])).all()
assert (conn_pre == np.array([5, 1, 1025, 1025, 1026])).all()
assert (conn_post == np.array([4, 2, 2, 258, 3])).all()
assert (con.connections[con.connections != 0] == np.array([2, 2, 2, 1, 1])).all()

# Remove connections to neurons
con.remove_all_connections_to([2])
camtype, conn_pre, conn_post = np.nonzero(con.connections)
assert (camtype == np.array([0, 1, 2])).all()
assert (conn_pre == np.array([5, 1025, 1026])).all()
assert (conn_post == np.array([4, 258, 3])).all()
assert (con.connections[con.connections != 0] == np.array([2, 1, 1])).all()

# Remove connections from neurons
con.remove_all_connections_from([1026])
camtype, conn_pre, conn_post = np.nonzero(con.connections)
assert (camtype == np.array([0, 1])).all()
assert (conn_pre == np.array([5, 1025])).all()
assert (conn_post == np.array([4, 258])).all()
assert (con.connections[con.connections != 0] == np.array([2, 1])).all()

# Test clearing
con.clear_connections([0])
camtype, conn_pre, conn_post = np.nonzero(con.connections)
assert (camtype == np.array([1])).all()
assert (conn_pre == np.array([1025])).all()
assert (conn_post == np.array([258])).all()
assert (con.connections[con.connections != 0] == np.array([1])).all()
con.clear_connections(range(1, 8))
assert np.sum(con.connections) == 0
camtype, conn_pre, conn_post = np.nonzero(con.connections_virtual)
assert (camtype == np.array(2048 * [3])).all()
assert (conn_pre == np.array(2048 * [0])).all()
assert (conn_post == np.arange(2048)).all()
assert (
    con.connections_virtual[con.connections_virtual != 0] == np.array([64] * 2048)
).all()

### --- Test layer

dt = 2 / 9 * 1e-4
dt_in = 1.0 / 360.0
neuron_ids = (
    list(range(3, 7)) + list(range(19, 23)) + list(range(24, 32)) + list(range(40, 48))
)
virtual_neuron_ids = list(range(1, 3))

scriptpath = Path(__file__).parent
con.load_biases(scriptpath / "files" / "biases.py")

con.silence_hot_neurons(range(64), duration=5)

# - Weights
w_in = np.hstack((np.ones((2, 8)), np.zeros((2, 16))))
w_ir = np.random.choice(3, p=(0.5, 0.4, 0.1), size=(8, 16))
w_rr = np.random.choice(3, p=(0.5, 0.4, 0.1), size=(16, 16))
w_irr = np.vstack((w_ir, w_rr))
w_rec = np.hstack((np.zeros((24, 8)), w_irr))

reservoir = RecDynapSEDemo(
    weights_in=w_in,
    weights_rec=w_rec,
    neuron_ids=neuron_ids,
    virtual_neuron_ids=virtual_neuron_ids,
    max_num_trials_batch=120,
    clearcores_list=None,
    controller=con,
    name="test",
    skip_weights=False,
)

# - AS layer
aslayer = FFUpDown(
    (2, 1),
    repeat_output=1,
    thr_up=0.1,
    thr_down=0.1,
    dt=1.0 / 720.0,
    tau_decay=np.array((None, None)),
    name="updown",
    multiplex_spikes=True,
)

# - Load data
classprobs = {0: 0.8, 1: 0.05, 2: 0, 3: 0.05, 4: 0.05, 5: 0, 18: 0.05}
annotations, recordings = rec.load_from_file(rec.save_path)
data_in, data_tgt, anno_curr = rec.generate_data(
    num_beats=1000,
    annotations=annotations,
    rec_data=recordings,
    use_recordings=None,
    probs=classprobs,
    include_bad_signal=False,
    min_len_segment=2,
    use_cont_segments=True,
)
rhythm_starts = anno_curr.idx_new_start

# - Convert to spikes
times = np.arange(data_in.shape[0]) * dt_in
spike_input = aslayer.evolve(TSContinuous(times, data_in))

# - Load to chip
reservoir.load_events(
    ts_as=spike_input,
    vt_rhythm_start=np.array(rhythm_starts) * dt_in,
    dur_total=(data_in.shape[0] + 1) * dt_in,
)
