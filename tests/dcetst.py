import numpy as np

from Rockpool import TSContinuous
from Rockpool.devices import dynapse_control_extd as dce
from Rockpool.layers import FFUpDown
from Rockpool.layers.internal.devices.dynap_hw import RecDynapSEDemo
import ECG.recordings as rec

### --- Test connections
con = dce.DynapseControl(init_chips=[0, 1])
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
assert (camtype == np.array([0, 1, 1, 2])).all()
assert (conn_pre == np.array([3, 1, 1, 2])).all()
assert (conn_post == np.array([3, 2, 258, 3])).all()
assert (
    con.connections_virtual[con.connections_virtual != 0] == np.array([1, 2, 1, 1])
).all()
# - Provoke aliasing
try:
    con.set_connections_from_weights([-1], [1025], [4], syn_inh=con.syn_inh_slow)
except ValueError:
    pass
else:
    raise AssertionError("Should have raised exception because of aliasing.")
# - Ignore aliasing
con.set_connections_from_weights(
    [-1], [1025], [4], syn_inh=con.syn_inh_slow, prevent_aliasing=False
)
camtype, conn_pre, conn_post = np.nonzero(con.connections)
assert (camtype == np.array([1, 1, 1, 2, 3, 3])).all()
assert (conn_pre == np.array([1, 1025, 1025, 1026, 1, 1025])).all()
assert (conn_post == np.array([2, 2, 258, 3, 4, 4])).all()
assert (con.connections[con.connections != 0] == np.array([1, 1, 2, 1, 1, 1])).all()
# Something is still wrong with weights...

# Test removing connections from/to neurons
# Test clearing

### --- Test layer

dt = 2 / 9 * 1e-4
dt_in = 1.0 / 360.0
neuron_ids = (
    list(range(3, 7)) + list(range(19, 23)) + list(range(24, 32)) + list(range(40, 48))
)
virtual_neuron_ids = list(range(1, 3)) + list(range(16, 17))
con = dce.DynapseControlExtd(dt, clearcores_list=None, initialize_chips=[0])
con.load_biases(
    "/home/felix/gitlab/Projects/AnomalyDetection/ECG/ECGDemo/hardware/realecg/networks/C30_0/biases.py"
)
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
    vnLayerNeuronIDs=neuron_ids,
    vnVirtualNeuronIDs=virtual_neuron_ids,
    nMaxTrialsPerBatch=120,
    lnClearCores=None,
    controller=con,
    name="test",
    bSkipWeights=False,
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
    tsAS=spike_input,
    vtRhythmStart=np.array(rhythm_starts) * dt_in,
    tTotalDuration=(data_in.shape[0] + 1) * dt_in,
)
