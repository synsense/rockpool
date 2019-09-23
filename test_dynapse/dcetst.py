"""
This unit test for the DynapseControl, RecDynapSE and RecDynapSEDemo classes
is intended to be run interactively, with an available cortexcontrol instance
on port 1301 and not through the normal unit test pipeline.
Some assertions will fail if chips other than 0 and 1 have previously been
initialized in the current cortexcontrol instance.
"""

from pathlib import Path
from typing import List
from warnings import warn

import numpy as np

from Rockpool import TSContinuous, TSEvent
from Rockpool.devices import dynapse_control_extd as dce
from Rockpool.layers import FFUpDown
from Rockpool.layers.internal.devices.dynap_hw import RecDynapSEDemo, RecDynapSE
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

# - get_connections method
conns = con.get_connections(ids_pre, ids_post, [1, 2])
camtype, conn_pre, conn_post = np.nonzero(conns)
assert (camtype == np.array([0, 0, 1])).all()
assert (conn_pre == np.array([0, 1, 2])).all()
assert (conn_post == np.array([0, 1, 2])).all()
conns_virt = con.get_connections([3], ids_post, syn_types=0, virtual_pre=True)
assert (conns_virt == np.array([[0, 0, 1]])).all()

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

## -- Neuron assignment
# - allocate neurons that are available
con.allocate_hw_neurons(range(1, 1024))
# - Try to request already allocated neurons
try:
    con.allocate_hw_neurons(range(10, 20))
except ValueError:
    pass
else:
    raise AssertionError("Did not notice request of already allocated neurons.")
# - Allocate the next 500 available neurons
con.allocate_hw_neurons(500)
# - Now there should be 523 available neurons (from 1524 to 2047) -> request 524, so that chip 2 gets initialized
con.allocate_hw_neurons(524)
# - Request neurons, some of which are on chip 3
con.allocate_hw_neurons([2050, 4000])
# - Request more neurons than can be made available
try:
    con.allocate_hw_neurons(2044)
except ValueError:
    pass
else:
    raise AssertionError("Did not notice request of too many neurons.")
# - Clear neuron allocation
con.clear_neuron_assignments(range(16))
assert np.sum(con.hwneurons_isavailable) == 409
### --- Dynapse layers

dt = 2 / 9 * 1e-4
dt_in = 1.0 / 360.0
neuron_ids = (
    list(range(3, 7)) + list(range(19, 23)) + list(range(24, 32)) + list(range(40, 48))
)
virtual_neuron_ids = [1, 2, 17, 18]

scriptpath = Path(__file__).parent
con.load_biases(scriptpath / "files" / "biases.py")

con.silence_hot_neurons(range(64), duration=5)

# - Weights
w_in = np.hstack((np.random.randint(3, size=(4, 8)), np.zeros((4, 16))))
w_in[1::2] *= -1
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
    fastmode=True,
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
    num_beats=10,
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
rhythm_sizes = np.diff(np.r_[rhythm_starts, data_in.shape[0]])

# - Stimulate in random order and collect activities
activities: List[TSEvent] = []
rndm_order = np.random.permutation(rhythm_sizes.size)
for idx_rhythm in rndm_order:
    num_ts = rhythm_sizes[idx_rhythm]
    activities.append(reservoir.evolve(idx_rhythm))
# - Make sure durations of recorded beats are correct
durations = [tse.duration * 360 for tse in activities]
activity_sizes = np.round(durations).astype(int)
if np.sum(np.abs(rhythm_sizes[rndm_order] - activity_sizes)) > 0:
    warn(
        "Rhythm times differ slightly from what is expected. This may be normal however."
    )

# - Test normal DynapSE layer
reservoir1 = RecDynapSE(
    weights_in=w_in,
    weights_rec=w_rec,
    neuron_ids=neuron_ids,
    virtual_neuron_ids=virtual_neuron_ids,
    max_num_trials_batch=120,
    clearcores_list=None,
    controller=con,
    name="test_normal",
    skip_weights=True,
    skip_neuron_allocation=True,
)

ts_out = reservoir1.evolve(spike_input)
assert isinstance(ts_out, TSEvent)
assert ts_out.num_channels == 24
assert ts_out.times.size > 100
