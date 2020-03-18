# - Test DynapseControl class and RecDynapSE layer
from warnings import warn
import os.path
import numpy as np
from rockpool import TSEvent

RUN_TEST = False

try:
    from rockpool.devices import DynapseControlExtd
    from rockpool.devices import dynapse_control as dc
except ImportError:
    warn("DynapseControl could not be imported. Maybe RPyC is not available.")
else:
    # - Generate DynapseControl instance and connect
    try:
        con = DynapseControlExtd(init_chips=[0], fpga_isibase=1e-5)
    except ConnectionRefusedError:
        warn("Could not connect to cortexcontrol. Not available?")
    else:
        #  - Load biases, silence hot neurons
        con.load_biases(
            os.path.join(
                os.path.abspath(os.path.dirname(__file__)), "files", "dynapse_biases.py"
            )
        )
        con.silence_hot_neurons(range(128), 2)

        RUN_TEST = True


def test_dynapse_control():
    """
    Test basic functions of dynapse_control module
    """

    if not RUN_TEST:
        return

    # - Parameters
    # Neuron population sized
    size_in = 2
    size_layer = 4

    # Arrangement of neurons on chip
    firstid = 52
    width = 2

    # - IDs of neurons to be used
    neuron_ids = dc.rectangular_neuron_arrangement(firstid, size_layer, width)
    assert neuron_ids == [52, 53, 68, 69]

    # - Weights
    weights_in = np.array([[1, -1, 2, 0], [-1, 0, 0, 2]])
    weights_rec = np.array([[0, 1, -1, 0], [0, 0, 0, 1], [1, 0, 0, -2], [0, 0, 0, 0]])

    # - Allocate neurons
    virtual_neurons: np.ndarray = con.allocate_virtual_neurons(2)
    virtual_ids = [vn.get_neuron_id() for vn in virtual_neurons]
    assert 0 not in virtual_ids
    shadow_neurons, hw_neurons = con.allocate_hw_neurons(neuron_ids)
    real_neuron_ids = [n.get_id() for n in shadow_neurons]
    assert real_neuron_ids == neuron_ids
    # - Test how allocation routine handles other formats
    con.allocate_hw_neurons(2)
    con.allocate_hw_neurons(range(1001, 1004))
    try:
        con.allocate_virtual_neurons([virtual_ids[1]])
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Double allocation of virtual neurons has not been detected"
        )
    try:
        con.allocate_hw_neurons([1001])
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Double allocation of hardware neurons has not been detected"
        )

    # - Set connections in cortexcontrol
    # External input to input layer
    con.set_connections_from_weights(
        weights=weights_in,
        neuron_ids=virtual_ids,
        neuron_ids_post=neuron_ids,
        syn_exc=con.syn_exc_fast,
        syn_inh=con.syn_inh_fast,
        virtual_pre=True,
        apply_diff=False,
    )

    # Recurrent connections
    con.add_connections_from_weights(
        weights=weights_rec,
        neuron_ids=neuron_ids,
        syn_exc=con.syn_exc_slow,
        syn_inh=con.syn_inh_fast,
        apply_diff=False,
    )

    # Add other connections to check whether existing connections are removed
    con.set_connections_from_weights(
        weights=weights_rec,
        neuron_ids=np.array(neuron_ids) + 40,
        syn_exc=con.syn_exc_slow,
        syn_inh=con.syn_inh_fast,
        apply_diff=True,
    )

    # Test connections
    neur = shadow_neurons[-1]
    cams = neur.get_cams()
    pre_neur = [c.get_pre_neuron_id() for c in cams]
    assert pre_neur == [2, 2, 53, 68, 68] + 59 * [0], "Connections not correct"
    types = [c.get_type() for c in cams]
    assert types[:5] == 2 * [con.syn_exc_fast] + [con.syn_exc_slow] + 2 * [
        con.syn_inh_fast
    ], "Connection types not correct"

    # Stimulate
    ts_in = TSEvent(np.arange(100) * 0.01, np.tile([0, 1], 50))
    ts_out = con.send_TSEvent(
        ts_in,
        t_record=1.2,
        virtual_neur_ids=virtual_ids,
        record_neur_ids=neuron_ids,
        record=True,
        return_ts=True,
    )
    times = ts_out.times
    channels = ts_out.channels
    assert len(times) == len(
        channels
    ), "Numbers of output channels and times don't match"
    assert len(times) > 100, "Too few events generated."

    # - Remove connections
    con.remove_all_connections_to([69], apply_diff=True)
    assert [c.get_type() for c in cams[:5]] == 5 * [con.syn_inh_slow]
    assert [c.get_pre_neuron_id() for c in cams[:5]] == 5 * [0]

    # - Reset first four cores
    con.reset_all()


def test_dynapse_layer():
    """
    Test functioning of RecDynapSE layer
    """

    if not RUN_TEST:
        return

    from rockpool.layers import RecDynapSE

    # - Layer generation
    dynap_lyr = RecDynapSE(
        weights_in=np.array([[1, 3, 1], [0, 3, 2]]),
        weights_rec=np.array([[0, 2, 0], [-1, 0, 1], [2, -1, 0]]),
        neuron_ids=[5, 7, 9],
        virtual_neuron_ids=[3, 4],
        dt=2e-5,
        controller=con,
    )

    # - Input
    ts_input = TSEvent(np.arange(3) * 1e-4, [0, 1, 0])

    # - Evolution
    tsOutput = dynap_lyr.evolve(ts_input)
