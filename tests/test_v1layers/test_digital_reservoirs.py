"""
Test recurrent iaf layers with constant leak
"""
import sys
import numpy as np

### --- Test iaf_cl.RecCLIAF


def test_imports():
    from rockpool.nn.layers.iaf_cl import FFCLIAF, RecCLIAF, CLIAF_Base
    from rockpool.nn.layers.softmaxlayer import SoftMaxLayer


def test_cliaf_evolve_subtracting():
    """
    Test initialization and evolution of RecCLIAF layer using subtraction after spikes.
    """
    from rockpool.nn.layers.iaf_cl import RecCLIAF
    from rockpool.timeseries import TSEvent

    # - Input weight matrix
    weights_in = np.array([[12, 0, 0], [0, 0, 10]])
    # - Recurrent weight matrix
    weights_rec = np.array([[0, 3, 0], [0, 0, 0], [0, 0, 0]])

    # - Generate layer
    rl = RecCLIAF(
        weights_in=weights_in,
        weights_rec=weights_rec,
        bias=-0.05,
        v_thresh=5,
        dt=0.1,
        monitor_id=True,
        v_subtract=5,
    )

    # - Input spike
    ts_input = TSEvent(times=[0.55, 0.8], channels=[0, 1], t_stop=1)

    # - Evolution
    tsOutput, new_state, _ = rl.evolve(ts_input, duration=0.8)

    # - Expectation: Input spike will cause neuron 0 to spike 2 times at t=0.55
    #                These spikes will cause neuron 1 to spike once at t=0.65
    #                Last input spike will not have effect because evolution
    #                stops beforehand
    print(tsOutput.times)
    assert np.allclose(
        tsOutput.times, np.array([0.55, 0.55, 0.65])
    ), "Output spike times not as expected"
    assert (
        tsOutput.channels == np.array([0, 0, 1])
    ).all(), "Output spike channels not as expected"

    # - Reset
    rl.reset_all()
    assert rl.t == 0, "Time has not been reset correctly"
    assert (rl.state == 0).all(), "State has not been reset correctly"


def test_cliaf_evolve_resetting():
    """
    Test initialization and evolution of RecCLIAF layer using reset after spikes.
    """
    from rockpool.nn.layers.iaf_cl import RecCLIAF
    from rockpool.timeseries import TSEvent

    # - Input weight matrix
    weights_in = np.array([[12, 0, 0], [0, 0, 10]])

    # - Recurrent weight matrix
    weights_rec = np.array([[0, 3, 0], [0, 0, 0], [0, 0, 0]])

    # - Generate layer
    rl = RecCLIAF(
        weights_in=weights_in,
        weights_rec=weights_rec,
        bias=-0.05,
        v_thresh=5,
        dt=0.1,
        monitor_id=True,
        v_subtract=None,
    )

    # - Input spike
    ts_input = TSEvent(times=[0.55, 0.8], channels=[0, 1], t_stop=1)

    # - Evolution
    tsOutput, new_state, _ = rl.evolve(ts_input, duration=0.7)

    # - Expectation: Input spike will cause neuron 0 to spike once at t=0.55
    #                This spike will not be enough to make other neuron spike.
    #                Last input spike will not have any effect do anything
    #                either because evolution stops beforehand
    assert np.allclose(
        tsOutput.times, np.array([0.55])
    ), "Output spike times not as expected"
    assert (
        tsOutput.channels == np.array([0])
    ).all(), "Output spike channels not as expected"

    # - Reset
    rl.reset_all()
    assert rl.t == 0, "Time has not been reset correctly"
    assert (rl.state == 0).all(), "State has not been reset correctly"


### --- Test iaf_digital.RecDIAF


def test_diaf_evolve_subtracting():
    """
    Test initialization and evolution of RecDIAF layer using subtraction after spikes.
    """
    from rockpool.nn.layers.iaf_digital import RecDIAF
    from rockpool.timeseries import TSEvent

    # - Input weight matrix
    weights_in = np.array([[16, 0, 0], [0, 0, 10]])
    # - Recurrent weight matrix
    weights_rec = np.array([[0, 5, 0], [0, 0, 0], [0, 0, 0]])

    # - Generate layer
    rl = RecDIAF(
        weights_in=weights_in,
        weights_rec=weights_rec,
        v_thresh=5,
        v_reset=0,
        v_subtract=5,
        delay=0.04,
        refractory=0.01,
        tau_leak=0.2,  # - Subtract leak every 0.2 seconds
        leak=1,
    )

    # - Input spike
    ts_input = TSEvent(times=[0.55, 0.8], channels=[0, 1], t_stop=1)

    # - Evolution
    tsOutput, new_state, _ = rl.evolve(ts_input, duration=0.7)

    # - Expectation: Input spike will cause neuron 0 to spike 2 times
    #                (not three times because of the leak that reduced state before),
    #                once at t=0.55, then at t=0.55 + refractory = 0.56.
    #                These spikes will cause neuron 1 to spike once at
    #                t = 0.56 + delay = 0.6.
    #                Last input spike will not have effect because evolution
    #                stops beforehand
    print(tsOutput.times)
    assert np.allclose(
        tsOutput.times, np.array([0.55, 0.56, 0.6])
    ), "Output spike times not as expected"
    assert (
        tsOutput.channels == np.array([0, 0, 1])
    ).all(), "Output spike channels not as expected"

    # - Reset
    rl.reset_all()
    assert rl.t == 0, "Time has not been reset correctly"
    assert (rl.state == 0).all(), "State has not been reset correctly"


def test_diaf_evolve_resetting():
    """
    Test initialization and evolution of RecDIAF layer using reset after spikes.
    """

    from rockpool.nn.layers.iaf_digital import RecDIAF
    from rockpool.timeseries import TSEvent

    # - Input weight matrix
    weights_in = np.array([[16, 0, 0], [0, 0, 10]])
    # - Recurrent weight matrix
    weights_rec = np.array([[0, 5, 0], [0, 0, 0], [0, 0, 0]])

    # - Generate layer
    rl = RecDIAF(
        weights_in=weights_in,
        weights_rec=weights_rec,
        v_thresh=5,
        v_reset=0,
        v_subtract=None,  # Reset instead of subtracting
        delay=0.04,
        refractory=0.01,
        tau_leak=0.2,  # - Subtract leak every 0.2 seconds
        leak=1,
    )

    # - Input spike
    ts_input = TSEvent(times=[0.55, 0.8], channels=[0, 1], t_stop=1)

    # - Evolution
    tsOutput, new_state, _ = rl.evolve(ts_input, duration=0.7)

    # - Expectation: Input spike will cause neuron 0 to spike once at t=0.55
    #                This spikes will not be enough to make other neuron spike.
    #                Last input spike will not have any effect do anything
    #                either because evolution stops beforehand
    assert np.allclose(
        tsOutput.times, np.array([0.55])
    ), "Output spike times not as expected"
    assert (
        tsOutput.channels == np.array([0])
    ).all(), "Output spike channels not as expected"

    # - Reset
    rl.reset_all()
    assert rl.t == 0, "Time has not been reset correctly"
    assert (rl.state == 0).all(), "State has not been reset correctly"


def test_diaf_evolve_vfvrest():
    """
    Test initialization and evolution of RecDIAF layer with resting potential and monitor.
    """

    from rockpool.nn.layers.iaf_digital import RecDIAF
    from rockpool.timeseries import TSEvent

    # - Input weight matrix
    weights_in = np.array([[16, 0, -10], [0, 90, 0]])
    # - Recurrent weight matrix
    weights_rec = np.array([[0, 5, 0], [7, 0, 0], [1, 0, -4]])

    # - Generate layer
    rl = RecDIAF(
        weights_in=weights_in,
        weights_rec=weights_rec,
        v_thresh=50,
        dt=0.001,
        v_reset=0,
        v_subtract=None,  # Reset instead of subtracting
        delay=0.04,
        refractory=0.01,
        tau_leak=0.02,  # - Subtract leak every 0.2 seconds
        leak=1,
        v_rest=0,
        monitor_id=True,
    )

    # - Input spike
    ts_input = TSEvent(times=[0.55], channels=[0], t_stop=0.6)

    # - Evolution
    tsOutput, new_state, _ = rl.evolve(ts_input, duration=0.9)

    # - Expectation: Input spike will cause the potential of neuron 0 to increase
    #                and of neuron 2 to decrease. Due to the leak, both potentials
    #                should have moved back to 0 after 0.32 s.
    assert np.allclose(rl.state, np.zeros(3)), "Final state not as expected"
