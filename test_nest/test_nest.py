"""
Test Nest-based spiking layers from layers.internal.iaf_nest
"""

import numpy as np

from Rockpool.timeseries import set_global_ts_plotting_backend

set_global_ts_plotting_backend("plt")


def test_chargeSingleNeuron():
    """
    single neuron test
    charge neuron exactly to threshold without crossing using the bias
    """

    from Rockpool.layers import FFIAFNest

    weights = [[1.0]]
    epsilon = 1e-5
    bias = [0.375 - epsilon]
    tau_mem = [0.01]
    vReset = -0.07
    vRest = -0.07
    vTh = -0.055
    fC = 0.25
    dt = 0.001
    refractory = 0.002

    fl0 = FFIAFNest(
        weights=weights,
        dt=dt,
        bias=bias,
        tau_mem=tau_mem,
        v_reset=vReset,
        v_rest=vRest,
        v_thresh=vTh,
        capacity=fC,
        refractory=refractory,
        record=True,
        name="test",
    )

    dFl0 = fl0.evolve(duration=1.0)

    assert fl0.state[0] > vTh - 0.00001
    assert fl0.state[0] <= vTh
    assert dFl0.isempty()

    fl0.terminate()


def test_chargeAndSpikeSingleNeuron():
    """
    single neuron test
    charge neuron exactly to threshold without crossing using the bias
    add small input to make it spike
    """

    from Rockpool import timeseries as ts
    from Rockpool.layers import FFIAFNest

    weights = [[1.0]]
    bias = [0.375]
    tau_mem = [0.01]
    vReset = -0.07
    vRest = -0.07
    vTh = -0.055
    fC = 0.25
    dt = 0.001
    refractory = 0.002

    fl0 = FFIAFNest(
        weights=weights,
        dt=dt,
        bias=bias,
        tau_mem=tau_mem,
        v_reset=vReset,
        v_rest=vRest,
        v_thresh=vTh,
        capacity=fC,
        refractory=refractory,
        record=True,
        name="test",
    )

    # - Input signal
    vTime = np.arange(0, 1, dt)
    vVal = np.zeros([len(vTime), 1])
    vVal[500] = 0.001

    tsInCont = ts.TSContinuous(vTime, vVal)

    dFl0 = fl0.evolve(tsInCont, duration=1.0)

    assert not dFl0.isempty()

    fl0.terminate()


def test_FFNestLayer():
    """ Test FFIAFNest"""
    from Rockpool import timeseries as ts
    from Rockpool.layers import FFIAFNest

    # - Generic parameters
    weights = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])
    bias = 0.01
    tau_mem = [0.02, 0.05, 0.1]

    # - Layer generation
    fl0 = FFIAFNest(
        weights=weights,
        dt=0.001,
        bias=bias,
        tau_mem=tau_mem,
        refractory=0.001,
        record=True,
    )

    # - Input signal
    tsInCont = ts.TSContinuous(times=np.arange(15) * 0.01, samples=np.ones((15, 2)))

    # - Compare states before and after
    vStateBefore = np.copy(fl0.state)
    dFl0 = fl0.evolve(tsInCont, duration=0.1)

    dFl0 = fl0.evolve(tsInCont, duration=0.1)

    assert fl0.t == 0.2
    assert (vStateBefore != fl0.state).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state).all()

    fl0.terminate()


def test_RecNestLayer():
    """ Test RecIAFNest"""
    from Rockpool.layers import RecIAFSpkInNest
    import numpy as np

    # - Generic parameters
    weights_in = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])
    weights_rec = np.random.rand(3, 3) * 0.01
    bias = 0.01
    tau_mem = [0.02, 0.05, 0.1]
    tau_syn_exc = [0.2, 0.01, 0.01]
    tau_syn_inh = tau_syn_exc

    # - Layer generation
    fl0 = RecIAFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=0.001,
        bias=bias,
        tau_mem=tau_mem,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        refractory=0.001,
        record=True,
    )

    # - Compare states before and after
    vStateBefore = np.copy(fl0.state)
    dFl0 = fl0.evolve(duration=0.2)

    assert fl0.t == 0.2
    assert (vStateBefore != fl0.state).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state).all()

    fl0.terminate()


def test_setWeightsIn():
    """ Test weight setting"""
    from Rockpool.layers import FFIAFNest, RecIAFSpkInNest, RecAEIFSpkInNest
    from Rockpool import TSEvent, TSContinuous
    import numpy as np

    # - Generic parameters
    weights_in = np.array([[-0.5, 0.02, 0.4], [0.2, -0.3, -0.15]])
    weights_rec = np.random.rand(3, 3) * 0.01
    bias = 0.01
    tau_mem = [0.02, 0.05, 0.1]
    tau_syn_exc = [0.2, 0.01, 0.01]
    tau_syn_inh = tau_syn_exc

    # - Different input weights for initialization of fl1
    weights_in1 = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])

    # - Input time series
    tsInpCont = TSContinuous(np.arange(15) * 0.01, np.ones(15) * 0.1)
    tsInp = TSEvent([0.1], [0])

    ## -- FFIAFNEst
    # - Layer generation
    fl0 = FFIAFNest(
        weights=weights_in, dt=0.001, bias=bias, tau_mem=tau_mem, refractory=0.001
    )
    fl1 = FFIAFNest(
        weights=weights_in1, dt=0.001, bias=bias, tau_mem=tau_mem, refractory=0.001
    )

    # - Set input weights to same as fl0
    fl1.weights = weights_in
    assert (fl1.weights_ == weights_in).all()

    # - Compare states before and after
    fl0.evolve(tsInpCont, duration=0.12)
    fl1.evolve(tsInpCont, duration=0.12)

    assert (fl0.state == fl1.state).all()

    fl0.terminate()
    fl1.terminate()

    ## -- RecIAFSpkInNest
    # - Layer generation
    fl0 = RecIAFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=0.001,
        bias=bias,
        tau_mem=tau_mem,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        refractory=0.001,
        record=True,
    )
    fl1 = RecIAFSpkInNest(
        weights_in=weights_in1,
        weights_rec=weights_rec,
        dt=0.001,
        bias=bias,
        tau_mem=tau_mem,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        refractory=0.001,
        record=True,
    )

    # - Set input weights to same as fl0
    fl1.weights_in = weights_in
    assert (fl1.weights_in_ == weights_in).all()

    # - Compare states before and after
    fl0.evolve(tsInp, duration=0.12)
    fl1.evolve(tsInp, duration=0.12)

    assert (fl0.state == fl1.state).all()

    fl0.terminate()
    fl1.terminate()

    ## -- RecAEIFSpkInNest
    # - Layer generation
    fl0 = RecAEIFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=0.001,
        bias=bias,
        tau_mem=tau_mem,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        refractory=0.001,
        record=True,
    )
    fl1 = RecAEIFSpkInNest(
        weights_in=weights_in1,
        weights_rec=weights_rec,
        dt=0.001,
        bias=bias,
        tau_mem=tau_mem,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        refractory=0.001,
        record=True,
    )

    # - Set input weights to same as fl0
    fl1.weights_in[0, 0] = weights_in[0, 0]
    assert (fl1.weights_in_ == weights_in).all()

    # - Compare states before and after
    fl0.evolve(tsInp, duration=0.12)
    fl1.evolve(tsInp, duration=0.12)

    assert (fl0.state == fl1.state).all()

    fl0.terminate()
    fl1.terminate()


def test_setWeightsRec():
    """ Test RecIAFNest"""
    from Rockpool.layers import RecIAFSpkInNest
    from Rockpool import TSEvent
    import numpy as np

    # - Generic parameters
    weights_in = np.array([[-0.5, 0.02, 0.4], [0.2, -0.3, -0.15]])
    weights_rec = np.random.rand(3, 3) * 0.01
    bias = 0.01
    tau_mem = [0.02, 0.05, 0.1]
    tau_syn_exc = [0.2, 0.01, 0.01]
    tau_syn_inh = tau_syn_exc

    # - Layer generation
    fl0 = RecIAFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=0.001,
        bias=bias,
        tau_mem=tau_mem,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        refractory=0.001,
        record=True,
    )

    # - Initialize with different recurrent weights
    weights_rec_1 = np.random.rand(3, 3) * 0.01
    weights_rec_1[0, 2] = 0

    # - Layer generation
    fl1 = RecIAFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec_1,
        dt=0.001,
        bias=bias,
        tau_mem=tau_mem,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        refractory=0.001,
        record=True,
    )
    # - Set recurrent weights to same as fl0
    fl1.weights_rec = fl0.weights_rec
    assert (fl1.weights_rec_ == weights_rec).all()

    tsInp = TSEvent([0.1], [0])

    # - Compare states before and after
    fl0.evolve(tsInp, duration=0.12)
    fl1.evolve(tsInp, duration=0.12)

    assert (fl0.state == fl1.state).all()

    fl0.terminate()
    fl1.terminate()


def test_FFToRecLayer():
    """ Test FFToRecNest"""

    from Rockpool import timeseries as ts
    from Rockpool.layers import FFIAFNest, RecIAFSpkInNest
    from Rockpool.networks import network as nw
    import numpy as np

    weights = [[0.0, 0.001, 0.0]]
    bias = 0.375
    tau_mem = 0.01
    vReset = -0.07
    vRest = -0.07
    vTh = -0.055
    fC = 0.25
    dt = 0.001
    refractory = 0.002

    fl0 = FFIAFNest(
        weights=weights,
        dt=dt,
        bias=bias,
        tau_mem=tau_mem,
        v_reset=vReset,
        v_rest=vRest,
        v_thresh=vTh,
        capacity=fC,
        refractory=refractory,
        record=True,
        name="FF",
    )

    weights_in = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.6], [0.0, 0.0, 0.0]]
    weights_rec = np.random.rand(3, 3) * 0.001
    vfBiasRec = 0.0
    vtTauNRec = [0.02, 0.05, 0.1]
    tau_syn_exc_rec = [0.2, 0.01, 0.01]
    tau_syn_inh_rec = [0.2, 0.01, 0.01]

    # - Layer generation
    fl1 = RecIAFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=0.001,
        bias=vfBiasRec,
        tau_mem=vtTauNRec,
        tau_syn_exc=tau_syn_exc_rec,
        tau_syn_inh=tau_syn_inh_rec,
        refractory=0.001,
        record=True,
        name="Rec",
    )

    net = nw.Network(fl0, fl1)

    # - Input signal
    vTime = np.arange(0, 1, dt)
    vVal = np.zeros([len(vTime), 1])
    vVal[500] = 0.01

    tsInCont = ts.TSContinuous(vTime, vVal)

    # - Compare states before and after
    vStateBefore = np.copy(fl1.state)

    dAct = net.evolve(tsInCont, duration=1.0)

    assert fl0.t == 1.0
    assert fl1.t == 1.0

    assert (vStateBefore != fl1.state).any()

    net.reset_all()
    assert fl0.t == 0
    assert fl1.t == 0
    assert (vStateBefore == fl1.state).all()

    fl0.terminate()


def test_randomizeStateRec():
    """ test Randomize State """

    """ Test RecIAFNest"""
    from Rockpool.layers import RecIAFSpkInNest
    import numpy as np

    # - Generic parameters
    weights_in = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])
    weights_rec = np.random.rand(3, 3) * 0.01
    bias = 0.01
    tau_mem = [0.02, 0.05, 0.1]
    tau_syn_exc = [0.2, 0.01, 0.01]
    tau_syn_inh = tau_syn_exc

    # - Layer generation
    fl0 = RecIAFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=0.0001,
        bias=bias,
        tau_mem=tau_mem,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        refractory=0.001,
        record=True,
    )

    # - Compare states before and after
    vStateBefore = np.copy(fl0.state)

    fl0.randomize_state()

    assert (vStateBefore != fl0.state).any()

    fl0.terminate()


def test_randomizeStateFF():
    """ Test FFIAFNest"""
    from Rockpool import timeseries as ts
    from Rockpool.layers import FFIAFNest

    # - Generic parameters
    weights = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])
    bias = 0.01
    tau_mem = [0.02, 0.05, 0.1]

    # - Layer generation
    fl0 = FFIAFNest(
        weights=weights,
        dt=0.001,
        bias=bias,
        tau_mem=tau_mem,
        refractory=0.001,
        record=True,
    )

    # - Compare states before and after
    vStateBefore = np.copy(fl0.state)

    fl0.randomize_state()

    assert (vStateBefore != fl0.state).any()

    fl0.terminate()


def test_recording():
    """ tests if the shape of recording is correct """

    import numpy as np
    from Rockpool import timeseries as ts
    from Rockpool.layers import FFIAFNest

    # - Generic parameters
    weights = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])
    bias = 0.01
    tau_mem = [0.02, 0.05, 0.1]

    # - Layer generation
    fl0 = FFIAFNest(
        weights=weights,
        dt=0.0001,
        bias=bias,
        tau_mem=tau_mem,
        refractory=0.001,
        record=True,
    )

    # - Input signal
    tsInCont = ts.TSContinuous(times=np.arange(15) * 0.01, samples=np.ones((15, 2)))

    # - Compare states before and after
    vStateBefore = np.copy(fl0.state)
    dFl0 = fl0.evolve(tsInCont, duration=0.1)

    assert np.shape(fl0.recorded_states.samples) == (1000, 3)

    fl0.terminate()


def test_FFToRecLayerRepeat():
    """ Test FFToRecNest"""

    from Rockpool import timeseries as ts
    from Rockpool.layers import FFIAFNest, RecIAFSpkInNest
    from Rockpool.networks import network as nw
    import numpy as np

    weights = [[0.0, 0.001, 0.0]]
    bias = 0.375
    tau_mem = 0.01
    vReset = -0.07
    vRest = -0.07
    vTh = -0.055
    fC = 0.25
    dt = 0.001
    refractory = 0.002

    fl0 = FFIAFNest(
        weights=weights,
        dt=dt,
        bias=bias,
        tau_mem=tau_mem,
        v_reset=vReset,
        v_rest=vRest,
        v_thresh=vTh,
        capacity=fC,
        refractory=refractory,
        record=True,
        name="FF",
    )

    weights_in = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.6], [0.0, 0.0, 0.0]]
    weights_rec = np.random.rand(3, 3) * 0.001
    vfBiasRec = 0.0
    vtTauNRec = [0.02, 0.05, 0.1]
    tau_syn_exc_rec = [0.2, 0.01, 0.01]
    tau_syn_inh_rec = tau_syn_exc_rec

    # - Layer generation
    fl1 = RecIAFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=0.001,
        bias=vfBiasRec,
        tau_mem=vtTauNRec,
        tau_syn_exc=tau_syn_exc_rec,
        tau_syn_inh=tau_syn_inh_rec,
        refractory=0.001,
        record=True,
        name="Rec",
    )

    net = nw.Network(fl0, fl1)

    # - Input signal
    vTime = np.arange(0, 1, dt)
    vVal = np.zeros([len(vTime), 1])
    vVal[500] = 0.01

    tsInCont = ts.TSContinuous(vTime, vVal)

    # - Compare states before and after
    vStateBefore = np.copy(fl1.state)

    for _ in range(10):
        dAct = net.evolve(tsInCont, duration=1.0 / 10)

    assert fl0.t == 1.0
    assert fl1.t == 1.0

    assert (vStateBefore != fl1.state).any()

    net.reset_all()
    assert fl0.t == 0
    assert fl1.t == 0
    assert (vStateBefore == fl1.state).all()

    fl0.terminate()


def test_DefaultParams():
    """ Test RecIAFNest"""
    from Rockpool.layers import RecIAFSpkInNest, FFIAFNest
    from Rockpool import timeseries as ts
    from Rockpool.networks import network as nw
    import numpy as np

    # - Generic parameters
    weights = np.ones([1, 2]) * 0.01
    weights_in = np.array([[-0.1, 0.02, 0.4], [0.2, -0.3, -0.15]])
    weights_rec = np.random.rand(3, 3) * 0.01
    bias = 0.0
    dt = 0.0001
    tau_mem = 0.02
    tau_syn_exc = 0.05
    tau_syn_inh = 0.05
    v_thresh = -0.055
    v_reset = -0.065
    v_rest = -0.065
    capacity = tau_mem
    refractory = 0.001

    fl0 = FFIAFNest(
        weights=weights,
        dt=dt,
        bias=bias,
        tau_mem=tau_mem,
        v_reset=v_reset,
        v_rest=v_rest,
        v_thresh=v_thresh,
        capacity=capacity,
        refractory=refractory,
        record=True,
        name="FF",
    )

    fl1 = RecIAFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=dt,
        bias=bias,
        tau_mem=tau_mem,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        v_thresh=v_thresh,
        v_reset=v_reset,
        v_rest=v_rest,
        capacity=capacity,
        refractory=refractory,
        record=True,
        name="Rec",
    )

    net0 = nw.Network(fl0, fl1)

    fl2 = FFIAFNest(weights=weights, record=True, name="FF")

    fl3 = RecIAFSpkInNest(
        weights_in=weights_in, weights_rec=weights_rec, record=True, name="Rec"
    )

    net1 = nw.Network(fl2, fl3)

    # - Input signal
    vTime = np.arange(0, 1, dt)
    vVal = np.zeros([len(vTime), 1])
    vVal[2000:7000] = 0.01

    tsInCont = ts.TSContinuous(vTime, vVal)

    eps = 1e-6

    assert (np.abs(fl0.state - fl2.state) < eps).all()
    assert (np.abs(fl1.state - fl3.state) < eps).all()

    # - Compare states before and after
    dAct0 = net0.evolve(tsInCont, duration=1.0)
    dAct1 = net1.evolve(tsInCont, duration=1.0)

    assert (np.abs(fl0.state - fl2.state) < eps).all()
    assert (np.abs(fl1.state - fl3.state) < eps).all()

    fl0.terminate()
    fl1.terminate()


def test_timeconstants():
    """ test delays """
    from Rockpool import timeseries as ts
    from Rockpool.layers import FFIAFNest, RecIAFSpkInNest
    from Rockpool.networks import network as nw
    import numpy as np

    weights = [[0.001]]
    bias = 0.375
    tau_mem = 0.01
    vReset = -0.07
    vRest = -0.07
    vTh = -0.055
    fC = 0.25
    dt = 0.001
    refractory = 0.002

    fl0 = FFIAFNest(
        weights=weights,
        dt=dt,
        bias=bias,
        tau_mem=tau_mem,
        v_reset=vReset,
        v_rest=vRest,
        v_thresh=vTh,
        capacity=fC,
        refractory=refractory,
        record=True,
        name="FF",
    )

    weights_in = [[0.001, -0.001]]
    weights_rec = [[0, 0], [0, 0]]
    vfBiasRec = 0.0
    vtTauNRec = [0.2, 0.2]
    tau_syn_exc_rec = [0.1, 0.1]
    tau_syn_inh_rec = [0.01, 0.01]

    # - Layer generation
    fl1 = RecIAFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=0.001,
        bias=vfBiasRec,
        tau_mem=vtTauNRec,
        tau_syn_exc=tau_syn_exc_rec,
        tau_syn_inh=tau_syn_inh_rec,
        refractory=0.001,
        v_reset=vReset,
        v_rest=vRest,
        v_thresh=vTh,
        record=True,
        name="Rec",
    )

    net = nw.Network(fl0, fl1)

    # - Input signal
    vTime = np.arange(0, 1, dt)
    vVal = np.zeros([len(vTime), 1])
    vVal[500] = 0.01

    tsInCont = ts.TSContinuous(vTime, vVal)
    dAct = net.evolve(tsInCont, duration=1.0)

    exc_input = np.abs(fl1.recorded_states.samples[:, 0] - vRest)
    inh_input = np.abs(fl1.recorded_states.samples[:, 1] - vRest)

    # excitatory input peak should be later than inhibitory as the synaptic TC is longer
    assert np.argmax(exc_input) > np.argmax(inh_input)

    fl0.terminate()
    fl1.terminate()


def test_delays():
    """ test delays """
    from Rockpool import timeseries as ts
    from Rockpool.layers import FFIAFNest, RecIAFSpkInNest
    from Rockpool.networks import network as nw
    import numpy as np

    weights = [[0.001, 0.0, 0.0, 0.0]]
    bias = 0.375
    tau_mem = 0.01
    vReset = -0.07
    vRest = -0.07
    vTh = -0.055
    fC = 0.25
    dt = 0.001
    refractory = 0.002

    fl0 = FFIAFNest(
        weights=weights,
        dt=dt,
        bias=bias,
        tau_mem=tau_mem,
        v_reset=vReset,
        v_rest=vRest,
        v_thresh=vTh,
        capacity=fC,
        refractory=refractory,
        record=True,
        name="FF",
    )

    weights_in = [
        [0.015, 0.015, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    weights_rec = [
        [0.0, 0.0, 0.015, 0.015],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0014],
    ]
    delay_in = [
        [0.001, 0.011, 0.001, 0.001],
        [0.001, 0.001, 0.001, 0.001],
        [0.001, 0.001, 0.001, 0.001],
        [0.001, 0.001, 0.001, 0.001],
    ]
    delay_rec = [
        [0.001, 0.001, 0.001, 0.011],
        [0.001, 0.001, 0.001, 0.001],
        [0.001, 0.001, 0.001, 0.001],
        [0.001, 0.001, 0.001, 0.001],
    ]
    vfBiasRec = 0.0
    vtTauNRec = [0.2, 0.2, 0.2, 0.2]
    tau_syn_exc_rec = [0.2, 0.2, 0.2, 0.2]
    tau_syn_inh_rec = tau_syn_exc_rec

    # - Layer generation
    fl1 = RecIAFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        delay_in=delay_in,
        delay_rec=delay_rec,
        dt=0.001,
        bias=vfBiasRec,
        tau_mem=vtTauNRec,
        capacity=0.1,
        tau_syn_exc=tau_syn_exc_rec,
        tau_syn_inh=tau_syn_inh_rec,
        refractory=0.001,
        record=True,
        name="Rec",
    )

    net = nw.Network(fl0, fl1)

    # - Input signal
    vTime = np.arange(0, 1, dt)
    vVal = np.zeros([len(vTime), 1])
    vVal[500] = 0.01

    tsInCont = ts.TSContinuous(vTime, vVal)
    dAct = net.evolve(tsInCont, duration=1.0)

    times = dAct["Rec"].times

    eps = 0.000001

    assert times[1] - times[0] - 0.01 < eps
    assert times[3] - times[2] - 0.01 < eps

    fl0.terminate()
    fl1.terminate()


def test_IAF2AEIFNest():
    """ Test RecIAFNest to RecAEIFNest """
    from Rockpool.layers import RecIAFSpkInNest
    from Rockpool.layers import RecAEIFSpkInNest
    from Rockpool import timeseries as ts
    import numpy as np

    # - Generic parameters
    weights_in = np.array([[-0.001, 0.002, 0.004], [0.03, -0.003, -0.0015]])
    weights_rec = np.random.rand(3, 3) * 0.001
    bias = 0.0
    tau_mem = [0.02, 0.05, 0.1]
    tau_syn_exc = [0.2, 0.01, 0.01]
    tau_syn_inh = tau_syn_exc
    dt = 0.001
    vThresh = -0.055
    vRest = -0.065

    # - Layer generation
    fl0 = RecIAFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=dt,
        bias=bias,
        tau_mem=tau_mem,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        v_thresh=vThresh,
        v_reset=vRest,
        v_rest=vRest,
        refractory=0.001,
        record=True,
    )

    # - Layer generation
    fl1 = RecAEIFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=dt,
        bias=bias,
        tau_mem=tau_mem,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        v_thresh=vThresh,
        v_reset=vRest,
        v_rest=vRest,
        subthresh_adapt=0.0,
        spike_adapt=0.0,
        delta_t=0.0,
        refractory=0.001,
        record=True,
    )

    # - Input signal
    spikeTimes = np.arange(0, 1, dt)
    spikeTrain0 = np.sort(np.random.choice(spikeTimes, size=20, replace=False))
    channels = np.random.choice([0, 1], 20, replace=True).astype(int)

    tsInEvent = ts.TSEvent(spikeTrain0, channels)

    # - Compare states before and after

    assert (np.abs(fl0.state - fl1.state) < 0.00001).all()

    dFl0 = fl0.evolve(tsInEvent, duration=1.0)
    dFl1 = fl1.evolve(tsInEvent, duration=1.0)

    assert (np.abs(fl0.state - fl1.state) < 0.00001).all()

    fl0.terminate()
    fl1.terminate()


def test_SaveLoad():
    """ Test save and load RecAEIFNest """
    from Rockpool.layers import RecIAFSpkInNest
    from Rockpool.layers import RecAEIFSpkInNest
    from Rockpool import timeseries as ts
    from Rockpool.networks import network as nw
    import numpy as np

    # - Generic parameters
    weights_in = np.array([[-0.001, 0.002, 0.004], [0.03, -0.003, -0.0015]])
    weights_rec = np.random.rand(3, 3) * 0.001
    bias = 0.0
    tau_mem = [0.02, 0.05, 0.1]
    tau_syn_exc = [0.2, 0.01, 0.01]
    tau_syn_inh = tau_syn_exc
    dt = 0.001
    vThresh = -0.055
    vRest = -0.065

    # - Layer generation
    fl0 = RecAEIFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=dt,
        bias=bias,
        tau_mem=tau_mem,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        v_thresh=vThresh,
        v_reset=vRest,
        v_rest=vRest,
        subthresh_adapt=0.0,
        spike_adapt=0.001,
        delta_t=0.0,
        refractory=0.001,
        record=True,
        name="lyrNest",
    )

    net0 = nw.Network(fl0)
    net0.save("tmp.model")

    net1 = nw.Network.load("tmp.model")

    # - Input signal
    spikeTimes = np.arange(0, 1, dt)
    spikeTrain0 = np.sort(np.random.choice(spikeTimes, size=20, replace=False))
    channels = np.random.choice([0, 1], 20, replace=True).astype(int)

    tsInEvent = ts.TSEvent(spikeTrain0, channels)

    # - Compare states before and after

    assert (np.abs(net0.input_layer.state - net1.input_layer.state) < 0.00001).all()

    dFl0 = net0.evolve(tsInEvent, duration=1.0)
    dFl1 = net1.evolve(tsInEvent, duration=1.0)

    assert (np.abs(net0.input_layer.state - net1.input_layer.state) < 0.00001).all()

    fl0.terminate()
    net1.lyrNest.terminate()
