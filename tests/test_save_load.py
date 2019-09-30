def test_save_load():
    """ Test RecIAFNest"""
    from rockpool.layers import RecIAFSpkInNest, FFIAFNest, FFExpSynTorch
    from rockpool import timeseries as ts
    from rockpool.networks import network as nw
    from rockpool.networks import Network as nws
    import numpy as np
    import pylab as plt

    # - Generic parameters
    weights = np.ones([1, 1]) * 0.01
    weights_in = [[0.1, 0, 0]]

    weights_rec = [[0, 0.1, 0], [0, 0, 0.1], [0.0, 0, 0]]
    mfWOut = [[1], [1], [1]]
    bias = 0.0
    dt = 0.001
    tau_mem = 0.02
    tau_syn = 0.05
    v_thresh = -0.055
    v_reset = -0.065
    v_rest = -0.065
    capacity = 100.0
    refractory = 0.001

    np.random.seed(0)

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
        num_cores=1,
        record=True,
        name="FF",
    )

    fl1 = RecIAFSpkInNest(
        weights_in=weights_in,
        weights_rec=weights_rec,
        dt=dt,
        bias=bias,
        tau_mem=tau_mem,
        tau_syn_exc=tau_syn,
        tau_syn_inh=tau_syn,
        v_thresh=v_thresh,
        v_reset=v_reset,
        v_rest=v_rest,
        capacity=capacity,
        refractory=refractory,
        num_cores=1,
        record=True,
        name="Rec",
    )

    net0 = nw.Network(fl0, fl1)

    net0.save("test_nw.json")
    net1 = nws.load("test_nw.json")
    fl2 = net1.FF
    fl3 = net1.Rec

    np.random.seed(0)

    # - Input signal
    vTime = np.arange(0, 1, dt)
    vVal = np.zeros([len(vTime), 1])
    vVal[200:201] = 0.25

    tsInCont = ts.TSContinuous(vTime, vVal)

    epsilon = 0.0000001

    # - Compare states before and after
    np.random.seed(0)
    dAct0 = net0.evolve(tsInCont, duration=1.0)

    np.random.seed(0)
    dAct1 = net1.evolve(tsInCont, duration=1.0)

    assert (
        np.abs(fl0.recorded_states.samples - fl2.recorded_states.samples) < epsilon
    ).all()
    assert (
        np.abs(fl1.recorded_states.samples - fl3.recorded_states.samples) < epsilon
    ).all()
