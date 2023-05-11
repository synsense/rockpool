import pytest


def test_XyloSamna_readout():
    pytest.importorskip("samna")
    pytest.importorskip("xylosim")

    from rockpool.devices.xylo.syns61300 import XyloSamna, config_from_specification
    import rockpool.devices.xylo.syns61300.xylo_devkit_utils as putils
    import numpy as np

    xylo_hdk_nodes = putils.find_xylo_boards()

    if len(xylo_hdk_nodes) == 0:
        pytest.skip("A connected Xylo HDK is required to run this test")

    daughterboard = xylo_hdk_nodes[0]

    # - Make a Xylo configuration
    Nin = 3
    Nhidden = 5
    Nout = 2
    dt = 1e-3

    config, valid, msg = config_from_specification(
        weights_in=np.random.uniform(-127, 127, size=(Nin, Nhidden, 2)),
        weights_out=np.random.uniform(-127, 127, size=(Nhidden, Nout)),
        weights_rec=np.random.uniform(-127, 127, size=(Nhidden, Nhidden, 2)),
        dash_mem=2 * np.ones(Nhidden),
        dash_mem_out=3 * np.ones(Nout),
        dash_syn=4 * np.ones(Nhidden),
        dash_syn_2=2 * np.ones(Nhidden),
        dash_syn_out=3 * np.ones(Nout),
        threshold=128 * np.ones(Nhidden),
        threshold_out=256 * np.ones(Nout),
        weight_shift_in=1,
        weight_shift_rec=1,
        weight_shift_out=1,
        aliases=None,
    )

    # - Make a XyloSamna module
    mod_xylo_Vmem = XyloSamna(daughterboard, config, dt, output_mode="Vmem")
    mod_xylo_Isyn = XyloSamna(daughterboard, config, dt, output_mode="Isyn")
    mod_xylo_Spike = XyloSamna(daughterboard, config, dt, output_mode="Spike")

    # - Simulate with random input
    T = 10
    f_rate = 0.01
    input_ts = np.random.rand(T, Nin) < 0.01
    mod_xylo_Vmem.reset_state()
    mod_xylo_Isyn.reset_state()
    mod_xylo_Spike.reset_state()

    output_ts_Vmem, _, rec_state_Vmem = mod_xylo_Vmem(input_ts, record=True)
    output_ts_Isyn, _, rec_state_Isyn = mod_xylo_Isyn(input_ts, record=True)
    output_ts_Spike, _, rec_state_Spike = mod_xylo_Spike(input_ts, record=True)
