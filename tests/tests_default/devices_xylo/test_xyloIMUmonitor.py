import pytest

pytest.importorskip("samna")


def test_XyloMonitor():
    from rockpool.devices.xylo.imu import XyloIMUMonitor, config_from_specification
    import rockpool.devices.xylo.imu.xylo_imu_devkit_utils as putils
    import numpy as np

    xylo_hdk_nodes = putils.find_xylo_imu_boards()

    if len(xylo_hdk_nodes) == 0:
        pytest.skip("A connected Xylo IMU HDK is required to run this test")

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

    # - Make a XyloMonitor module
    mod_xylo = XyloIMUMonitor(device=daughterboard, config=config, dt=dt, output_mode="Vmem")

    # - Simulate with random input
    T = 10
    input_ts = np.random.rand(T, Nin)
    # mod_xylo.reset_state()
    output_ts, _, _ = mod_xylo(input_ts)
