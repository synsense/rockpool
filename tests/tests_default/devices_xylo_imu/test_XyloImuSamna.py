import pytest

pytest.importorskip("samna")


def test_imports():
    from rockpool.devices.xylo.imu import (
        save_config,
        load_config,
        XyloIMUSamna,
    )
    import rockpool.devices.xylo.imu.xylo_imu_devkit_utils as putils


def test_XyloSamna():
    from rockpool.devices.xylo.imu import XyloIMUSamna, config_from_specification
    import rockpool.devices.xylo.imu.xylo_imu_devkit_utils as putils
    from rockpool import TSEvent, TSContinuous

    import numpy as np

    # - Get a Xylo HDK board
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

    # - Make a XyloImuSamna module
    modXyloSamna = XyloIMUSamna(
        device=daughterboard, config=config, dt=dt, output_mode="Vmem"
    )

    # - Simulate with random input
    T = 100
    f = 0.4
    input_spikes = np.random.rand(T, Nin) < f
    output_ts, _, _ = modXyloSamna(input_spikes)
    print(output_ts)


def test_save_load():
    from rockpool.devices.xylo.imu import (
        config_from_specification,
        save_config,
        load_config,
    )
    import numpy as np

    # - Make a Xylo configuration
    Nin = 3
    Nhidden = 5
    Nout = 2

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

    save_config(config, "../test_samna_config.json")
    conf2 = load_config("../test_samna_config.json")

    # - Test configuration should be equal
    np.testing.assert_allclose(config.input.weights, conf2.input.weights)
    np.testing.assert_allclose(
        config.input.weight_bit_shift, conf2.input.weight_bit_shift
    )
    np.testing.assert_allclose(config.hidden.weights, conf2.hidden.weights)
    np.testing.assert_allclose(
        config.hidden.weight_bit_shift, conf2.hidden.weight_bit_shift
    )
    np.testing.assert_allclose(
        config.hidden.neurons[0].threshold,
        conf2.hidden.neurons[0].threshold,
    )
    np.testing.assert_allclose(
        config.hidden.neurons[0].i_syn_decay,
        conf2.hidden.neurons[0].i_syn_decay,
    )
    np.testing.assert_allclose(
        config.hidden.neurons[0].v_mem_decay,
        conf2.hidden.neurons[0].v_mem_decay,
    )

    np.testing.assert_allclose(config.readout.weights, conf2.readout.weights)
    np.testing.assert_allclose(
        config.readout.weight_bit_shift, conf2.readout.weight_bit_shift
    )
    np.testing.assert_allclose(
        config.readout.neurons[0].threshold,
        conf2.readout.neurons[0].threshold,
    )
    np.testing.assert_allclose(
        config.readout.neurons[0].i_syn_decay,
        conf2.readout.neurons[0].i_syn_decay,
    )
    np.testing.assert_allclose(
        config.readout.neurons[0].v_mem_decay,
        conf2.readout.neurons[0].v_mem_decay,
    )


def test_xylo_vs_xylosim():
    # - Samna imports
    import samna

    from rockpool.devices.xylo.imu import xylo_imu_devkit_utils as putils
    import rockpool.devices.xylo.imu as x

    import numpy as np

    # Make a Xylo configuration
    Nin = 3
    Nhidden = 5
    Nout = 2
    T = 1000

    config, valid, msg = x.config_from_specification(
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

    # - Create XyloSim object
    mod_xylo_sim_vmem = x.XyloSim.from_config(config, output_mode="Vmem", dt=1e-3)
    mod_xylo_sim_isyn = x.XyloSim.from_config(config, output_mode="Isyn", dt=1e-3)
    mod_xylo_sim_spike = x.XyloSim.from_config(config, dt=1e-3)
    mod_xylo_sim_vmem.timed()
    mod_xylo_sim_isyn.timed()
    mod_xylo_sim_spike.timed()

    # - Generate random input
    input_raster = np.random.randint(0, 16, (T, Nin))

    # - Simulate the evolution of the network on Xylo
    # mod_xylo_sim_spike.reset_state()
    out_sim, _, rec_sim = mod_xylo_sim_spike.evolve(
        input_raster.clip(0, 15), record=True
    )

    # - Get a Xylo HDK board
    xylo_hdk_nodes = putils.find_xylo_imu_boards()

    if len(xylo_hdk_nodes) == 0:
        pytest.skip("A connected Xylo IMU HDK is required to run this test")

    daughterboard = xylo_hdk_nodes[0]

    # - Init Xylo
    # mod_xylo_vmem = x.XyloIMUSamna(daughterboard, config, dt=1e-3, output_mode="Vmem")
    # mod_xylo_isyn = x.XyloIMUSamna(daughterboard, config, dt=1e-3, output_mode="Isyn")
    mod_xylo_spike = x.XyloIMUSamna(daughterboard, config, dt=1e-3)

    # - Evolve Xylo
    mod_xylo_spike.reset_state()
    out_xylo, _, rec_xylo = mod_xylo_spike.evolve(input_raster, record=True)

    # - Assert equality for all outputs and recordings
    assert np.all(out_sim == out_xylo)
    assert np.all(rec_sim["Vmem"] == rec_xylo["Vmem"])
    assert np.all(rec_sim["Isyn"] == rec_xylo["Isyn"])
    assert np.all(rec_sim["Vmem_out"] == rec_xylo["Vmem_out"])
    assert np.all(rec_sim["Isyn_out"] == rec_xylo["Isyn_out"])
    assert np.all(rec_sim["Spikes"] == rec_xylo["Spikes"])
