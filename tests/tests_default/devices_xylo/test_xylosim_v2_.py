import pytest

pytest.importorskip("samna")
pytest.importorskip("xylosim")


def test_imports_V2():
    from rockpool.devices.xylo.syns61201 import XyloSim, mapper


def test_specification_V2():
    # - Samna imports
    from rockpool.devices.xylo.syns61201 import XyloSim

    import numpy as np

    Nin = 8
    Nhidden = 3
    Nout = 2

    # - Test minimal spec
    spec = {
        "weights_in": np.ones((Nin, Nhidden, 2), "int"),
        "weights_out": np.ones((Nhidden, Nout), "int"),
    }

    mod_xylo_sim = XyloSim.from_specification(**spec)

    # - Test complete spec
    spec = {
        "weights_in": np.ones((Nin, Nhidden, 2), "int"),
        "weights_rec": np.ones((Nhidden, Nhidden, 2), "int"),
        "weights_out": np.ones((Nhidden, Nout), "int"),
        "dash_mem": np.ones(Nhidden, "int"),
        "dash_mem_out": np.ones(Nout, "int"),
        "dash_syn": np.ones(Nhidden, "int"),
        "dash_syn_2": np.ones(Nhidden, "int"),
        "dash_syn_out": np.ones(Nout, "int"),
        "threshold": np.ones(Nhidden, "int"),
        "threshold_out": np.ones(Nout, "int"),
        "bias": np.ones(Nhidden, "int"),
        "bias_out": np.ones(Nout, "int"),
        "weight_shift_in": 0,
        "weight_shift_rec": 0,
        "weight_shift_out": 0,
        "aliases": None,
    }

    mod_xylo_sim_vmem = XyloSim.from_specification(**spec, output_mode="Vmem")
    mod_xylo_sim_isyn = XyloSim.from_specification(**spec, output_mode="Isyn")
    mod_xylo_sim_spike = XyloSim.from_specification(**spec, output_mode="Spike")

    # - Simulate the evolution of the network on Xylo
    T = 1000
    input_rate = 0.01
    input_raster = np.random.rand(T, Nin) < input_rate
    output_raster_vmem, _, _ = mod_xylo_sim_vmem(input_raster)
    output_raster_isyn, _, _ = mod_xylo_sim_isyn(input_raster)
    output_raster_spike, _, _ = mod_xylo_sim_spike(input_raster)


def test_xylo_vs_xylosim():
    # - Samna imports
    import samna
    from samna.xyloCore2 import validate_configuration

    from rockpool.devices.xylo.syns61201 import xa2_devkit_utils as xu
    import rockpool.devices.xylo.syns61201 as x

    import numpy as np

    T = 1000
    Nin = 8
    Nhidden = 3
    Nout = 2

    # - Test full random spec
    spec = {
        "weights_in": np.random.randint(-128, 128, (Nin, Nhidden, 2)),
        "weights_rec": np.random.randint(-128, 128, (Nhidden, Nhidden, 2)),
        "weights_out": np.random.randint(-128, 128, (Nhidden, Nout)),
        "dash_mem": np.random.randint(1, 8, Nhidden),
        "dash_mem_out": np.random.randint(1, 8, Nout),
        "dash_syn": np.random.randint(1, 8, Nhidden),
        "dash_syn_2": np.random.randint(1, 8, Nhidden),
        "dash_syn_out": np.random.randint(1, 8, Nout),
        "threshold": np.random.randint(1, 1000, Nhidden),
        "threshold_out": np.ones(Nout) * 1000,
        "weight_shift_in": np.random.randint(1, 8),
        "weight_shift_rec": np.random.randint(1, 8),
        "weight_shift_out": np.random.randint(1, 8),
        "aliases": [[], [2], []],
    }

    # - Create configuration object
    conf, _, _ = x.config_from_specification(**spec)

    # - Check for validity
    valid, message = validate_configuration(conf)
    assert valid

    # - Create XyloSim object
    mod_xylo_sim_vmem = x.XyloSim.from_config(conf, output_mode="Vmem", dt=1e-3)
    mod_xylo_sim_isyn = x.XyloSim.from_config(conf, output_mode="Isyn", dt=1e-3)
    mod_xylo_sim_spike = x.XyloSim.from_config(conf, dt=1e-3)
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
    xylo_hdk_nodes = xu.find_xylo_a2_boards()

    if len(xylo_hdk_nodes) == 0:
        pytest.skip("This test requires a connected Xylo board to continue.")

    db = xylo_hdk_nodes[0]

    # - Init Xylo
    mod_xylo_vmem = x.XyloSamna(db, conf, dt=1e-3, output_mode="Vmem")
    mod_xylo_isyn = x.XyloSamna(db, conf, dt=1e-3, output_mode="Isyn")
    mod_xylo_spike = x.XyloSamna(db, conf, dt=1e-3)

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
