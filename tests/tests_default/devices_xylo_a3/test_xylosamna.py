import pytest


def test_imports():
    import pytest

    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import (
        save_config,
        load_config,
        XyloSamna,
        config_from_specification,
    )
    import rockpool.devices.xylo.syns65302.xa3_devkit_utils as putils


def test_XyloSamna():
    import pytest

    pytest.importorskip("samna")

    from rockpool.devices.xylo.syns65302 import XyloSamna, config_from_specification
    import rockpool.devices.xylo.syns65302.xa3_devkit_utils as putils
    from rockpool import TSEvent, TSContinuous

    import numpy as np
    import samna

    # - Get a Xylo HDK board
    xylo_hdk_nodes = putils.find_xylo_a3_boards()

    if len(xylo_hdk_nodes) == 0:
        pytest.skip("A connected XyloAudio 3 HDK is required to run this test")

    daughterboard = xylo_hdk_nodes[0]

    # - Make a Xylo configuration
    Nin = 3
    Nhidden = 5
    Nout = 2
    dt = 1e-3

    config, valid, msg = config_from_specification(
        weights_in=np.random.uniform(-127, 127, size=(Nin, Nhidden, 1)),
        weights_out=np.random.uniform(-127, 127, size=(Nhidden, Nout)),
        weights_rec=np.random.uniform(-127, 127, size=(Nhidden, Nhidden, 1)),
        dash_mem=2 * np.ones(Nhidden),
        dash_mem_out=3 * np.ones(Nout),
        dash_syn=4 * np.ones(Nhidden),
        dash_syn_out=3 * np.ones(Nout),
        threshold=128 * np.ones(Nhidden),
        threshold_out=256 * np.ones(Nout),
        weight_shift_in=1,
        weight_shift_rec=1,
        weight_shift_out=1,
        aliases=None,
    )

    config.input_source = samna.xyloAudio3.InputSource.SpikeEvents
    config.operation_mode = samna.xyloAudio3.OperationMode.AcceleratedTime

    # - Simulate with random input
    T = 100
    f = 0.4
    input_spikes = np.random.rand(T, Nin) < f

    # - Make a XyloSamna module for Vmem
    modXyloSamna = XyloSamna(
        device=daughterboard, config=config, dt=dt, output_mode="Vmem", record=True
    )
    output_ts, _, _ = modXyloSamna.evolve(input_spikes)

    # - Make a XyloSamna module for Isyn
    modXyloSamna = XyloSamna(
        device=daughterboard, config=config, dt=dt, output_mode="Isyn", record=True
    )
    output_ts, _, _ = modXyloSamna.evolve(input_spikes)

    # - Make a XyloSamna module for Spike
    modXyloSamna = XyloSamna(device=daughterboard, config=config, dt=dt, record=True)
    output_ts, _, _ = modXyloSamna.evolve(input_spikes)


def test_save_load():
    import pytest

    pytest.importorskip("samna")

    from rockpool.devices.xylo.syns65302 import (
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
        weights_in=np.random.uniform(-127, 127, size=(Nin, Nhidden, 1)),
        weights_out=np.random.uniform(-127, 127, size=(Nhidden, Nout)),
        weights_rec=np.random.uniform(-127, 127, size=(Nhidden, Nhidden, 1)),
        dash_mem=2 * np.ones(Nhidden),
        dash_mem_out=3 * np.ones(Nout),
        dash_syn=4 * np.ones(Nhidden),
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


def test_xylo_vs_xylosim_acceleratedtime():
    import pytest

    pytest.importorskip("samna")
    pytest.importorskip("xylosim")

    # - Samna imports
    import samna

    from rockpool.devices.xylo.syns65302 import xa3_devkit_utils as putils
    import rockpool.devices.xylo.syns65302 as x

    import numpy as np

    # Make a Xylo configuration
    Nin = 3
    Nhidden = 5
    Nout = 2
    T = 1000

    config, valid, msg = x.config_from_specification(
        weights_in=np.random.uniform(-127, 127, size=(Nin, Nhidden, 1)),
        weights_out=np.random.uniform(-127, 127, size=(Nhidden, Nout)),
        weights_rec=np.random.uniform(-127, 127, size=(Nhidden, Nhidden, 1)),
        dash_mem=2 * np.ones(Nhidden),
        dash_mem_out=3 * np.ones(Nout),
        dash_syn=4 * np.ones(Nhidden),
        dash_syn_out=3 * np.ones(Nout),
        threshold=128 * np.ones(Nhidden),
        threshold_out=256 * np.ones(Nout),
        weight_shift_in=1,
        weight_shift_rec=1,
        weight_shift_out=1,
        aliases=None,
    )

    config.input_source = samna.xyloAudio3.InputSource.SpikeEvents
    config.operation_mode = samna.xyloAudio3.OperationMode.AcceleratedTime

    # - Create XyloSim object
    mod_xylo_sim_vmem = x.XyloSim.from_config(config, output_mode="Vmem", dt=1.0 / 200)
    mod_xylo_sim_isyn = x.XyloSim.from_config(config, output_mode="Isyn", dt=1.0 / 200)
    mod_xylo_sim_spike = x.XyloSim.from_config(config, dt=1.0 / 200)
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
    xylo_hdk_nodes = putils.find_xylo_a3_boards()

    if len(xylo_hdk_nodes) == 0:
        pytest.skip("A connected XyloAudio 3 HDK is required to run this test")

    daughterboard = xylo_hdk_nodes[0]

    # - Make sure the board is in a clean state after running other tests
    daughterboard.reset_board_soft()

    # - Init Xylo
    mod_xylo_spike = x.XyloSamna(daughterboard, config, dt=1.0 / 200, record=True)

    print(len(input_raster))

    # - Evolve Xylo
    mod_xylo_spike.reset_state()
    out_xylo, _, rec_xylo = mod_xylo_spike.evolve(input_raster)

    # - Assert equality for all outputs and recordings
    assert np.all(out_sim == out_xylo)
    assert np.all(rec_sim["Vmem"] == rec_xylo["Vmem"])
    assert np.all(rec_sim["Isyn"] == rec_xylo["Isyn"])
    assert np.all(rec_sim["Vmem_out"] == rec_xylo["Vmem_out"])
    assert np.all(rec_sim["Isyn_out"] == rec_xylo["Isyn_out"])
    assert np.all(rec_sim["Spikes"] == rec_xylo["Spikes"])


def test_xylo_vs_xylosim_manual():
    import pytest

    pytest.importorskip("samna")
    pytest.importorskip("xylosim")

    # - Samna imports
    import samna

    from rockpool.devices.xylo.syns65302 import xa3_devkit_utils as putils
    import rockpool.devices.xylo.syns65302 as x

    import numpy as np

    # Make a Xylo configuration
    Nin = 3
    Nhidden = 5
    Nout = 2
    T = 1000

    config, valid, msg = x.config_from_specification(
        weights_in=np.random.uniform(-127, 127, size=(Nin, Nhidden, 1)),
        weights_out=np.random.uniform(-127, 127, size=(Nhidden, Nout)),
        weights_rec=np.random.uniform(-127, 127, size=(Nhidden, Nhidden, 1)),
        dash_mem=2 * np.ones(Nhidden),
        dash_mem_out=3 * np.ones(Nout),
        dash_syn=4 * np.ones(Nhidden),
        dash_syn_out=3 * np.ones(Nout),
        threshold=128 * np.ones(Nhidden),
        threshold_out=256 * np.ones(Nout),
        weight_shift_in=1,
        weight_shift_rec=1,
        weight_shift_out=1,
        aliases=None,
    )

    config.input_source = samna.xyloAudio3.InputSource.SpikeEvents
    config.operation_mode = samna.xyloAudio3.OperationMode.Manual

    # - Create XyloSim object
    mod_xylo_sim_vmem = x.XyloSim.from_config(config, output_mode="Vmem", dt=1.0 / 200)
    mod_xylo_sim_isyn = x.XyloSim.from_config(config, output_mode="Isyn", dt=1.0 / 200)
    mod_xylo_sim_spike = x.XyloSim.from_config(config, dt=1.0 / 200)
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
    xylo_hdk_nodes = putils.find_xylo_a3_boards()

    if len(xylo_hdk_nodes) == 0:
        pytest.skip("A connected XyloAudio 3 HDK is required to run this test")

    daughterboard = xylo_hdk_nodes[0]

    # - Make sure the board is in a clean state after running other tests
    daughterboard.reset_board_soft()

    # - Init Xylo
    mod_xylo_spike = x.XyloSamna(daughterboard, config, dt=1.0 / 200, record=True)

    # - Evolve Xylo
    mod_xylo_spike.reset_state()
    out_xylo, _, rec_xylo = mod_xylo_spike._evolve_manual(input_raster)

    # - Assert equality for all outputs and recordings
    assert np.all(out_sim == out_xylo)
    # FIXME: These values are shifited by 1 event!
    # assert np.all(rec_sim["Vmem"] == rec_xylo["Vmem"])
    # assert np.all(rec_sim["Isyn"] == rec_xylo["Isyn"])
    # assert np.all(rec_sim["Vmem_out"] == rec_xylo["Vmem_out"])
    # assert np.all(rec_sim["Isyn_out"] == rec_xylo["Isyn_out"])
    # assert np.all(rec_sim["Spikes"] == rec_xylo["Spikes"])


def test_config_from_specification():
    import pytest

    pytest.importorskip("samna")

    from rockpool.devices.xylo.syns65302 import config_from_specification, mapper
    from rockpool.transform import quantize_methods as q
    from rockpool.nn.modules import LIFTorch, LinearTorch
    from rockpool.nn.combinators import Sequential, Residual

    Nin = 2
    Nhidden = 4
    Nout = 2
    dt = 1e-2

    net = Sequential(
        LinearTorch((Nin, Nhidden), has_bias=False),
        LIFTorch(Nhidden, dt=dt),
        Residual(
            LinearTorch((Nhidden, Nhidden), has_bias=False),
            LIFTorch(Nhidden, has_rec=True, threshold=1.0, dt=dt),
        ),
        LinearTorch((Nhidden, Nout), has_bias=False),
        LIFTorch(Nout, dt=dt),
    )

    spec = mapper(
        net.as_graph(),
        weight_dtype="float",
        threshold_dtype="float",
        dash_dtype="float",
    )
    spec.update(q.global_quantize(**spec))

    config, is_valid, msg = config_from_specification(**spec)
    if not is_valid:
        print(msg)
