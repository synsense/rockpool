def test_imports():
    import pytest

    pytest.importorskip("samna")
    pytest.importorskip("xylosim")

    from rockpool.devices.xylo.syns61300 import (
        config_from_specification,
        save_config,
        load_config,
        XyloSamna,
    )
    import rockpool.devices.xylo.syns61300.xylo_devkit_utils as putils


def test_from_specification():
    import pytest

    pytest.importorskip("samna")
    pytest.importorskip("xylosim")

    from rockpool.devices.xylo.syns61300 import config_from_specification
    import numpy as np

    Nin = 3
    Nhidden = 5
    Nout = 2

    config, valid, msg = config_from_specification(
        weights_in=np.zeros((Nin, Nhidden, 2)),
        weights_out=np.zeros((Nhidden, Nout)),
        weights_rec=np.zeros((Nhidden, Nhidden, 2)),
        dash_mem=np.ones(Nhidden),
        dash_mem_out=np.ones(Nout),
        dash_syn=np.ones(Nhidden),
        dash_syn_2=np.ones(Nhidden),
        dash_syn_out=np.ones(Nout),
        threshold=np.ones(Nhidden),
        threshold_out=np.ones(Nout),
        weight_shift_in=1,
        weight_shift_rec=1,
        weight_shift_out=1,
        aliases=None,
    )


def test_save_load():
    import pytest

    pytest.importorskip("samna")
    pytest.importorskip("xylosim")

    from rockpool.devices.xylo.syns61300 import (
        config_from_specification,
        save_config,
        load_config,
    )
    import numpy as np

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
    np.testing.assert_allclose(config.input.syn2_weights, conf2.input.syn2_weights)
    np.testing.assert_allclose(
        config.input.weight_bit_shift, conf2.input.weight_bit_shift
    )
    np.testing.assert_allclose(config.reservoir.weights, conf2.reservoir.weights)
    np.testing.assert_allclose(
        config.reservoir.syn2_weights, conf2.reservoir.syn2_weights
    )
    np.testing.assert_allclose(
        config.reservoir.weight_bit_shift, conf2.reservoir.weight_bit_shift
    )
    np.testing.assert_allclose(
        config.reservoir.neurons[0].i_syn2_decay,
        conf2.reservoir.neurons[0].i_syn2_decay,
    )
    np.testing.assert_allclose(
        config.reservoir.neurons[0].threshold,
        conf2.reservoir.neurons[0].threshold,
    )
    np.testing.assert_allclose(
        config.reservoir.neurons[0].i_syn_decay,
        conf2.reservoir.neurons[0].i_syn_decay,
    )
    np.testing.assert_allclose(
        config.reservoir.neurons[0].v_mem_decay,
        conf2.reservoir.neurons[0].v_mem_decay,
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
