import pytest


def test_imports():
    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import AFESamna


def test_afe_samna_module():
    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import AFESamna
    from rockpool.devices.xylo.syns65302.xa3_devkit_utils import find_xylo_a3_boards

    import numpy as np

    a3_boards = find_xylo_a3_boards()

    if len(a3_boards) == 0:
        pytest.skip("A connect XyloAudio 3 HDK is required for this test")

    # - Create and configure a module
    mod_afe = AFESamna(a3_boards[0], dt=1e-3)

    # - Check parameter access
    mod_afe.parameters()
    mod_afe.simulation_parameters()
    mod_afe.state()

    # - Try to record
    out, _, _ = mod_afe.evolve(np.zeros([0, 100, 0]))

    # AFESamna uses live microphone so hard to identify what will be captured
    # However, we assume something should be capture, and the sum of the output spikes must be bigger than zero
    assert np.sum(out) > 0


def test_afe_samna_module_flip_and_encode():
    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import AFESamna
    from rockpool.devices.xylo.syns65302.xa3_devkit_utils import find_xylo_a3_boards

    import numpy as np

    a3_boards = find_xylo_a3_boards()

    if len(a3_boards) == 0:
        pytest.skip("A connect XyloAudio 3 HDK is required for this test")

    # - Create and configure a module
    mod_afe = AFESamna(a3_boards[0], dt=1e-3)

    # - Check parameter access
    mod_afe.parameters()
    mod_afe.simulation_parameters()
    mod_afe.state()

    # - Try to record
    out, _, _ = mod_afe.evolve(np.zeros([0, 100, 0]), flip_and_encode=True)

    # AFESamna uses live microphone so hard to identify what will be captured
    # However, we assume something should be capture, and the sum of the output spikes must be bigger than zero
    assert np.sum(out) > 0


def test_afe_samna_save_config():
    import numpy as np

    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import AFESamna
    from rockpool.devices.xylo.syns65302.xa3_devkit_utils import find_xylo_a3_boards
    from rockpool.devices.xylo.syns65302 import (
        load_config,
    )

    a3_boards = find_xylo_a3_boards()

    if len(a3_boards) == 0:
        pytest.skip("A connect XyloAudio 3 HDK is required for this test")

    # - Create and configure a module
    mod_afe = AFESamna(a3_boards[0], dt=1e-3)

    # - Check parameter access
    mod_afe.parameters()
    mod_afe.simulation_parameters()
    mod_afe.state()

    # - Try to save and load config
    mod_afe.save_config(filename="test_samna_config")

    loaded_config = load_config("test_samna_config")

    # - Test configuration should be equal
    np.testing.assert_allclose(
        mod_afe._config.input.weights, loaded_config.input.weights
    )
    np.testing.assert_allclose(
        mod_afe._config.input.syn2_weights, loaded_config.input.syn2_weights
    )
    np.testing.assert_allclose(
        mod_afe._config.input.weight_bit_shift, loaded_config.input.weight_bit_shift
    )
    np.testing.assert_allclose(
        mod_afe._config.readout.weights, loaded_config.readout.weights
    )
    np.testing.assert_allclose(
        mod_afe._config.readout.weight_bit_shift, loaded_config.readout.weight_bit_shift
    )
    np.testing.assert_allclose(
        mod_afe._config.readout.neurons[0].threshold,
        loaded_config.readout.neurons[0].threshold,
    )
    np.testing.assert_allclose(
        mod_afe._config.readout.neurons[0].i_syn_decay,
        loaded_config.readout.neurons[0].i_syn_decay,
    )
    np.testing.assert_allclose(
        mod_afe._config.readout.neurons[0].v_mem_decay,
        loaded_config.readout.neurons[0].v_mem_decay,
    )

    assert mod_afe._config.operation_mode, loaded_config.operation_mode
    assert mod_afe._config.input_source, loaded_config.input_source


def test_afe_samna_module_record():
    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import AFESamna
    from rockpool.devices.xylo.syns65302.xa3_devkit_utils import find_xylo_a3_boards

    import numpy as np

    a3_boards = find_xylo_a3_boards()

    if len(a3_boards) == 0:
        pytest.skip("A connect XyloAudio 3 HDK is required for this test")

    # - Create and configure a module
    mod_afe = AFESamna(a3_boards[0], dt=1e-3)

    # - Check parameter access
    mod_afe.parameters()
    mod_afe.simulation_parameters()
    mod_afe.state()

    # - Try to record
    out, _, rec = mod_afe.evolve(np.zeros([0, 100, 0]), record=True)

    # AFESamna uses live microphone so hard to identify what will be captured
    # However, we assume something should be capture, and the sum of the output spikes must be bigger than zero
    assert np.sum(out) > 0
    assert "neuron_ids" in rec
    assert rec["timesteps"][0] >= 0
    assert rec["timesteps"][-1] <= 100
