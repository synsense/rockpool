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
    # However, we assume something should be capture, and the sum of the spikes must be bigger than zero
    assert np.sum(out) > 0
