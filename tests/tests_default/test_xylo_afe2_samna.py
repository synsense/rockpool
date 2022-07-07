import pytest

pytest.importorskip("samna")


def test_imports():
    from rockpool.devices.xylo.syns61201 import AFESamna


def test_afe2_module():
    from rockpool.devices.xylo.syns61201 import AFESamna
    from rockpool.devices.xylo.syns61201.xa2_devkit_utils import find_xylo_a2_boards

    # import samna
    # from samna.afe2.configuration import AfeConfiguration

    import numpy as np

    a2_boards = find_xylo_a2_boards()

    if len(a2_boards) == 0:
        pytest.skip("A connect Xylo-A2 HDK is required for this test")

    # - Create and configure a module
    mod_afe2 = AFESamna(a2_boards[0], dt=1e-3)

    # - Check parameter access
    mod_afe2.parameters()
    mod_afe2.simulation_parameters()
    mod_afe2.state()

    # - Try to record
    out, _, _ = mod_afe2.evolve(np.zeros([0, 100, 0]))
