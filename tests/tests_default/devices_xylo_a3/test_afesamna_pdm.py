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


def test_afesamna_pdm():
    import pytest

    pytest.importorskip("samna")

    from rockpool.devices.xylo.syns65302 import (
        AFESamnaPDM,
    )
    import rockpool.devices.xylo.syns65302.xa3_devkit_utils as putils
    from rockpool.nn.combinators import Sequential
    from rockpool.nn.modules import LIF, Linear
    from rockpool.transform.quantize_methods import channel_quantize

    import numpy as np
    import samna

    # - Get a Xylo HDK board
    xylo_hdk_nodes = putils.find_xylo_a3_boards()

    if len(xylo_hdk_nodes) == 0:
        pytest.skip("A connected XyloAudio 3 HDK is required to run this test")

    daughterboard = xylo_hdk_nodes[0]

    # - Create a Xylo module with PDM input
    dn = True
    xmod = AFESamnaPDM(daughterboard, dt=1024e-6, dn_active=dn)

    assert xmod != None

    input_pdm = np.loadtxt("tests/tests_default/models/xylo_a3_input_pdm.txt")

    _, _, rd = xmod(input_pdm, record=True, flip_and_encode=True)
    del xmod

    dur = 200e-3
    # result of the same data simulation with flip_and_encode flag
    a = np.loadtxt("tests/tests_default/models/xylo_a3_afe_sim_pdm_output.txt")
    b = np.sum(rd["Spikes_in"].T, axis=1) / dur

    result = [abs(i - j) / i for i, j in zip(a, b)]

    # some error marging accepted
    b = np.array([element < 0.05 for element in result])

    assert np.all(b)
