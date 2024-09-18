def test_imports():
    import pytest

    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import (
        save_config,
        load_config,
        XyloSamna,
        config_from_specification,
    )
    import rockpool.devices.xylo.syns65302.xylo_a3_devkit_utils as putils


def test_xylosamna_pdm():
    import pytest

    pytest.importorskip("samna")

    from rockpool.devices.xylo.syns65302 import (
        XyloSamnaPDM,
        config_from_specification,
        mapper,
    )
    import rockpool.devices.xylo.syns65302.xylo_a3_devkit_utils as putils
    from rockpool import TSEvent, TSContinuous
    from rockpool.nn.combinators import Sequential
    from rockpool.nn.modules import LIF, Linear
    from rockpool.transform.quantize_methods import channel_quantize

    import numpy as np
    import samna
    import os

    # - Get a Xylo HDK board
    xylo_hdk_nodes = putils.find_xylo_a3_boards()

    if len(xylo_hdk_nodes) == 0:
        pytest.skip("A connected Xylo Audio3 HDK is required to run this test")

    daughterboard = xylo_hdk_nodes[0]

    # create a network
    net = Sequential(
        Linear((16, 63)),
        LIF((63, 63)),
        Linear((63, 32)),
        LIF(32),
    )

    net[0].weight *= 0.05

    spec = mapper(net.as_graph())
    Q_spec = spec
    Q_spec.update(channel_quantize(**Q_spec))
    config, is_valid, msg = config_from_specification(**Q_spec)

    if not is_valid:
        print(msg)

    # - Create a Xylo module with PDM input
    dn = True
    config.operation_mode = samna.xyloAudio3.OperationMode.AcceleratedTime
    xmod = XyloSamnaPDM(daughterboard, config, dt=1024e-6, dn_active=dn)

    assert xmod != None

    input_pdm = np.loadtxt("tests/tests_default/models/xylo_a3_input_pdm.txt")

    out, _, rd = xmod(input_pdm, record=True)
    dur = 200e-3
    # result of the same data simulation
    a = np.loadtxt("tests/tests_default/models/xylo_a3_afe_sim_pdm_output.txt")
    b = np.sum(rd["Spikes_in"].T, axis=1) / dur

    result = [abs(i - j) / i for i, j in zip(a, b)]

    b = np.array([element < 0.05 for element in result])

    # some error marging accepted
    assert np.all(b)
