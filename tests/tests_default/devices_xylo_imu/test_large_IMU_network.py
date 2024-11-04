import pytest

pytest.importorskip("samna")


def test_large_IMU_network_config():
    from rockpool.devices.xylo.syns63300 import (
        XyloSamna,
        mapper,
        config_from_specification,
        XyloIMUMonitor,
    )

    import rockpool.devices.xylo.syns63300.xylo_imu_devkit_utils as putils

    from rockpool.transform.quantize_methods import channel_quantize

    from rockpool.nn.modules import LIF, Linear
    from rockpool.nn.combinators import Sequential

    import numpy as np
    import matplotlib.pyplot as plt

    # - Get a Xylo HDK board
    xylo_hdk_nodes = putils.find_xylo_imu_boards()

    if len(xylo_hdk_nodes) == 0:
        pytest.skip("A connected Xylo IMU device is required for this test")

    net = Sequential(
        Linear((16, 63)),
        LIF(63),
        Linear((63, 63)),
        LIF(63),
        Linear((63, 63)),
        LIF(63),
        Linear((63, 1)),
        LIF(1),
    )
    net

    spec = mapper(net.as_graph())
    Q_spec = spec
    Q_spec.update(channel_quantize(**Q_spec))

    config, is_valid, msg = config_from_specification(**Q_spec)

    if not is_valid:
        raise ValueError(msg)

    xmod = XyloSamna.from_config(xylo_hdk_nodes[0], config, dt=1e-3)
    xmod = XyloIMUMonitor.from_config(xylo_hdk_nodes[0], config, dt=1e-3)
