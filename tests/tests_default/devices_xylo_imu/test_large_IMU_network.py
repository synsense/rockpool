def test_large_IMU_network():
    from rockpool.devices.xylo.syns63300 import (
        XyloSamna,
        mapper,
        config_from_specification,
        XyloIMUMonitor,
    )

    # from rockpool.devices.xylo.syns61201 import XyloSamna, mapper, config_from_specification, AFESamna, XyloMonitor
    from rockpool.devices.xylo import find_xylo_hdks

    from rockpool.transform.quantize_methods import channel_quantize

    from rockpool.nn.modules import LIF, Linear
    from rockpool.nn.combinators import Sequential

    import numpy as np
    import matplotlib.pyplot as plt

    hdks, modules, versions = find_xylo_hdks()

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
