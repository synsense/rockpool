import pytest


def test_imports_syns61201():
    from rockpool.devices.xylo.syns61201 import cycles_model, est_clock_freq


def test_imports_syns63300():
    from rockpool.devices.xylo.syns63300 import cycles_model, est_clock_freq


def test_cycles_model_syns61201():
    pytest.importorskip("samna")

    from rockpool.devices.xylo.syns61201 import (
        cycles_model,
        est_clock_freq,
        mapper,
        config_from_specification,
    )
    from rockpool.transform.quantize_methods import channel_quantize
    from rockpool.nn.modules import LIF, Linear
    from rockpool.nn.combinators import Sequential

    net = Sequential(
        Linear((16, 64)),
        LIF((64, 32)),
        Linear((32, 64)),
        LIF((64, 32), has_rec=True),
        Linear((32, 8)),
        LIF(8),
    )

    spec = mapper(net.as_graph())
    config, _, _ = config_from_specification(**channel_quantize(**spec))
    cycles, isyn2ops = cycles_model(config)

    assert isyn2ops > 0.0, f"Expected isyn2ops > 0, received {isyn2ops}."
    assert cycles > 0.0, f"Expected cycles > 0, received {cycles}"

    cycles_model(config, input_sp=0.5, hidden_sp=0.5, output_sp=0.5)

    clk_freq = est_clock_freq(config, 1e-3)
    assert (
        clk_freq / 1e6 < 50.0
    ), f"Expected master clock frequency < 50 MHz, received {clk_freq / 1e6:.2f} MHz"

    est_clock_freq(config, 1e-3, margin=0.0)
    est_clock_freq(config, 1e-3, margin=1.0)


def test_cycles_model_syns63300():
    pytest.importorskip("samna")

    from rockpool.devices.xylo.syns63300 import (
        cycles_model,
        est_clock_freq,
        mapper,
        config_from_specification,
    )
    from rockpool.transform.quantize_methods import channel_quantize
    from rockpool.nn.modules import LIF, Linear
    from rockpool.nn.combinators import Sequential

    net = Sequential(
        Linear((16, 32)),
        LIF(32),
        Linear((32, 32)),
        LIF(32, has_rec=True),
        Linear((32, 16)),
        LIF(16),
    )

    spec = mapper(net.as_graph())
    config, _, _ = config_from_specification(**channel_quantize(**spec))
    cycles = cycles_model(config)

    assert cycles > 0.0, f"Expected cycles > 0, received {cycles}"

    cycles_model(config, input_sp=0.5, hidden_sp=0.5, output_sp=0.5)

    clk_freq = est_clock_freq(config, 1e-3)
    assert (
        clk_freq / 1e6 < 50.0
    ), f"Expected master clock frequency < 50 MHz, received {clk_freq / 1e6:.2f} MHz"

    est_clock_freq(config, 1e-3, margin=0.0)
    est_clock_freq(config, 1e-3, margin=1.0)
