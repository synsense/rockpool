import pytest


def test_imports():
    from rockpool.devices.pollen import (
        PollenCim,
        config_from_specification,
        load_config,
        save_config,
    )


def test_from_specification():
    from rockpool.devices.pollen import PollenCim
    import numpy as np

    Nin = 3
    Nhidden = 5
    Nout = 2

    modCim = PollenCim.from_specification(
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
        dt=1e-3,
    )

    T = 10
    f_rate = 0.1
    input = np.random.rand(T, Nin) < f_rate
    _, _, _ = modCim(input)


def test_from_config():
    from rockpool.devices.pollen import PollenCim, config_from_specification
    import numpy as np

    Nin = 3
    Nhidden = 5
    Nout = 2

    config, _, _ = config_from_specification(
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

    modCim = PollenCim.from_config(config, dt=1e-3)

    T = 10
    f_rate = 0.1
    input = np.random.rand(T, Nin) < f_rate
    _, _, _ = modCim(input)
