import pytest

pytest.importorskip("scipy")
pytest.importorskip("samna")
pytest.importorskip("xylosim")


def test_imports():
    pass


def test_init():
    from rockpool.devices.xylo import AFESim

    afe = AFESim()


def test_evolve():
    from rockpool.devices.xylo import AFESim
    from rockpool.timeseries import TSContinuous
    import numpy as np

    Q = 5
    fc1 = 100.0
    f_factor = 1.325
    thr_up = 1.0
    leakage = 5.0
    LNA_gain = 0.0
    fs = 48000
    digital_counter = 8
    num_filters: int = 16
    manual_scaling = None
    add_noise = True
    seed = 1

    afe = AFESim(
        shape=num_filters,
        Q=Q,
        fc1=fc1,
        f_factor=f_factor,
        thr_up=thr_up,
        leakage=leakage,
        LNA_gain=LNA_gain,
        fs=fs,
        digital_counter=digital_counter,
        manual_scaling=manual_scaling,
        add_noise=add_noise,
        seed=seed,
    ).timed()

    # create chirp
    T = 1
    f0 = 1
    f1 = 7800.0
    dt = 1 / fs
    c = (f0 + f1) / T
    p0 = 0
    time = np.arange(0, T, dt)
    inp = np.sin(p0 + 2 * np.pi * ((c / 2) * time**2 + f0 * time))

    ts_inp = TSContinuous.from_clocked(inp, dt=dt, t_start=0)
    out, state, rec = afe.evolve(ts_inp)

    assert out.raster(dt).shape == (T * fs, num_filters)


def test_zero_input():
    from rockpool.devices.xylo import AFESim
    import numpy as np

    T = 1000
    Nout = 16
    afe = AFESim(Nout)
    out, state, rec = afe(np.zeros((T, Nout)))

    assert out.shape == (T, Nout)
