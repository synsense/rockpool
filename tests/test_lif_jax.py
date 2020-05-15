"""
Test rate-based Euler models in iaf_jax.py
"""

import numpy as np
import pytest


def test_imports():
    from rockpool.layers import RecLIFJax
    from rockpool.layers import RecLIFCurrentInJax
    from rockpool.layers import RecLIFJax_IO
    from rockpool.layers import RecLIFCurrentInJax_IO


def test_RecLIFJax():
    """ Test RecIAFExpJax """
    from rockpool import TSContinuous, TSEvent
    from rockpool.layers import RecLIFJax

    # - Generic parameters
    net_size = 2
    dt = 1e-3

    w_recurrent = 2 * np.random.rand(net_size, net_size) - 1
    bias = 2 * np.random.rand(net_size) - 1
    tau_m = 20e-3 * np.ones(net_size)
    tau_s = 20e-3 * np.ones(net_size)

    # - Layer generation
    fl0 = RecLIFJax(
        w_recurrent=w_recurrent,
        bias=bias,
        noise_std=0.1,
        tau_mem=tau_m,
        tau_syn=tau_s,
        dt=dt,
    )

    # - Input signal
    tsInSp = TSEvent(
        times=np.arange(15) * dt,
        channels=np.ones(15) * (net_size - 1),
        t_start=0.0,
        t_stop=16 * dt,
    )

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.state["Vmem"])
    ts_output = fl0.evolve(tsInSp, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state["Vmem"]).any()

    # - Test TS only evolution
    fl0.reset_all()
    ts_output = fl0.evolve(tsInSp)
    assert fl0.t == 16 * dt

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state["Vmem"]).all()

    # - Test evolution with only duration
    fl0.evolve(duration=1.0)

    # - Test that some errors are caught
    with pytest.raises(AssertionError):
        fl1 = RecLIFJax(
            w_recurrent=np.zeros((3, 2)),
            tau_mem=np.zeros(3),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = RecLIFJax(
            w_recurrent=np.zeros((2, 2)),
            tau_mem=np.zeros(3),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = RecLIFJax(
            w_recurrent=np.zeros((2, 2)),
            tau_mem=np.zeros(2),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )


def test_RecLIFCurrentInJax():
    """ Test RecLIFCurrentInJax """
    from rockpool import TSContinuous
    from rockpool.layers import RecLIFCurrentInJax

    # - Generic parameters
    net_size = 2
    dt = 1e-3

    w_recurrent = 2 * np.random.rand(net_size, net_size) - 1
    bias = 2 * np.random.rand(net_size) - 1
    tau_m = 20e-3 * np.ones(net_size)
    tau_s = 20e-3 * np.ones(net_size)

    # - Layer generation
    fl0 = RecLIFCurrentInJax(
        w_recurrent=w_recurrent,
        bias=bias,
        noise_std=0.1,
        tau_mem=tau_m,
        tau_syn=tau_s,
        dt=dt,
    )

    # - Input signal
    tsInCont = TSContinuous(times=np.arange(100), samples=np.ones((100, net_size)))

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.state["Vmem"])
    ts_output = fl0.evolve(tsInCont, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state["Vmem"]).any()

    # - Test TS only evolution
    fl0.reset_all()
    ts_output = fl0.evolve(tsInCont)
    assert fl0.t == 99

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state["Vmem"]).all()

    # - Test that some errors are caught
    with pytest.raises(AssertionError):
        fl1 = RecLIFCurrentInJax(
            w_recurrent=np.zeros((3, 2)),
            tau_mem=np.zeros(3),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = RecLIFCurrentInJax(
            w_recurrent=np.zeros((2, 2)),
            tau_mem=np.zeros(3),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = RecLIFCurrentInJax(
            w_recurrent=np.zeros((2, 2)),
            tau_mem=np.zeros(2),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )


def test_RecLIFJax_IO():
    """ Test RecLIFJax_IO """
    from rockpool import TSContinuous, TSEvent
    from rockpool.layers import RecLIFJax_IO

    # - Generic parameters
    in_size = 3
    net_size = 2
    out_size = 4
    dt = 1e-3

    w_in = 2 * np.random.rand(in_size, net_size) - 1
    w_recurrent = 2 * np.random.rand(net_size, net_size) - 1
    w_out = 2 * np.random.rand(net_size, out_size) - 1
    bias = 2 * np.random.rand(net_size) - 1
    tau_m = 20e-3 * np.ones(net_size)
    tau_s = 20e-3 * np.ones(net_size)

    # - Layer generation
    fl0 = RecLIFJax_IO(
        w_in=w_in,
        w_recurrent=w_recurrent,
        w_out=w_out,
        bias=bias,
        noise_std=0.1,
        tau_mem=tau_m,
        tau_syn=tau_s,
        dt=dt,
    )

    # - Input signal
    tsInSp = TSEvent(
        times=np.arange(15) * dt,
        channels=np.ones(15) * (in_size-1),
        t_start=0.0,
        t_stop=16 * dt,
    )

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.state["Vmem"])
    ts_output = fl0.evolve(tsInSp, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state["Vmem"]).any()

    # - Test TS only evolution
    fl0.reset_all()
    ts_output = fl0.evolve(tsInSp)
    assert fl0.t == 16 * dt

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state["Vmem"]).all()

    # - Test that some errors are caught
    with pytest.raises(AssertionError):
        fl1 = RecLIFJax_IO(
            w_in=np.zeros((2, 3)),
            w_recurrent=np.zeros((3, 2)),
            w_out=np.zeros((3, 2)),
            tau_mem=np.zeros(3),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = RecLIFJax_IO(
            w_in=np.zeros((2, 3)),
            w_recurrent=np.zeros((2, 2)),
            w_out=np.zeros((3, 2)),
            tau_mem=np.zeros(3),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = RecLIFJax_IO(
            w_in=np.zeros((2, 3)),
            w_recurrent=np.zeros((2, 2)),
            w_out=np.zeros((3, 2)),
            tau_mem=np.zeros(2),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )


def test_RecLIFCurrentInJax_IO():
    """ Test RecLIFJax_IO """
    from rockpool import TSContinuous, TSEvent
    from rockpool.layers import RecLIFCurrentInJax_IO

    # - Generic parameters
    in_size = 3
    net_size = 2
    out_size = 4
    dt = 1e-3

    w_in = 2 * np.random.rand(in_size, net_size) - 1
    w_recurrent = 2 * np.random.rand(net_size, net_size) - 1
    w_out = 2 * np.random.rand(net_size, out_size) - 1
    bias = 2 * np.random.rand(net_size) - 1
    tau_m = 20e-3 * np.ones(net_size)
    tau_s = 20e-3 * np.ones(net_size)

    # - Layer generation
    fl0 = RecLIFCurrentInJax_IO(
        w_in=w_in,
        w_recurrent=w_recurrent,
        w_out=w_out,
        bias=bias,
        noise_std=0.1,
        tau_mem=tau_m,
        tau_syn=tau_s,
        dt=dt,
    )

    # - Input signal
    tsInCont = TSContinuous(
        times=np.arange(15) * dt,
        samples=np.ones((15, in_size)),
        t_start=0.0,
        t_stop=16 * dt,
    )

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.state["Vmem"])
    ts_output = fl0.evolve(tsInCont, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state["Vmem"]).any()

    # - Test TS only evolution
    fl0.reset_all()
    ts_output = fl0.evolve(tsInCont)
    assert fl0.t == 16 * dt

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state["Vmem"]).all()

    # - Test that some errors are caught
    with pytest.raises(AssertionError):
        fl1 = RecLIFCurrentInJax_IO(
            w_in=np.zeros((2, 3)),
            w_recurrent=np.zeros((3, 2)),
            w_out=np.zeros((3, 2)),
            tau_mem=np.zeros(3),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = RecLIFCurrentInJax_IO(
            w_in=np.zeros((2, 3)),
            w_recurrent=np.zeros((2, 2)),
            w_out=np.zeros((3, 2)),
            tau_mem=np.zeros(3),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = RecLIFCurrentInJax_IO(
            w_in=np.zeros((2, 3)),
            w_recurrent=np.zeros((2, 2)),
            w_out=np.zeros((3, 2)),
            tau_mem=np.zeros(2),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )


def test_FFLIFJax_IO():
    """ Test test_FFLIFJax_IO """
    from rockpool import TSContinuous, TSEvent
    from rockpool.layers import FFLIFJax_IO

    # - Generic parameters
    in_size = 5
    net_size = 2
    out_size = 4
    dt = 1e-3

    w_in = 2 * np.random.rand(in_size, net_size) - 1
    w_out = 2 * np.random.rand(net_size, out_size) - 1
    bias = 2 * np.random.rand(net_size) - 1
    tau_m = 20e-3 * np.ones(net_size)
    tau_s = 20e-3 * np.ones(net_size)

    # - Layer generation
    fl0 = FFLIFJax_IO(
        w_in=w_in,
        w_out=w_out,
        bias=bias,
        noise_std=0.1,
        tau_mem=tau_m,
        tau_syn=tau_s,
        dt=dt,
    )

    # - Input signal
    tsInSp = TSEvent(
        times=np.arange(15) * dt,
        channels=np.ones(15) * (in_size-1),
        t_start=0.0,
        t_stop=16 * dt,
    )

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.state["Vmem"])
    ts_output = fl0.evolve(tsInSp, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state["Vmem"]).any()

    # - Test TS only evolution
    fl0.reset_all()
    ts_output = fl0.evolve(tsInSp)
    assert fl0.t == 16 * dt

    print(tsInSp)
    print(tsInSp.raster(dt=dt).shape)

    fl0.reset_all()
    fl0._evolve_functional(fl0._pack(), fl0.state, tsInSp.raster(dt=dt), )

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state["Vmem"]).all()

    # - Test that some errors are caught
    with pytest.raises(AssertionError):
        fl1 = FFLIFJax_IO(
            w_in=np.zeros((2, 3)),
            w_out=np.zeros((3, 2)),
            tau_mem=np.zeros(3),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = FFLIFJax_IO(
            w_in=np.zeros((2, 3)),
            w_out=np.zeros((3, 2)),
            tau_mem=np.zeros(3),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = FFLIFJax_IO(
            w_in=np.zeros((2, 3)),
            w_out=np.zeros((3, 2)),
            tau_mem=np.zeros(2),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )


def test_largescale():
    from rockpool import TSEvent, TSContinuous
    from rockpool.layers import RecLIFCurrentInJax, RecLIFJax, RecLIFJax_IO

    # Numpy
    import numpy as np

    # - Define network
    N = 200
    Nin = 500
    Nout = 1

    tau_mem = 50e-3
    tau_syn = 100e-3
    bias = 0.0

    def rand_params(N, Nin, Nout, tau_mem, tau_syn, bias):
        return {
            "w_in": np.random.rand(Nin, N) - 0.5,
            "w_recurrent": 0.1 * np.random.randn(N, N) / np.sqrt(N),
            "w_out": 2 * np.random.rand(N, Nout) - 1,
            "tau_mem": tau_mem,
            "tau_syn": tau_syn,
            "bias": (np.ones(N) * bias).reshape(N),
        }

    # - Build a random network
    params0 = rand_params(N, Nin, Nout, tau_mem, tau_syn, bias)
    lyrIO = RecLIFJax_IO(**params0)

    # - Define input and target
    numRepeats = 1
    dur_input = 1000e-3
    dt = 1e-3
    T = int(np.round(dur_input / dt))

    timebase = np.linspace(0, T * dt, T)

    trigger = np.atleast_2d(timebase < 50e-3).T

    chirp = np.atleast_2d(np.sin(timebase * 2 * np.pi * (timebase * 10))).T
    target_ts = TSContinuous(timebase, chirp, periodic=True, name="Target")

    spiking_prob = 0.01
    sp_in_ts = np.random.rand(T * numRepeats, Nin) < spiking_prob * trigger
    spikes = np.argwhere(sp_in_ts)
    input_sp_ts = TSEvent(
        timebase[spikes[:, 0]],
        spikes[:, 1],
        name="Input",
        periodic=True,
        t_start=0.0,
        t_stop=dur_input,
    )

    lyrIO.evolve(input_sp_ts)


def test_training_FFLIFJax_IO():
    from rockpool import TSEvent, TSContinuous
    from rockpool.layers import RecLIFCurrentInJax, RecLIFJax, RecLIFJax_IO, FFLIFJax_IO
    import numpy as np

    Nin = 10
    N = 200
    Nout = 3

    tau_mem = 50e-3
    tau_syn = 100e-3
    bias = 0.0
    dt = 1e-3

    def rand_params(N, Nin, Nout, tau_mem, tau_syn, bias):
        return {
            "w_in": (np.random.rand(Nin, N) - 0.5) / Nin,
            "w_out": 2 * np.random.rand(N, Nout) - 1,
            "tau_mem": tau_mem,
            "tau_syn": tau_syn,
            "bias": (np.ones(N) * bias).reshape(N),
        }

    # - Generate a network
    params0 = rand_params(N, Nin, Nout, tau_mem, tau_syn, bias)
    lyrIO = FFLIFJax_IO(**params0, dt=dt)

    # - Define input and target
    numRepeats = 1
    dur_input = 1000e-3
    dt = 1e-3
    T = int(np.round(dur_input / dt))

    timebase = np.linspace(0, T * dt, T)

    trigger = np.atleast_2d(timebase < dur_input).T

    chirp = np.atleast_2d(np.sin(timebase * 2 * np.pi * (timebase * 10))).T
    target_ts = TSContinuous(timebase, chirp, periodic=True, name="Target")

    spiking_prob = 0.01
    sp_in_ts = np.random.rand(T * numRepeats, Nin) < spiking_prob * trigger
    input_sp_ts = TSEvent.from_raster(
        sp_in_ts,
        dt =dt,
        name="Input",
        periodic=True,
        t_start=0,
        t_stop=dur_input,
    )


    # - Simulate initial network state
    lyrIO.randomize_state()
    lyrIO.evolve(input_sp_ts)

    def loss(params, output_t, target_t):
        return 0.

    # - Train
    steps = 3
    for t in range(steps):
        lyrIO.randomize_state()
        l_fcn, g_fcn, o_fcn = lyrIO.train_output_target(
            input_sp_ts, target_ts, is_first=(t == 0), loss_fcn=loss,
        )

        l_fcn()
        g_fcn()
        o_fcn()


def test_training_RecLIFCurrentInJax_IO():
    from rockpool import TSEvent, TSContinuous
    from rockpool.layers import RecLIFCurrentInJax_IO
    import numpy as np

    Nin = 10
    N = 200
    Nout = 3

    tau_mem = 50e-3
    tau_syn = 100e-3
    bias = 0.0
    dt = 1e-3

    def rand_params(N, Nin, Nout, tau_mem, tau_syn, bias):
        return {
            "w_in": (np.random.rand(Nin, N) - 0.5) / Nin,
            "w_recurrent": np.random.randn(N, N),
            "w_out": 2 * np.random.rand(N, Nout) - 1,
            "tau_mem": tau_mem,
            "tau_syn": tau_syn,
            "bias": (np.ones(N) * bias).reshape(N),
        }

    # - Generate a network
    params0 = rand_params(N, Nin, Nout, tau_mem, tau_syn, bias)
    lyrIO = RecLIFCurrentInJax_IO(**params0, dt=dt)

    # - Define input and target
    numRepeats = 1
    dur_input = 1000e-3
    dt = 1e-3
    T = int(np.round(dur_input / dt))

    timebase = np.linspace(0, T * dt, T)

    chirp = np.atleast_2d(np.sin(timebase * 2 * np.pi * (timebase * 10))).T
    target_ts = TSContinuous(timebase, chirp, periodic=True, name="Target")

    tsInCont = TSContinuous.from_clocked(
        np.random.rand(T, Nin),
        dt=dt,
        periodic=True,
    )

    # - Simulate initial network state
    lyrIO.randomize_state()
    lyrIO.evolve(tsInCont)

    def loss(params, output_t, target_t):
        return 0.

    # - Train
    steps = 3
    for t in range(steps):
        lyrIO.randomize_state()
        l_fcn, g_fcn, o_fcn = lyrIO.train_output_target(
            tsInCont, target_ts, is_first=(t == 0), loss_fcn=loss,
        )

        l_fcn()
        g_fcn()
        o_fcn()
