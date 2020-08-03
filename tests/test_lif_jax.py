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
    from rockpool.layers import FFLIFJax_IO
    from rockpool.layers import FFExpSynCurrentInJax
    from rockpool.layers import FFExpSynJax


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
        channels=np.ones(15) * (in_size - 1),
        t_start=0.0,
        t_stop=16 * dt,
    )

    # - Compare states and time before and after
    output_at_t0 = fl0._get_outputs_from_state(fl0.state)[0]
    vStateBefore = np.copy(fl0.state["Vmem"])
    ts_output = fl0.evolve(tsInSp, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state["Vmem"]).any()
    assert (ts_output(0) == output_at_t0).all()

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
    tsInCont.beyond_range_exception = False

    # - Compare states and time before and after
    output_at_t0 = fl0._get_outputs_from_state(fl0.state)[0]
    vStateBefore = np.copy(fl0.state["Vmem"])
    ts_output = fl0.evolve(tsInCont, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state["Vmem"]).any()
    assert (ts_output(0) == output_at_t0).all()

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
        channels=np.ones(15) * (in_size - 1),
        t_start=0.0,
        t_stop=16 * dt,
    )

    # - Compare states and time before and after
    output_at_t0 = fl0._get_outputs_from_state(fl0.state)[0]
    vStateBefore = np.copy(fl0.state["Vmem"])
    ts_output = fl0.evolve(tsInSp, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state["Vmem"]).any()
    assert (ts_output(0) == output_at_t0).all()

    # - Test TS only evolution
    fl0.reset_all()
    ts_output = fl0.evolve(tsInSp)
    assert fl0.t == 16 * dt

    print(tsInSp)
    print(tsInSp.raster(dt=dt).shape)

    fl0.reset_all()
    fl0._evolve_functional(
        fl0._pack(), fl0.state, tsInSp.raster(dt=dt),
    )

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


def test_FFLIFCurrentInJax_SO():
    """ Test test_FFLIFCurrentInJax_SO """
    from rockpool import TSContinuous, TSEvent
    from rockpool.layers import FFLIFCurrentInJax_SO

    # - Generic parameters
    in_size = 3
    net_size = 2
    dt = 1e-3

    w_in = 2 * np.random.rand(in_size, net_size) - 1
    bias = 2 * np.random.rand(net_size) - 1
    tau_m = 20e-3 * np.ones(net_size)
    tau_s = 20e-3 * np.ones(net_size)

    # - Layer generation
    fl0 = FFLIFCurrentInJax_SO(
        w_in=w_in, bias=bias, noise_std=0.1, tau_mem=tau_m, tau_syn=tau_s, dt=dt,
    )

    # - Input signal
    tsInCont = TSContinuous(
        times=np.arange(15) * dt,
        samples=np.ones((15, in_size)),
        t_start=0.0,
        t_stop=16 * dt,
    )
    tsInCont.beyond_range_exception = False

    # - Compare states and time before and after
    output_at_t0 = fl0._get_outputs_from_state(fl0.state)[0]
    vStateBefore = np.copy(fl0.state["Vmem"])
    ts_output = fl0.evolve(tsInCont, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state["Vmem"]).any()
    assert (ts_output(0) == output_at_t0).all()

    # - Test TS only evolution
    fl0.reset_all()
    ts_output = fl0.evolve(tsInCont)
    assert fl0.t == 16 * dt

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state["Vmem"]).all()

    # - Test that some errors are caught
    with pytest.raises(AssertionError):
        fl1 = FFLIFCurrentInJax_SO(
            w_in=np.zeros((2, 4)),
            tau_mem=np.zeros(3),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = FFLIFCurrentInJax_SO(
            w_in=np.zeros((2, 3)),
            tau_mem=np.zeros(4),
            tau_syn=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(AssertionError):
        fl1 = FFLIFCurrentInJax_SO(
            w_in=np.zeros((2, 3)),
            tau_mem=np.zeros(3),
            tau_syn=np.zeros(4),
            bias=np.zeros(3),
        )


def test_FFExpSynCurrentInJax():
    from rockpool import TSEvent, TSContinuous
    from rockpool.layers import FFExpSynCurrentInJax

    # Numpy
    import numpy as np

    # - Define network
    Nin = 200
    Nout = 5
    tau = 100e-3

    dt = 1e-3

    def rand_params(
        Nin, Nout, tau,
    ):
        return {
            "w_out": 2 * np.random.rand(Nin, Nout) - 1,
            "tau": tau,
        }

    params0 = rand_params(Nin, Nout, tau)
    lyrExpSyn = FFExpSynCurrentInJax(**params0, dt=dt)

    # - Define input
    numRepeats = 1
    dur_input = 1000e-3
    dt = 1e-3
    T = int(np.round(dur_input / dt))

    timebase = np.linspace(0, T * dt, T)
    # chirp = np.atleast_2d(np.sin(timebase * 2 * np.pi * (timebase * 10))).T
    # target_ts = TSContinuous.from_clocked(chirp, dt = dt, periodic=True, name="Target")

    output_at_t0 = np.dot(lyrExpSyn.state["Isyn"], lyrExpSyn._w_out)
    I_in_ts = np.random.rand(T * numRepeats, Nin)
    input_ts = TSContinuous.from_clocked(I_in_ts, dt=dt, periodic=True, name="Input")
    ts_output = lyrExpSyn.evolve(input_ts)
    assert (ts_output(0) == output_at_t0).all()


def test_FFExpSynJax():
    from rockpool import TSEvent, TSContinuous
    from rockpool.layers import FFExpSynJax

    # Numpy
    import numpy as np

    # - Define network
    Nin = 200
    Nout = 5
    tau = 100e-3

    dt = 1e-3

    def rand_params(
        Nin, Nout, tau,
    ):
        return {
            "w_out": 2 * np.random.rand(Nin, Nout) - 1,
            "tau": tau,
        }

    params0 = rand_params(Nin, Nout, tau)
    lyrExpSyn = FFExpSynJax(**params0, dt=dt)

    # - Define input
    numRepeats = 1
    dur_input = 1000e-3
    dt = 1e-3
    T = int(np.round(dur_input / dt))

    spike_prob = 0.1
    input_ts = TSEvent.from_raster(
        np.random.rand(T * numRepeats, Nin) < spike_prob,
        dt=dt,
        periodic=True,
        name="Input",
    )
    output_at_t0 = np.dot(lyrExpSyn.state["Isyn"], lyrExpSyn._w_out)
    ts_output = lyrExpSyn.evolve(input_ts)
    assert (ts_output(0) == output_at_t0).all()


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


def test_save_load_FFLIFJax_IO():
    from rockpool.layers import FFLIFJax_IO
    from rockpool.timeseries import TSEvent

    n_inp = 4
    n_neurons = 10
    n_out = 2
    dt = 0.0001

    w_in = np.random.rand(n_inp, n_neurons)
    w_out = np.random.rand(n_neurons, n_out)
    tau_mem = np.random.rand(n_neurons) + 10 * dt
    tau_syn = np.random.rand(n_neurons) + 10 * dt
    bias = np.random.rand(n_neurons)
    std_noise = np.random.rand()

    lyr = FFLIFJax_IO(
        w_in=w_in,
        w_out=w_out,
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        bias=bias,
        noise_std=std_noise,
        dt=dt,
        name="test layer",
    )

    lyr.save_layer("test.json")

    lyr_loaded = FFLIFJax_IO.load_from_file("test.json")

    assert np.all(lyr.w_in == lyr_loaded.w_in)
    assert np.all(lyr.w_out == lyr_loaded.w_out)
    assert np.all(lyr.tau_mem == lyr_loaded.tau_mem)
    assert np.all(lyr.tau_syn == lyr_loaded.tau_syn)
    assert np.all(lyr.bias == lyr_loaded.bias)
    assert np.all(lyr.noise_std == lyr_loaded.noise_std)
    assert np.all(lyr.dt == lyr_loaded.dt)
    assert np.all(lyr.name == lyr_loaded.name)
    assert np.all(lyr._rng_key == lyr_loaded._rng_key)

    t_spikes = np.arange(0, 0.01, dt)
    channels = np.random.randint(n_inp, size=len(t_spikes))
    ts_inp = TSEvent(t_spikes, channels)

    ts_out = lyr.evolve(ts_inp, duration=0.1)
    ts_out_loaded = lyr_loaded.evolve(ts_inp, duration=0.1)

    assert np.all(ts_out.samples == ts_out_loaded.samples)


def test_save_load_RecLIFCurrentInJax_IO():
    from rockpool.layers import RecLIFCurrentInJax_IO

    n_inp = 4
    n_neurons = 10
    n_out = 2
    dt = 0.0001

    w_in = np.random.rand(n_inp, n_neurons)
    w_rec = np.random.rand(n_neurons, n_neurons)
    w_out = np.random.rand(n_neurons, n_out)
    tau_mem = np.random.rand(n_neurons) + 10 * dt
    tau_syn = np.random.rand(n_neurons) + 10 * dt
    bias = np.random.rand(n_neurons)
    std_noise = np.random.rand()

    lyr = RecLIFCurrentInJax_IO(
        w_in=w_in,
        w_recurrent=w_rec,
        w_out=w_out,
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        bias=bias,
        noise_std=std_noise,
        dt=dt,
        name="test layer",
    )

    lyr.save_layer("test.json")

    lyr_loaded = RecLIFCurrentInJax_IO.load_from_file("test.json")

    assert np.all(lyr.w_in == lyr_loaded.w_in)
    assert np.all(lyr.w_recurrent == lyr_loaded.w_recurrent)
    assert np.all(lyr.w_out == lyr_loaded.w_out)
    assert np.all(lyr.tau_mem == lyr_loaded.tau_mem)
    assert np.all(lyr.tau_syn == lyr_loaded.tau_syn)
    assert np.all(lyr.bias == lyr_loaded.bias)
    assert np.all(lyr.noise_std == lyr_loaded.noise_std)
    assert np.all(lyr.dt == lyr_loaded.dt)
    assert np.all(lyr.name == lyr_loaded.name)
    assert np.all(lyr._rng_key == lyr_loaded._rng_key)


def test_save_load_RecLIFJax_IO():
    from rockpool.layers import RecLIFJax_IO

    n_inp = 4
    n_neurons = 10
    n_out = 2
    dt = 0.0001

    w_in = np.random.rand(n_inp, n_neurons)
    w_rec = np.random.rand(n_neurons, n_neurons)
    w_out = np.random.rand(n_neurons, n_out)
    tau_mem = np.random.rand(n_neurons) + 10 * dt
    tau_syn = np.random.rand(n_neurons) + 10 * dt
    bias = np.random.rand(n_neurons)
    std_noise = np.random.rand()

    lyr = RecLIFJax_IO(
        w_in=w_in,
        w_recurrent=w_rec,
        w_out=w_out,
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        bias=bias,
        noise_std=std_noise,
        dt=dt,
        name="test layer",
    )

    lyr.save_layer("test.json")

    lyr_loaded = RecLIFJax_IO.load_from_file("test.json")

    assert np.all(lyr.w_in == lyr_loaded.w_in)
    assert np.all(lyr.w_recurrent == lyr_loaded.w_recurrent)
    assert np.all(lyr.w_out == lyr_loaded.w_out)
    assert np.all(lyr.tau_mem == lyr_loaded.tau_mem)
    assert np.all(lyr.tau_syn == lyr_loaded.tau_syn)
    assert np.all(lyr.bias == lyr_loaded.bias)
    assert np.all(lyr.noise_std == lyr_loaded.noise_std)
    assert np.all(lyr.dt == lyr_loaded.dt)
    assert np.all(lyr.name == lyr_loaded.name)
    assert np.all(lyr._rng_key == lyr_loaded._rng_key)


def test_save_load_RecLIFCurrentInJax():
    from rockpool.layers import RecLIFCurrentInJax

    n_neurons = 10
    dt = 0.0001

    w_rec = np.random.rand(n_neurons, n_neurons)
    tau_mem = np.random.rand(n_neurons) + 10 * dt
    tau_syn = np.random.rand(n_neurons) + 10 * dt
    bias = np.random.rand(n_neurons)
    std_noise = np.random.rand()

    lyr = RecLIFCurrentInJax(
        w_recurrent=w_rec,
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        bias=bias,
        noise_std=std_noise,
        dt=dt,
        name="test layer",
    )

    lyr.save_layer("test.json")

    lyr_loaded = RecLIFCurrentInJax.load_from_file("test.json")

    assert np.all(lyr.w_recurrent == lyr_loaded.w_recurrent)
    assert np.all(lyr.tau_mem == lyr_loaded.tau_mem)
    assert np.all(lyr.tau_syn == lyr_loaded.tau_syn)
    assert np.all(lyr.bias == lyr_loaded.bias)
    assert np.all(lyr.noise_std == lyr_loaded.noise_std)
    assert np.all(lyr.dt == lyr_loaded.dt)
    assert np.all(lyr.name == lyr_loaded.name)
    assert np.all(lyr._rng_key == lyr_loaded._rng_key)


def test_save_load_RecLIFJax():
    from rockpool.layers import RecLIFJax
    import numpy as np

    n_neurons = 10
    dt = 0.0001

    w_rec = np.random.rand(n_neurons, n_neurons)
    tau_mem = np.random.rand(n_neurons) + 10 * dt
    tau_syn = np.random.rand(n_neurons) + 10 * dt
    bias = np.random.rand(n_neurons)
    std_noise = np.random.rand()

    lyr = RecLIFJax(
        w_recurrent=w_rec,
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        bias=bias,
        noise_std=std_noise,
        dt=dt,
        name="test layer",
    )

    lyr.save_layer("test.json")

    lyr_loaded = RecLIFJax.load_from_file("test.json")

    assert np.all(lyr.w_recurrent == lyr_loaded.w_recurrent)
    assert np.all(lyr.tau_mem == lyr_loaded.tau_mem)
    assert np.all(lyr.tau_syn == lyr_loaded.tau_syn)
    assert np.all(lyr.bias == lyr_loaded.bias)
    assert np.all(lyr.noise_std == lyr_loaded.noise_std)
    assert np.all(lyr.dt == lyr_loaded.dt)
    assert np.all(lyr.name == lyr_loaded.name)
    assert np.all(lyr._rng_key == lyr_loaded._rng_key)


def test_grads_FFLIFJax_IO():
    from rockpool.layers import FFLIFJax_IO
    from rockpool.timeseries import TSEvent, TSContinuous

    lyr = FFLIFJax_IO(
        w_in=np.array([[3, 4, 5], [7, 8, 9]]),
        w_out=np.array([[1, 2, 3], [4, 5, 6]]).T,
        tau_mem=100e-3,
        tau_syn=200e-3,
        dt=1e-3,
    )

    input_sp_ts = TSEvent.from_raster(np.array([[1, 0, 0, 1], [1, 1, 0, 0]]), dt=1e-3,)

    lyr.evolve(input_sp_ts, num_timesteps=2)

    # - Known-value test
    assert np.allclose(
        lyr.i_syn_last_evolution.samples[-2:, :],
        [[3.0, 4.0, 5.0], [12.9850378, 15.98005009, 18.97506142],],
    )
    assert np.allclose(lyr.spikes_last_evolution.channels, [])
    assert np.allclose(lyr.spikes_last_evolution.times, [])
    assert np.allclose(
        lyr.surrogate_last_evolution.samples[-2:, :],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    assert np.allclose(
        lyr.v_mem_last_evolution.samples[-2:, :],
        [
            [-0.96999997, -0.95999998, -0.94999999],
            [-0.84044957, -0.80059946, -0.76074934],
        ],
    )

    # - Rasterise input and target for control over training
    inps = input_sp_ts.raster(dt=1e-3, channels=np.array([0, 1]))
    target_ts = TSContinuous.from_clocked(np.array([[1, 2, 3], [2, 3, 1]]).T, dt=1e-3,)
    target = target_ts([0, 1e-3])

    # - Perform one sample of SGD training
    loss, grads, o_fcn = lyr.train_output_target(inps, target)

    # - Known-value test
    assert np.allclose(loss, 10.624139, rtol=1e-4)
    assert np.allclose(
        grads["bias"], [-0.00681818, -0.00573616, -0.15497251], rtol=1e-4
    )
    assert np.allclose(grads["tau_mem"], [1.2602061, 1.3096639, 47.103264], rtol=1e-4)
    assert np.allclose(
        grads["tau_syn"], [-0.00295216, -0.00303672, -0.11853314], rtol=1e-4
    )
    assert np.allclose(
        grads["w_in"],
        [[0.09129014, 0.1260704, -0.06462787], [0.23143218, 0.26513225, 0.22329542]],
        rtol=1e-4,
    )
    assert np.allclose(
        grads["w_out"],
        [[0.03333307, 0.13333295], [0.06659944, 0.16656671], [0.08346391, 0.17541301]],
        rtol=1e-4,
    )
    assert np.allclose(grads["w_recurrent"], 0.0, rtol=1e-4)
