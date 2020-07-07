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
        channels=np.ones(15) * in_size,
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


def test_training_FFwd():
    from rockpool import TSEvent, TSContinuous
    from rockpool.layers import RecLIFCurrentInJax, RecLIFJax, RecLIFJax_IO, FFLIFJax_IO
    import numpy as np

    N = 100
    Nin = 100
    Nout = 1

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
    spikes = np.argwhere(sp_in_ts)
    input_sp_ts = TSEvent(
        timebase[spikes[:, 0]],
        spikes[:, 1],
        name="Input",
        periodic=True,
        t_start=0,
        t_stop=dur_input,
    )

    # - Simulate initial network state
    lyrIO.randomize_state()
    lyrIO.evolve(input_sp_ts)

    # - Add training shim
    from rockpool.layers.training import add_shim_lif_jax_sgd

    lyrIO = add_shim_lif_jax_sgd(lyrIO)

    # - Train
    steps = 100
    for t in range(steps):
        lyrIO.randomize_state()
        l_fcn, g_fcn = lyrIO.train_output_target(
            input_sp_ts, target_ts, is_first=(t == 0)
        )

        l_fcn()
        g_fcn()


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


    lyr = FFLIFJax_IO(w_in=w_in,
                      w_out=w_out,
                      tau_mem=tau_mem,
                      tau_syn=tau_syn,
                      bias=bias,
                      noise_std=std_noise,
                      dt=dt,
                      name="test layer")

    
    lyr.save_layer("test.json")

    lyr_loaded = FFLIFJax_IO.load_from_file("test.json")

    assert(np.all(lyr.w_in == lyr_loaded.w_in))
    assert(np.all(lyr.w_out == lyr_loaded.w_out))
    assert(np.all(lyr.tau_mem == lyr_loaded.tau_mem))
    assert(np.all(lyr.tau_syn == lyr_loaded.tau_syn))
    assert(np.all(lyr.bias == lyr_loaded.bias))
    assert(np.all(lyr.noise_std == lyr_loaded.noise_std))
    assert(np.all(lyr.dt == lyr_loaded.dt))
    assert(np.all(lyr.name == lyr_loaded.name))
    assert(np.all(lyr._rng_key == lyr_loaded._rng_key))


    t_spikes = np.arange(0, 0.01, dt)
    channels = np.random.randint(n_inp, size=len(t_spikes)) 
    ts_inp = TSEvent(t_spikes, channels)

    ts_out = lyr.evolve(ts_inp, duration=0.1)
    ts_out_loaded = lyr_loaded.evolve(ts_inp, duration=0.1)

    assert(np.all(ts_out.samples == ts_out_loaded.samples))



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


    lyr = RecLIFCurrentInJax_IO(w_in=w_in,
                                w_recurrent=w_rec,
                                w_out=w_out,
                                tau_mem=tau_mem,
                                tau_syn=tau_syn,
                                bias=bias,
                                noise_std=std_noise,
                                dt=dt,
                                name="test layer")

    
    lyr.save_layer("test.json")

    lyr_loaded = RecLIFCurrentInJax_IO.load_from_file("test.json")

    assert(np.all(lyr.w_in == lyr_loaded.w_in))
    assert(np.all(lyr.w_recurrent == lyr_loaded.w_recurrent))
    assert(np.all(lyr.w_out == lyr_loaded.w_out))
    assert(np.all(lyr.tau_mem == lyr_loaded.tau_mem))
    assert(np.all(lyr.tau_syn == lyr_loaded.tau_syn))
    assert(np.all(lyr.bias == lyr_loaded.bias))
    assert(np.all(lyr.noise_std == lyr_loaded.noise_std))
    assert(np.all(lyr.dt == lyr_loaded.dt))
    assert(np.all(lyr.name == lyr_loaded.name))
    assert(np.all(lyr._rng_key == lyr_loaded._rng_key))



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


    lyr = RecLIFJax_IO(w_in=w_in,
                                w_recurrent=w_rec,
                                w_out=w_out,
                                tau_mem=tau_mem,
                                tau_syn=tau_syn,
                                bias=bias,
                                noise_std=std_noise,
                                dt=dt,
                                name="test layer")

    
    lyr.save_layer("test.json")

    lyr_loaded = RecLIFJax_IO.load_from_file("test.json")

    assert(np.all(lyr.w_in == lyr_loaded.w_in))
    assert(np.all(lyr.w_recurrent == lyr_loaded.w_recurrent))
    assert(np.all(lyr.w_out == lyr_loaded.w_out))
    assert(np.all(lyr.tau_mem == lyr_loaded.tau_mem))
    assert(np.all(lyr.tau_syn == lyr_loaded.tau_syn))
    assert(np.all(lyr.bias == lyr_loaded.bias))
    assert(np.all(lyr.noise_std == lyr_loaded.noise_std))
    assert(np.all(lyr.dt == lyr_loaded.dt))
    assert(np.all(lyr.name == lyr_loaded.name))
    assert(np.all(lyr._rng_key == lyr_loaded._rng_key))


       
def test_save_load_RecLIFCurrentInJax():
    from rockpool.layers import RecLIFCurrentInJax

    n_neurons = 10
    dt = 0.0001

    w_rec = np.random.rand(n_neurons, n_neurons)
    tau_mem = np.random.rand(n_neurons) + 10 * dt
    tau_syn = np.random.rand(n_neurons) + 10 * dt
    bias = np.random.rand(n_neurons)
    std_noise = np.random.rand()


    lyr = RecLIFCurrentInJax(w_recurrent=w_rec,
                    tau_mem=tau_mem,
                    tau_syn=tau_syn,
                    bias=bias,
                    noise_std=std_noise,
                    dt=dt,
                    name="test layer")

    
    lyr.save_layer("test.json")

    lyr_loaded = RecLIFCurrentInJax.load_from_file("test.json")

    assert(np.all(lyr.w_recurrent == lyr_loaded.w_recurrent))
    assert(np.all(lyr.tau_mem == lyr_loaded.tau_mem))
    assert(np.all(lyr.tau_syn == lyr_loaded.tau_syn))
    assert(np.all(lyr.bias == lyr_loaded.bias))
    assert(np.all(lyr.noise_std == lyr_loaded.noise_std))
    assert(np.all(lyr.dt == lyr_loaded.dt))
    assert(np.all(lyr.name == lyr_loaded.name))
    assert(np.all(lyr._rng_key == lyr_loaded._rng_key))



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


    lyr = RecLIFJax(w_recurrent=w_rec,
                    tau_mem=tau_mem,
                    tau_syn=tau_syn,
                    bias=bias,
                    noise_std=std_noise,
                    dt=dt,
                    name="test layer")

    
    lyr.save_layer("test.json")

    lyr_loaded = RecLIFJax.load_from_file("test.json")

    assert(np.all(lyr.w_recurrent == lyr_loaded.w_recurrent))
    assert(np.all(lyr.tau_mem == lyr_loaded.tau_mem))
    assert(np.all(lyr.tau_syn == lyr_loaded.tau_syn))
    assert(np.all(lyr.bias == lyr_loaded.bias))
    assert(np.all(lyr.noise_std == lyr_loaded.noise_std))
    assert(np.all(lyr.dt == lyr_loaded.dt))
    assert(np.all(lyr.name == lyr_loaded.name))
    assert(np.all(lyr._rng_key == lyr_loaded._rng_key))


