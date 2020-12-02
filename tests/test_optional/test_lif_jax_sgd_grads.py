def test_training_FFLIFJax_IO():
    from rockpool import TSEvent, TSContinuous
    from nn.layers import FFLIFJax_IO
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
        dt=dt,
        name="Input",
        periodic=True,
        t_start=0,
        t_stop=dur_input,
    )

    # - Simulate initial network state
    lyrIO.randomize_state()
    lyrIO.evolve(input_sp_ts)

    # - Train
    steps = 3
    for t in range(steps):
        lyrIO.randomize_state()
        loss, grads, o_fcn = lyrIO.train_output_target(
            input_sp_ts,
            target_ts,
            is_first=(t == 0),
            debug_nans=True,
        )

        o_fcn()

    # - Train with batches
    steps = 3
    for t in range(steps):
        lyrIO.randomize_state()
        inp_batch = np.repeat(np.expand_dims(sp_in_ts, 0), 2, axis=0)
        tgt_batch = np.repeat(np.expand_dims(target_ts.samples, 0), 2, axis=0)
        loss, grads, o_fcn = lyrIO.train_output_target(
            inp_batch,
            tgt_batch,
            is_first=(t == 0),
            debug_nans=True,
            batch_axis=0,
        )

        o_fcn()


def test_training_RecLIFJax():
    from rockpool import TSEvent, TSContinuous
    from nn.layers import RecLIFJax
    import numpy as np

    Nin = 10
    N = 200
    Nout = 3

    tau_mem = 50e-3
    tau_syn = 100e-3
    bias = 0.0
    dt = 1e-3

    def rand_params(N, tau_mem, tau_syn, bias):
        return {
            "w_recurrent": np.random.randn(N, N) / N,
            "tau_mem": tau_mem,
            "tau_syn": tau_syn,
            "bias": (np.ones(N) * bias).reshape(N),
        }

    # - Generate a network
    params0 = rand_params(N, tau_mem, tau_syn, bias)
    lyrIO = RecLIFJax(**params0, dt=dt)

    # - Define input and target
    dur_input = 1000e-3
    dt = 1e-3
    T = int(np.round(dur_input / dt))

    timebase = np.arange(0, T) * dt

    chirp = np.atleast_2d(np.sin(timebase * 2 * np.pi * (timebase * 10))).T
    target_ts = TSContinuous(timebase, chirp, periodic=True, name="Target")

    sp_in_ts = np.random.rand(T, N) < 0.1
    tsInSpikes = TSEvent.from_raster(
        sp_in_ts,
        dt=dt,
        periodic=True,
    )

    # - Simulate initial network state
    lyrIO.randomize_state()
    lyrIO.evolve(tsInSpikes)

    # - Train
    steps = 3
    for t in range(steps):
        lyrIO.randomize_state()
        loss, grads, o_fcn = lyrIO.train_output_target(
            tsInSpikes,
            target_ts,
            is_first=(t == 0),
            debug_nans=True,
        )

        o_fcn()

    # - Train with batches
    steps = 3
    for t in range(steps):
        lyrIO.randomize_state()
        inp_batch = np.repeat(np.expand_dims(sp_in_ts, 0), 2, axis=0)
        tgt_batch = np.repeat(np.expand_dims(target_ts.samples, 0), 2, axis=0)
        loss, grads, o_fcn = lyrIO.train_output_target(
            inp_batch,
            tgt_batch,
            is_first=(t == 0),
            debug_nans=True,
            batch_axis=0,
        )

        o_fcn()


def test_training_RecLIFCurrentInJax():
    from rockpool import TSContinuous
    from nn.layers import RecLIFCurrentInJax
    import numpy as np

    Nin = 10
    N = 200
    Nout = 3

    tau_mem = 50e-3
    tau_syn = 100e-3
    bias = 0.0
    dt = 1e-3

    def rand_params(N, tau_mem, tau_syn, bias):
        return {
            "w_recurrent": np.random.randn(N, N) / N,
            "tau_mem": tau_mem,
            "tau_syn": tau_syn,
            "bias": (np.ones(N) * bias).reshape(N),
        }

    # - Generate a network
    params0 = rand_params(N, tau_mem, tau_syn, bias)
    lyrIO = RecLIFCurrentInJax(**params0, dt=dt)

    # - Define input and target
    dur_input = 1000e-3
    dt = 1e-3
    T = int(np.round(dur_input / dt))

    timebase = np.arange(0, T) * dt

    chirp = np.atleast_2d(np.sin(timebase * 2 * np.pi * (timebase * 10))).T
    target_ts = TSContinuous(timebase, chirp, periodic=True, name="Target")

    tsInCont = TSContinuous.from_clocked(
        np.random.rand(T, N),
        dt=dt,
        periodic=True,
    )

    # - Simulate initial network state
    lyrIO.randomize_state()
    lyrIO.evolve(tsInCont)

    # - Train
    steps = 3
    for t in range(steps):
        lyrIO.randomize_state()
        loss, grads, o_fcn = lyrIO.train_output_target(
            tsInCont,
            target_ts,
            is_first=(t == 0),
            debug_nans=True,
        )

        o_fcn()

    # - Train with batches
    steps = 3
    for t in range(steps):
        lyrIO.randomize_state()
        inp_batch = np.repeat(np.expand_dims(tsInCont.samples, 0), 2, axis=0)
        tgt_batch = np.repeat(np.expand_dims(target_ts.samples, 0), 2, axis=0)
        loss, grads, o_fcn = lyrIO.train_output_target(
            inp_batch,
            tgt_batch,
            is_first=(t == 0),
            debug_nans=True,
            batch_axis=0,
        )

        o_fcn()


def test_training_RecLIFCurrentInJax_IO():
    from rockpool import TSContinuous
    from nn.layers import RecLIFCurrentInJax_IO
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
            "w_recurrent": np.random.randn(N, N) / N,
            "w_out": 2 * np.random.rand(N, Nout) - 1,
            "tau_mem": tau_mem,
            "tau_syn": tau_syn,
            "bias": (np.ones(N) * bias).reshape(N),
        }

    # - Generate a network
    params0 = rand_params(N, Nin, Nout, tau_mem, tau_syn, bias)
    lyrIO = RecLIFCurrentInJax_IO(**params0, dt=dt)

    # - Define input and target
    dur_input = 1000e-3
    dt = 1e-3
    T = int(np.round(dur_input / dt))

    timebase = np.arange(0, T) * dt

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

    # - Train
    steps = 3
    for t in range(steps):
        lyrIO.randomize_state()
        loss, grads, o_fcn = lyrIO.train_output_target(
            tsInCont,
            target_ts,
            is_first=(t == 0),
            debug_nans=True,
        )

        o_fcn()

    # - Train with batches
    steps = 3
    for t in range(steps):
        lyrIO.randomize_state()
        inp_batch = np.repeat(np.expand_dims(tsInCont.samples, 0), 2, axis=0)
        tgt_batch = np.repeat(np.expand_dims(target_ts.samples, 0), 2, axis=0)
        loss, grads, o_fcn = lyrIO.train_output_target(
            inp_batch,
            tgt_batch,
            is_first=(t == 0),
            debug_nans=True,
            batch_axis=0,
        )

        o_fcn()


def test_training_RecLIFJax_IO():
    from rockpool import TSEvent, TSContinuous
    from nn.layers import RecLIFJax_IO
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
            "w_recurrent": np.random.randn(N, N) / N,
            "w_out": 2 * np.random.rand(N, Nout) - 1,
            "tau_mem": tau_mem,
            "tau_syn": tau_syn,
            "bias": (np.ones(N) * bias).reshape(N),
        }

    # - Generate a network
    params0 = rand_params(N, Nin, Nout, tau_mem, tau_syn, bias)
    lyrIO = RecLIFJax_IO(**params0, dt=dt)

    # - Define input and target
    dur_input = 1000e-3
    dt = 1e-3
    T = int(np.round(dur_input / dt))

    timebase = np.arange(0, T) * dt

    chirp = np.atleast_2d(np.sin(timebase * 2 * np.pi * (timebase * 10))).T
    target_ts = TSContinuous(timebase, chirp, periodic=True, name="Target")

    ts_inp_sp = np.random.rand(T, Nin) < 0.1
    tsInSpikes = TSEvent.from_raster(
        ts_inp_sp,
        dt=dt,
        periodic=True,
    )

    # - Simulate initial network state
    lyrIO.randomize_state()
    lyrIO.evolve(tsInSpikes)

    # - Train
    steps = 3
    for t in range(steps):
        lyrIO.randomize_state()
        loss, grads, o_fcn = lyrIO.train_output_target(
            tsInSpikes,
            target_ts,
            is_first=(t == 0),
            debug_nans=True,
        )

        o_fcn()

    # - Train with batches
    steps = 3
    for t in range(steps):
        lyrIO.randomize_state()
        inp_batch = np.repeat(np.expand_dims(ts_inp_sp, 0), 2, axis=0)
        tgt_batch = np.repeat(np.expand_dims(target_ts.samples, 0), 2, axis=0)
        loss, grads, o_fcn = lyrIO.train_output_target(
            inp_batch,
            tgt_batch,
            is_first=(t == 0),
            debug_nans=True,
            batch_axis=0,
        )

        o_fcn()


def test_training_FFLIFCurrentInJax_SO():
    from rockpool import TSContinuous
    from nn.layers import FFLIFCurrentInJax_SO
    import numpy as np

    Nin = 10
    N = 200
    Nout = 3

    tau_mem = 50e-3
    tau_syn = 100e-3
    bias = 0.0
    dt = 1e-3

    def rand_params(N, Nin, tau_mem, tau_syn, bias):
        return {
            "w_in": (np.random.rand(Nin, N) - 0.5) / N,
            "tau_mem": tau_mem,
            "tau_syn": tau_syn,
            "bias": (np.ones(N) * bias).reshape(N),
        }

    # - Generate a network
    params0 = rand_params(N, Nin, tau_mem, tau_syn, bias)
    lyrIO = FFLIFCurrentInJax_SO(**params0, dt=dt)

    # - Define input and target
    dur_input = 1000e-3
    dt = 1e-3
    T = int(np.round(dur_input / dt))

    timebase = np.arange(0, T) * dt

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

    # - Train
    steps = 3
    for t in range(steps):
        lyrIO.randomize_state()
        loss, grads, o_fcn = lyrIO.train_output_target(
            tsInCont,
            target_ts,
            is_first=(t == 0),
            debug_nans=True,
        )

        o_fcn()

    # - Train with batches
    steps = 3
    for t in range(steps):
        lyrIO.randomize_state()
        inp_batch = np.repeat(np.expand_dims(tsInCont.samples, 0), 2, axis=0)
        tgt_batch = np.repeat(np.expand_dims(target_ts.samples, 0), 2, axis=0)
        loss, grads, o_fcn = lyrIO.train_output_target(
            inp_batch,
            tgt_batch,
            is_first=(t == 0),
            debug_nans=True,
            batch_axis=0,
        )

        o_fcn()
