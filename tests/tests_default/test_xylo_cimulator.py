def test_imports():
    from rockpool.devices import xylo
    from rockpool.devices.xylo import XyloCim, XyloSamna


def test_configure():
    # - Samna imports
    from samna.xylo.configuration import ReservoirNeuron
    from samna.xylo.configuration import XyloConfiguration
    from samna.xylo import validate_configuration
    from rockpool.devices import xylo
    import numpy as np

    # - Build a network
    dt = 1e-3
    Nin = 10
    Nhidden = 3

    # - Quantise the network

    # - Build a Xylo configuration
    c = XyloConfiguration()

    w = np.ones((Nin, Nhidden), "int")
    print(w.shape)
    c.input.weights = w
    c.input.syn2_weights = w
    c.synapse2_enable = True

    hidden_neurons = []
    for _ in range(Nhidden):
        n = ReservoirNeuron()
        n.i_syn_decay = 2
        n.v_mem_decay = 4
        n.threshold = 484
        hidden_neurons.append(n)

    c.reservoir.neurons = hidden_neurons

    print(validate_configuration(c))

    # - Build a simulated Xylo Module
    mod_cimulator = xylo.XyloCim.from_config(c, dt=dt)

    # - Simulate the evolution of the network on Xylo
    T = 1000
    input_rate = 0.01
    input_raster = np.random.rand(T, Nin) < input_rate
    output_raster, _, _ = mod_cimulator(input_raster)


def test_specification():
    # - Samna imports
    from rockpool.devices import xylo
    import numpy as np

    Nin = 8
    Nhidden = 3
    Nout = 2

    # - Test minimal spec
    spec = {
        "weights_in": np.ones((Nin, Nhidden, 2), "int"),
        "weights_out": np.ones((Nhidden, Nout), "int"),
    }

    mod_cimulator = xylo.XyloCim.from_specification(**spec)

    # - Test complete spec
    spec = {
        "weights_in": np.ones((Nin, Nhidden, 2), "int"),
        "weights_rec": np.ones((Nhidden, Nhidden, 2), "int"),
        "weights_out": np.ones((Nhidden, Nout), "int"),
        "dash_mem": np.ones(Nhidden, "int"),
        "dash_mem_out": np.ones(Nout, "int"),
        "dash_syn": np.ones(Nhidden, "int"),
        "dash_syn_2": np.ones(Nhidden, "int"),
        "dash_syn_out": np.ones(Nout, "int"),
        "threshold": np.zeros(Nhidden, "int"),
        "threshold_out": np.zeros(Nout, "int"),
        "weight_shift_in": 0,
        "weight_shift_rec": 0,
        "weight_shift_out": 0,
        "aliases": None,
    }

    mod_cimulator = xylo.XyloCim.from_specification(**spec)

    # - Simulate the evolution of the network on Xylo
    T = 1000
    input_rate = 0.01
    input_raster = np.random.rand(T, Nin) < input_rate
    output_raster, _, _ = mod_cimulator(input_raster)


def test_from_config():
    # - Samna imports
    from rockpool.devices import xylo
    import numpy as np

    Nin = 8
    Nhidden = 3
    Nout = 2

    # - Test minimal spec
    spec = {
        "weights_in": np.ones((Nin, Nhidden, 2), "int"),
        "weights_out": np.ones((Nhidden, Nout), "int"),
    }

    c, _, _ = xylo.config_from_specification(**spec)
    mod_cimulator = xylo.XyloCim.from_config(c)

    mod_cimulator.timed()
