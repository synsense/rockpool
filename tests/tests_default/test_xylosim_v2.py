def test_imports_V2():
    from rockpool.devices.xylo.syns61201 import XyloSim, mapper


def test_specification_V2():
    # - Samna imports
    from rockpool.devices.xylo.syns61201 import XyloSim

    import numpy as np

    Nin = 8
    Nhidden = 3
    Nout = 2

    # - Test minimal spec
    spec = {
        "weights_in": np.ones((Nin, Nhidden, 2), "int"),
        "weights_out": np.ones((Nhidden, Nout), "int"),
    }

    mod_xylo_sim = XyloSim.from_specification(**spec)

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
        "threshold": np.ones(Nhidden, "int"),
        "threshold_out": np.ones(Nout, "int"),
        "bias": np.ones(Nhidden, "int"),
        "bias_out": np.ones(Nout, "int"),
        "weight_shift_in": 0,
        "weight_shift_rec": 0,
        "weight_shift_out": 0,
        "aliases": None,
    }

    mod_xylo_sim_vmem = XyloSim.from_specification(**spec, output_mode="Vmem")
    mod_xylo_sim_isyn = XyloSim.from_specification(**spec, output_mode="Isyn")
    mod_xylo_sim_spike = XyloSim.from_specification(**spec, output_mode="Spike")

    # - Simulate the evolution of the network on Xylo
    T = 1000
    input_rate = 0.01
    input_raster = np.random.rand(T, Nin) < input_rate
    output_raster_vmem, _, _ = mod_xylo_sim_vmem(input_raster)
    output_raster_isyn, _, _ = mod_xylo_sim_isyn(input_raster)
    output_raster_spike, _, _ = mod_xylo_sim_spike(input_raster)
