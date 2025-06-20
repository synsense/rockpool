def test_configure():
    import pytest

    pytest.importorskip("samna")
    pytest.importorskip("xylosim")

    # - Samna imports
    from samna.xyloCore2.configuration import ReservoirNeuron, OutputNeuron
    from samna.xyloCore2.configuration import XyloConfiguration
    from samna.xyloCore2 import validate_configuration
    from rockpool.devices.xylo.syns61201 import XyloSim
    import numpy as np

    # - Build a network
    dt = 1e-3
    Nin = 10
    Nhidden = 3
    Nout = 2

    # - Quantise the network

    # - Build a Xylo configuration
    c = XyloConfiguration()

    w_in = np.ones((Nin, Nhidden), "int")
    c.input.weights = w_in
    c.input.syn2_weights = w_in
    c.synapse2_enable = True

    w_rec = np.zeros((Nhidden, Nhidden), "int")
    c.reservoir.weights = w_rec
    c.reservoir.syn2_weights = w_rec

    hidden_neurons = []
    for _ in range(Nhidden):
        n = ReservoirNeuron()
        n.i_syn_decay = 2
        n.v_mem_decay = 4
        n.threshold = 484
        hidden_neurons.append(n)

    c.reservoir.neurons = hidden_neurons

    w_out = np.ones((Nhidden, Nout), "int")
    c.readout.weights = w_out

    readout_neurons = []
    for _ in range(Nout):
        n = OutputNeuron()
        n.i_syn_decay = 2
        n.v_mem_decay = 4
        n.threshold = 484
        readout_neurons.append(n)

    c.readout.neurons = readout_neurons

    valid, message = validate_configuration(c)
    assert valid, message

    # - Build a simulated Xylo Module
    mod_xylo_sim = XyloSim.from_config(c, dt=dt)

    # - Simulate the evolution of the network on Xylo
    T = 1000
    input_rate = 0.01
    input_raster = np.random.rand(T, Nin) < input_rate
    output_raster, _, _ = mod_xylo_sim(input_raster)

    # - Build a simulated Xylo Module, specifying output mode
    mod_xylo_sim_vmem = XyloSim.from_config(c, dt=dt, output_mode="Vmem")
    mod_xylo_sim_isyn = XyloSim.from_config(c, dt=dt, output_mode="Isyn")
    mod_xylo_sim_spike = XyloSim.from_config(c, dt=dt, output_mode="Spike")

    # - Simulate the evolution of the network on Xylo
    T = 1000
    input_rate = 0.01
    input_raster = np.random.rand(T, Nin) < input_rate
    output_raster_vmem, _, _ = mod_xylo_sim_vmem(input_raster)
    output_raster_isyn, _, _ = mod_xylo_sim_vmem(input_raster)
    output_raster_spike, _, _ = mod_xylo_sim_vmem(input_raster)


def test_specification():
    import pytest

    pytest.importorskip("samna")
    pytest.importorskip("xylosim")

    # - Rockpool imports
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
    mod_xylo_sim_vmem = XyloSim.from_specification(**spec, output_mode="Vmem")
    mod_xylo_sim_isyn = XyloSim.from_specification(**spec, output_mode="Isyn")
    mod_xylo_sim_spike = XyloSim.from_specification(**spec, output_mode="Spike")

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
        "weight_shift_in": 0,
        "weight_shift_rec": 0,
        "weight_shift_out": 0,
        "aliases": None,
    }

    mod_xylo_sim = XyloSim.from_specification(**spec)
    mod_xylo_sim_vmem = XyloSim.from_specification(**spec, output_mode="Vmem")
    mod_xylo_sim_isyn = XyloSim.from_specification(**spec, output_mode="Isyn")
    mod_xylo_sim_spike = XyloSim.from_specification(**spec, output_mode="Spike")

    # - Simulate the evolution of the network on Xylo
    T = 1000
    input_rate = 0.01
    input_raster = np.random.rand(T, Nin) < input_rate
    output_raster, _, _ = mod_xylo_sim(input_raster)


def test_from_config():
    import pytest

    pytest.importorskip("samna")
    pytest.importorskip("xylosim")

    # - Samna imports
    from rockpool.devices.xylo.syns61201 import XyloSim, config_from_specification
    from samna.xyloCore2 import validate_configuration
    import numpy as np

    Nin = 8
    Nhidden = 3
    Nout = 2

    # - Test minimal spec
    spec = {
        "weights_in": np.ones((Nin, Nhidden, 2), "int"),
        "weights_out": np.ones((Nhidden, Nout), "int"),
    }

    c, _, _ = config_from_specification(**spec)
    valid, message = validate_configuration(c)
    assert valid, message

    mod_xylo_sim_vmem = XyloSim.from_config(c, output_mode="Vmem")
    mod_xylo_sim_isyn = XyloSim.from_config(c, output_mode="Isyn")
    mod_xylo_sim_spike = XyloSim.from_config(c, output_mode="Spike")

    mod_xylo_sim = XyloSim.from_config(c)
    mod_xylo_sim.timed()

    # - Simulate the evolution of the network on Xylo
    T = 1000
    input_rate = 0.01
    input_raster = np.random.rand(T, Nin) < input_rate
    output_raster, _, _ = mod_xylo_sim(input_raster)


def test_FF_equality_torch():
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("xylosim")
    import torch
    import numpy as np
    from xylosim.v1 import XyloSynapse, XyloLayer

    quant_scaling = 100
    bitshift = 4

    # - parameter
    n_synapses = 1
    n_neurons = 1
    n_batches = 1
    T = 100
    tau_mem = 0.008
    tau_syn = 0.016
    dt = 1e-3
    threshold = 60 * quant_scaling
    weight = quant_scaling

    # - init LIFTorch
    from rockpool.nn.modules import LIFBitshiftTorch
    from rockpool.nn.modules.torch.lif_bitshift_torch import calc_bitshift_decay

    # - init rockpool neuron
    lif_torch = LIFBitshiftTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        bias=0.0,
        has_rec=False,
        dt=dt,
        noise_std=0.0,
        threshold=threshold,
    )

    # - create Xylo Synapse
    syn = XyloSynapse(target_neuron_id=0, target_synapse_id=0, weight=weight)

    # - calculate corresponding dashes for bitshift decay
    dash_mem = calc_bitshift_decay(torch.Tensor([tau_mem]), dt).item()
    dash_syn = calc_bitshift_decay(torch.Tensor([tau_syn]), dt).item()

    # - init XyloLayer from XyloSim
    lif_xylo = XyloLayer(
        synapses_in=[[syn]],
        synapses_rec=[[]],
        synapses_out=[[]],
        aliases=[[]],
        threshold=[threshold],
        threshold_out=[],
        weight_shift_inp=bitshift,
        weight_shift_rec=0,
        weight_shift_out=0,
        dash_mem=[dash_mem],
        dash_mem_out=[],
        dash_syns=[[dash_syn]],
        dash_syns_out=[],
        name="test",
    )

    # - define one input spike at time 0
    inp = np.zeros((T, 1), int)
    inp[0, 0] = 1

    # - evolve XyloSim
    out_xylo = lif_xylo.evolve(inp)

    # - get recordings
    out_xylo = lif_xylo.rec_recurrent_spikes
    vmem_xylo = lif_xylo.rec_v_mem[0]
    isyn_xylo = lif_xylo.rec_i_syn[0]

    # - weight the spike for rockpool model
    inp[0, 0] = (inp[0, 0] * weight) << bitshift

    # - evolve rockpool model
    out_torch, state, rec = lif_torch.evolve(torch.Tensor(np.array(inp)), record=True)

    # - get recordings
    vmem_torch = rec["vmem"][0, :, 0].detach().int().numpy()
    isyn_torch = rec["isyn"][0, :, 0, 0].detach().int().numpy()
    out_torch = out_torch.detach().int().numpy()[0]

    # - assert similarity on spikes, states and recordings
    assert np.all(out_torch == out_xylo)

    assert np.allclose(
        [(i >> bitshift) / quant_scaling for i in isyn_torch],
        [(i >> bitshift) / quant_scaling for i in isyn_xylo],
        atol=1e-2,
        rtol=1e-2,
    )

    assert np.allclose(
        [(i >> bitshift) / quant_scaling for i in vmem_torch],
        [(i >> bitshift) / quant_scaling for i in vmem_xylo],
        atol=1e-1,
        rtol=1e-1,
    )


def test_Rec_equality_torch():
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("xylosim")
    import torch
    import numpy as np
    from xylosim.v1 import XyloSynapse, XyloLayer

    quant_scaling = 100
    bitshift = 4

    # - parameter
    n_synapses = 1
    n_neurons = 1
    n_batches = 1
    T = 100
    tau_mem = 0.008
    tau_syn = 0.016
    dt = 1e-3
    threshold = 60 * quant_scaling
    weight = int(1.0 * quant_scaling)
    weight_rec = int(0.4 * quant_scaling)

    # - init LIFTorch
    from rockpool.nn.modules import LIFBitshiftTorch
    from rockpool.nn.modules.torch.lif_bitshift_torch import calc_bitshift_decay

    lif_torch = LIFBitshiftTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        bias=0.0,
        has_rec=True,
        w_rec=torch.ones(1, 1) * (weight_rec << bitshift),
        dt=dt,
        noise_std=0.0,
        threshold=threshold,
    )

    syn = XyloSynapse(target_neuron_id=0, target_synapse_id=0, weight=weight)

    syn_rec = XyloSynapse(target_neuron_id=0, target_synapse_id=0, weight=weight_rec)

    dash_mem = calc_bitshift_decay(torch.Tensor([tau_mem]), dt).item()
    dash_syn = calc_bitshift_decay(torch.Tensor([tau_syn]), dt).item()

    lif_xylo = XyloLayer(
        synapses_in=[[syn]],
        synapses_rec=[[syn_rec]],
        synapses_out=[[]],
        aliases=[[]],
        threshold=[threshold],
        threshold_out=[],
        weight_shift_inp=bitshift,
        weight_shift_rec=bitshift,
        weight_shift_out=0,
        dash_mem=[dash_mem],
        dash_mem_out=[],
        dash_syns=[[dash_syn]],
        dash_syns_out=[],
        name="test",
    )

    inp = np.zeros((T, 1), int)
    inp[0, 0] = 1

    out_xylo = lif_xylo.evolve(inp)
    out_xylo = lif_xylo.rec_recurrent_spikes
    vmem_xylo = lif_xylo.rec_v_mem[0]
    isyn_xylo = lif_xylo.rec_i_syn[0]

    inp[0, 0] = (inp[0, 0] * weight) << bitshift

    out_torch, state, rec = lif_torch.evolve(torch.Tensor([inp]), record=True)

    vmem_torch = rec["vmem"][0, :, 0].detach().int().numpy()
    isyn_torch = rec["isyn"][0, :, 0, 0].detach().int().numpy()
    out_torch = out_torch.detach().int().numpy()[0]

    assert np.all(out_torch == out_xylo)

    assert np.allclose(
        [(i >> bitshift) / quant_scaling for i in isyn_torch],
        [(i >> bitshift) / quant_scaling for i in isyn_xylo],
        atol=1e-2,
        rtol=1e-2,
    )

    assert np.allclose(
        [(i >> bitshift) / quant_scaling for i in vmem_torch],
        [(i >> bitshift) / quant_scaling for i in vmem_xylo],
        atol=1e-1,
        rtol=1e-1,
    )


def test_FF_equality_slayer():
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("sinabs.exodus")
    pytest.importorskip("xylosim")
    import torch
    import numpy as np

    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA to continue.")

    quant_scaling = 100
    bitshift = 4

    # - parameter
    n_synapses = 1
    n_neurons = 1
    n_batches = 1
    T = 100
    tau_mem = 0.008
    tau_syn = 0.016
    dt = 1e-3
    threshold = 50 * quant_scaling
    weight = quant_scaling

    # - init LIFTorch
    from rockpool.nn.modules import LIFExodus
    from rockpool.nn.modules.torch.lif_bitshift_torch import calc_bitshift_decay

    dash_mem = calc_bitshift_decay(torch.Tensor([tau_mem]), dt).item()
    dash_syn = calc_bitshift_decay(torch.Tensor([tau_syn]), dt).item()

    alpha_bitshift = 1 - 1 / (2 ** calc_bitshift_decay(torch.Tensor([tau_mem]), dt))
    beta_bitshift = 1 - 1 / (2 ** calc_bitshift_decay(torch.Tensor([tau_syn]), dt))

    tau_mem_slayer = (-dt / torch.log(alpha_bitshift)).item()
    tau_syn_slayer = (-dt / torch.log(beta_bitshift)).item()

    lif_torch = LIFExodus(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem_slayer,
        tau_syn=tau_syn_slayer,
        bias=0.0,
        has_rec=False,
        dt=dt,
        noise_std=0.0,
        threshold=float(threshold),
    ).cuda()

    from xylosim.v1 import XyloSynapse, XyloLayer

    syn = XyloSynapse(target_neuron_id=0, target_synapse_id=0, weight=weight)

    lif_xylo = XyloLayer(
        synapses_in=[[syn]],
        synapses_rec=[[]],
        synapses_out=[[]],
        aliases=[[]],
        threshold=[threshold],
        threshold_out=[],
        weight_shift_inp=bitshift,
        weight_shift_rec=0,
        weight_shift_out=0,
        dash_mem=[dash_mem],
        dash_mem_out=[],
        dash_syns=[[dash_syn]],
        dash_syns_out=[],
        name="test",
    )

    inp = np.zeros((T, 1), int)
    inp[0, 0] = 1

    out_xylo = lif_xylo.evolve(inp)
    out_xylo = lif_xylo.rec_recurrent_spikes
    vmem_xylo = lif_xylo.rec_v_mem[0]
    isyn_xylo = lif_xylo.rec_i_syn[0]

    inp[0, 0] = (inp[0, 0] * weight) << bitshift

    out_torch, state, rec = lif_torch.evolve(torch.Tensor([inp]).cuda(), record=True)

    vmem_torch = rec["vmem"][0, :, 0].detach().cpu().int().numpy()
    isyn_torch = rec["isyn"][0, :, 0, 0].detach().cpu().int().numpy()
    out_torch = out_torch.detach().cpu().int().numpy()[0]

    assert np.all(out_torch == out_xylo)

    assert np.allclose(
        [(i >> bitshift) / quant_scaling for i in isyn_torch],
        [(i >> bitshift) / quant_scaling for i in isyn_xylo],
        atol=1e-2,
        rtol=1e-2,
    )

    assert np.allclose(
        [(i >> bitshift) / quant_scaling for i in vmem_torch],
        [(i >> bitshift) / quant_scaling for i in vmem_xylo],
        atol=1e-1,
        rtol=1e-1,
    )
