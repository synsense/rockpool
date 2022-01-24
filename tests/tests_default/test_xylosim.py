def test_imports():
    from rockpool.devices import xylo
    import rockpool.devices.xylo.xylo_sim
    import rockpool.devices.xylo.xylo_samna
    from rockpool.devices.xylo import XyloSim, XyloSamna


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
    mod_xylo_sim = xylo.XyloSim.from_config(c, dt=dt)

    # - Simulate the evolution of the network on Xylo
    T = 1000
    input_rate = 0.01
    input_raster = np.random.rand(T, Nin) < input_rate
    output_raster, _, _ = mod_xylo_sim(input_raster)


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

    mod_xylo_sim = xylo.XyloSim.from_specification(**spec)

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

    mod_xylo_sim = xylo.XyloSim.from_specification(**spec)

    # - Simulate the evolution of the network on Xylo
    T = 1000
    input_rate = 0.01
    input_raster = np.random.rand(T, Nin) < input_rate
    output_raster, _, _ = mod_xylo_sim(input_raster)


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
    mod_xylo_sim = xylo.XyloSim.from_config(c)

    mod_xylo_sim.timed()



def test_FF_equality_torch():
    import torch
    import numpy as np
    
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
    
    from xylosim.v1 import XyloSynapse, XyloLayer
    
    syn = XyloSynapse(target_neuron_id=0,
                      target_synapse_id=0,
                      weight=weight)
    
    dash_mem = calc_bitshift_decay(torch.Tensor([tau_mem]), dt).item() 
    dash_syn = calc_bitshift_decay(torch.Tensor([tau_syn]), dt).item() 
    
    lif_xylo = XyloLayer(synapses_in = [[syn]], 
                         synapses_rec = [[]], 
                         synapses_out = [[]], 
                         aliases = [[]], 
                         threshold = [threshold], 
                         threshold_out = [], 
                         weight_shift_inp = bitshift, 
                         weight_shift_rec = 0, 
                         weight_shift_out = 0, 
                         dash_mem = [dash_mem], 
                         dash_mem_out = [], 
                         dash_syns = [[dash_syn]], 
                         dash_syns_out = [], 
                         name = "test") 
    
    
    
    inp = np.zeros((T, 1), int)
    inp[0, 0] = 1
    
    out_xylo = lif_xylo.evolve(inp)
    out_xylo = lif_xylo.rec_recurrent_spikes
    vmem_xylo = lif_xylo.rec_v_mem[0]
    isyn_xylo = lif_xylo.rec_i_syn[0]
    
    inp[0, 0] = (inp[0, 0] * weight) << bitshift
    
    out_torch, state, rec = lif_torch.evolve(torch.Tensor([inp]), record=True)
    
    vmem_torch = rec['vmem'][0, :, 0].detach().int().numpy()
    isyn_torch = rec['isyn'][0, :, 0, 0].detach().int().numpy()
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
    
def test_Rec_equality_torch():
    import torch
    import numpy as np
    
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
    
    from xylosim.v1 import XyloSynapse, XyloLayer
    
    
    syn = XyloSynapse(target_neuron_id=0,
                      target_synapse_id=0,
                      weight=weight)
    
    syn_rec = XyloSynapse(target_neuron_id=0,
                          target_synapse_id=0,
                          weight=weight_rec)
    
    dash_mem = calc_bitshift_decay(torch.Tensor([tau_mem]), dt).item() 
    dash_syn = calc_bitshift_decay(torch.Tensor([tau_syn]), dt).item() 
    
    lif_xylo = XyloLayer(synapses_in = [[syn]], 
                         synapses_rec = [[syn_rec]], 
                         synapses_out = [[]], 
                         aliases = [[]], 
                         threshold = [threshold], 
                         threshold_out = [], 
                         weight_shift_inp = bitshift, 
                         weight_shift_rec = bitshift, 
                         weight_shift_out = 0, 
                         dash_mem = [dash_mem], 
                         dash_mem_out = [], 
                         dash_syns = [[dash_syn]], 
                         dash_syns_out = [], 
                         name = "test") 
    
    
    
    inp = np.zeros((T, 1), int)
    inp[0, 0] = 1
    
    out_xylo = lif_xylo.evolve(inp)
    out_xylo = lif_xylo.rec_recurrent_spikes
    vmem_xylo = lif_xylo.rec_v_mem[0]
    isyn_xylo = lif_xylo.rec_i_syn[0]
    
    inp[0, 0] = (inp[0, 0] * weight) << bitshift
    
    out_torch, state, rec = lif_torch.evolve(torch.Tensor([inp]), record=True)
    
    vmem_torch = rec['vmem'][0, :, 0].detach().int().numpy()
    isyn_torch = rec['isyn'][0, :, 0, 0].detach().int().numpy()
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
    import torch
    import numpy as np
    
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
    from rockpool.nn.modules import LIFSlayer, LIFBitshiftTorch
    from rockpool.nn.modules.torch.lif_bitshift_torch import calc_bitshift_decay
    
    dash_mem = calc_bitshift_decay(torch.Tensor([tau_mem]), dt).item() 
    dash_syn = calc_bitshift_decay(torch.Tensor([tau_syn]), dt).item() 
    
    alpha_bitshift =  1 - 1 / (2 ** calc_bitshift_decay(torch.Tensor([tau_mem]), dt))
    beta_bitshift =  1 - 1 / (2 ** calc_bitshift_decay(torch.Tensor([tau_syn]), dt))
    
    tau_mem_slayer = (-dt / torch.log(alpha_bitshift)).item()
    tau_syn_slayer = (-dt / torch.log(beta_bitshift)).item()
    
    lif_torch = LIFSlayer(
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
    
    syn = XyloSynapse(target_neuron_id=0,
                      target_synapse_id=0,
                      weight=weight)
    
    lif_xylo = XyloLayer(synapses_in = [[syn]], 
                         synapses_rec = [[]], 
                         synapses_out = [[]], 
                         aliases = [[]], 
                         threshold = [threshold], 
                         threshold_out = [], 
                         weight_shift_inp = bitshift, 
                         weight_shift_rec = 0, 
                         weight_shift_out = 0, 
                         dash_mem = [dash_mem], 
                         dash_mem_out = [], 
                         dash_syns = [[dash_syn]], 
                         dash_syns_out = [], 
                         name = "test") 
    
    
    
    inp = np.zeros((T, 1), int)
    inp[0, 0] = 1
    
    out_xylo = lif_xylo.evolve(inp)
    out_xylo = lif_xylo.rec_recurrent_spikes
    vmem_xylo = lif_xylo.rec_v_mem[0]
    isyn_xylo = lif_xylo.rec_i_syn[0]
    
    inp[0, 0] = (inp[0, 0] * weight) << bitshift
    
    out_torch, state, rec = lif_torch.evolve(torch.Tensor([inp]).cuda(), record=True)
    
    vmem_torch = rec['vmem'][0, :, 0].detach().cpu().int().numpy()
    isyn_torch = rec['isyn'][0, :, 0, 0].detach().cpu().int().numpy()
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
    
