def test_LIFTorch_shapes():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    n_synapses = 5
    n_neurons = 10
    n_batches = 3
    T = 20
    tau_mem = torch.rand(n_neurons)
    tau_syn = 0.05

    # - Test maximal initialisation
    mod = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        has_bias=False,
        has_rec=False,
        dt=1e-3,
        noise_std=0.0,
        device="cpu",
    )

    # - Generate some data
    input_data = torch.rand(n_batches, T, n_synapses * n_neurons, requires_grad=True)

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    out.sum().backward()

    assert out.shape == (n_batches, T, n_neurons)
    assert ns["Isyn"].shape == (n_neurons, n_synapses)
    assert ns["Vmem"].shape == (n_neurons,)
    assert rd["Isyn"].shape == (n_batches, T, n_neurons, n_synapses)
    assert rd["Vmem"].shape == (n_batches, T, n_neurons)


def test_LIFTorch_bias():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    n_synapses = 5
    n_neurons = 10
    n_batches = 3
    T = 20
    tau_mem = torch.rand(n_neurons)
    tau_syn = 0.02
    bias = torch.ones(n_neurons) * 0.1
    dt = 1e-3

    mod = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        bias=bias,
        has_bias=True,
        dt=dt,
        noise_std=0.0,
        device="cpu",
    )

    # - Generate some data
    input_data = torch.zeros(n_batches, T, n_synapses * n_neurons, requires_grad=True)

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    out.sum().backward()

    assert torch.all(ns["Isyn"] == 0)
    assert torch.all(rd["Isyn"] == 0)
    assert torch.all(rd["Vmem"][:, 0] == 0.1)  # match bias in the fist timestep
    assert torch.all(
        rd["Vmem"][:, 1] == 0.1 * torch.exp(-dt / tau_mem) + 0.1
    )  # decay one timestep + bias

    # bias has gradients
    assert not torch.all(mod.bias == 0)


def test_LIFTorch_recurrent():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    n_synapses = 2
    n_neurons = 5
    n_batches = 1
    T = 20
    tau_mem = 0.01
    tau_syn = 0.02

    # more recurrent input to neurons with higher id
    w_rec = torch.zeros(n_neurons, n_neurons, n_synapses)
    w_rec[0, 1, 0] = 1  # neuron 1, synapse 0
    w_rec[0, 1, 1] = 2  # neuron 1, synapse 1
    w_rec[0, 2, 0] = 3  # neuron 2, synapse 0
    w_rec[0, 2, 1] = 4  # neuron 2, synapse 1
    w_rec[0, 3, 0] = 5  # neuron 3, synapse 0
    w_rec[0, 4, 1] = 6  # neuron 4, synapse 1

    w_rec = w_rec.reshape(n_neurons, n_neurons * n_synapses)

    dt = 1e-3

    mod = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        has_bias=False,
        has_rec=True,
        w_rec=w_rec,
        dt=dt,
        noise_std=0.0,
        device="cpu",
    )

    # - Generate some data
    input_data = torch.zeros(n_batches, T, n_synapses * n_neurons, requires_grad=True)
    with torch.no_grad():
        input_data[:, 0, 0] = 100

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    out.sum().backward()

    # assert neurons are increasingly active (per neuron id)
    assert torch.all(out[:, :, 1] <= out[:, :, 2])
    assert torch.all(out[:, :, 3] <= out[:, :, 4])

    # assert w_rec has gradients
    assert not torch.all(mod.w_rec.grad == 0)


def test_LIFTorch_noise():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    n_synapses = 5
    n_neurons = 10
    n_batches = 3
    T = 20
    tau_mem = torch.rand(n_neurons)
    tau_syn = torch.rand(n_neurons, n_synapses)
    dt = 1e-3

    # - Test maximal initialisation
    mod = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        has_bias=False,
        dt=dt,
        noise_std=0.1,
        device="cpu",
    )

    # - Generate some data
    input_data = torch.zeros(n_batches, T, n_synapses * n_neurons, requires_grad=True)

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    out.sum().backward()

    # no input but vmem not zero due to noise
    assert not torch.all(rd["Vmem"] == 0)


def test_LIFTorch_tau_syn_shape_1():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    n_synapses = 5
    n_neurons = 10
    n_batches = 3
    T = 20
    tau_mem = torch.rand(n_neurons)
    tau_syn = torch.rand(n_neurons, n_synapses)
    dt = 1e-3

    # - Test maximal initialisation
    mod = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        has_bias=False,
        dt=dt,
        noise_std=0.1,
        device="cpu",
    )

    # - Generate some data
    input_data = torch.zeros(n_batches, T, n_synapses * n_neurons, requires_grad=True)

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    out.sum().backward()

    # assert correct shape
    assert mod.tau_syn.shape == (n_neurons, n_synapses)


def test_LIFTorch_tau_syn_shape_2():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    n_synapses = 5
    n_neurons = 10
    n_batches = 3
    T = 20
    tau_mem = torch.rand(n_neurons)
    tau_syn = 0.03
    dt = 1e-3

    # - Test maximal initialisation
    mod = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        has_bias=False,
        dt=dt,
        noise_std=0.1,
        device="cpu",
    )

    # - Generate some data
    input_data = torch.ones(n_batches, T, n_synapses * n_neurons, requires_grad=True)

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    out.sum().backward()

    # assert correct shape
    assert mod.tau_syn.shape == (n_neurons, n_synapses)


def test_LIFTorch_threshold_shape_1():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    n_synapses = 5
    n_neurons = 10
    n_batches = 3
    T = 20
    tau_mem = torch.rand(n_neurons)
    tau_syn = 0.03
    threshold = 0.5
    dt = 1e-3

    # - Test maximal initialisation
    mod = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        has_bias=False,
        dt=dt,
        threshold=threshold,
        noise_std=0.1,
        device="cpu",
    )

    # - Generate some data
    input_data = torch.zeros(n_batches, T, n_synapses * n_neurons, requires_grad=True)

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    out.sum().backward()

    # assert correct shape
    assert mod.threshold.shape == (n_neurons,)


def test_LIFTorch_threshold_shape_2():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    n_synapses = 5
    n_neurons = 2
    n_batches = 3
    T = 20
    tau_mem = torch.ones(n_neurons) * 0.05
    tau_syn = 0.03
    threshold = torch.Tensor([0.5, 1.0])
    dt = 1e-3

    # - Test maximal initialisation
    mod = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        has_bias=False,
        dt=dt,
        threshold=threshold,
        noise_std=0.1,
        device="cpu",
    )

    # - Generate some data
    input_data = torch.ones(n_batches, T, n_synapses * n_neurons, requires_grad=True)

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    out.sum().backward()

    # assert correct shape
    assert mod.threshold.shape == (n_neurons,)

    # assert output makes sense (low threshold produces higher activity)
    assert torch.all(out[:, :, 0] >= out[:, :, 1])
    assert not torch.any(out[:, :, 0] < out[:, :, 1])
