import pytest

pytest.importorskip("torch")


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
        has_rec=False,
        dt=1e-3,
        noise_std=0.0,
    )

    # - Generate some data
    input_data = torch.rand(n_batches, T, n_synapses * n_neurons, requires_grad=True)

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    out.sum().backward()

    assert out.shape == (n_batches, T, n_neurons)
    assert ns["isyn"].shape == (n_neurons, n_synapses)
    assert ns["vmem"].shape == (n_neurons,)
    assert rd["isyn"].shape == (n_batches, T, n_neurons, n_synapses)
    assert rd["vmem"].shape == (n_batches, T, n_neurons)


def test_LIFTorch_bias():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    n_synapses = 1
    n_neurons = 1
    n_batches = 1
    T = 20
    tau_mem = torch.rand(n_neurons)
    tau_syn = 0.02
    bias = torch.ones(n_neurons) * 0.1
    dt = 1e-3

    mod = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        threshold=1.0,
        bias=bias,
        dt=dt,
        noise_std=0.0,
    )

    # - Generate some data
    input_data = torch.zeros(n_batches, T, n_synapses * n_neurons, requires_grad=True)

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    out.sum().backward()

    assert torch.all(ns["isyn"] == 0)
    assert torch.all(rd["isyn"] == 0)
    assert torch.all(rd["vmem"][:, 0] == 0.1)  # match bias in the fist timestep
    assert torch.all(
        rd["vmem"][:, 1] == 0.1 * torch.exp(-dt / tau_mem) + 0.1
    )  # decay one timestep + bias

    # assert bias has gradients
    assert not torch.all(mod.bias.grad == 0)


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
        has_rec=True,
        w_rec=w_rec,
        dt=dt,
        noise_std=0.0,
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
        dt=dt,
        noise_std=0.1,
    )

    # - Generate some data
    input_data = torch.zeros(n_batches, T, n_synapses * n_neurons, requires_grad=True)

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    out.sum().backward()

    # no input but vmem not zero due to noise
    assert not torch.all(rd["vmem"] == 0)


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
        dt=dt,
        noise_std=0.1,
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
    tau_syn = torch.rand(n_neurons, n_synapses)
    dt = 1e-3

    # - Test maximal initialisation
    mod = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        dt=dt,
        noise_std=0.1,
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
    threshold = 0.5 * torch.ones(n_neurons)
    dt = 1e-3

    # - Test maximal initialisation
    mod = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        dt=dt,
        threshold=threshold,
        noise_std=0.1,
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

    n_synapses = 1
    n_neurons = 2
    n_batches = 1
    T = 20
    tau_mem = torch.ones(n_neurons) * 0.05
    tau_syn = 0.03
    threshold = torch.Tensor([0.1, 10.0])
    dt = 1e-3

    # - Test maximal initialisation
    mod = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        dt=dt,
        threshold=threshold,
        noise_std=0.0,
    )

    # - Generate some data
    input_data = 2 * torch.ones(
        n_batches, T, n_synapses * n_neurons, requires_grad=True
    )

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    out.sum().backward()

    # assert correct shape
    assert mod.threshold.shape == (n_neurons,)

    print(out[:, :, 0] - out[:, :, 1])
    print(out)

    # assert output makes sense (low threshold produces higher activity)
    assert torch.all(out[:, :, 0] >= out[:, :, 1])


def test_LIFTorch_reset():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required to test reset to device")

    mod = LIFTorch(10).to("cuda")
    device = mod.isyn.device
    print(device)

    mod.reset_state()
    assert mod.isyn.device == device
    assert mod.vmem.device == device

    mod.reset_parameters()
    assert mod.tau_syn.device == device
    assert mod.tau_mem.device == device
    assert mod.threshold.device == device
    assert mod.bias.device == device


def test_LIFTorch_tc_training():
    from rockpool.nn.modules.torch import LIFTorch
    from rockpool.parameters import Constant

    mod = LIFTorch(1, bias=Constant(0), threshold=Constant(1.0))
    assert "tau_mem" in mod.parameters().keys()
    assert "tau_syn" in mod.parameters().keys()


def test_LIFTorch_decay_training():
    from rockpool.nn.modules.torch import LIFTorch
    from rockpool.parameters import Constant

    mod = LIFTorch(1, bias=Constant(0), threshold=Constant(1.0), leak_mode="decays")
    assert "alpha" in mod.parameters().keys()
    assert "beta" in mod.parameters().keys()

    mod = LIFTorch(1, bias=Constant(0), threshold=Constant(1.0), leak_mode="taus")
    assert "alpha" not in mod.parameters().keys()
    assert "beta" not in mod.parameters().keys()


def test_LIFTorch_bitshift_training():
    from rockpool.nn.modules.torch import LIFTorch
    from rockpool.parameters import Constant

    mod = LIFTorch(
        1,
        bias=Constant(0),
        threshold=Constant(1.0),
        leak_mode="bitshifts",
    )

    assert "dash_mem" in mod.parameters().keys()
    assert "dash_syn" in mod.parameters().keys()

    mod = LIFTorch(
        1,
        bias=Constant(0),
        dash_mem=Constant(0),
        dash_syn=Constant(0.02),
        threshold=Constant(1.0),
        leak_mode="bitshifts",
    )

    assert mod.parameters() == {}
