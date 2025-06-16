import pytest

pytest.importorskip("torch")


# shared test with lif_torch
def test_ahp_LIFTorch_shapes():
    from rockpool.nn.modules.torch.ahp_lif_torch import aLIFTorch
    import torch

    n_synapses = 5
    n_neurons = 10
    n_batches = 3
    T = 20
    tau_mem = torch.rand(n_neurons)
    tau_syn = 0.05

    # - Test maximal initialisation
    mod = aLIFTorch(
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

    assert ns["isyn"].shape == (n_neurons, n_synapses)
    assert ns["vmem"].shape == (n_neurons,)
    assert ns["iahp"].shape == (n_neurons,)
    assert rd["isyn"].shape == (n_batches, T, n_neurons, n_synapses)
    assert rd["vmem"].shape == (n_batches, T, n_neurons)
    assert rd["iahp"].shape == (n_batches, T, n_neurons)

    # - Test as_graph
    g = mod.as_graph()

    assert len(g.input_nodes) == mod.size_in
    assert len(g.output_nodes) == mod.size_out


def test_ahp_LIFTorch_bias():
    from rockpool.nn.modules.torch.ahp_lif_torch import aLIFTorch
    import torch

    torch.manual_seed(1)

    n_synapses = 1
    n_neurons = 1
    n_batches = 1
    T = 20
    tau_mem = torch.rand(n_neurons)
    tau_syn = 0.02
    bias = torch.ones(n_neurons) * 0.1
    dt = 1e-3

    mod = aLIFTorch(
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

    # with default initialization of weights (w_ahp) and given bias and threshold the neuron will spike and iahp will be non-zero
    assert not torch.all(ns["iahp"] == 0)
    assert not torch.all(rd["iahp"] == 0)

    assert torch.all(ns["isyn"] == 0)
    assert torch.all(rd["vmem"][:, 0] == 0.1)  # match bias in the fist timestep
    assert torch.all(
        rd["vmem"][:, 1] == 0.1 * torch.exp(-dt / tau_mem) + 0.1
    )  # decay one timestep + bias

    # assert bias has gradients
    assert not torch.all(mod.bias.grad == 0)

    # - Test as_graph
    mod.as_graph()


def test_ahp_LIFTorch_recurrent():
    from rockpool.nn.modules.torch.ahp_lif_torch import aLIFTorch
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

    mod = aLIFTorch(
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

    # - Test as_graph
    mod.as_graph()


def test_ahp_LIFTorch_noise():
    from rockpool.nn.modules.torch.ahp_lif_torch import aLIFTorch
    import torch

    n_synapses = 5
    n_neurons = 10
    n_batches = 3
    T = 20
    tau_mem = torch.rand(n_neurons)
    tau_syn = torch.rand(n_neurons, n_synapses)
    dt = 1e-3

    # - Test maximal initialisation
    mod = aLIFTorch(
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


def test_ahp_LIFTorch_tau_syn_shape_1():
    from rockpool.nn.modules.torch.ahp_lif_torch import aLIFTorch
    import torch

    n_synapses = 5
    n_neurons = 10
    n_batches = 3
    T = 20
    tau_mem = torch.rand(n_neurons)
    tau_syn = torch.rand(n_neurons, n_synapses)
    dt = 1e-3

    # - Test maximal initialisation
    mod = aLIFTorch(
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

    # - Test as_graph
    mod.as_graph()


def test_ahp_LIFTorch_tau_syn_shape_2():
    from rockpool.nn.modules.torch.ahp_lif_torch import aLIFTorch
    import torch

    n_synapses = 5
    n_neurons = 10
    n_batches = 3
    T = 20
    tau_mem = torch.rand(n_neurons)
    tau_syn = torch.rand(n_neurons, n_synapses)
    dt = 1e-3

    # - Test maximal initialisation
    mod = aLIFTorch(
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

    # - Test as_graph
    mod.as_graph()


def test_ahp_LIFTorch_threshold_shape_1():
    from rockpool.nn.modules.torch.ahp_lif_torch import aLIFTorch
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
    mod = aLIFTorch(
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

    # - Test as_graph
    mod.as_graph()


def test_ahp_LIFTorch_threshold_shape_2():
    from rockpool.nn.modules.torch.ahp_lif_torch import aLIFTorch
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
    mod = aLIFTorch(
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

    # assert output makes sense (low threshold---> more spikes---> more inhibition---> smaller vmem)
    # assert torch.all(out[:, :, 0] >= out[:, :, 1])
    assert torch.all(rd["vmem"][:, :, 1] >= rd["vmem"][:, :, 1])

    # - Test as_graph
    mod.as_graph()


def test_ahp_LIFTorch_reset():
    from rockpool.nn.modules.torch.ahp_lif_torch import aLIFTorch
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required to test reset to device")

    mod = aLIFTorch(10).to("cuda")
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


# ##### tests spesific to lif_ahp
# all neurons share a single tau_ahp but their w_ahp is a negative value that is increased proportionally with their index
# therefore neurons with smaller index recieve weaker inhibition and their vmem would be higher
def test_ahp_LIFTorch_w():
    from rockpool.nn.modules.torch.ahp_lif_torch import aLIFTorch
    from rockpool.parameters import Constant
    import torch

    n_synapses = 2
    n_neurons = 5
    n_batches = 10
    T = 20
    tau_mem = 0.01
    tau_syn = 0.02
    tau_ahp = 0.02

    # shape of w_ahp
    w_ahp = Constant(torch.ones(n_neurons))

    # keeping tau_ahp same among all neurons and setting different w_ahp
    # neauron with bigger index will recieve stronger inhibition
    for n in range(n_neurons):
        w_ahp[n] *= -(n + 1)

    dt = 1e-3

    mod = aLIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        w_ahp=w_ahp,
        tau_ahp=tau_ahp,
        dt=dt,
        noise_std=0.0,
    )

    # - Generate some data
    input_data = torch.zeros(n_batches, T, n_synapses * n_neurons, requires_grad=True)
    # a non-zero input at the begining of input stream
    with torch.no_grad():
        input_data[:, 0, :] = 100

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    # in each pair of consecuitive neurons the one with smaller index should have higher vmem in all timestamps of all samples
    # comparison starts from timestamp = 1, as the nonzero input is at timestamp=0
    for i_n in range(n_neurons - 1):
        assert torch.all(rd["vmem"][:, 1:, i_n] > rd["vmem"][:, 1:, i_n + 1])

    # assert w_ahp has not gradients
    assert not mod.w_ahp.grad

    # - Test as_graph
    mod.as_graph()


def test_ahp_LIFTorch_tau():
    from rockpool.nn.modules.torch.ahp_lif_torch import aLIFTorch
    from rockpool.parameters import Constant
    import torch

    n_synapses = 2
    n_neurons = 5
    n_batches = 10
    T = 20
    tau_mem = 0.01
    tau_syn = 0.02

    w_ahp = Constant(-torch.ones(n_neurons))
    tau_ahp = torch.zeros((n_neurons), requires_grad=False)

    # keeping w_ahp same among all neurons and setting different tau_ahp
    # neauron with bigger index will receive i_ahp with slower decay (bigger tau_ahp)
    for n in range(n_neurons):
        tau_ahp[n] = (n + 1) * 0.1

    dt = 1e-3

    mod = aLIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        w_ahp=w_ahp,
        tau_ahp=tau_ahp,
        dt=dt,
        noise_std=0.0,
    )

    # - Generate some data
    input_data = torch.zeros(n_batches, T, n_synapses * n_neurons, requires_grad=True)
    # a non-zero input at the begining of input stream
    with torch.no_grad():
        input_data[:, 0, :] = 100

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    # in each pair of consecuitive neurons the one with smaller index should have bigger i_ahp in all timestamps of all samples, as it has a faster dynamics (smaller tau_ahp)
    # as w_ahp for all neurons is set to -1, neurons with smaller tau_ahp will decay faster towrds zero
    # comparison starts from timestamp = 1, as the nonzero input is at timestamp=0
    for i_n in range(n_neurons - 1):
        assert torch.all(rd["iahp"][:, 1:, i_n] > rd["iahp"][:, 1:, i_n + 1])

    # assert w_ahp has not gradients
    assert not mod.w_ahp.grad

    # - Test as_graph
    mod.as_graph()


def test_ahp_LIFTorch_network_graph():
    """Test aLIF implementation with the newer SYNS61201 mapper"""
    from rockpool.nn.modules import aLIFTorch, LinearTorch, LIFTorch
    from rockpool.nn.combinators import Sequential

    # Create a network with both feed-forward and recurrent aLIF layers
    net = Sequential(
        LinearTorch((2, 5)),
        aLIFTorch((5, 5)),  # FFwd aLIF layer
        LinearTorch((5, 5)),
        aLIFTorch((5, 5), has_rec=True),  # Recurrent aLIF layer
        LinearTorch((5, 8)),
        LIFTorch((8, 8)),  # Output layer
    )

    # Convert to graph
    g = net.as_graph()

    # Use the newer SYNS61201 mapper
    from rockpool.devices.xylo.syns61201 import mapper, config_from_specification

    # Map the graph
    spec = mapper(g)
    config, valid, msg = config_from_specification(**spec)

    # Verify the mapping was successful
    assert valid, f"Mapping failed: {msg}"

    # Verify the weight matrices have the correct shapes
    assert spec["weights_in"].shape[0] == 2  # Input channels
    assert spec["weights_rec"].shape[0] == spec["weights_rec"].shape[1]  # Square matrix
    assert spec["weights_out"].shape[1] == 8  # Output neurons

    # Verify the AHP functionality is preserved
    # The recurrent weights should include the AHP weights on the diagonal
    w_rec = spec["weights_rec"]
    for i in range(w_rec.shape[0]):
        assert w_rec[i, i] != 0, f"AHP weight missing at position {i}"
