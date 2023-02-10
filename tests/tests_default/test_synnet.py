import pytest

pytest.importorskip("torch")


def test_synnet_import():
    from rockpool.nn.networks import SynNet


def test_synnet_init():
    from rockpool.nn.networks import SynNet
    from rockpool.nn.modules import LIFTorch

    # model init
    model = SynNet(
        n_classes=3,
        n_channels=12,
        size_hidden_layers=[60, 20, 3],
        time_constants_per_layer=[3, 7, 1],
        tau_syn_base=0.031,
        tau_mem=0.01,
        tau_syn_out=0.004,
        threshold=1.3,
        train_threshold=True,
        neuron_model=LIFTorch,
        max_spikes_per_dt=13,
        max_spikes_per_dt_out=2,
        p_dropout=0.2,
        dt=10e-3,
    )


def test_synnet_forward():
    from rockpool.nn.networks import SynNet
    import torch

    n_neurons_input = 6

    model = SynNet(
        n_classes=2,
        n_channels=n_neurons_input,
        size_hidden_layers=[20, 8],
        time_constants_per_layer=[8, 1],
        tau_syn_base=0.0032,
        tau_mem=0.0015,
        tau_syn_out=0.0026,
        threshold=1.0,
        max_spikes_per_dt=15,
        max_spikes_per_dt_out=12,
        p_dropout=0.1,
        dt=1e-3,
    )

    # input params
    n_batches = 3
    T = 20

    # input
    inp = torch.ones(n_batches, T, n_neurons_input) * 100

    # forward
    out, state, rec = model(inp, record=True)
    layer_LIF = [
        label for label in rec.keys() if "LIFTorch" in label and "output" not in label
    ]
    # pick last LIFTorch layer as readout layer
    label_last_LIF = sorted(layer_LIF)[-1]

    assert torch.allclose(out, rec[label_last_LIF]["spikes"])
    assert torch.allclose(
        state["seq"][label_last_LIF]["spikes"], rec[label_last_LIF]["spikes"][0, -1]
    )
    assert torch.allclose(
        state["seq"][label_last_LIF]["vmem"], rec[label_last_LIF]["vmem"][0, -1]
    )


def test_synnet_record():
    from rockpool.nn.networks import SynNet
    import torch

    n_neurons_input = 6

    model = SynNet(
        n_classes=2,
        n_channels=n_neurons_input,
        size_hidden_layers=[20, 8, 8],
        time_constants_per_layer=[8, 1, 3],
        tau_syn_base=0.0032,
        tau_mem=0.0015,
        tau_syn_out=0.0026,
        threshold=1.0,
        max_spikes_per_dt=15,
        max_spikes_per_dt_out=12,
        p_dropout=0.1,
        dt=1e-3,
    )

    # input params
    n_batches = 2
    T = 20

    # input
    inp = torch.ones(n_batches, T, n_neurons_input) * 100

    # forward
    out, state, rec = model(inp)
    assert len(rec) == 0
    assert all([len(d) == 0 for d in model._record_dict.values()])

    model = SynNet(
        n_classes=2,
        n_channels=n_neurons_input,
        size_hidden_layers=[20, 8],
        time_constants_per_layer=[8, 1],
        tau_syn_base=0.0032,
        tau_mem=0.0015,
        tau_syn_out=0.0026,
        threshold=1.0,
        # threshold_out=100,
        max_spikes_per_dt=15,
        max_spikes_per_dt_out=12,
        p_dropout=0.0,
        dt=2 * 1e-3,
    )

    # forward
    out, state, rec = model(inp, record=True)
    print(rec)
    print(model._record_dict)

    assert len(rec) > 0


def test_synnet_backward():
    from rockpool.nn.networks import SynNet
    from rockpool.nn.modules import LIFTorch
    import torch

    # - Ensure deterministic testing
    torch.manual_seed(1)

    n_neurons_input = 6

    model = SynNet(
        n_classes=2,
        n_channels=n_neurons_input,
        size_hidden_layers=[20, 8],
        time_constants_per_layer=[8, 1],
        neuron_model=LIFTorch,
        tau_syn_base=0.032,
        tau_mem=0.075,
        tau_syn_out=0.046,
        threshold=1.0,
        train_threshold=True,
        max_spikes_per_dt=15,
        max_spikes_per_dt_out=12,
        p_dropout=0.1,
        dt=1e-8,
    )

    # input params
    n_batches = 2
    T = 20

    # input
    torch.manual_seed(0)
    inp = torch.rand(n_batches, T, n_neurons_input) * 10
    inp.requires_grad = True

    # forward
    out, state, rec = model(inp, record=True)

    # backward
    out.sum().backward()

    assert not torch.all(inp.grad == 0)
    assert not torch.all(model.seq[0].weight.grad == 0)
    assert not torch.all(model.seq[3].weight.grad == 0)
    assert not torch.all(model.seq[6].weight.grad == 0)

    assert not torch.all(model.seq[1].threshold.grad == 0)
    assert not torch.all(model.seq[4].threshold.grad == 0)


def test_synnet_trainables():
    from rockpool.nn.networks import SynNet
    import torch

    # - Ensure deterministic testing
    torch.manual_seed(1)

    n_neurons_input = 6

    model = SynNet(
        n_classes=2,
        n_channels=n_neurons_input,
        size_hidden_layers=[20, 8],
        time_constants_per_layer=[8, 1],
        tau_syn_base=0.0032,
        tau_mem=0.0015,
        tau_syn_out=0.0026,
        threshold=1.0,
        max_spikes_per_dt=15,
        max_spikes_per_dt_out=12,
        p_dropout=0.1,
        dt=1e-3,
    )

    assert model.seq[0].weight.requires_grad is True
    assert model.seq[3].weight.requires_grad is True
    assert model.seq[6].weight.requires_grad is True

    assert model.seq[1].bias.requires_grad is False
    assert model.seq[4].bias.requires_grad is False
    assert model.seq[7].bias.requires_grad is False

    assert model.seq[1].threshold.requires_grad is False
    assert model.seq[4].threshold.requires_grad is False
    assert model.seq[7].threshold.requires_grad is False

    assert model.seq[1].tau_mem.requires_grad is False
    assert model.seq[4].tau_mem.requires_grad is False
    assert model.seq[7].tau_mem.requires_grad is False

    assert model.seq[1].tau_syn.requires_grad is False
    assert model.seq[4].tau_syn.requires_grad is False
    assert model.seq[7].tau_syn.requires_grad is False

    assert model.seq[1].max_spikes_per_dt.requires_grad is False
    assert model.seq[4].max_spikes_per_dt.requires_grad is False
    assert model.seq[7].max_spikes_per_dt.requires_grad is False


def test_synnet_reset():
    from rockpool.nn.networks import SynNet
    import torch

    # - Ensure deterministic testing
    torch.manual_seed(1)

    n_neurons_input = 6

    model = SynNet(
        n_classes=2,
        n_channels=n_neurons_input,
        size_hidden_layers=[20, 8],
        time_constants_per_layer=[8, 1],
        tau_syn_base=0.0032,
        tau_mem=0.0015,
        tau_syn_out=0.0026,
        threshold=1.0,
        train_threshold=True,
        max_spikes_per_dt=15,
        max_spikes_per_dt_out=12,
        p_dropout=0.1,
        dt=1e-3,
    )

    # input params
    n_batches = 2
    T = 20

    # input
    torch.manual_seed(0)
    inp = torch.rand(n_batches, T, n_neurons_input) * 10

    # forward
    out, state, rec = model(inp)

    # assert that first spk layers state is not reset
    layer_LIF = [
        label
        for label in state["seq"].keys()
        if "LIFTorch" in label and "output" not in label
    ]
    # pick last LIFTorch layer as readout layer
    label_first_LIF = sorted(layer_LIF)[0]
    assert not torch.all(state["seq"][label_first_LIF]["vmem"] == 0)

    model.reset_state()

    # get state
    state = model.state()

    # assert first spk layers state is reset
    assert torch.all(state["seq"][label_first_LIF]["vmem"] == 0)


def test_synnet_graph():
    from rockpool.nn.networks import SynNet
    import torch
    import copy

    # - Ensure deterministic testing
    torch.manual_seed(1)

    n_neurons_input = 6

    model = SynNet(
        n_classes=2,
        n_channels=n_neurons_input,
        size_hidden_layers=[20, 8],
        time_constants_per_layer=[8, 1],
        tau_syn_base=0.0032,
        tau_mem=0.0015,
        tau_syn_out=0.0026,
        threshold=1.0,
        train_threshold=True,
        max_spikes_per_dt=15,
        max_spikes_per_dt_out=12,
        p_dropout=0.1,
        dt=1e-3,
    )

    model.as_graph()
    copy.deepcopy(model.as_graph())


def test_synnet_time_constants():
    import torch
    from rockpool.nn.networks import SynNet
    from rockpool.nn.modules import LIFTorch

    tau_mem = 0.01
    tau_syn_base = 0.003
    tau_syn_out = 0.004
    dt = 1.3 * 1e-3
    size_hidden_layers = [60, 3]
    time_constants_per_layer = [3, 1]

    model = SynNet(
        n_classes=3,
        n_channels=12,
        size_hidden_layers=size_hidden_layers,
        time_constants_per_layer=time_constants_per_layer,
        tau_syn_base=tau_syn_base,
        tau_mem=tau_mem,
        tau_syn_out=tau_syn_out,
        quantize_time_constants=False,
        threshold=1.3,
        train_threshold=True,
        neuron_model=LIFTorch,
        max_spikes_per_dt=13,
        max_spikes_per_dt_out=2,
        p_dropout=0.2,
        dt=dt,
    )

    assert model.seq[1].tau_mem == tau_mem
    assert model.seq[4].tau_mem == tau_mem
    assert model.seq[7].tau_mem == tau_mem

    # check values and frequencies of time constants
    tau_syn_hidden0 = torch.tensor(
        [
            (tau_syn_base / dt) ** j * dt
            for j in range(1, time_constants_per_layer[0] + 1)
        ]
    )
    torch.sort(tau_syn_hidden0)
    tau_unique, frequencies = model.seq[1].tau_syn.unique(return_counts=True)
    assert torch.allclose(tau_unique, tau_syn_hidden0)
    n_tau = time_constants_per_layer[0]
    s = size_hidden_layers[0]
    while n_tau >= 1:
        n = torch.ceil(torch.tensor(s / n_tau))
        assert torch.sum(frequencies[-n_tau]) == n
        n_tau -= n

    tau_syn_hidden1 = torch.tensor(
        [
            (tau_syn_base / dt) ** j * dt
            for j in range(1, time_constants_per_layer[1] + 1)
        ]
    )
    torch.sort(tau_syn_hidden1)
    tau_unique, frequencies = model.seq[4].tau_syn.unique(return_counts=True)
    assert torch.allclose(tau_unique, tau_syn_hidden1)
    n_tau = time_constants_per_layer[1]
    s = size_hidden_layers[1]
    while n_tau >= 1:
        n = torch.ceil(torch.tensor(s / n_tau))
        assert torch.sum(frequencies[-n_tau]) == n
        n_tau -= n

    assert model.seq[7].tau_syn == tau_syn_out


def test_synnet_time_constants_quantized():
    import torch
    from rockpool.nn.networks import SynNet
    from rockpool.nn.modules import LIFTorch
    from rockpool.nn.modules.torch.lif_torch import tau_to_bitshift, bitshift_to_tau

    tau_mem = 0.01
    tau_syn_base = 0.003
    tau_syn_out = 0.004
    dt = 1.3 * 1e-3
    size_hidden_layers = [60, 3]
    time_constants_per_layer = [3, 1]

    model = SynNet(
        n_classes=3,
        n_channels=12,
        size_hidden_layers=size_hidden_layers,
        time_constants_per_layer=time_constants_per_layer,
        tau_syn_base=tau_syn_base,
        tau_mem=tau_mem,
        tau_syn_out=tau_syn_out,
        quantize_time_constants=True,
        threshold=1.3,
        train_threshold=True,
        neuron_model=LIFTorch,
        max_spikes_per_dt=13,
        max_spikes_per_dt_out=2,
        p_dropout=0.2,
        dt=dt,
    )

    tau_mem_bitshift = torch.round(tau_to_bitshift(dt, torch.tensor(tau_mem))[0]).int()
    tau_mem_rounded = bitshift_to_tau(dt, tau_mem_bitshift)[0].item()
    assert model.seq[1].tau_mem == tau_mem_rounded
    assert model.seq[4].tau_mem == tau_mem_rounded
    assert model.seq[7].tau_mem == tau_mem_rounded

    tau_syn_bitshift = torch.round(
        tau_to_bitshift(dt, torch.tensor(tau_syn_base))[0]
    ).int()
    tau_syn_rounded = bitshift_to_tau(dt, tau_syn_bitshift)[0].item()
    assert model.seq[1].tau_syn[0] == tau_syn_rounded
    assert model.seq[4].tau_syn[1] == tau_syn_rounded


def test_synnet_output():
    import torch
    from rockpool.nn.networks import SynNet
    from rockpool.nn.modules import LIFTorch
    from numpy.testing import assert_equal

    tau_mem = 0.01
    tau_syn_base = 0.003
    tau_syn_out = 0.004
    dt = 1.3 * 1e-3
    size_hidden_layers = [60, 3]
    time_constants_per_layer = [3, 1]
    threshold = 100
    n_batches = 3
    T = 20
    n_channels = 12

    model_spikes = SynNet(
        n_classes=3,
        n_channels=n_channels,
        size_hidden_layers=size_hidden_layers,
        time_constants_per_layer=time_constants_per_layer,
        tau_syn_base=tau_syn_base,
        tau_mem=tau_mem,
        tau_syn_out=tau_syn_out,
        quantize_time_constants=False,
        threshold=threshold,
        train_threshold=True,
        neuron_model=LIFTorch,
        max_spikes_per_dt=13,
        max_spikes_per_dt_out=2,
        p_dropout=0.0,
        dt=dt,
    )

    model_vmem = SynNet(
        n_classes=3,
        n_channels=n_channels,
        size_hidden_layers=size_hidden_layers,
        time_constants_per_layer=time_constants_per_layer,
        tau_syn_base=tau_syn_base,
        tau_mem=tau_mem,
        tau_syn_out=tau_syn_out,
        quantize_time_constants=False,
        threshold=threshold,
        train_threshold=True,
        neuron_model=LIFTorch,
        max_spikes_per_dt=13,
        max_spikes_per_dt_out=2,
        p_dropout=0.0,
        dt=dt,
        output="vmem",
    )

    assert_equal(
        model_spikes.seq[-1].threshold.detach().numpy(),
        model_vmem.seq[-1].threshold.detach().numpy(),
    )

    # models need to have the same parameter
    with torch.no_grad():
        model_vmem.seq[0].weight.data = model_spikes.seq[0].weight.data
        model_vmem.seq[1].bias.data = model_spikes.seq[1].bias.data
        model_vmem.seq[3].weight.data = model_spikes.seq[3].weight.data
        model_vmem.seq[4].bias.data = model_spikes.seq[4].bias.data
        model_vmem.seq[6].weight.data = model_spikes.seq[6].weight.data
        model_vmem.seq[7].bias.data = model_spikes.seq[7].bias.data

    inp = torch.ones(n_batches, T, n_channels) * 100
    out_spikes, _, rec_spikes = model_spikes(inp, record=True)
    out_vmem, _, rec_vmem = model_vmem(inp, record=True)

    layer_LIF = [
        label
        for label in rec_spikes.keys()
        if "LIFTorch" in label and "output" not in label
    ]
    # pick last LIFTorch layer as readout layer
    label_last_LIF = sorted(layer_LIF)[-1]

    assert torch.allclose(out_vmem, rec_spikes[label_last_LIF]["vmem"])
    assert torch.allclose(
        rec_vmem[label_last_LIF]["vmem"], rec_spikes[label_last_LIF]["vmem"]
    )
    assert torch.allclose(
        rec_spikes[label_last_LIF]["spikes"],
        torch.zeros_like(rec_spikes[label_last_LIF]["spikes"]),
    )


def test_synnet_graph_extraction_spike_readout():
    """
    test_synnet_graph_extraction checks if the extracted computational graph and the sequential module collection is the same
    The test obtains the graph from the model, copies the structure.
    It checks if the parameters observed in the graph and the parameters given to the model is the same.

    Note :
    * In the early version, the `deepcopy` operation was raising errors. It is mainly related to saving and returning a non-empty record dictionary.
    * The dropout is intentionally excluded from the graph.
    """

    # - Rockpool
    from rockpool.nn.networks import SynNet
    from rockpool.nn.modules import LIFTorch
    from rockpool.nn.modules import LinearTorch
    from rockpool.nn.modules.torch import TimeStepDropout
    from rockpool.graph.utils import bag_graph

    # - External
    import torch
    import numpy as np
    from numpy.testing import assert_equal
    from copy import deepcopy

    # - Implement the paper model
    n_classes = 4
    n_channels = 16
    size_hidden_layers = [24, 24, 24]
    time_constants_per_layer = [2, 4, 8]

    # - Spike read-out
    model_spikes = SynNet(
        n_classes=n_classes,
        n_channels=n_channels,
        size_hidden_layers=size_hidden_layers,
        time_constants_per_layer=time_constants_per_layer,
        output="spikes",
        neuron_model=LIFTorch,
    )

    # - Evolve the network with dummy input making sure that it returns a record dictionary
    out, state, rec = model_spikes(torch.Tensor(np.ones((250, 1))), record=True)
    graph_spikes = deepcopy(model_spikes.as_graph())
    nodes, modules = bag_graph(graph_spikes)

    # - Check if the extracted computational graph and the internal sequential combinator is the same
    i = 0
    for module in model_spikes.seq:
        # - Skip if dropout, trace the extracted modules if not
        if isinstance(module, TimeStepDropout):
            continue
        else:
            module_from_graph = modules[i].computational_module
            i += 1

        # - Check the weights if Linear
        if isinstance(module, LinearTorch):
            assert_equal(
                module.weight.detach().numpy(),
                module_from_graph.weight.detach().numpy(),
            )

        # - Check the time constants, bias and threshold if LIF
        elif isinstance(module, LIFTorch):
            assert_equal(
                module.tau_syn.detach().numpy(),
                module_from_graph.tau_syn.detach().numpy(),
            )
            assert_equal(
                module.tau_mem.detach().numpy(),
                module_from_graph.tau_mem.detach().numpy(),
            )
            assert_equal(
                module.bias.detach().numpy(),
                module_from_graph.bias.detach().numpy(),
            )
            assert_equal(
                module.threshold.detach().numpy(),
                module_from_graph.threshold.detach().numpy(),
            )

        else:
            raise ValueError("Unintended computational model found!")


def test_synnet_graph_extraction_vmem_readout():
    """
    The same test with the ``test_synnet_graph_extraction_spike_readout()`` please check the test above for explanations.
    ONLY the output of the SynNet model is different. `output="vmem"` instead of `output="spikes"`
    """

    # - Rockpool
    from rockpool.nn.networks import SynNet
    from rockpool.nn.modules import LIFTorch
    from rockpool.nn.modules import LinearTorch
    from rockpool.nn.modules.torch import TimeStepDropout
    from rockpool.graph.utils import bag_graph

    # - External
    import torch
    import numpy as np
    from numpy.testing import assert_equal
    from copy import deepcopy

    # - Implement the paper model
    n_classes = 4
    n_channels = 16
    size_hidden_layers = [24, 24, 24]
    time_constants_per_layer = [2, 4, 8]

    # - Spike read-out
    model_spikes = SynNet(
        n_classes=n_classes,
        n_channels=n_channels,
        size_hidden_layers=size_hidden_layers,
        time_constants_per_layer=time_constants_per_layer,
        output="vmem",
        neuron_model=LIFTorch,
    )

    # - Evolve the network with dummy input making sure that it returns a record dictionary
    out, state, rec = model_spikes(torch.Tensor(np.ones((250, 1))), record=True)
    graph_spikes = deepcopy(model_spikes.as_graph())
    nodes, modules = bag_graph(graph_spikes)

    # - Check if the extracted computational graph and the internal sequential combinator is the same
    i = 0
    for module in model_spikes.seq:
        # - Skip if dropout, trace the extracted modules if not
        if isinstance(module, TimeStepDropout):
            continue
        else:
            module_from_graph = modules[i].computational_module
            i += 1

        # - Check the weights if Linear
        if isinstance(module, LinearTorch):
            assert_equal(
                module.weight.detach().numpy(),
                module_from_graph.weight.detach().numpy(),
            )

        # - Check the time constants, bias and threshold if LIF
        elif isinstance(module, LIFTorch):
            assert_equal(
                module.tau_syn.detach().numpy(),
                module_from_graph.tau_syn.detach().numpy(),
            )
            assert_equal(
                module.tau_mem.detach().numpy(),
                module_from_graph.tau_mem.detach().numpy(),
            )
            assert_equal(
                module.bias.detach().numpy(),
                module_from_graph.bias.detach().numpy(),
            )
            assert_equal(
                module.threshold.detach().numpy(),
                module_from_graph.threshold.detach().numpy(),
            )

        else:
            raise ValueError("Unintended computational model found!")
