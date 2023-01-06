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
        tau_syn_base=3,
        tau_mem=10,
        tau_syn_out=4,
        threshold=1.3,
        threshold_out=7.2,
        train_threshold=True,
        neuron_model=LIFTorch,
        max_spikes_per_dt=13,
        max_spikes_per_dt_out=2,
        p_dropout=0.2,
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
        tau_syn_base=3.2,
        tau_mem=1.5,
        tau_syn_out=2.6,
        threshold=1.0,
        threshold_out=0.001,
        max_spikes_per_dt=15,
        max_spikes_per_dt_out=12,
        p_dropout=0.1,
    )

    # input params
    n_batches = 3
    T = 20

    # input
    inp = torch.ones(n_batches, T, n_neurons_input) * 100

    # forward
    out, state, rec = model(inp, record=True)

    assert torch.allclose(out, rec["spk_out"]["spikes"])
    assert torch.allclose(state["spk_out"]['spikes'], rec["spk_out"]["spikes"][0, -1])
    assert torch.allclose(state["spk_out"]['vmem'], rec["spk_out"]["vmem"][0, -1])


def test_synnet_record():
    from rockpool.nn.networks import SynNet
    import torch

    n_neurons_input = 6

    model = SynNet(
        n_classes=2,
        n_channels=n_neurons_input,
        size_hidden_layers=[20, 8, 8],
        time_constants_per_layer=[8, 1, 3],
        tau_syn_base=3.2,
        tau_mem=1.5,
        tau_syn_out=2.6,
        threshold=1.0,
        threshold_out=100,
        max_spikes_per_dt=15,
        max_spikes_per_dt_out=12,
        p_dropout=0.1,
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
        tau_syn_base=3.2,
        tau_mem=1.5,
        tau_syn_out=2.6,
        threshold=1.0,
        threshold_out=100,
        max_spikes_per_dt=15,
        max_spikes_per_dt_out=12,
        p_dropout=0.0,
    )

    # forward
    out, state, rec = model(inp, record=True)
    print(rec)
    print(model._record_dict)

    assert len(rec) > 0


def test_synnet_backward():
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
        tau_syn_base=3.2,
        tau_mem=1.5,
        tau_syn_out=2.6,
        threshold=1.0,
        threshold_out=1.0,
        train_threshold=True,
        max_spikes_per_dt=15,
        max_spikes_per_dt_out=12,
        p_dropout=0.1,
    )

    # input params
    n_batches = 2
    T = 20

    # input
    torch.manual_seed(0)
    inp = torch.rand(n_batches, T, n_neurons_input) * 10
    inp.requires_grad = True

    # forward
    out, state, rec = model(inp)

    # backward
    out.sum().backward()

    assert not torch.all(inp.grad == 0)
    assert not torch.all(model.lin0.weight.grad == 0)
    assert not torch.all(model.lin1.weight.grad == 0)
    assert not torch.all(model.lin_out.weight.grad == 0)

    assert not torch.all(model.spk0.threshold.grad == 0)
    assert not torch.all(model.spk1.threshold.grad == 0)


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
        tau_syn_base=3.2,
        tau_mem=1.5,
        tau_syn_out=2.6,
        threshold=1.0,
        threshold_out=1.0,
        max_spikes_per_dt=15,
        max_spikes_per_dt_out=12,
        p_dropout=0.1,
    )

    assert model.lin0.weight.requires_grad is True
    assert model.lin1.weight.requires_grad is True
    assert model.lin_out.weight.requires_grad is True

    assert model.spk0.bias.requires_grad is False
    assert model.spk1.bias.requires_grad is False
    assert model.spk_out.bias.requires_grad is False

    assert model.spk0.threshold.requires_grad is False
    assert model.spk1.threshold.requires_grad is False
    assert model.spk_out.threshold.requires_grad is False

    assert model.spk0.tau_mem.requires_grad is False
    assert model.spk1.tau_mem.requires_grad is False
    assert model.spk_out.tau_mem.requires_grad is False

    assert model.spk0.tau_syn.requires_grad is False
    assert model.spk1.tau_syn.requires_grad is False
    assert model.spk_out.tau_syn.requires_grad is False

    assert model.spk0.max_spikes_per_dt.requires_grad is False
    assert model.spk1.max_spikes_per_dt.requires_grad is False
    assert model.spk_out.max_spikes_per_dt.requires_grad is False


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
        tau_syn_base=3.2,
        tau_mem=1.5,
        tau_syn_out=2.6,
        threshold=1.0,
        threshold_out=1.0,
        train_threshold=True,
        max_spikes_per_dt=15,
        max_spikes_per_dt_out=12,
        p_dropout=0.1,
    )

    # input params
    n_batches = 2
    T = 20

    # input
    torch.manual_seed(0)
    inp = torch.rand(n_batches, T, n_neurons_input) * 10

    # forward
    out, state, rec = model(inp)

    # assert first spk layers state is not reset
    assert not torch.all(state["spk0"]["vmem"] == 0)

    model.reset_state()

    # get state
    state = model.state()

    # assert first spk layers state is reset
    assert torch.all(state["spk0"]["vmem"] == 0)


def test_synnet_graph():
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
        tau_syn_base=3.2,
        tau_mem=1.5,
        tau_syn_out=2.6,
        threshold=1.0,
        threshold_out=1.0,
        train_threshold=True,
        max_spikes_per_dt=15,
        max_spikes_per_dt_out=12,
        p_dropout=0.1,
    )

    model.as_graph()
