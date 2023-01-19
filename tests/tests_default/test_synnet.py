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
        threshold_out=7.2,
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
        threshold_out=0.001,
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

    assert torch.allclose(out, rec["spk_out"]["spikes"])
    assert torch.allclose(state["spk_out"]["spikes"], rec["spk_out"]["spikes"][0, -1])
    assert torch.allclose(state["spk_out"]["vmem"], rec["spk_out"]["vmem"][0, -1])


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
        threshold_out=100,
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
        threshold_out=100,
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
        threshold_out=1.0,
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

    print(model.spk0.tau_syn, model.spk0.tau_mem)
    print(model.spk1.tau_syn, model.spk1.tau_mem)
    print(model.lin1.weight.grad)
    print(rec["spk0"])
    # print(out)
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
        tau_syn_base=0.0032,
        tau_mem=0.0015,
        tau_syn_out=0.0026,
        threshold=1.0,
        threshold_out=1.0,
        max_spikes_per_dt=15,
        max_spikes_per_dt_out=12,
        p_dropout=0.1,
        dt=1e-3,
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
        tau_syn_base=0.0032,
        tau_mem=0.0015,
        tau_syn_out=0.0026,
        threshold=1.0,
        threshold_out=1.0,
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
        tau_syn_base=0.0032,
        tau_mem=0.0015,
        tau_syn_out=0.0026,
        threshold=1.0,
        threshold_out=1.0,
        train_threshold=True,
        max_spikes_per_dt=15,
        max_spikes_per_dt_out=12,
        p_dropout=0.1,
        dt=1e-3,
    )

    model.as_graph()


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
        threshold_out=7.2,
        train_threshold=True,
        neuron_model=LIFTorch,
        max_spikes_per_dt=13,
        max_spikes_per_dt_out=2,
        p_dropout=0.2,
        dt=dt,
    )

    assert model.spk0.tau_mem == tau_mem
    assert model.spk1.tau_mem == tau_mem
    assert model.spk_out.tau_mem == tau_mem

    # check values and frequencies of time constants
    tau_syn_hidden0 = torch.tensor(
        [
            (tau_syn_base / dt) ** j * dt
            for j in range(1, time_constants_per_layer[0] + 1)
        ]
    )
    torch.sort(tau_syn_hidden0)
    tau_unique, frequencies = model.spk0.tau_syn.unique(return_counts=True)
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
    tau_unique, frequencies = model.spk1.tau_syn.unique(return_counts=True)
    assert torch.allclose(tau_unique, tau_syn_hidden1)
    n_tau = time_constants_per_layer[1]
    s = size_hidden_layers[1]
    while n_tau >= 1:
        n = torch.ceil(torch.tensor(s / n_tau))
        assert torch.sum(frequencies[-n_tau]) == n
        n_tau -= n

    assert model.spk_out.tau_syn == tau_syn_out


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
        threshold_out=7.2,
        train_threshold=True,
        neuron_model=LIFTorch,
        max_spikes_per_dt=13,
        max_spikes_per_dt_out=2,
        p_dropout=0.2,
        dt=dt,
    )

    tau_mem_bitshift = torch.round(tau_to_bitshift(dt, torch.tensor(tau_mem))[0]).int()
    tau_mem_rounded = bitshift_to_tau(dt, tau_mem_bitshift)[0].item()
    assert model.spk0.tau_mem == tau_mem_rounded
    assert model.spk1.tau_mem == tau_mem_rounded
    assert model.spk_out.tau_mem == tau_mem_rounded

    tau_syn_bitshift = torch.round(
        tau_to_bitshift(dt, torch.tensor(tau_syn_base))[0]
    ).int()
    tau_syn_rounded = bitshift_to_tau(dt, tau_syn_bitshift)[0].item()
    assert model.spk0.tau_syn[0] == tau_syn_rounded
    assert model.spk1.tau_syn[1] == tau_syn_rounded
