def test_wavesense_import():
    from rockpool.nn.networks import WaveSenseNet


def test_wavesense_init():
    from rockpool.nn.networks import WaveSenseNet
    from rockpool.nn.modules import LIFTorch

    # model params
    dilations = [2, 16]
    n_out_neurons = 2
    n_inp_neurons = 3
    n_neurons = 4
    kernel_size = 2
    tau_mem = 0.002
    base_tau_syn = 0.002
    tau_lp = 0.01
    threshold = 1.0
    dt = 0.001
    device = "cpu"

    # model init
    model = WaveSenseNet(
        dilations=dilations,
        n_classes=n_out_neurons,
        n_channels_in=n_inp_neurons,
        n_channels_res=n_neurons,
        n_channels_skip=n_neurons,
        n_hidden=n_neurons,
        kernel_size=kernel_size,
        has_bias=True,
        smooth_output=True,
        tau_mem=tau_mem,
        base_tau_syn=base_tau_syn,
        tau_lp=tau_lp,
        threshold=threshold,
        neuron_model=LIFTorch,
        dt=dt,
        device=device,
    )


def test_wavesense_forward():
    from rockpool.nn.networks import WaveSenseNet
    from rockpool.nn.modules import LIFTorch
    import torch

    # model params
    dilations = [2, 16]
    n_out_neurons = 2
    n_inp_neurons = 3
    n_neurons = 4
    kernel_size = 2
    tau_mem = 0.002
    base_tau_syn = 0.002
    tau_lp = 0.01
    threshold = 1.0
    dt = 0.001
    device = "cpu"

    # model init
    model = WaveSenseNet(
        dilations=dilations,
        n_classes=n_out_neurons,
        n_channels_in=n_inp_neurons,
        n_channels_res=n_neurons,
        n_channels_skip=n_neurons,
        n_hidden=n_neurons,
        kernel_size=kernel_size,
        has_bias=True,
        smooth_output=True,
        tau_mem=tau_mem,
        base_tau_syn=base_tau_syn,
        tau_lp=tau_lp,
        threshold=threshold,
        neuron_model=LIFTorch,
        dt=dt,
        device=device,
    )

    # input params
    n_batches = 2
    T = 20

    # input
    inp = torch.ones(n_batches, T, n_inp_neurons) * 100

    # forward
    out, state, rec = model(inp)

    assert len(rec) == 0


def test_wavesense_record():
    from rockpool.nn.networks import WaveSenseNet
    from rockpool.nn.modules import LIFTorch
    import torch

    # model params
    dilations = [2, 16]
    n_out_neurons = 2
    n_inp_neurons = 3
    n_neurons = 4
    kernel_size = 2
    tau_mem = 0.002
    base_tau_syn = 0.002
    tau_lp = 0.01
    threshold = 1.0
    dt = 0.001
    device = "cpu"

    # model init
    model = WaveSenseNet(
        dilations=dilations,
        n_classes=n_out_neurons,
        n_channels_in=n_inp_neurons,
        n_channels_res=n_neurons,
        n_channels_skip=n_neurons,
        n_hidden=n_neurons,
        kernel_size=kernel_size,
        has_bias=True,
        smooth_output=True,
        tau_mem=tau_mem,
        base_tau_syn=base_tau_syn,
        tau_lp=tau_lp,
        threshold=threshold,
        neuron_model=LIFTorch,
        dt=dt,
        device=device,
    )

    # input params
    n_batches = 2
    T = 20

    # input
    inp = torch.rand(n_batches, T, n_inp_neurons) * 10

    # forward
    out, state, rec = model(inp, record=True)

    assert len(rec) > 0


def test_wavesense_backward():
    from rockpool.nn.networks import WaveSenseNet
    from rockpool.nn.modules import LIFTorch
    import torch

    # model params
    dilations = [2, 16]
    n_out_neurons = 2
    n_inp_neurons = 3
    n_neurons = 4
    kernel_size = 2
    tau_mem = 0.002
    base_tau_syn = 0.002
    tau_lp = 0.01
    threshold = 1.0
    dt = 0.001
    device = "cpu"

    # model init
    model = WaveSenseNet(
        dilations=dilations,
        n_classes=n_out_neurons,
        n_channels_in=n_inp_neurons,
        n_channels_res=n_neurons,
        n_channels_skip=n_neurons,
        n_hidden=n_neurons,
        kernel_size=kernel_size,
        has_bias=True,
        smooth_output=True,
        tau_mem=tau_mem,
        base_tau_syn=base_tau_syn,
        tau_lp=tau_lp,
        threshold=threshold,
        neuron_model=LIFTorch,
        dt=dt,
        device=device,
    )

    # input params
    n_batches = 2
    T = 20

    # input
    inp = torch.rand(n_batches, T, n_inp_neurons) * 10
    inp.requires_grad = True

    # forward
    out, state, rec = model(inp)

    # backward
    out.sum().backward()

    assert not torch.all(inp.grad == 0)


def test_wavesense_save_load():
    from rockpool.nn.networks import WaveSenseNet
    from rockpool.nn.modules import LIFTorch
    import torch
    import os

    # model params
    dilations = [2, 16]
    n_out_neurons = 2
    n_inp_neurons = 3
    n_neurons = 4
    kernel_size = 2
    tau_mem = 0.002
    base_tau_syn = 0.002
    tau_lp = 0.01
    threshold = 1.0
    dt = 0.001
    device = "cpu"

    # model init
    model = WaveSenseNet(
        dilations=dilations,
        n_classes=n_out_neurons,
        n_channels_in=n_inp_neurons,
        n_channels_res=n_neurons,
        n_channels_skip=n_neurons,
        n_hidden=n_neurons,
        kernel_size=kernel_size,
        has_bias=True,
        smooth_output=True,
        tau_mem=tau_mem,
        base_tau_syn=base_tau_syn,
        tau_lp=tau_lp,
        threshold=threshold,
        neuron_model=LIFTorch,
        dt=dt,
        device=device,
    )

    model2 = WaveSenseNet(
        dilations=dilations,
        n_classes=n_out_neurons,
        n_channels_in=n_inp_neurons,
        n_channels_res=n_neurons,
        n_channels_skip=n_neurons,
        n_hidden=n_neurons,
        kernel_size=kernel_size,
        has_bias=True,
        smooth_output=True,
        tau_mem=tau_mem,
        base_tau_syn=base_tau_syn,
        tau_lp=tau_lp,
        threshold=threshold,
        neuron_model=LIFTorch,
        dt=dt,
        device=device,
    )

    # input params
    n_batches = 2
    T = 20

    # input
    inp = torch.rand(n_batches, T, n_inp_neurons) * 10
    inp.requires_grad = True

    # forward
    out, _, _ = model(inp)

    # forward model 2
    out2, _, _ = model2(inp)

    # assert not all outputs are equal
    assert not torch.all(out == out2)

    # save model
    model.save("tmp.json")

    # load parameters to model2
    model2.load("tmp.json")

    # forward model 2
    model2.reset_state()
    out2, _, _ = model2(inp)

    # assert all outputs are equal
    assert torch.all(out == out2)

    # cleanup
    os.remove("tmp.json")


def test_wavesense_reset():
    from rockpool.nn.networks import WaveSenseNet
    from rockpool.nn.modules import LIFTorch
    import torch

    # model params
    dilations = [2, 16]
    n_out_neurons = 2
    n_inp_neurons = 3
    n_neurons = 4
    kernel_size = 2
    tau_mem = 0.002
    base_tau_syn = 0.002
    tau_lp = 0.01
    threshold = 1.0
    dt = 0.001
    device = "cpu"

    # model init
    model = WaveSenseNet(
        dilations=dilations,
        n_classes=n_out_neurons,
        n_channels_in=n_inp_neurons,
        n_channels_res=n_neurons,
        n_channels_skip=n_neurons,
        n_hidden=n_neurons,
        kernel_size=kernel_size,
        has_bias=True,
        smooth_output=True,
        tau_mem=tau_mem,
        base_tau_syn=base_tau_syn,
        tau_lp=tau_lp,
        threshold=threshold,
        neuron_model=LIFTorch,
        dt=dt,
        device=device,
    )

    # input params
    n_batches = 2
    T = 20

    # input
    inp = torch.rand(n_batches, T, n_inp_neurons) * 10

    # forward
    out, state, rec = model(inp)

    # assert first spk layers state is not reset
    assert not torch.all(state["spk1"]["vmem"] == 0)

    # assert first spk layers state of first wave block is not reset
    assert not torch.all(state["wave0"]["spk1"]["vmem"] == 0)

    model.reset_state()

    # get state
    state = model.state()

    # assert first spk layers state is reset
    assert torch.all(state["spk1"]["vmem"] == 0)

    # assert first spk layers state of first wave block is reset
    assert torch.all(state["wave0"]["spk1"]["vmem"] == 0)


def test_wavenet():
    from rockpool.nn.networks.wavesense import WaveNet
    import torch
    
    dilations = [2]
    n_neurons = 64
    kernel_size = 2
    device = "cpu"
    
    T_total = 10
    T_stim = 3
    N_batch = 2
    
    model = WaveNet(
        dilations=dilations,
        n_classes=n_neurons,
        n_channels_in=n_neurons,
        n_channels_res=n_neurons,
        n_channels_skip=n_neurons,
        n_hidden=n_neurons,
        kernel_size=kernel_size,
        bias=False,
    ).to(device)
    
    inp = torch.zeros(N_batch, T_total, n_neurons)
    inp[:, T_stim, 0] = 1
    
    out, _, _ = model(inp)
    
    # assert correct shape
    assert out.shape == inp.shape
    
    # assert output at time T_stim
    assert not any(out[:, T_stim, :].detach().numpy().ravel() == 0)
    
    # assert output at time T_stim + dilation
    assert not any(out[:, T_stim + dilations[0], :].detach().numpy().ravel() == 0)
    
    # assert other outputs to be zero
    assert all(out[:, T_stim + dilations[0] + 1, :].detach().numpy().ravel() == 0)
    assert all(out[:, T_stim + dilations[0] - 1, :].detach().numpy().ravel() == 0)
    assert all(out[:, 0, :].detach().numpy().ravel() == 0)
    assert all(out[:, 1, :].detach().numpy().ravel() == 0)
