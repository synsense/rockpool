import pytest


def test_single_neuron():
    from rockpool.nn.modules.torch.lif_bitshift_torch import LIFBitshiftTorch
    import torch

    N = 1
    Nsyn = 2
    tau_mem = [0.04]
    tau_syn = [[0.02], [0.03]]
    threshold = [10.0]
    learning_window = [0.5]

    lyr = LIFBitshiftTorch(
        n_neurons=N,
        n_synapses=Nsyn,
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        threshold=threshold,
        learning_window=learning_window,
        batch_size=1,
        dt=0.01,
        device="cpu",
    )

    inp = torch.zeros((10, 1, 2, 1)).cpu()
    inp[1, :, :, :] = 1
    out, states, recs = lyr(inp, record=True)


def test_backward():
    from rockpool.nn.modules.torch.lif_bitshift_torch import LIFBitshiftTorch
    import torch

    N = 1
    Nsyn = 2
    tau_mem = [0.04]
    tau_syn = [[0.02]]
    threshold = [10.0]
    learning_window = [0.5]

    lyr = LIFBitshiftTorch(
        n_neurons=N,
        n_synapses=Nsyn,
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        threshold=threshold,
        learning_window=learning_window,
        batch_size=1,
        dt=0.01,
        device="cpu",
    )

    inp = torch.rand(50, 1, Nsyn, N).cpu()

    inp.requires_grad = True
    out, states, recs = lyr(inp, record=True)

    out.sum().backward()
