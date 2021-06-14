import pytest


def test_LIFTorch():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    n_synapses = 5
    n_neurons = 10
    n_batches = 3
    T = 20
    tau_mem = torch.rand(n_neurons)
    tau_syn = torch.rand(n_synapses)
    bias = torch.rand(n_neurons)
    w_syn = torch.rand(n_synapses, n_neurons)
    w_rec = torch.rand(n_neurons, n_synapses)

    # - Test minimal initialisation
    mod = LIFTorch((n_synapses, n_neurons))

    # - Test maximal initialisation
    mod = LIFTorch(
        shape=(n_synapses, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        bias=bias,
        has_bias=True,
        w_syn=w_syn,
        w_rec=w_rec,
        has_rec=True,
        dt=1e-3,
        noise_std=0.1,
        device=None,
        dtype=None,
    )

    # - Generate some data
    input_data = torch.rand(n_batches, T, n_synapses, requires_grad=True)

    # - Test torch interface
    out = mod.forward(input_data)

    out.sum().backward()

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    assert out.shape == (n_batches, T, n_neurons)
    assert ns["Isyn"].shape == (1, n_synapses)
    assert ns["Vmem"].shape == (1, n_neurons)
    assert rd["Isyn"].shape == (n_batches, T, n_synapses)
    assert rd["Vmem"].shape == (n_batches, T, n_neurons)
