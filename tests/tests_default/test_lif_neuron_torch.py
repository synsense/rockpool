import pytest


def test_LIFNeuronTorch():
    from rockpool.nn.modules.torch.lif_neuron_torch import LIFNeuronTorch
    import torch

    n_neurons = 10
    n_batches = 3
    T = 20
    tau_mem = torch.rand(1, n_neurons)
    bias = torch.rand(1, n_neurons)

    # - Test minimal initialisation
    mod = LIFNeuronTorch((n_neurons,))

    # - Test maximal initialisation
    mod = LIFNeuronTorch(
        shape=(n_neurons,),
        tau_mem=tau_mem,
        bias=bias,
        has_bias=True,
        dt=1e-3,
        noise_std=0.1,
        device=None,
        dtype=None,
        record=True,
    )

    # - Generate some data
    input_data = torch.rand(n_batches, T, n_neurons, requires_grad=True)

    # - Test torch interface
    out = mod.forward(input_data)

    out.sum().backward()

    # - Test Rockpool interface
    out, ns, rd = mod.evolve(input_data)

    assert out.shape == input_data.shape
    for _, obj in ns.items():
        assert obj.shape == (1, n_neurons)
    for _, obj in rd.items():
        assert obj.shape == input_data.shape
