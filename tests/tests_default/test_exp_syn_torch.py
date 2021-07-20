import pytest


def test_ExpSynTorch():
    from rockpool.nn.modules.torch.exp_syn_torch import ExpSynTorch
    import torch

    n_synapses = 10
    n_batches = 3
    T = 20
    tau_syn = torch.rand(n_synapses)

    # - Test minimal initialisation
    mod = ExpSynTorch(n_synapses)

    # - Test maximal initialisation
    mod = ExpSynTorch(
        shape=(n_synapses,),
        tau_syn=tau_syn,
        dt=1e-3,
        device=None,
        dtype=None,
    )

    # - Generate some data
    input_data = torch.rand(n_batches, T, n_synapses, requires_grad=True)

    # - Test torch interface
    out = mod.forward(input_data)

    out.sum().backward()

    # - Test Rockpool interface
    out, ns, rd = mod(input_data)

    assert out.shape == input_data.shape

    for _, obj in ns.items():
        assert obj.shape == (1, n_synapses)

    for _, obj in rd.items():
        assert obj.shape == input_data.shape
