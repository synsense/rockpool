import pytest

pytest.importorskip("torch")


def test_import_ExpSynExodus():
    from rockpool.nn.modules.sinabs.lif_exodus import ExpSynExodus
    from rockpool.nn.modules.sinabs import ExpSynExodus

def test_ExpSynExodus():
    from rockpool.nn.modules.sinabs.lif_exodus import ExpSynExodus
    from rockpool.parameters import Constant
    import torch

    n_synapses = 10
    n_batches = 3
    T = 20
    tau_syn = torch.rand(n_synapses)

    # - Test minimal initialisation
    mod = ExpSynExodus(n_synapses)

    # - Test maximal initialisation
    mod = ExpSynExodus(shape=(n_synapses,), tau_syn=tau_syn, dt=1e-3)

    # - Generate some data
    input_data = torch.rand(n_batches, T, n_synapses, requires_grad=True)

    # - Test Rockpool interface
    out, ns, rd = mod(input_data)

    assert out.shape == input_data.shape

    for _, obj in ns.items():
        assert obj.shape == (n_synapses,)

    for _, obj in rd.items():
        assert obj.shape == input_data.shape

    # - Test torch interface
    mod = mod.to_torch()
    out = mod(input_data)
    out.sum().backward()

    # - Test scalar tau_syn
    tau_syn = 20e-3

    # TODO: Test trainable tau_syn as soon as this feature is enabled
    # mod = ExpSynExodus(n_synapses, tau_syn=tau_syn, dt=1e-3)

    # # - Test torch interface
    # mod = mod.to_torch()
    # out = mod(input_data)
    # out.sum().backward()

    # - Test constant tau_syn
    mod = ExpSynExodus(n_synapses, tau_syn=Constant(tau_syn), dt=1e-3)
    mod = mod.to_torch()
    out = mod(input_data)

    out.sum().backward()

