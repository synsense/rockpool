import pytest

pytest.importorskip("torch")


def test_import_ExpSynExodus():
    from rockpool.nn.modules.sinabs.lif_exodus import ExpSynExodus
    from rockpool.nn.modules.sinabs import ExpSynExodus
    from rockpool.nn.modules import ExpSynExodus


def test_ExpSynExodus():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Exodus tests")

    from rockpool.nn.modules.sinabs.lif_exodus import ExpSynExodus

    # - parameter
    n_synapses = 2
    n_batches = 3
    T = 100
    tau_syn = 0.05

    # - init ExpSynExodus
    lm_exodus = ExpSynExodus(
        shape=n_synapses,
        tau=tau_syn,
        dt=1e-3,
    ).to("cuda")

    # - Generate some data
    input_data = torch.rand(n_batches, T, n_synapses, requires_grad=True).cuda() * 0.01

    out_sinabs, ns_sinabs, rd_sinabs = lm_exodus(input_data)
    out_sinabs.sum().backward()
    lm_exodus.tau_syn.grad

    assert out_sinabs.shape == (n_batches, T, n_synapses)
