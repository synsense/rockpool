import pytest

pytest.importorskip("torch")
pytest.importorskip("sinabs")
pytest.importorskip("sinabs.exodus")


@pytest.mark.parametrize("n_synapses", (1, 2))
def test_FF_equality_exodus(n_synapses):
    import torch
    import numpy as np

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Exodus tests")

    # - parameter
    n_synapses = 1
    n_neurons = 10
    n_batches = 3
    T = 100
    tau_mem = 0.01
    tau_syn = 0.05
    threshold = 1.0
    bias = torch.arange(n_neurons) * 0.05

    # - init LIFTorch
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    from rockpool.parameters import Constant

    lif_torch = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        has_rec=False,
        dt=1e-3,
        noise_std=0.0,
        threshold=threshold,
        bias=bias,
    ).to("cuda")

    # - init LIFExodus
    from rockpool.nn.modules.sinabs.lif_exodus import LIFExodus

    lif_sinabs = LIFExodus(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        has_rec=False,
        dt=1e-3,
        noise_std=0.0,
        threshold=threshold,
        bias=bias,
    ).to("cuda")

    # - Generate some data
    input_data = torch.rand(n_batches, T, n_synapses * n_neurons).cuda() * 0.01
    # Separate (but equal) input tensor for exodus, to compare gradients
    input_data_torch = input_data.clone()
    input_data_sinabs = input_data.clone()
    input_data_torch.requires_grad_(True)
    input_data_sinabs.requires_grad_(True)

    # - run LIFTorch and LIFExodus
    out_torch, ns_torch, rd_torch = lif_torch(input_data_torch, record=True)
    out_sinabs, ns_sinabs, rd_sinabs = lif_sinabs(input_data_sinabs, record=True)

    assert np.allclose(out_torch.detach().cpu(), out_sinabs.detach().cpu())

    for key in ns_sinabs.keys():
        if not ns_torch[key] == None and not ns_sinabs[key] == None:
            assert np.allclose(
                ns_torch[key].detach().cpu(),
                ns_sinabs[key].detach().cpu(),
                atol=1e-5,
                rtol=1e-5,
            )

    for key in rd_sinabs.keys():
        if not rd_torch[key] == None and not rd_sinabs[key] == None:
            assert np.allclose(
                rd_torch[key].detach().cpu(),
                rd_sinabs[key].detach().cpu(),
                atol=1e-5,
                rtol=1e-5,
            )

    # - Backward pass
    out_torch.sum().backward()
    out_sinabs.sum().backward()

    assert torch.allclose(lif_torch.bias.grad, lif_sinabs.bias.grad)
    assert torch.allclose(input_data_torch.grad, input_data_sinabs.grad)


def test_lif_slayer():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Exodus tests")

    from rockpool.nn.modules import LIFSlayer

    # - Expect deprecation warning
    with pytest.warns(DeprecationWarning):
        mod = LIFSlayer(2)


def test_exodus_membrane():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Exodus tests")

    from rockpool.nn.modules import LIFMembraneExodus

    # - parameter
    n_synapses = 2
    n_neurons = 10
    n_batches = 3
    T = 100
    tau_mem = 0.01
    tau_syn = 0.05
    bias = 0.1

    # - init LIFExodus
    lm_exodus = LIFMembraneExodus(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        dt=1e-3,
        bias=bias,
    ).to("cuda")

    # - Generate some data
    input_data = (
        torch.rand(n_batches, T, n_synapses * n_neurons, requires_grad=True).cuda()
        * 0.01
    )

    out_sinabs, ns_sinabs, rd_sinabs = lm_exodus(input_data)
    out_sinabs.sum().backward()
    assert lm_exodus.tau_mem.grad is not None
    assert lm_exodus.tau_syn.grad is not None
    assert lm_exodus.bias.grad is not None

