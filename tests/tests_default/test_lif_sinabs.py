import pytest

pytest.importorskip("torch")
pytest.importorskip("sinabs")
pytest.importorskip("sinabs.exodus")


def test_FF_equality_exodus():
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
    ).to("cuda")

    # - Generate some data
    input_data = (
        torch.rand(n_batches, T, n_synapses * n_neurons, requires_grad=True).cuda()
        * 0.01
    )

    # - run LIFTorch and LIFExodus
    out_torch, ns_torch, rd_torch = lif_torch(input_data, record=True)
    out_sinabs, ns_sinabs, rd_sinabs = lif_sinabs(input_data, record=True)

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


def test_FF_multisyn_equality_exodus():
    import torch
    import numpy as np

    from rockpool.parameters import Constant

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Exodus tests")

    # - parameter
    n_synapses = 2
    n_neurons = 10
    n_batches = 3
    T = 20
    tau_mem = 0.01
    tau_syn = torch.Tensor([[0.02, 0.03]]).repeat(n_neurons, 1)
    threshold = 0.1

    # - init LIFTorch
    from rockpool.nn.modules.torch.lif_torch import LIFTorch

    lif_torch = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        bias=Constant(0.0),
        has_rec=False,
        dt=1e-3,
        noise_std=0.0,
        threshold=threshold,
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
    ).to("cuda")

    # - Generate some data
    input_data = (
        torch.rand(n_batches, T, n_synapses * n_neurons, requires_grad=True).cuda()
        * 0.1
    )

    # - run LIFTorch and LIFSlayer
    out_torch, ns_torch, rd_torch = lif_torch(input_data, record=True)
    out_sinabs, ns_sinabs, rd_sinabs = lif_sinabs(input_data, record=True)

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

    # - init LIFExodus
    from rockpool.nn.modules import LIFMembraneExodus

    # - parameter
    n_synapses = 2
    n_neurons = 10
    n_batches = 3
    T = 100
    tau_mem = 0.01
    tau_syn = 0.05

    # - init LIFTorch
    lm_exodus = LIFMembraneExodus(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        dt=1e-3,
    ).to("cuda")

    # - Generate some data
    input_data = (
        torch.rand(n_batches, T, n_synapses * n_neurons, requires_grad=True).cuda()
        * 0.01
    )

    out_sinabs, ns_sinabs, rd_sinabs = lm_exodus(input_data)
    out_sinabs.sum().backward()
    lm_exodus.tau_mem.grad
    lm_exodus.tau_syn.grad

