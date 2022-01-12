# def test_FF_equality():
#    import torch
#
#    # - parameter
#    n_synapses = 1
#    n_neurons = 10
#    n_batches = 3
#    T = 20
#    tau_mem = torch.rand(n_neurons)
#    tau_syn = 0.05
#
#    # - init LIFTorch
#
#    from rockpool.nn.modules.torch.lif_torch import LIFTorch
#    lif_torch  = LIFTorch(
#        shape=(n_synapses * n_neurons, n_neurons),
#        tau_mem=tau_mem,
#        tau_syn=tau_syn,
#        has_bias=False,
#        has_rec=False,
#        dt=1e-3,
#        noise_std=0.0,
#        device="cpu",
#    )
#
#
#    # - init LIFSinabs
#
#    from rockpool.nn.modules.torch.lif_sinabs import LIFSinabs
#    lif_sinabs = LIFSinabs(
#        shape=(n_synapses * n_neurons, n_neurons),
#        tau_mem=tau_mem,
#        tau_syn=tau_syn,
#        has_bias=False,
#        has_rec=False,
#        dt=1e-3,
#        noise_std=0.0,
#        device="cpu",
#    )
#
#
#    # - Generate some data
#    input_data = torch.rand(n_batches, T, n_synapses * n_neurons, requires_grad=True)
#
#    # - run LIFTorch and LIFSinabs
#    out_torch, ns_torch, rd_torch = lif_torch(input_data, record=True)
#    out_sinabs, ns_sinabs, rd_sinabs = lif_sinabs(input_data, record=True)
#
#    assert torch.all(out_torch == out_sinabs)
#    assert torch.all(ns_torch == ns_sinabs)
#    assert torch.all(rd_torch == rd_sinabs)


def test_FF_equality_slayer():
    import torch
    import numpy as np
    
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
        bias=Constant(0.0),
        has_rec=False,
        dt=1e-3,
        noise_std=0.0,
        threshold=threshold,
    ).to("cuda")
    
    # - init LIFSlayer
    from rockpool.nn.modules.sinabs.lif_slayer import LIFSlayer
    
    lif_sinabs = LIFSlayer(
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
    
    # - run LIFTorch and LIFSlayer
    out_torch, ns_torch, rd_torch = lif_torch(input_data, record=True)
    out_sinabs, ns_sinabs, rd_sinabs = lif_sinabs(input_data, record=True)
    
    assert np.allclose(out_torch.detach().cpu(), out_sinabs.detach().cpu())
    
    for key in ns_torch.keys():
        assert np.allclose(
            ns_torch[key].detach().cpu(),
            ns_sinabs[key].detach().cpu(),
            atol=1e-5,
            rtol=1e-5,
        )
    
    for key in rd_torch.keys():
        assert np.allclose(
            rd_torch[key].detach().cpu(),
            rd_sinabs[key].detach().cpu(),
            atol=1e-5,
            rtol=1e-5,
        )


def test_FF_multisyn_equality_slayer():
    import torch
    import numpy as np

    from rockpool.parameters import Constant

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

    # - init LIFSlayer
    from rockpool.nn.modules.sinabs.lif_slayer import LIFSlayer

    lif_sinabs = LIFSlayer(
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

    for key in ns_torch.keys():
        assert np.allclose(
            ns_torch[key].detach().cpu(),
            ns_sinabs[key].detach().cpu(),
            atol=1e-5,
            rtol=1e-5,
        )

    for key in rd_torch.keys():
        assert np.allclose(
            rd_torch[key].detach().cpu(),
            rd_sinabs[key].detach().cpu(),
            atol=1e-5,
            rtol=1e-5,
        )
