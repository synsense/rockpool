import pytest


def test_imports():
    from rockpool.nn.modules.torch.lif_bitshift_torch import LIFBitshiftTorch
    from rockpool.nn.modules import LIFBitshiftTorch


def test_LIFBitshiftTorch_Forward_Backward():
    from rockpool.nn.modules.torch.lif_bitshift_torch import LIFBitshiftTorch
    import numpy as np
    import torch

    N = 10
    Nsyn = 2
    tau_mem = 0.01
    tau_syn = torch.Tensor([0.005, 0.015])
    mod = LIFBitshiftTorch(
        shape=(N * Nsyn, N),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        threshold=1.0,
        has_bias=True,
        has_rec=True,
        noise_std=0.1,
        learning_window=0.5,
        dt=0.001,
        device="cpu",
    )

    # - Generate some data
    T = 100
    num_batches = 1
    input_data = torch.rand(num_batches, T, Nsyn * N).cpu() * 100
    input_data.requires_grad = True

    # - Test Rockpool interface
    out, ns, rd = mod.evolve(input_data)

    # - Test backward
    out.sum().backward()


def test_LIFBitshiftTorch_single_neuron():
    from rockpool.nn.modules.torch.lif_bitshift_torch import LIFBitshiftTorch
    import numpy as np
    import torch

    N = 1
    Nsyn = 2
    tau_mem = 0.01
    tau_syn = torch.Tensor([0.002, 0.004])
    mod = LIFBitshiftTorch(
        shape=(N * Nsyn, N),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        threshold=1000.0,
        has_bias=False,
        has_rec=False,
        noise_std=0.0,
        learning_window=0.5,
        dt=0.001,
        device="cpu",
    )

    # - Generate some data
    T = 10
    num_batches = 1
    input_data = torch.zeros(1, T, Nsyn * N).cpu()
    input_data[:, 0, :] = mod.tau_syn / mod.dt

    # - Test Rockpool interface
    out, state, rec = mod.evolve(input_data, record=True)

    # make sure the values decayed correctly
    assert rec["Isyn"][0, 1, 0, 0] == 0.5
    assert rec["Isyn"][0, 2, 0, 0] == 0.25

    assert rec["Isyn"][0, 1, 1, 0] == 0.75
    assert rec["Isyn"][0, 2, 1, 0] == 0.5625
