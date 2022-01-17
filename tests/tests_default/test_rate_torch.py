import pytest


def test_imports():
    from rockpool.nn.modules import RateTorch


def test_evolve():
    from rockpool.nn.modules import RateTorch
    import torch

    # - Test instantiation
    N = 2
    T = 100
    batches = 3
    mod = RateTorch(N)

    data = torch.randn(batches, T, N)
    out, _, _ = mod(data)
    assert out.shape == (batches, T, N)

    mod.to_torch()
    out = mod(data)
    assert out.shape == (batches, T, N)

    out.sum().backward()

    assert torch.all(mod.tau.grad != 0.0)
    assert torch.all(mod.bias.grad != 0.0)
    assert torch.all(mod.threshold.grad != 0.0)

    # - Test recurrence
    mod = RateTorch(N, has_rec=True).to_torch()
    out = mod(data)
    assert out.shape == (batches, T, N)
    out.sum().backward()

    assert torch.all(mod.tau.grad != 0.0)
    assert torch.all(mod.bias.grad != 0.0)
    assert torch.all(mod.threshold.grad != 0.0)
    assert torch.all(mod.w_rec.grad != 0.0)
