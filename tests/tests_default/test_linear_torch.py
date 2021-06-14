from pytest import raises


def test_imports():
    from rockpool.nn.modules.torch.torch_module import TorchModule


def test_linear_torch():
    from rockpool.nn.modules.torch.linear_torch import LinearTorch

    import torch

    mod = LinearTorch((2, 10))

    input = torch.rand(1, 10, 2)

    output, ns, r_d = mod.evolve(input)
    mod.set_attributes(ns)
    params = mod.parameters().astorch()

    assert output.shape == (1, 10, 10)

    with raises(ValueError):
        mod = LinearTorch(10)

    with raises(ValueError):
        mod = LinearTorch((10,))

    with raises(ValueError):
        mod = LinearTorch((10, 10, 10))
