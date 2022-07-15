import pytest

pytest.importorskip("torch")

from rockpool.utilities.backend_management import torch_version_satisfied

if not torch_version_satisfied(1, 12, 0):
    pytest.skip("This test requires torch >= 1.12.0.")


def test_imports():
    from rockpool.transform import torch_transform


def test_dropout():
    from rockpool.transform.torch_transform import dropout
    import torch

    t = torch.rand((10, 10))
    t.requires_grad = True
    o = dropout(t)
    o.sum().backward()

    assert not torch.all(t.grad == 0), "Gradients are all zero"


def test_transform_mod():
    import torch
    import rockpool.transform.torch_transform as tt
    from rockpool.nn.modules import LinearTorch, LIFTorch

    Nin = 3
    Nout = 4
    T = 10

    # - Test LinearTorch
    mod = LinearTorch((Nin, Nout))
    tconf = tt.make_param_T_config(mod, tt.stochastic_rounding)
    tmod = tt.make_param_T_network(mod, tconf)

    out, _, _ = tmod(torch.ones(T, Nin))
    out.sum().backward()

    assert not torch.all(tmod._mod.weight.grad == 0), "Gradients are all zero"

    # - Test LIFTorch
    mod = LIFTorch(Nin)
    tconf = tt.make_param_T_config(mod, tt.stochastic_rounding, "taus")
    tmod = tt.make_param_T_network(mod, tconf)

    # - Check that `as_graph()` works
    tmod.as_graph()

    # - Test evolution
    out, _, _ = tmod(torch.ones((T, Nin)))
    out.sum().backward()

    # - Make sure gradients are not zeroed
    assert not torch.all(tmod._mod.tau_mem.grad == 0), "Gradients are all zero"

    # - Apply the transformation manually
    tmod = tmod.apply_T()


def test_transform_net():
    import torch
    import rockpool.transform.torch_transform as tt
    import rockpool.utilities.tree_utils as tu
    from rockpool.nn.modules import LinearTorch, LIFTorch
    from rockpool.nn.combinators import Sequential, Residual

    # - Ensure deterministic testing
    torch.manual_seed(1)

    net = Sequential(
        LinearTorch((3, 5)),
        LIFTorch(5),
        Residual(
            LinearTorch((5, 10)),
            LIFTorch((10, 5)),
        ),
        LinearTorch((5, 5)),
        LIFTorch(5),
    )

    tconf = tt.make_param_T_config(net, tt.stochastic_rounding, "weights")
    tu.tree_update(tconf, tt.make_param_T_config(net, tt.stochastic_rounding, "biases"))
    tu.tree_update(tconf, tt.make_param_T_config(net, tt.stochastic_rounding, "taus"))

    tnet = tt.make_param_T_network(net, tconf)

    out, _, _ = tnet(torch.ones(1, 10, 3))
    out.sum().backward()

    for p in tnet.parameters().astorch():
        assert not torch.all(p.grad == 0), f"Gradients are all zero for {p}"

    # - Test applying transformation manually to network
    tnet = tt.apply_T(tnet)

    # - Test removing transformations
    undo_tnet = tt.remove_T_net(tnet)


def test_act_transform_net():
    import torch
    import rockpool.transform.torch_transform as tt
    from rockpool.nn.modules import LinearTorch, LIFTorch
    from rockpool.nn.combinators import Sequential, Residual

    net = Sequential(
        LinearTorch((3, 5)),
        LIFTorch(5),
        Residual(
            LinearTorch((5, 10)),
            LIFTorch((10, 5)),
        ),
        LinearTorch((5, 5)),
        LIFTorch(5),
    )

    # - Test null config
    tconf = tt.make_act_T_config(net)

    # - Build a specific config
    tconf = tt.make_act_T_config(net, tt.deterministic_rounding, LinearTorch)

    # - Patch the network
    tnet = tt.make_act_T_network(net, tconf)

    out, _, _ = tnet(torch.ones(1, 10, 3))
    out.sum().backward()

    for p in tnet.parameters().astorch():
        assert not torch.all(p.grad == 0), f"Gradients are all zero for {p}"
