import pytest

pytest.importorskip("torch")


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


def test_stochastic_rounding():
    import torch
    from rockpool.transform.torch_transform import stochastic_rounding

    value = torch.randn(1000)
    # verifying n_levels
    all_n_levels = [2**n - 1 for n in range(2, 9)]
    for n in all_n_levels:
        rounded_levels = stochastic_rounding(value, num_levels=n)
        assert len(torch.unique(rounded_levels)) <= n

    # verifying the output when input range and output range are None
    n = 16
    rounded_levels = stochastic_rounding(value, num_levels=n)
    assert -abs(value).max() <= rounded_levels.min()
    assert rounded_levels.max() <= abs(value).max()

    # verifying mainain_zero with a non asymetric input:
    value = torch.linspace(0, 1, 11)
    n = 11
    output_range = [-10, 10]
    rounded_levels = stochastic_rounding(
        value, output_range=output_range, maintain_zero=False, num_levels=11
    )
    assert rounded_levels.min() == output_range[0]

    rounded_levels = stochastic_rounding(
        value, output_range=output_range, maintain_zero=True, num_levels=11
    )
    assert rounded_levels.min() == value.min()


def test_stochastic_cahnnel_rounding():
    import torch
    from rockpool.transform.torch_transform import stochastic_channel_rounding

    value = torch.randn(size=(5, 5))
    output_range = [-10, 10]

    # verifying n_levels
    all_n_levels = [2**n - 1 for n in range(2, 9)]
    for n in all_n_levels:
        rounded_levels = stochastic_channel_rounding(
            value, output_range=output_range, num_levels=n
        )
        assert len(torch.unique(rounded_levels)) <= n

    # verifying mainain_zero with a non asymetric input:
    value = torch.rand(size=(5, 5))
    n = 11
    rounded_levels = stochastic_channel_rounding(
        value, output_range=output_range, maintain_zero=False, num_levels=11
    )
    assert rounded_levels.min() == output_range[0]

    rounded_levels = stochastic_channel_rounding(
        value, output_range=output_range, maintain_zero=True, num_levels=11
    )
    assert rounded_levels.min() == value.min()


def test_deterministic_rounding():
    import torch
    from rockpool.transform.torch_transform import deterministic_rounding

    value = torch.randn(1000)
    # verifying n_levels
    all_n_levels = [2**n - 1 for n in range(2, 9)]
    for n in all_n_levels:
        rounded_levels = deterministic_rounding(value, num_levels=n)
        assert len(torch.unique(rounded_levels)) <= n

    # verifying the output when input range and output range are None
    n = 16
    rounded_levels = deterministic_rounding(value, num_levels=n)
    assert -abs(value).max() <= rounded_levels.min()
    assert rounded_levels.max() <= abs(value).max()

    # verifying mainain_zero with a non asymetric input:
    value = torch.linspace(0, 1, 11)
    n = 11
    output_range = [-10, 10]
    rounded_levels = deterministic_rounding(
        value, output_range=output_range, maintain_zero=False, num_levels=11
    )
    assert rounded_levels.min() == output_range[0]

    rounded_levels = deterministic_rounding(
        value, output_range=output_range, maintain_zero=True, num_levels=11
    )
    assert rounded_levels.min() == value.min()


def test_int_quant():
    import torch
    from rockpool.transform.torch_transform import int_quant

    value = torch.randn(1000)
    nbits = torch.range(2, 8)

    # verify n_levels
    for nbit in nbits:
        N_max = 2 ** (nbit - 1) - 1
        rounded_levels = int_quant(value, n_bits=nbit)

        assert len(torch.unique(rounded_levels)) <= 2**nbit - 1
        assert all(
            i in torch.range(-N_max, N_max) for i in torch.unique(rounded_levels)
        )

    # verifying mainain_zero with a non asymetric input
    value = torch.cat((torch.rand(1000), torch.tensor([0])))

    nbit = 8
    N_max = 2 ** (nbit - 1) - 1

    rounded_levels = int_quant(value, maintain_zero=True, n_bits=nbit)
    assert rounded_levels.min() == value.min()

    rounded_levels = int_quant(value, maintain_zero=False, n_bits=nbit)
    assert rounded_levels.min() == -N_max

    max_n_bits = 8
    max_range = 2 ** (nbit - 1) - 1
    rounded_levels = int_quant(value, maintain_zero=False, map_to_max=True, n_bits=nbit)
    assert rounded_levels.min() == -max_range
    assert rounded_levels.max() == max_range


def net_test_int_quant():
    import torch
    from rockpool.transform.torch_transform import (
        int_quant,
        make_param_T_config,
        make_param_T_network,
        apply_T,
    )
    from rockpool.nn.modules import LIFTorch, LinearTorch
    from rockpool.parameters import Constant
    from rockpool.nn.combinators import Sequential

    nbit = 8
    N_max = 2 ** (nbit - 1) - 1

    net = Sequential(LinearTorch((5, 5)), LIFTorch(5, threshold=Constant(1)))
    tconfig = make_param_T_config(net, lambda p: int_quant(p, n_bits=nbit), "weights")
    t_net = make_param_T_network(net, tconfig)
    qmodel = apply_T(t_net)

    assert len(torch.unique(qmodel[0]._mod.weight)) <= 2**nbit - 1
    assert all(
        i in torch.range(-N_max, N_max) for i in torch.unique(qmodel[0]._mod.weight)
    )


def test_t_decay():
    import torch
    from rockpool.transform.torch_transform import t_decay

    alpha = torch.rand(10).clip(0.5, 0.99)
    q_decay = t_decay(alpha)
    allowed_decays = [1 - (1 / (2**bitshift)) for bitshift in range(1, 16)]

    assert all(decay in allowed_decays for decay in q_decay)


def test_t_decay_net():
    import torch
    from rockpool.nn.modules import LinearTorch, LIFTorch
    from rockpool.nn.combinators import Sequential
    from rockpool.transform.torch_transform import (
        t_decay,
        make_param_T_config,
        make_param_T_network,
        apply_T,
    )

    init_alpha, init_beta = torch.tensor(0.8), torch.tensor(0.6)

    net = Sequential(
        LinearTorch((2, 2)),
        LIFTorch(2, leak_mode="decays", alpha=init_alpha, beta=init_beta),
    )
    tconfig = make_param_T_config(net, lambda p: t_decay(p), "decays")
    t_net = make_param_T_network(net, tconfig)
    qmodel = apply_T(t_net)

    assert qmodel[1]._mod.alpha == t_decay(init_alpha)
    assert qmodel[1]._mod.beta == t_decay(init_beta)
