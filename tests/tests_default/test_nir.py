import pytest

pytest.importorskip("nir")
pytest.importorskip("nirtorch")
pytest.importorskip("torch")


def test_imports():
    from rockpool.nn.modules import to_nir, from_nir


def test_from_sequential_to_nir():
    import rockpool
    from rockpool.nn.modules import LinearTorch, ExpSynTorch, to_nir
    from rockpool.nn.combinators import Sequential
    import torch
    import nir

    Nin = 2
    Nhidden = 4
    Nout = 2
    dt = 1e-3
    net = Sequential(
        LinearTorch((Nin, Nhidden), has_bias=False),
        ExpSynTorch(Nhidden, dt=dt),
        LinearTorch((Nhidden, Nout), has_bias=False),
        ExpSynTorch(Nout, dt=dt),
    )
    graph = to_nir(net, torch.randn(1, 2))
    assert len(graph.nodes) == 6
    assert isinstance(graph.nodes["0_LinearTorch"], nir.Linear)
    assert isinstance(graph.nodes["1_ExpSynTorch"], nir.LI)
    assert isinstance(graph.nodes["2_LinearTorch"], nir.Linear)
    assert isinstance(graph.nodes["3_ExpSynTorch"], nir.LI)
    assert len(graph.edges) == 5


def test_from_sequential_to_nir_2():
    import rockpool
    from rockpool.nn.combinators import Sequential
    from rockpool.nn.modules import LinearTorch, LIFTorch, to_nir
    import torch
    import nir

    Nin = 2
    Nhidden = 4
    Nout = 2
    dt = 1e-3
    net = Sequential(
        LinearTorch((Nin, Nhidden), has_bias=False),
        LIFTorch(Nhidden, dt=dt),
        LinearTorch((Nhidden, Nout), has_bias=False),
        LIFTorch(Nout, dt=dt),
    )
    graph = to_nir(net, torch.randn(1, 2))
    assert len(graph.nodes) == 6
    assert isinstance(graph.nodes["0_LinearTorch"], nir.Linear)
    assert isinstance(graph.nodes["1_LIFTorch"], nir.CubaLIF)
    assert isinstance(graph.nodes["2_LinearTorch"], nir.Linear)
    assert isinstance(graph.nodes["3_LIFTorch"], nir.CubaLIF)
    assert len(graph.edges) == 5


def test_from_linear_to_nir():
    import torch
    from rockpool.nn.modules import LinearTorch, to_nir

    in_features = 2
    out_features = 3
    m = LinearTorch(shape=(in_features, out_features))
    graph = to_nir(m, torch.randn(1, in_features))
    assert len(graph.nodes) == 3
    assert graph.nodes["rockpool"].weight.shape == torch.Size(
        [out_features, in_features]
    )


def test_from_nir_to_linear():
    import torch
    from rockpool.nn.modules import LinearTorch, to_nir, from_nir

    in_features = 2
    out_features = 3
    m = LinearTorch(shape=(in_features, out_features))
    graph = to_nir(m, torch.randn(1, in_features))
    m2 = from_nir(graph)
    assert m2.rockpool.weight.shape == torch.Size([in_features, out_features])


def test_from_nir_to_sequential():
    from rockpool.nn.combinators import Sequential
    from rockpool.nn.modules import LinearTorch, ExpSynTorch, LIFTorch, to_nir, from_nir
    import torch
    import numpy as np

    timesteps = 6

    orig_model = Sequential(
        LinearTorch(shape=(2, 4)),
        ExpSynTorch(tau=torch.ones((4)) * 10.0, shape=4),
        LinearTorch(shape=(4, 8)),
        LIFTorch(tau_mem=torch.ones((8)) * 10.0, shape=8),
    )
    nir_graph = to_nir(orig_model, torch.randn(timesteps, 2))

    convert_model = from_nir(nir_graph)

    def compare_params(orig, converted):
        for key, param in orig.parameters().items():
            assert hasattr(
                converted, key
            ), f"Parameter {key} not found in converted model."
            assert np.allclose(
                torch.tensor(param).detach().numpy(),
                torch.tensor(getattr(converted, key)).detach().numpy(),
            ), f"Parameter {key} in converted model doesn't match original.\nFound {param} and {getattr(converted, key)}."

    for key, mod in orig_model.modules().items():
        assert (
            key in dict(convert_model.named_children()).keys()
        ), f"Key {key} not found in converted model."
        compare_params(mod, getattr(convert_model, key))


def test_complex_net():
    import rockpool
    from rockpool.nn.modules import LinearTorch, LIFTorch, to_nir, from_nir
    from rockpool.nn.combinators import Sequential
    import torch

    num_in = 2
    num_hidden_1 = 4
    num_hidden_2 = 6
    num_out = 8
    dt = 0.01
    tau_mem = 0.02
    tau_syn = 0.02
    threshold = 1

    net = Sequential(
        LinearTorch((num_in, num_hidden_1)),
        LIFTorch(
            (num_hidden_1,),
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            dt=dt,
            threshold=threshold,
        ),
        LinearTorch((num_hidden_1, num_hidden_2)),
        LIFTorch(
            (num_hidden_2,),
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            dt=dt,
            threshold=threshold,
        ),
        LinearTorch((num_hidden_2, num_out)),
        LIFTorch(
            (num_out,),
            tau_mem=tau_mem,
            tau_syn=tau_mem,
            dt=dt,
            threshold=threshold,
        ),
    )

    time_steps = 10
    torch.manual_seed(1)
    in_test = torch.rand(time_steps, num_in)

    nir_graph = to_nir(net, in_test)
    convert_model = from_nir(nir_graph)


def test_snntorch_nir_rockpool():
    pytest.importorskip("snntorch")
    from snntorch import export_to_nir
    import snntorch as snn
    import torch.nn as nn
    import torch
    from rockpool.nn.modules import from_nir
    import numpy as np

    alpha = 0.5
    beta = 0.5
    num_in = 2
    num_hidden_1 = 4
    num_hidden_2 = 6
    num_out = 8

    net_snntorch = nn.Sequential(
        nn.Linear(num_in, num_hidden_1),
        snn.Synaptic(alpha=alpha, beta=beta, init_hidden=True),
        nn.Linear(num_hidden_1, num_hidden_2),
        snn.Synaptic(alpha=alpha, beta=beta, init_hidden=True),
        nn.Linear(num_hidden_2, num_out),
        snn.Synaptic(alpha=alpha, beta=beta, init_hidden=True),
    )

    x = torch.zeros(2)

    net_nir = export_to_nir(net_snntorch, x)

    net_rockpool = from_nir(net_nir)

    def compare_params(orig, converted):
        for id, (param_orig, param_converted) in enumerate(
            zip(list(orig.parameters()), list(converted.parameters()))
        ):
            assert np.allclose(
                torch.tensor(param_orig.T).detach().numpy(),
                torch.tensor(param_converted).detach().numpy(),
            ), f"Parameter {id} in converted model doesn't match original.\nFound {param_orig} and {param_converted}."

    for mod_id in range(6):
        mod_snntorch = net_snntorch[mod_id]
        mod_rockpool = net_rockpool.get_submodule(f"{mod_id}")
        compare_params(mod_snntorch, mod_rockpool)


def test_import_rnn():
    from rockpool.nn.modules import from_nir
    import torch

    m = from_nir("tests/tests_default/nir_graphs/braille.nir")
    m(torch.empty(1, 1, 12))
