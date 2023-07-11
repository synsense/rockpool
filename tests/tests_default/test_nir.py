import nir
import torch
# from rockpool.nn.combinators import Sequential
import rockpool
from rockpool.nn.modules import LinearTorch, ExpSynTorch, LIFTorch, to_nir, from_nir

def test_from_sequential_to_nir():
    Nin = 2
    Nhidden = 4
    Nout = 2
    dt = 1e-3
    net = rockpool.nn.combinators.Sequential(
        LinearTorch((Nin, Nhidden), has_bias=False),
        ExpSynTorch(Nhidden, dt=dt),
        LinearTorch((Nhidden, Nout), has_bias=False),
        ExpSynTorch(Nout, dt=dt),
    )
    graph = to_nir(net, torch.randn(1, 2))
    assert len(graph.nodes) == 4
    assert isinstance(graph.nodes[0], nir.Linear)
    assert isinstance(graph.nodes[1], nir.LI)
    assert isinstance(graph.nodes[2], nir.Linear)
    assert isinstance(graph.nodes[3], nir.LI)
    assert len(graph.edges) == 3


def test_from_sequential_to_nir_2():
    Nin = 2
    Nhidden = 4
    Nout = 2
    dt = 1e-3
    net = rockpool.nn.combinators.Sequential(
        LinearTorch((Nin, Nhidden), has_bias=False),
        LIFTorch(Nhidden, dt=dt),
        LinearTorch((Nhidden, Nout), has_bias=False),
        LIFTorch(Nout, dt=dt),
    )
    graph = to_nir(net, torch.randn(1, 2))
    assert len(graph.nodes) == 4
    assert isinstance(graph.nodes[0], nir.Linear)
    assert isinstance(graph.nodes[1], nir.CubaLIF)
    assert isinstance(graph.nodes[2], nir.Linear)
    assert isinstance(graph.nodes[3], nir.CubaLIF)
    assert len(graph.edges) == 3



def test_from_linear_to_nir():
    in_features = 2
    out_features = 3
    m = LinearTorch(shape=(in_features, out_features))
    graph = to_nir(m, torch.randn(1, in_features))
    assert len(graph.nodes) == 1
    assert graph.nodes[0].weight.shape == torch.Size([in_features, out_features])


def test_from_nir_to_sequential():
    timesteps=6

    orig_model = rockpool.nn.combinators.Sequential(
        LinearTorch(shape=(5, 4)),
        ExpSynTorch(tau=10.0, shape=4),
        LinearTorch(shape=(4, 3)),
        LIFTorch(tau_mem=10.0, shape=3),
    )
    nir_graph = to_nir(orig_model, torch.randn(timesteps, 5))

    convert_model = from_nir(nir_graph)

    for (key1, value1), (key2, value2) in zip(orig_model.modules().items(), convert_model.modules().items()):
        assert key1 == key2
        assert value2 == value2

    torch.testing.assert_allclose(orig_model[0].weight, convert_model[0].weight)
    # torch.testing.assert_allclose(orig_model[0].bias, convert_model[0].bias)
    assert type(orig_model[1]) == type(convert_model[1])
    assert type(orig_model[2]) == type(convert_model[2])
    torch.testing.assert_allclose(orig_model[2].weight, convert_model[2].weight)
    # torch.testing.assert_allclose(orig_model[2].bias, convert_model[2].bias)
    # TODO: noise is not working
