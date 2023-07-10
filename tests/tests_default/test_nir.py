import nir
import torch
import pytest
from rockpool.nn.combinators import Sequential
from rockpool.nn.modules import LinearTorch, ExpSynTorch, LIFTorch, to_nir

def test_from_sequential_to_nir():
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
    net = Sequential(
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
