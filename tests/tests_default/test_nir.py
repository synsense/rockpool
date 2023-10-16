import nir
import torch
from rockpool.nn.combinators import Sequential
import rockpool
from rockpool.nn.modules import LinearTorch, ExpSynTorch, LIFTorch, to_nir, from_nir
import snntorch as snn
import torch.nn as nn
from snntorch import export
import numpy as np

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
    assert len(graph.nodes) == 6
    assert isinstance(graph.nodes['0_LinearTorch'], nir.Linear)
    assert isinstance(graph.nodes['1_ExpSynTorch'], nir.LI)
    assert isinstance(graph.nodes['2_LinearTorch'], nir.Linear)
    assert isinstance(graph.nodes['3_ExpSynTorch'], nir.LI)
    assert len(graph.edges) == 5


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
    assert len(graph.nodes) == 6
    assert isinstance(graph.nodes['0_LinearTorch'], nir.Linear)
    assert isinstance(graph.nodes['1_LIFTorch'], nir.CubaLIF)
    assert isinstance(graph.nodes['2_LinearTorch'], nir.Linear)
    assert isinstance(graph.nodes['3_LIFTorch'], nir.CubaLIF)
    assert len(graph.edges) == 5


def test_from_linear_to_nir():
    in_features = 2
    out_features = 3
    m = LinearTorch(shape=(in_features, out_features))
    graph = to_nir(m, torch.randn(1, in_features))
    assert len(graph.nodes) == 3
    assert graph.nodes["rockpool"].weight.shape == torch.Size([out_features, in_features])


def test_from_nir_to_linear():
    in_features = 2
    out_features = 3
    m = LinearTorch(shape=(in_features, out_features))
    graph = to_nir(m, torch.randn(1, in_features))
    m2 = from_nir(graph)
    assert m2.rockpool.weight.shape == torch.Size([in_features, out_features])

def test_from_nir_to_sequential():
    timesteps=6

    orig_model = Sequential(
        LinearTorch(shape=(2, 4)),
        ExpSynTorch(tau=torch.ones((4)) * 10.0, shape=4),
        LinearTorch(shape=(4, 8)),
        LIFTorch(tau_mem=torch.ones((8)) * 10.0, shape=8),
    )
    nir_graph = to_nir(orig_model, torch.randn(timesteps, 2))

    convert_model = from_nir(nir_graph)

    for (key1, value1), (key2, value2) in zip(orig_model.modules().items(), dict(convert_model.named_children()).items()):
        assert key1 == key2
        assert value2 == value2

    convert_children = list(convert_model.children())
    torch.testing.assert_allclose(orig_model[0].weight, convert_children[0].weight)
    # torch.testing.assert_allclose(orig_model[0].bias, convert_model[0].bias)
    assert type(orig_model[1]) == type(convert_children[1])
    assert type(orig_model[2]) == type(convert_children[2])
    torch.testing.assert_allclose(orig_model[2].weight, convert_children[2].weight)
    # TODO: Bias not working
    # torch.testing.assert_allclose(orig_model[2].bias, convert_children[2].bias)

def test_complex_net():
    num_in = 2
    num_hidden_1 = 4
    num_hidden_2 = 6
    num_out = 8
    dt = 0.01
    tau_mem = 0.02
    tau_syn = 0.02
    threshold = 1

    net = rockpool.nn.combinators.Sequential(
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
    print(convert_model)


def test_snntorch_nir_rockpool():

    alpha = 0.5
    beta = 0.5
    num_in = 2
    num_hidden_1 = 4
    num_hidden_2 = 6
    num_out = 8

    net_snntorch = nn.Sequential(nn.Linear(num_in, num_hidden_1),
                                 snn.Synaptic(alpha=alpha, beta=beta, init_hidden=True),
                                 nn.Linear(num_hidden_1, num_hidden_2),
                                 snn.Synaptic(alpha=alpha, beta=beta, init_hidden=True),
                                 nn.Linear(num_hidden_2, num_out),
                                 snn.Synaptic(alpha=alpha, beta=beta, init_hidden=True))

    x = torch.zeros(2)

    net_nir = export.to_nir(net_snntorch, x)

    net_rockpool = from_nir(net_nir)

    test0 = (net_snntorch[0].weight.data.T == net_rockpool.modules().get('0_LinearTorch').weight.data)
    test2 = (net_snntorch[2].weight.data.T == net_rockpool.modules().get('2_LinearTorch').weight.data)
    test4 = (net_snntorch[4].weight.data.T == net_rockpool.modules().get('4_LinearTorch').weight.data)
    assert torch.sum(test0).detach().numpy() == num_in * num_hidden_1
    assert torch.sum(test2).detach().numpy() == num_hidden_1 * num_hidden_2
    assert torch.sum(test4).detach().numpy() == num_hidden_2 * num_out

def test_import_rnn():
    m = from_nir("rockpool_nir/tests/tests_default/nir_graphs/braille.nir")
    m(torch.empty(1, 1, 12))


if __name__ == '__main__':
    test_from_linear_to_nir()
    test_from_nir_to_linear()
    test_from_nir_to_sequential()
    test_from_sequential_to_nir()
    test_from_sequential_to_nir_2()
    test_import_rnn()
    test_snntorch_nir_rockpool()
    test_complex_net()
