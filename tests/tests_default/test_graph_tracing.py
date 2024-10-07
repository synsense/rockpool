import pytest


def test_tracing():
    from rockpool.nn.modules import (
        LIF,
        Linear,
    )
    from rockpool.nn.combinators import Sequential
    from rockpool.graph import bag_graph

    # - Sequential jax
    mod = Sequential(
        Linear((2, 3)),
        LIF((3,)),
        Linear((3, 4)),
        LIF((4, 4)),
        Linear((4, 5)),
    )

    # - Get a graph
    g = mod.as_graph()

    # - Test the graph
    nodes, mods = bag_graph(g)
    print(mods)
    assert len(mods) == 5, "Wrong number of modules found"


def test_tracing_jax():
    pytest.importorskip("jax")

    from rockpool.nn.modules import (
        LIFJax,
        LinearJax,
    )
    from rockpool.nn.combinators import Sequential
    from rockpool.graph import bag_graph

    # - Sequential jax
    mod = Sequential(
        LinearJax((2, 3)),
        LIFJax((3,)),
        LinearJax((3, 4)),
        LIFJax((4, 4)),
        LinearJax((4, 5)),
    )

    # - Get a graph
    g = mod.as_graph()

    # - Test the graph
    nodes, mods = bag_graph(g)
    print(mods)
    assert len(mods) == 5, "Wrong number of modules found"


def test_tracing_torch():
    pytest.importorskip("jax")
    pytest.importorskip("torch")

    from rockpool.nn.modules import LinearTorch, LIFTorch
    from rockpool.nn.combinators import Sequential
    from rockpool.graph import bag_graph

    # - Sequential torch
    mod = Sequential(
        LinearTorch((2, 3)),
        LIFTorch((3,)),
        LinearTorch((3, 4)),
        LIFTorch((4, 4), has_rec=True),
        LinearTorch((4, 5)),
    )

    # - Get a graph
    g = mod.as_graph()

    # - Test the graph
    nodes, mods = bag_graph(g)
    assert len(mods) == 6, "Wrong number of modules found"
