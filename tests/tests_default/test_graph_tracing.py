import pytest


def test_tracing():
    from rockpool.nn.modules import (
        LIFJax,
        LIFTorch,
        LinearJax,
        LinearTorch,
        RateEulerJax,
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
