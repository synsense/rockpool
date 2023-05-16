import pytest


def test_imports():
    from rockpool.graph.utils import (
        connect_modules,
        bag_graph,
        find_recurrent_modules,
        find_modules_of_subclass,
        replace_module,
    )


def test_connect_modules():
    from rockpool.graph import connect_modules, GraphModule, as_GraphHolder

    gm1 = GraphModule._factory(2, 3)
    gm2 = GraphModule._factory(3, 4)
    connect_modules(gm1, gm2)

    assert gm1.output_nodes == gm2.input_nodes, "Nodes not merged on connection"

    # - GraphHolders should be discarded
    gm1 = GraphModule._factory(2, 3)
    gm2 = GraphModule._factory(3, 4)
    gh2 = as_GraphHolder(gm2)
    connect_modules(gm1, gh2)

    assert (
        gh2 not in gm1.output_nodes[0].sink_modules
    ), "GraphHolder module should have been discarded"

    assert (
        gh2 not in gm2.output_nodes[0].source_modules
    ), "GraphHolder module should have been discarded"

    assert gm1.output_nodes == gm2.input_nodes, "Nodes not merged on connection"

    assert (
        gh2.input_nodes == gm2.input_nodes
    ), "Graph holder nodes not merged on connection"

    # - GraphHolders should be discarded
    gm1 = GraphModule._factory(2, 3)
    gm2 = GraphModule._factory(3, 4)
    gh1 = as_GraphHolder(gm1)
    connect_modules(gh1, gm2)

    assert (
        gh1 not in gm2.input_nodes[0].source_modules
    ), "GraphHolder module should have been discarded"

    assert (
        gh1 not in gm2.output_nodes[0].sink_modules
    ), "GraphHolder module should have been discarded"

    assert gm1.output_nodes == gm2.input_nodes, "Nodes not merged on connection"

    assert (
        gh1.output_nodes == gm2.input_nodes
    ), "Graph holder nodes not merged on connection"

    # - In / Out dimensionality must match
    with pytest.raises(ValueError):
        gm1 = GraphModule._factory(2, 3, "mod1")
        gm2 = GraphModule._factory(2, 3, "mod2")
        connect_modules(gm1, gm2)


def test_connect_modules_with_partial_nodes_connected():
    from rockpool.graph import connect_modules, GraphModule

    # - 1 - Check the situation where part of source nodes connect
    gm1 = GraphModule._factory(2, 4)
    gm2 = GraphModule._factory(2, 4)
    connect_modules(gm1, gm2, [1, 2], None)

    # - Checking from source module output nodes
    for i in range(4):
        if gm1.output_nodes[i].sink_modules:
            assert (
                gm1.output_nodes[i].sink_modules[0] == gm2
            ), f"Nodes {gm1.output_nodes[i]} not merged on dest module, GraphHolder module should have been discarded"

    # - Checking from dest module input nodes
    for i in range(2):
        assert (
            gm2.input_nodes[i].source_modules[0] == gm1
        ), f"Nodes {gm2.input_nodes[i]} not merged on source module, GraphHolder module should have been discarded"

    # - Checking directly if nodes merge
    assert [
        gm1.output_nodes[i] for i in [1, 2]
    ] == gm2.input_nodes, "Nodes not merged on connection"

    # - 2 - Check the situation where part of dest nodes connect
    gm1 = GraphModule._factory(2, 4)
    gm2 = GraphModule._factory(8, 4)
    connect_modules(gm1, gm2, None, [0, 2, 4, 6])

    # - Checking from dest module output nodes
    for i in range(4):
        assert (
            gm1.output_nodes[i].sink_modules[0] == gm2
        ), f"Nodes {gm1.output_nodes[i]} not merged on dest module, GraphHolder module should have been discarded"

    # - Checking from dest module input nodes
    for i in range(8):
        if gm2.input_nodes[i].source_modules:
            assert (
                gm2.input_nodes[i].source_modules[0] == gm1
            ), f"Nodes {gm2.input_nodes[i]} not merged on source module, GraphHolder module should have been discarded"

    # - Checking directly if nodes merge
    assert gm1.output_nodes == [
        gm2.input_nodes[i] for i in [0, 2, 4, 6]
    ], "Nodes not merged on connection"

    # - 3 - Check the situation where part of source and part of dest nodes connect
    gm1 = GraphModule._factory(2, 4)
    gm2 = GraphModule._factory(4, 4)
    connect_modules(gm1, gm2, [0, 3], [1, 2])

    # - Checking from source module output nodes
    for i in range(4):
        if gm1.output_nodes[i].sink_modules:
            assert (
                gm1.output_nodes[i].sink_modules[0] == gm2
            ), f"Nodes {gm1.output_nodes[i]} not merged on dest module, GraphHolder module should have been discarded"

    # - Checking from dest module input nodes
    for i in range(4):
        if gm2.input_nodes[i].source_modules:
            assert (
                gm2.input_nodes[i].source_modules[0] == gm1
            ), f"Nodes {gm2.input_nodes[i]} not merged on source module, GraphHolder module should have been discarded"

    # - Checking directly if nodes merge
    assert [gm1.output_nodes[i] for i in [0, 3]] == [
        gm2.input_nodes[i] for i in [1, 2]
    ], "Nodes not merged on connection"


def test_bag_graph():
    from rockpool.graph import (
        bag_graph,
        connect_modules,
        GraphModule,
        as_GraphHolder,
        GraphHolder,
    )

    # - Make some modules
    gm1 = GraphModule._factory(2, 3, "mod1")
    gm2 = GraphModule._factory(3, 4, "mod2")
    gm3 = GraphModule._factory(4, 2, "mod3")

    # - Make a linear graph
    connect_modules(gm1, gm2)
    connect_modules(gm2, gm3)
    g = GraphHolder(gm1.input_nodes, gm3.output_nodes, "graph", None)

    # - test bagging
    nodes, mods = bag_graph(g)
    assert len(nodes) == 11
    assert len(mods) == 3

    # - Make a loop graph
    connect_modules(gm3, gm1)

    # - Should still work
    nodes, mods = bag_graph(gm1)
    assert len(nodes) == 9
    assert len(mods) == 3


def test_find_modules_of_subclass():
    from rockpool.graph import (
        GraphModule,
        LinearWeights,
        find_modules_of_subclass,
        connect_modules,
        GraphHolder,
    )
    import numpy as np

    gm1 = GraphModule._factory(2, 3, "mod1", None)
    lw = LinearWeights._factory(3, 4, "linear", None, np.empty((3, 4)))
    gm2 = GraphModule._factory(4, 5, "mod2", None)

    connect_modules(gm1, lw)
    connect_modules(lw, gm2)

    g = GraphHolder(gm1.input_nodes, gm2.output_nodes, "graph", None)

    # - Search for linear weights
    mods = find_modules_of_subclass(g, LinearWeights)
    assert len(mods) == 1
    assert mods[0] == lw
    assert type(mods[0]) is LinearWeights

    # - Search for base class (finds everything)
    mods = find_modules_of_subclass(g, GraphModule)
    assert len(mods) == 3


def test_replace_module():
    from rockpool.graph import (
        replace_module,
        GraphModule,
        bag_graph,
        connect_modules,
        GraphHolder,
    )

    gm1 = GraphModule._factory(2, 3, "mod1")
    gm2 = GraphModule._factory(3, 4, "mod2")
    gm2_1 = GraphModule._factory(3, 4, "mod2_1")
    gm3 = GraphModule._factory(4, 5, "mod3")

    connect_modules(gm1, gm2)
    connect_modules(gm2, gm3)

    g = GraphHolder(gm1.input_nodes, gm3.output_nodes, "graph", None)

    replace_module(gm2, gm2_1)

    # - Check that the module was replaced
    nodes, mods = bag_graph(g)

    assert gm2 not in mods, "The original module was not removed"
    assert gm2_1 in mods, "The new module was not wired in"


def test_find_recurrent_modules():
    from rockpool.graph import (
        GraphModule,
        GraphHolder,
        bag_graph,
        connect_modules,
        find_recurrent_modules,
    )

    # - Build a graph with a recurrent loop
    gm1 = GraphModule._factory(2, 3, "mod1")
    gm2_1 = GraphModule._factory(3, 3, "mod2_1")
    gm2_2 = GraphModule(
        input_nodes=gm2_1.output_nodes,
        output_nodes=gm2_1.input_nodes,
        name="mod2_2",
        computational_module=None,
    )
    gm3 = GraphModule._factory(3, 4, "mod3")

    connect_modules(gm1, gm2_1)
    connect_modules(gm2_1, gm3)

    g = GraphHolder(gm1.input_nodes, gm3.output_nodes, "graph", None)

    # - Search for the recurrent modules
    _, rec_mods = find_recurrent_modules(g)
    assert len(rec_mods) == 2, "Found too many modules"
    assert gm2_1 in rec_mods, "Did not find first recurrent module"
    assert gm2_2 in rec_mods, "Did not find second recurrent module"
