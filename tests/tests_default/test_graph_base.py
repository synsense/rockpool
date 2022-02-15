import pytest


def test_imports():
    from rockpool import graph


def test_SetList():
    from rockpool.graph import SetList

    # - Bare object
    sl = SetList()

    # - From list
    sl = SetList([0, 1, 2])

    # - With non-unique init
    sl = SetList([0, 1, 1, 2])
    assert len(sl) == 3, "SetList did not reject repeated inputs"

    # - Test append
    sl = SetList([0, 1])
    sl.append(2)
    assert sl[-1] == 2

    sl = SetList([0, 1])
    sl.append(0)
    assert sl[-1] == 1
    assert len(sl) == 2

    # - Test extend
    sl = SetList([0, 1])
    sl.extend([2, 1, 0])
    assert sl[-1] == 2
    assert len(sl) == 3


def test_GraphModuleBase():
    from rockpool.graph import GraphModuleBase

    gmb = GraphModuleBase._factory(2, 3, "test")

    gmb = GraphModuleBase._factory(0, 3, "test")

    from rockpool.graph import GraphNode

    gn0 = GraphNode()
    gn1 = GraphNode()
    gmb.add_input(gn0)
    gmb.add_output(gn1)

    assert gn0 in gmb.input_nodes
    assert gn1 in gmb.output_nodes

    gmb.remove_input(gn0)
    assert gn0 not in gmb.input_nodes

    gn = GraphNode()
    gmb.remove_input(gn)
    gmb.remove_output(gn)

    assert len(gmb.output_nodes) == 4


def test_GraphNode():
    from rockpool.graph import GraphNode, GraphModuleBase

    gn = GraphNode()
    gmb = GraphModuleBase([], [], "test", None)

    gn.add_source(gmb)
    gn.add_sink(gmb)
    gn.remove_source(gmb)
    gn.remove_sink(gmb)


def test_GraphModule():
    from rockpool.graph import GraphModule, GraphNode

    gm = GraphModule._factory(2, 3, "test")

    with pytest.raises(ValueError):
        gm = GraphModule._convert_from(gm)

    gn0 = GraphNode()
    gn1 = GraphNode()

    gm.add_input(gn0)
    assert gn0 in gm.input_nodes
    assert gm in gn0.sink_modules

    gm.remove_input(gn0)
    assert gn0 not in gm.input_nodes
    assert gm not in gn0.sink_modules

    gm.clear_inputs()
    assert len(gm.input_nodes) == 0

    gm.add_output(gn1)
    assert gn1 in gm.output_nodes
    assert gm in gn1.source_modules

    gm.remove_output(gn1)
    assert gn1 not in gm.output_nodes
    assert gm not in gn1.source_modules

    gm.clear_outputs()
    assert len(gm.output_nodes) == 0


def test_GraphHolder():
    from rockpool.graph import GraphHolder, GraphNode

    GraphHolder._factory(2, 3, "test")
    GraphHolder(
        [GraphNode() for _ in range(2)],
        [GraphNode() for _ in range(3)],
        "test",
        None,
    )


def test_as_GraphHolder():
    from rockpool.graph import GraphModule, as_GraphHolder

    gm = GraphModule._factory(2, 3, "test")

    gh = as_GraphHolder(gm)

    assert gh.input_nodes == gm.input_nodes
    assert gh.output_nodes == gm.output_nodes
