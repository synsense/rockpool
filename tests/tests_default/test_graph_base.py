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

    gmb.add_input(GraphNode())
    gmb.add_input(GraphNode())

    gn = GraphNode()
    gmb.remove_input(gn)
    gmb.remove_output(gn)


def test_GraphNode():
    from rockpool.graph import GraphNode, GraphModuleBase

    gn = GraphNode()
    gmb = GraphModuleBase([], [], "test")

    gn.add_source(gmb)
    gn.add_sink(gmb)
    gn.remove_source(gmb)
    gn.remove_sink(gmb)


def test_GraphModule():
    from rockpool.graph import GraphModule, GraphNode

    gm = GraphModule._factory(2, 3, "test")

    with pytest.raises(ValueError):
        gm = GraphModule._convert_from(gm)


def test_GraphHolder():
    from rockpool.graph import GraphHolder, GraphNode

    GraphHolder._factory(2, 3, "test")
    GraphHolder(
        [GraphNode() for _ in range(2)],
        [GraphNode() for _ in range(3)],
        "test",
    )


def test_as_GraphHolder():
    from rockpool.graph import GraphModule, as_GraphHolder

    gm = GraphModule._factory(2, 3, "test")

    gh = as_GraphHolder(gm)

    assert gh.input_nodes == gm.input_nodes
    assert gh.output_nodes == gm.output_nodes
