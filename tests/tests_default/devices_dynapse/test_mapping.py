""" 
Dynap-SE2 API supports converting multi-layer `LIFJax` and `DynapSim` networks to a 2 layer Sequential `LinearJax`, `DynapSim` combination by preserving the connectivity.
The tests listed here explore the limits of the conversion and make sure that the weight placement is healthy.
The tests here use the functions that `devices.dynapse.mapper` uses under the hood to be able to address the problems easily.

:Future reference:
    Now only from `LIFJax` to `DynapSim` conversion support exists; however, when more graph conversion support is added, add more tests here as well!

In case of a failure, first check
    * `devices.dynapse.mapper` and its dependencies
        * `rockpool.devices.dynapse.mapping.container.DynapseGraphContainer`
        * `rockpool.devices.dynapse.mapping.utils.converter.lifnet_to_dynapsim`
    * DynapSE-2 graph implementation `devices.dynapse.DynapseNeurons`
    * LinearJax and LIFJax graph implementations
"""

import pytest
from numpy.testing import assert_equal


def test_imports():
    """
    test_imports is to first make sure that none of the imports raise any errors
    """
    import pytest

    pytest.importorskip("samna")
    pytest.importorskip("jax")
    from rockpool.devices.dynapse import mapper, DynapSim
    from rockpool.nn.modules import LinearJax, LIFJax
    from rockpool.nn.combinators import Sequential


def test_zero():
    """
    test_zero is a negative tests which tries to map an empty network
    """
    import pytest

    pytest.importorskip("samna")
    pytest.importorskip("jax")
    from rockpool.devices.dynapse import mapper, DynapSim

    with pytest.raises(ValueError):
        mapper(DynapSim(0).as_graph())


def test_single_dynapsim():
    """
    test_single_dynapsim creates a standalone `DynapSim` layer and maps it
    """
    import pytest

    pytest.importorskip("samna")
    pytest.importorskip("jax")
    from rockpool.devices.dynapse import mapper, DynapSim

    # - Parameters
    n = 10

    # - Construction
    layer = DynapSim(n)
    spec = mapper(layer.as_graph())

    # - Check atttributes
    assert spec["weights_in"] is None
    assert spec["weights_rec"] is None
    assert spec["Iscale"] == layer.Iscale
    for _, val in spec["unclustered"].items():
        assert len(val) == n

    # - Check once again with recurrent weights
    layer = DynapSim(n, has_rec=True)
    spec = mapper(layer.as_graph())

    # - #
    assert spec["weights_rec"].any()
    assert spec["weights_rec"].shape == (n, n)


def test_dynapsim_net_minimal():
    """
    test_dynapsim_net_minimal creates a minimal sequential `LinearJax` + `DynapSim` network and maps it
    """
    import pytest

    pytest.importorskip("samna")
    pytest.importorskip("jax")
    from rockpool.nn.modules import LinearJax
    from rockpool.nn.combinators import Sequential
    from rockpool.devices.dynapse import mapper, DynapSim

    # - Parameters
    n_in = 10
    n_rec = 5

    # - Construction
    net = Sequential(LinearJax((n_in, n_rec)), DynapSim(n_rec, has_rec=False))
    spec = mapper(net.as_graph())

    # - Check atttributes
    assert spec["weights_in"].shape == (n_in, n_rec)
    assert spec["weights_rec"] is None
    for _, val in spec["unclustered"].items():
        assert len(val) == n_rec

    # - Check once again with recurrent weights
    net = Sequential(LinearJax((n_in, n_rec)), DynapSim(n_rec, has_rec=True))
    spec = mapper(net.as_graph())

    # - #
    assert spec["weights_rec"].any()
    assert spec["weights_rec"].shape == (n_rec, n_rec)


def test_lif_net_minimal():
    """
    test_lif_net_minimal creates a minimal sequential `LinearJax` + `LIFJax` network and maps it
    It includes an extra step of conversion from `LIFJax` layer to `DynapSim` layer
    """
    import pytest

    pytest.importorskip("samna")
    pytest.importorskip("jax")
    from rockpool.nn.modules import LIFJax, LinearJax
    from rockpool.nn.combinators import Sequential
    from rockpool.devices.dynapse import mapper

    # - Parameters
    n_in = 10
    n_rec = 5

    # - Construction
    net = Sequential(LinearJax((n_in, n_rec)), LIFJax(n_rec, has_rec=False))
    spec = mapper(net.as_graph())

    # - Check atttributes
    assert spec["weights_in"].shape == (n_in, n_rec)

    ##  recurrent weight matrix should be all zeros
    assert spec["weights_rec"].shape == (n_rec, n_rec)
    assert not spec["weights_rec"].any()

    for _, val in spec["unclustered"].items():
        assert len(val) == n_rec

    # - Check once again with recurrent weights
    net = Sequential(LinearJax((n_in, n_rec)), LIFJax(n_rec, has_rec=True))
    spec = mapper(net.as_graph())

    # - #
    assert spec["weights_rec"].shape == (n_rec, n_rec)
    assert spec["weights_rec"].any()


def test_dynapsim_net_multi():
    """
    test_dynapsim_net_multi creates a multi-layer sequential `LinearJax` + `DynapSim` network and maps it
    It includes an extra step of concatanating the layers together in one layer
    """
    import pytest

    pytest.importorskip("samna")
    pytest.importorskip("jax")
    from rockpool.nn.modules import LinearJax
    from rockpool.nn.combinators import Sequential
    from rockpool.devices.dynapse import mapper, DynapSim

    # - Parameters
    n_in = 32
    r1 = 16
    r2 = 8
    r3 = 4
    r4 = 2

    # - Construction
    ## - store the layer objects independently
    l_0 = LinearJax((n_in, r1))
    l_1 = DynapSim(r1, has_rec=True)
    l_2 = LinearJax((r1, r2))
    l_3 = DynapSim(r2, has_rec=True)
    l_4 = LinearJax((r2, r3))
    l_5 = DynapSim(r3, has_rec=True)
    l_6 = LinearJax((r3, r4))
    l_7 = DynapSim(r4, has_rec=True)

    ## - #
    net = Sequential(l_0, l_1, l_2, l_3, l_4, l_5, l_6, l_7)
    spec = mapper(net.as_graph())

    ## - Overspannig variables
    __r4 = r1 + r2 + r3 + r4
    __r3 = r1 + r2 + r3
    __r2 = r1 + r2

    ## - Shape and existance check
    assert spec["weights_in"].shape == (n_in, __r4)
    assert spec["weights_rec"].shape == (__r4, __r4)
    assert spec["weights_in"].any()
    assert spec["weights_rec"].any()

    ## - Placement Check
    # Note : to grasp the idea better, you can visualize the weight placement using
    # `seaborn.heatmap(spec["weights_rec"])`

    ### - Input weight matrix should be filled up to r1, then empty
    assert_equal(l_0.weight, spec["weights_in"][:, :r1])
    assert not spec["weights_in"][:, r1:].any()

    ### - Recurrent weight matrix

    #### l1 placement creates an empty space below
    assert_equal(l_1.w_rec, spec["weights_rec"][:r1, :r1])
    assert not spec["weights_rec"][r1:, :r1].any()

    #### l1 + l2 placement creates a empty space on the right hand side
    assert_equal(l_2.weight, spec["weights_rec"][:r1, r1:__r2])
    assert not spec["weights_rec"][:r1, __r2:].any()

    #### l3 placement creates an empty space below
    assert_equal(l_3.w_rec, spec["weights_rec"][r1:__r2, r1:__r2])
    assert not spec["weights_rec"][__r2:, :__r2].any()

    #### l4 placement creates an empty space on both right hand side (left covered by l1-below)
    assert_equal(l_4.weight, spec["weights_rec"][r1:__r2, __r2:__r3])
    assert not spec["weights_rec"][:__r2, __r3:].any()

    #### l5 placement creates an empty space below
    assert_equal(l_5.w_rec, spec["weights_rec"][__r2:__r3, __r2:__r3])
    assert not spec["weights_rec"][__r3:, :__r3].any()

    #### l6 and l7 placements are leaned to the corners bottom and right corners
    assert_equal(l_6.weight, spec["weights_rec"][__r2:__r3, __r3:__r4])
    assert_equal(l_7.w_rec, spec["weights_rec"][__r3:__r4, __r3:__r4])


def test_lif_net_multi():
    """
    test_lif_net_multi is almost the same test `test_dynapsim_net_multi()`.
    The only difference is it includes an extra step of conversion from `LIFJax` layer to `DynapSim` layer
    """
    import pytest

    pytest.importorskip("samna")
    pytest.importorskip("jax")
    from rockpool.nn.modules import LIFJax, LinearJax
    from rockpool.nn.combinators import Sequential
    from rockpool.devices.dynapse import mapper, DynapSim

    # - Parameters
    n_in = 32
    r1 = 16
    r2 = 8
    r3 = 4
    r4 = 2

    # - Construction
    ## - store the layer objects independently
    l_0 = LinearJax((n_in, r1))
    l_1 = LIFJax(r1, has_rec=True)
    l_2 = LinearJax((r1, r2))
    l_3 = LIFJax(r2, has_rec=True)
    l_4 = LinearJax((r2, r3))
    l_5 = LIFJax(r3, has_rec=True)
    l_6 = LinearJax((r3, r4))
    l_7 = LIFJax(r4, has_rec=True)

    ## - #
    net = Sequential(l_0, l_1, l_2, l_3, l_4, l_5, l_6, l_7)
    spec = mapper(net.as_graph())

    ## - Overspannig variables
    __r4 = r1 + r2 + r3 + r4
    __r3 = r1 + r2 + r3
    __r2 = r1 + r2

    ## - Shape and existance check
    assert spec["weights_in"].shape == (n_in, __r4)
    assert spec["weights_rec"].shape == (__r4, __r4)
    assert spec["weights_in"].any()
    assert spec["weights_rec"].any()

    ## - Placement Check
    # Note : to grasp the idea better, you can visualize the weight placement using
    # `seaborn.heatmap(spec["weights_rec"])`

    ### - Input weight matrix should be filled up to r1, then empty
    assert_equal(l_0.weight, spec["weights_in"][:, :r1])
    assert not spec["weights_in"][:, r1:].any()

    ### - Recurrent weight matrix

    #### l1 placement creates an empty space below
    assert_equal(l_1.w_rec, spec["weights_rec"][:r1, :r1])
    assert not spec["weights_rec"][r1:, :r1].any()

    #### l1 + l2 placement creates a empty space on the right hand side
    assert_equal(l_2.weight, spec["weights_rec"][:r1, r1:__r2])
    assert not spec["weights_rec"][:r1, __r2:].any()

    #### l3 placement creates an empty space below
    assert_equal(l_3.w_rec, spec["weights_rec"][r1:__r2, r1:__r2])
    assert not spec["weights_rec"][__r2:, :__r2].any()

    #### l4 placement creates an empty space on both right hand side (left covered by l1-below)
    assert_equal(l_4.weight, spec["weights_rec"][r1:__r2, __r2:__r3])
    assert not spec["weights_rec"][:__r2, __r3:].any()

    #### l5 placement creates an empty space below
    assert_equal(l_5.w_rec, spec["weights_rec"][__r2:__r3, __r2:__r3])
    assert not spec["weights_rec"][__r3:, :__r3].any()

    #### l6 and l7 placements are leaned to the corners bottom and right corners
    assert_equal(l_6.weight, spec["weights_rec"][__r2:__r3, __r3:__r4])
    assert_equal(l_7.w_rec, spec["weights_rec"][__r3:__r4, __r3:__r4])


def test_negative_consecutive_lif():
    """
    test_negative_consecutive_lif tries to map a network with consecutive `LIFJax` layers and results raising an error
    """
    import pytest

    pytest.importorskip("samna")
    pytest.importorskip("jax")

    from rockpool.nn.modules import LIFJax, LinearJax
    from rockpool.nn.combinators import Sequential
    from rockpool.devices.dynapse import mapper

    # - Parameters
    n_in = 32
    n_rec = 16

    # - Construction
    net = Sequential(
        LinearJax((n_in, n_rec)),
        LIFJax(n_rec, has_rec=True),
        LIFJax(n_rec, has_rec=True),
    )

    # - Get the error
    with pytest.raises(ValueError):
        spec = mapper(net.as_graph())


def test_negative_consecutive_linear():
    """
    test_negative_consecutive_linear tries to map a network with consecutive `LinearJax` layers and results raising an error
    """
    import pytest

    pytest.importorskip("samna")
    pytest.importorskip("jax")

    from rockpool.nn.modules import LIFJax, LinearJax
    from rockpool.nn.combinators import Sequential
    from rockpool.devices.dynapse import mapper

    # - Parameters
    n_in = 32
    r1 = 16
    r2 = 8

    # - Construction
    net = Sequential(
        LinearJax((n_in, r1)), LinearJax((r1, r2)), LIFJax(r2, has_rec=True)
    )

    # - Get the error
    with pytest.raises(ValueError):
        spec = mapper(net.as_graph())
