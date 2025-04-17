import pytest

pytest.importorskip("xylosim")


def test_imports():
    from rockpool.devices.xylo.syns61201 import mapper

    from rockpool.devices.xylo.syns61201 import (
        Xylo2Neurons,
        Xylo2HiddenNeurons,
        Xylo2OutputNeurons,
    )


def test_mapper():
    from rockpool.nn.modules import LinearTorch, LIFTorch
    from rockpool.nn.combinators import Sequential
    from rockpool.devices.xylo.syns61201 import mapper

    smod = Sequential(
        LinearTorch((5, 6)),
        LIFTorch(
            (6, 6),
            has_rec=True,
        ),
        LinearTorch((6, 6)),
        LIFTorch((6,)),
        LinearTorch((6, 8)),
        LIFTorch((8,), has_rec=True),
        LinearTorch((8, 3)),
        LIFTorch((3, 3), has_rec=False),
    )

    g = smod.as_graph()

    specs = mapper(g)


def test_output_nodes_have_neurons_as_source():
    from rockpool.nn.modules import LinearTorch, LIFTorch
    from rockpool.nn.combinators import Sequential
    from rockpool.devices.xylo.syns61201 import mapper, DRCError

    # - Output nodes with weights as last layer
    smod = Sequential(
        LinearTorch((2, 3)),
        LIFTorch((3,)),
        LinearTorch((3, 4)),
    )

    with pytest.raises(DRCError):
        mapper(smod.as_graph())


def test_input_to_neurons_is_a_weight():
    from rockpool.nn.modules import LinearTorch, LIFTorch
    from rockpool.nn.combinators import Sequential
    from rockpool.devices.xylo.syns61201 import mapper, DRCError

    # - Neuron to neuron connection with no weights
    smod = Sequential(
        LinearTorch((2, 3)),
        LIFTorch((3, 4)),
        LIFTorch((4, 5)),
    )

    with pytest.raises(DRCError):
        mapper(smod.as_graph())


def test_first_module_is_a_weight():
    from rockpool.nn.modules import LinearTorch, LIFTorch
    from rockpool.nn.combinators import Sequential
    from rockpool.devices.xylo.syns61201 import mapper, DRCError

    # - Network with no weights on input
    smod = Sequential(
        LIFTorch((2,)),
        LinearTorch((2, 3)),
        LIFTorch((3, 4)),
    )

    with pytest.raises(DRCError):
        mapper(smod.as_graph())


def test_le_16_input_channels():
    from rockpool.nn.modules import LinearTorch, LIFTorch
    from rockpool.nn.combinators import Sequential
    from rockpool.devices.xylo.syns61201 import mapper, DRCWarning

    # - Network with no weights on input
    smod = Sequential(
        LinearTorch((32, 2)),
        LIFTorch((2,)),
        LinearTorch((2, 3)),
        LIFTorch((3, 3)),
    )

    with pytest.warns(DRCWarning):
        mapper(smod.as_graph())


def test_le_8_output_channels():
    from rockpool.nn.modules import LinearTorch, LIFTorch
    from rockpool.nn.combinators import Sequential
    from rockpool.devices.xylo.syns61201 import mapper, DRCError, DRCWarning

    # - Network with > 8 output channels
    smod = Sequential(
        LinearTorch((4, 2)),
        LIFTorch((2,)),
        LinearTorch((2, 32)),
        LIFTorch((32, 32)),
    )

    with pytest.warns(DRCWarning):
        mapper(smod.as_graph(), max_output_neurons=32)


def test_network_too_large():
    from rockpool.nn.modules import LinearTorch, LIFTorch
    from rockpool.nn.combinators import Sequential
    from rockpool.devices.xylo.syns61201 import mapper, DRCError

    # - Network with too many hidden neurons
    smod = Sequential(
        LinearTorch((2, 2000)),
        LIFTorch((2000,)),
        LinearTorch((2000, 4)),
        LIFTorch((4,)),
    )

    with pytest.raises(DRCError):
        mapper(smod.as_graph())

    mapper(smod.as_graph(), max_hidden_neurons=2000)

    # - Network with too many output neurons
    smod = Sequential(
        LinearTorch((2, 3)),
        LIFTorch((3,)),
        LinearTorch((3, 64)),
        LIFTorch((64,)),
    )

    with pytest.raises(DRCError):
        mapper(smod.as_graph())

    mapper(smod.as_graph(), max_output_neurons=64)


def test_all_neurons_have_same_dt():
    from rockpool.nn.modules import LinearTorch, LIFTorch
    from rockpool.nn.combinators import Sequential
    from rockpool.devices.xylo.syns61201 import mapper, DRCError

    # - Network with no weights on input
    smod = Sequential(
        LinearTorch((2, 3)),
        LIFTorch((3,), dt=10e-3),
        LinearTorch((3, 4)),
        LIFTorch((4,), dt=20e-3),
    )

    with pytest.raises(DRCError):
        mapper(smod.as_graph())


def test_output_neurons_cannot_be_recurrent():
    from rockpool.nn.modules import LinearTorch, LIFTorch
    from rockpool.nn.combinators import Sequential
    from rockpool.devices.xylo.syns61201 import mapper, DRCError

    # - Network with no weights on input
    smod = Sequential(
        LinearTorch((2, 3)),
        LIFTorch((3,)),
        LinearTorch((3, 4)),
        LIFTorch((4, 4), has_rec=True),
    )

    with pytest.raises(DRCError):
        mapper(smod.as_graph())


def test_network_too_large():
    from rockpool.nn.modules import LinearTorch, LIFTorch
    from rockpool.nn.combinators import Sequential
    from rockpool.devices.xylo.syns61201 import mapper, DRCError

    # - Network with too many hidden neurons
    smod = Sequential(
        LinearTorch((2, 2000)),
        LIFTorch((2000,)),
        LinearTorch((2000, 4)),
        LIFTorch((4,)),
    )

    with pytest.raises(DRCError):
        mapper(smod.as_graph())

    # - Network with too many output neurons
    smod = Sequential(
        LinearTorch((2, 3)),
        LIFTorch((3,)),
        LinearTorch((3, 64)),
        LIFTorch((64,)),
    )

    with pytest.raises(DRCError):
        mapper(smod.as_graph())


def test_XyloSim_creation():
    from rockpool.utilities.backend_management import backend_available

    if not backend_available("xylosim", "samna"):
        return

    from rockpool.devices.xylo.syns61201 import mapper, XyloSim
    from rockpool.nn.modules import LinearTorch, LIFTorch
    from rockpool.nn.combinators import Sequential

    smod = Sequential(
        LinearTorch((5, 6)),
        LIFTorch(
            (6, 6),
            has_rec=True,
        ),
        LinearTorch((6, 6)),
        LIFTorch((6,)),
        LinearTorch((6, 8)),
        LIFTorch((8,), has_rec=True),
        LinearTorch((8, 3)),
        LIFTorch((3, 3), has_rec=False),
    )

    specs = mapper(
        smod.as_graph(), weight_dtype=int, dash_dtype=int, threshold_dtype=int
    )
    specs.pop("mapped_graph")
    xcmod = XyloSim.from_specification(**specs)
