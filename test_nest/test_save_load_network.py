"""
Test the saving and loading of network model
"""


def test_load_save_net():
    from rockpool.layers import FFIAFTorch, FFIAFNest, RecIAFSpkInTorch, RecIAFSpkInNest
    from rockpool import networks as nw

    # lfftorch = FFIAFNest([[1, 2], [3, 4]], name="torchff")
    # lrectorch = RecIAFSpkInNest([[1, 2], [3, 4]], [[1, 2], [3, 4]], name="torchrec")
    lffnest = FFIAFTorch([[1]], name="nestff")
    lrecnest = RecIAFSpkInTorch([[1]], [[1]], name="nestrec")

    net = nw.Network(lffnest, lrecnest)

    net.save("test_net_torch_config")

    net1 = nw.Network.load("test_net_torch_config")

    assert net.evol_order[0].name == net1.evol_order[0].name
    assert (net.evol_order[0].weights == net1.evol_order[0].weights).all()
    # net = nw.Network(lfftorch, lrectorch)
    #
    # net.save("test_net_nest_config")
    #
    # net1 = nw.Network.load("test_net_nest_config")
    #
    # assert net.evol_order[0].name == net1.evol_order[0].name


def test_load_save_net_structure():
    """
    Test whether network is saved and loaded correctly, including the correct
    network structure.
    """
    from rockpool.layers import FFRateEuler
    from rockpool import Network
    import numpy as np

    l1 = FFRateEuler(np.random.rand(3, 4))
    l2 = FFRateEuler(np.random.rand(4, 2))
    l3 = FFRateEuler(np.random.rand(4, 3))
    l4 = FFRateEuler(np.random.rand(3, 3))

    net = Network(l1, l3, l4, dt=2)
    net.add_layer(l2, input_layer=l1)

    net.save("test_net_save_structure")

    net1 = Network.load("test_net_save_structure")

    assert net.dt == net1.dt

    for lyr in net.evol_order:
        lyr1 = getattr(net1, lyr.name)
        if lyr.pre_layer is None:
            assert lyr1.pre_layer is None
        else:
            assert lyr.pre_layer.name == lyr1.pre_layer.name
        assert lyr1.external_input == lyr.external_input
