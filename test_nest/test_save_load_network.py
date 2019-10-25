"""
Test the saving and loading of network model
"""

def test_load_save_net():
    from rockpool.layers import FFIAFTorch, FFIAFNest, RecIAFSpkInTorch, RecIAFSpkInNest
    from rockpool import networks as nw

    lfftorch = FFIAFNest([[1,2],[3,4]], name="torchff")
    lrectorch = RecIAFSpkInNest([[1,2],[3,4]],[[1,2],[3,4]], name="torchrec")
    lffnest = FFIAFTorch([[1]], name="nestff")
    lrecnest = RecIAFSpkInTorch([[1]],[[1]], name="nestrec")

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