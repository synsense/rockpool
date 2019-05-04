"""
Test the saving and loading of network model
"""

def test_load_save_net():
    from NetworksPython.layers import FFIAFTorch, FFIAFNest, RecIAFSpkInTorch, RecIAFSpkInNest
    from NetworksPython import networks as nw

    lfftorch = FFIAFNest([[1,2],[3,4]], strName="torchff")
    lrectorch = RecIAFSpkInNest([[1,2],[3,4]],[[1,2],[3,4]], strName="torchrec")
    lffnest = FFIAFTorch([[1]], strName="nestff")
    lrecnest = RecIAFSpkInTorch([[1]],[[1]], strName="nestrec")

    net = nw.Network(lffnest, lrecnest)

    net.save("test_net_torch_config")

    net1 = nw.Network.load("test_net_torch_config")

    assert net.lEvolOrder[0].strName == net1.lEvolOrder[0].strName
    assert (net.lEvolOrder[0].mfW == net1.lEvolOrder[0].mfW).all()
    # net = nw.Network(lfftorch, lrectorch)
    #
    # net.save("test_net_nest_config")
    #
    # net1 = nw.Network.load("test_net_nest_config")
    #
    # assert net.lEvolOrder[0].strName == net1.lEvolOrder[0].strName
