"""
Test the saving and loading of network model
"""

def test_load_save_net():
    from NetworksPython.layers import FFIAFTorch, FFIAFNest, RecIAFSpkInTorch, RecIAFSpkInNest
    from NetworksPython import networks as nw

    lfftorch = FFIAFTorch([1], strName="torchff")
    lrectorch = RecIAFSpkInTorch([1],[1], strName="torchrec")
    lffnest = FFIAFNest([[1]], strName="nestff")
    lrecnest = RecIAFSpkInNest([[1]],[[1]], strName="nestrec")

    net = nw.Network(lffnest, lrecnest)

    net.save("test_net_torch_config")

    net1 = nw.Network.load("test_net_torch_config")

    assert net.lEvolOrder[0].strName == net1.lEvolOrder[0].strName
    net = nw.Network(lfftorch, lrectorch)

    net.save("test_net_nest_config")

    net1 = nw.Network.load("test_net_nest_config")

    assert net.lEvolOrder[0].strName == net1.lEvolOrder[0].strName
