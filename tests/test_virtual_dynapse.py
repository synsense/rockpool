import numpy as np

from NetworksPython.layers import VirtualDynapse


def test_virtualdynapse_instantiation():
    # - Instantiation
    # TODO
    vd = VirtualDynapse()

    # - Weight validation
    wegihts_ok_fanin = np.zeros((4096, 4096))
    wegihts_ok_fanin[0, :30] = 1
    wegihts_ok_fanin[0, 30:64] = -1
    assert vd.validate_connections(wegihts_ok_fanin) == 0
    wegihts_high_fanin = wegihts_ok_fanin.copy()
    wegihts_high_fanin[0, 65] = -1
    assert vd.validate_connections(wegihts_high_fanin) == 1
    weights_ok_fanout = np.zeros((4096, 4096))
    weights_ok_fanout[0, [2, 1050, 2100]] = [1, 2, -1]
    assert vd.validate_connections(weights_ok_fanout) == 0
    weights_wrong_fanout = weights_ok_fanout.copy()
    weights_wrong_fanout[1000, 4000] = 1
    assert vd.validate_connections(weights_wrong_fanout) == 2
    weights_no_aliasing = np.zeros((4096, 4096))
    weights_no_aliasing[1024, 1] = 1
    weights_no_aliasing[
        2049, 4
    ] = 1  # Different presyn. ID and different presyn. chip -> no aliasing
    weights_no_aliasing[
        1025, 4
    ] = 1  # Different presyn. ID but same presyn. chip -> no aliasing
    weights_no_aliasing[
        1024, 5
    ] = 1  # Different presyn. ID but same presyn. chip -> no aliasing
    weights_no_aliasing[
        2048, 257
    ] = (
        1
    )  # Same presyn. ID, different presyn. chip but different postsyn. core -> no aliasing
    assert vd.validate_connections(weights_no_aliasing) == 0
    weights_aliasing = weights_no_aliasing.copy()
    weights_aliasing[2048, 1] = 1  # Same presyn. ID, same postsyn. core -> aliasing
    assert vd.validate_connections(weights_aliasing) == 4
