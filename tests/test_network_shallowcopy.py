"""
Test the shallow_copy method of the `Network` class
"""


def test_load_save_net_structure():
    """
    Test whether network is correctly replicated through the `shallow_copy` method.
    """
    from rockpool.layers import FFRateEuler
    from rockpool import Network
    import numpy as np

    l1 = FFRateEuler(np.random.rand(3, 4))
    l2 = FFRateEuler(np.random.rand(4, 2))
    l3 = FFRateEuler(np.random.rand(4, 3))
    l4 = FFRateEuler(np.random.rand(3, 3))

    net = Network([l1, l3, l4], dt=2)
    net.add_layer(l2, input_layer=l1)

    net1 = net.shallow_copy()

    assert net.dt == net1.dt
    assert net.input_layer is net.input_layer
    for lyr, lyr1 in zip(net.evol_order, net1.evol_order):
        assert lyr is lyr1
