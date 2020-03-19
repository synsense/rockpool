"""
Test the methods of the `Network` class
"""

from rockpool import Network
import numpy as np


def test_shallow_copy():
    """
    Test whether network is correctly replicated through the `shallow_copy` method.
    """
    from rockpool.layers import FFRateEuler

    l1 = FFRateEuler(np.random.rand(3, 4))
    l2 = FFRateEuler(np.random.rand(4, 2))
    l3 = FFRateEuler(np.random.rand(4, 3))
    l4 = FFRateEuler(np.random.rand(3, 3))

    net = Network(l1, l3, l4, dt=2)
    net.add_layer(l2, input_layer=l1)

    net1 = net.shallow_copy()

    assert net.dt == net1.dt
    assert net.input_layer is net.input_layer
    for lyr, lyr1 in zip(net.evol_order, net1.evol_order):
        assert lyr is lyr1


def test_evolve_on_disk():
    from rockpool import TSContinuous, TSDictOnDisk
    from rockpool.layers import FFRateEuler

    l1 = FFRateEuler(np.random.rand(3, 4))
    l2 = FFRateEuler(np.random.rand(4, 3))
    l3 = FFRateEuler(np.random.rand(3, 3))

    net = Network(l1, l2, l3, dt=2, evolve_on_disk=True)

    # - Generate an input
    time_trace = np.linspace(0, 10, 100)
    ts_input = TSContinuous(time_trace, np.random.rand(100))

    # - Evolve the network
    resp = net.evolve(ts_input)
    assert isinstance(resp, TSDictOnDisk)
