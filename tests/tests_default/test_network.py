"""
Test the methods of the `Network` class
"""

import numpy as np


def test_imports():
    from rockpool import Network


def test_load_save_net():
    from nn.layers import FFRateEuler, RecRateEuler
    from nn import networks as nw

    lff = FFRateEuler([[1]], name="nestff")
    lrec = RecRateEuler([[1]], [[1]], name="nestrec")

    net = nw.Network([lff, lrec])

    net.save("test_net_torch_config")

    net1 = nw.Network.load("test_net_torch_config")

    assert net.evol_order[0].name == net1.evol_order[0].name
    assert (net.evol_order[0].weights == net1.evol_order[0].weights).all()


def test_load_save_net_structure():
    """
    Test whether network is saved and loaded correctly, including the correct
    network structure.
    """
    from nn.layers import FFRateEuler

    l1 = FFRateEuler(np.random.rand(3, 4))
    l2 = FFRateEuler(np.random.rand(4, 2))
    l3 = FFRateEuler(np.random.rand(4, 3))
    l4 = FFRateEuler(np.random.rand(3, 3))

    net = Network([l1, l3, l4], dt=2)
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


def test_shallow_copy():
    """
    Test whether network is correctly replicated through the `shallow_copy` method.
    """
    from nn.layers import FFRateEuler

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


def test_evolve_on_disk():
    from rockpool import TSContinuous, TSDictOnDisk
    from nn.layers import FFRateEuler

    l1 = FFRateEuler(np.random.rand(3, 4))
    l2 = FFRateEuler(np.random.rand(4, 3))
    l3 = FFRateEuler(np.random.rand(3, 3))

    net = Network([l1, l2, l3], dt=2, evolve_on_disk=True)

    # - Generate an input
    time_trace = np.linspace(0, 10, 100)
    ts_input = TSContinuous(time_trace, np.random.rand(100))

    # - Evolve the network
    resp = net.evolve(ts_input)
    assert isinstance(resp, TSDictOnDisk)
