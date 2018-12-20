# - Test DynapseControl class and RecDynapSE layer

def test_dynapse_control():
    from NetworksPython.dynapse_control import DynapseControl
    from NetworksPython import dynapse_control
    import os.path
    dc = dynapse_control.DynapseControl()
    dc.load_biases(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "files", "dynapse_biases.py")
    )
    dc.allocate_virtual_neurons(2)
    dc.allocate_hw_neurons(5)
    dc.allocate_hw_neurons(range(10,16))
    try:
        dc.allocate_virtual_neurons([2])
    except MemoryError:
        pass
    else:
        raise AssertionError("Double allocation of virtual neurons has not been detected")
    try:
        dc.allocate_hw_neurons([11])
    except MemoryError:
        pass
    else:
        raise AssertionError("Double allocation of hardware neurons has not been detected")
    dc.connect_to_virtual(2, range(10,13), dc.synSE)
    dc.remove_all_connections_to(range(12, 15), bApplyDiff=True)


def test_dynapse_layer():
    from NetworksPython.layers import RecDynapSE
    from NetworksPython import TSEvent
    import numpy as np

    # - Layer generation
    rlDynap = RecDynapSE(
        mfWIn=np.array([[1,3,1], [0,3,2]]),
        mfWRec=np.array([[0,2,0], [-1,0,1], [2,-1,0]]),
        vnLayerNeuronIDs=[5,7,9],
        vnVirtualNeuronIDs=[3,4],
        tDt=2e-5,
    )

    # - Input
    tsInput = TSEvent(np.arange(3)*1e-4, [0,1,0])

    # - Evolution
    tsOutput = rlDynap.evolve(tsInput)