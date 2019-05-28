# - Test DynapseControl class and RecDynapSE layer
from warnings import warn

def test_dynapse_control():
    try:
        from NetworksPython.dynapse_control import DynapseControl
        from NetworksPython import dynapse_control
    except ImportError:
        warn("DynapseControl could not be imported. Maybe cortexcontrol is not available.")
    else:
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
    try:
        from NetworksPython.layers import RecDynapSE
    except ImportError:
        warn("RecDynapSE could not be imported. Maybe cortexcontrol is not available.")
    else:
        from NetworksPython import TSEvent
        import numpy as np

        # - Layer generation
        rlDynap = RecDynapSE(
            weights_in=np.array([[1,3,1], [0,3,2]]),
            weights_rec=np.array([[0,2,0], [-1,0,1], [2,-1,0]]),
            vnLayerNeuronIDs=[5,7,9],
            vnVirtualNeuronIDs=[3,4],
            dt=2e-5,
        )

        # - Input
        ts_input = TSEvent(np.arange(3)*1e-4, [0,1,0])

        # - Evolution
        tsOutput = rlDynap.evolve(ts_input)
