import numpy as np

class DynapSE1NeuronSynapseJax(JaxModule):
    """
    Implements the chip dynamical equations for the DPI neuron and synapse models
    Receives configuration as bias currents
    As few HW restrictions as possible
    [ ] TODO: What is the initial configuration of biases?
    [ ] TODO: How to convert from bias current parameters to high-level parameters and vice versa?
    [ ] TODO: Provides mismatch simulation (as second step)
    As a utility function that operates on a set of parameters?
    As a method on the class?
    """
    def __init__():
        None

    def evolve(self, input: np.ndarray, record: bool = False):
        """
        Simulate a temporal chunk of data
        """

        def forward():
            """
            Single time-step neuron and synapse dynamics
            """
  
    @property
    def tau_mem():
        """ Setter"""
        # - Convert tau -> Itau
        # - Set Itau

    @tau_mem.getter
    def tau_mem():
        """ Convert from Itau -> time constant in seconds """
        # - Convert using code from teili  / brian2 module