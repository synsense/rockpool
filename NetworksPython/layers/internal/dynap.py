###
# dynapse.py - Classes implementing layers via ctxctl for controlling the DynapSE
###

from ..layer import Layer
from ...timeseries import TSEvent

import numpy as np

import CtxDynapse
import NeuronNeuronConnector
from CtxDynapse import DynapseCamType as SynapseTypes

class RecDynapSE(Layer):
    """
    RecDynapSE - Recurrent layer implemented on DynapSE
    """
    def __init__(self,
                 mfW: np.ndarray,
                 ):
        # - Convert weight matrix to connectivity list (excitatory)
        vnPreSynE, vnPostSynE = connectivity_matrix_to_prepost_lists(mfW > 0)

        # - Convert weight matrix to connectivity list (inhibitory)
        vnPreSynI, vnPostSynI = connectivity_matrix_to_prepost_lists(mfW < 0)

        # - Get a list of HW neurons on the DynapSE
        lHWNeurons = CtxDynapse.model.get_neurons()

        # - Get a connector object
        connector = NeuronNeuronConnector.DynapseConnector()

        # - Map neuron indices to neurons
        # lPreSynE = lHWNeurons[vnPreSynE]
        # lPostSynE = lHWNeurons[vnPostSynE]
        # lPreSynI = lHWNeurons[vnPreSynI]
        # lPostSynI = lHWNeurons[vnPostSynI]

        # - Connect the excitatory neurons
        connector.add_connection_from_list(lHWNeurons[vnPreSynE],
                                           lHWNeurons[vnPostSynE],
                                           [SynapseTypes.SLOW_EXC]
                                           )

        # - Connect the inhibitory neurons
        connector.add_connection_from_list(lHWNeurons[vnPreSynI],
                                           lHWNeurons[vnPostSynI],
                                           [SynapseTypes.SLOW_EXC]
                                           )

    def evolve(self,
               tsInput: TSEvent = None,
               tDuration: float = None,
               ):
        """
        evolve - Evolve the layer by queueing spikes, stimulating and recording

        :param tsInput:
        :param tDuration:
        :return:
        """
        # - Get input events from tsInput


        pass


def connectivity_matrix_to_prepost_lists(mfW: np.ndarray):
    return np.nonzero(mfW)

def TSEvent_to_spike_list(tsSeries: TSEvent):
    """
    TSEvent_to_spike_list - Convert a TSEvent object to a ctxctl spike list

    :param tsSeries:
    :return:
    """
    pass