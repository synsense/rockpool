# ----
# dynap_hw.py - Implementation of HW FF and Rec layers for DynapSE, via ctxCTL
# ----

from ..layer import Layer
from ...timeseries import TSEvent

import numpy as np
from warnings import warn
from typing import List

# - Imports from ctxCTL
import CtxDynapse
import NeuronNeuronConnector
from CtxDynapse import DynapseCamType as SynapseTypes
from CtxDynapse import DynapseFpgaSpikeGen, DynapseNeuron, EventFilter

def init_dynapse() -> dict:
    """
    init_dynapse - Initialisation function

    :return: dict Global dictionary containing DynapSE HW models
    """
    # - Initialise HW dictionary
    dDynapse = {}

    dDynapse['model'] = CtxDynapse.model()
    lFPGAModules = dDynapse['model'].get_fpga_modules()

    # - Find a spike generator module
    vbIsSpikeGenModule = [isinstance(m, DynapseFpgaSpikeGen) for m in lFPGAModules]
    if not np.any(vbIsSpikeGenModule):
        # - There is no spike generator, so we can't use this Python layer on the HW
        assert ModuleNotFoundError

    else:
        # - Get first spike generator module
        dDynapse['fpgaSpikeGen'] = lFPGAModules[np.argwhere(vbIsSpikeGenModule)[0][0]]

    # - Get all neurons
    dDynapse['lAllNeurons'] = dDynapse['model'].get_neurons()

    # - Initialise neuron allocation
    dDynapse['vbFreeNeurons'] = np.array(True * len(dDynapse['lAllNeurons']))

    # - Wipe configuration
    warn('DynapSE configuration is not wiped -- IMPLEMENT ME --')

    # - Return dictionary
    return dDynapse

# -- Create global dictionary, only initialise on first import of this module
global DHW_dDynapse
if 'DHW_dDynapse' not in dir():
    DHW_dDynapse = init_dynapse()


def allocate_neurons(nNumNeurons: int) -> DynapseNeuron:
    """
    allocate_neurons - Return a list of neurons that may be used. These are guaranteed not to already be assigned.

    :param nNumNeurons: int     The number of neurons requested
    :return:            list    A list of neurons that may be used
    """
    # - Are there sufficient unallocated neurons?
    if np.sum(DHW_dDynapse['vbFreeNeurons']) < nNumNeurons:
        raise MemoryError('Insufficient unallocated neurons available. {}'.format(nNumNeurons) + ' requested.')

    # - Pick the first available neurons
    vnNeuronsToAllocate = np.nonzero(DHW_dDynapse['vbFreeNeurons'])[:nNumNeurons]

    # - Mark these as allocated
    DHW_dDynapse['vbFreeNeurons'][vnNeuronsToAllocate] = False

    # - Return these neurons
    return DHW_dDynapse['lAllNeurons'][vnNeuronsToAllocate]


# -- Define the HW layer class for recurrent networks
class RecDynapSE(Layer):
    """
    RecDynapSE - Recurrent layer implemented on DynapSE
    """
    def __init__(self,
                 mfW: np.ndarray,
                 tDt: float = None,
                 fNoiseStd: float = None,
                 strName: str = 'unnamed',
                 ):
        """
        RecDynapSE - Recurrent layer implemented on DynapSE


        :param mfW:         ndarray NxN matrix of recurrent weights
        :param tDt:         float   Dummy time step. Not used in layer evolution
        :param fNoiseStd    float   Dummy noise to inject. Not used in layer evolution
        :param strName:     str     Layer name
        """
        # - Check supplied arguments
        if tDt is not None:
            warn('Caution: `tDt` is ignored during DynapSE layer evolution.')
        else:
            tDt = 1e-6

        if fNoiseStd is not None:
            warn('Caution: `fNoiseStd` is ignored during DynapSE layer evolution.')
        else:
            fNoiseStd = 0.

        # - Initialise superclass
        super().__init__(mfW, tDt, fNoiseStd, strName)

        # - Convert weight matrix to connectivity list (excitatory)
        vnPreSynE, vnPostSynE = connectivity_matrix_to_prepost_lists(mfW > 0)

        # - Convert weight matrix to connectivity list (inhibitory)
        vnPreSynI, vnPostSynI = connectivity_matrix_to_prepost_lists(mfW < 0)

        # - Get a connector object
        connector = NeuronNeuronConnector.DynapseConnector()

        # - Map neuron indices to neurons
        self._lHWNeurons = allocate_neurons(self.nSize)

        # - Connect the excitatory neurons
        connector.add_connection_from_list(self._lHWNeurons[vnPreSynE],
                                           self._lHWNeurons[vnPostSynE],
                                           [SynapseTypes.SLOW_EXC]
                                           )

        # - Connect the inhibitory neurons
        connector.add_connection_from_list(self._lHWNeurons[vnPreSynI],
                                           self._lHWNeurons[vnPostSynI],
                                           [SynapseTypes.XXX]
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
        # - Convert events to fpga representation
        spikeList = TSEvent_to_spike_list(tsInput, self._lHWNeurons)

        # - Send event sequence to fpga module
        DHW_dDynapse['fpgaSpikeGen'].preload_stimulus(spikeList)
        DHW_dDynapse['fpgaSpikeGen'].set_repeat_mode(False)
        DHW_dDynapse['fpgaSpikeGen'].set_base_addr(0)

        # - Define recording callback
        lEvents = []
        def func_event_callback(events):
            # - Append these events to list
            lEvents.append(events)

        # - Configure recording callback
        oFilter = EventFilter(DHW_dDynapse['model'],
                              callback_function = func_event_callback,
                              id_list = self._lHWNeurons,
                              )

        # - Stimulate / record for desired duration
        DHW_dDynapse['fpgaSpikeGen'].start()
        # - wait for required time
        DHW_dDynapse['fpgaSpikeGen'].stop()

        # - Convert recorded events into TSEvent object
        tsResponse = TSEvent(...)

        # - Trim recorded events if necessary
        tsResponse = tsResponse.clip([0, tDuration])

        # - Return recorded events
        return tsResponse


def connectivity_matrix_to_prepost_lists(mfW: np.ndarray):
    return np.nonzero(mfW)


def TSEvent_to_spike_list(tsSeries: TSEvent, lNeurons: List[DynapseNeuron]):
    """
    TSEvent_to_spike_list - Convert a TSEvent object to a ctxctl spike list

    :param tsSeries:
    :return:
    """
    pass

