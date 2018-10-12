# ----
# dynap_hw.py - Implementation of HW FF and Rec layers for DynapSE, via ctxCTL
# ----

from ..layer import Layer
from ...timeseries import TSEvent

import numpy as np
from warnings import warn
from typing import List, Optional

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

    dDynapse['model'] = CtxDynapse.model
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
    dDynapse['vbFreeInputNeurons'] = np.array(True * 256)
    dDynapse['vbFreeLayerNeurons'] = np.array(True * len(dDynapse['lAllNeurons']))
    dDynapse['vbFreeLayerNeurons'][:256] = False

    # - Wipe configuration
    warn('DynapSE configuration is not wiped -- IMPLEMENT ME --')

    # - Return dictionary
    return dDynapse

# -- Create global dictionary, only initialise on first import of this module
global DHW_dDynapse
if 'DHW_dDynapse' not in dir():
    DHW_dDynapse = init_dynapse()


def allocate_layer_neurons(nNumNeurons: int) -> DynapseNeuron:
    """
    allocate_layer_neurons - Return a list of neurons that may be used. These are guaranteed not to already be assigned.

    :param nNumNeurons: int     The number of neurons requested
    :return:            list    A list of neurons that may be used
    """
    # - Are there sufficient unallocated neurons?
    if np.sum(DHW_dDynapse['vbFreeLayerNeurons']) < nNumNeurons:
        raise MemoryError('Insufficient unallocated neurons available. {}'.format(nNumNeurons) + ' requested.')

    # - Pick the first available neurons
    vnNeuronsToAllocate = np.nonzero(DHW_dDynapse['vbFreeLayerNeurons'])[:nNumNeurons]

    # - Mark these as allocated
    DHW_dDynapse['vbFreeLayerNeurons'][vnNeuronsToAllocate] = False

    # - Return these neurons
    return DHW_dDynapse['lAllNeurons'][vnNeuronsToAllocate]


def allocate_input_neurons(nNumNeurons: int) -> DynapseNeuron:
    """
    allocate_input_neurons - Return a list of neurons that may be used. These are guaranteed not to already be assigned.

    :param nNumNeurons: int     The number of neurons requested
    :return:            list    A list of neurons that may be used
    """
    # - Are there sufficient unallocated neurons?
    if np.sum(DHW_dDynapse['vbFreeInputNeurons']) < nNumNeurons:
        raise MemoryError('Insufficient unallocated neurons available. {}'.format(nNumNeurons) + ' requested.')

    # - Pick the first available neurons
    vnNeuronsToAllocate = np.nonzero(DHW_dDynapse['vbFreeInputNeurons'])[:nNumNeurons]

    # - Mark these as allocated
    DHW_dDynapse['vbFreeInputNeurons'][vnNeuronsToAllocate] = False

    # - Return these neurons
    return DHW_dDynapse['lAllNeurons'][vnNeuronsToAllocate]


# -- Define the HW layer class for recurrent networks
class RecDynapSE(Layer):
    """
    RecDynapSE - Recurrent layer implemented on DynapSE
    """
    def __init__(self,
                 mfWRec: np.ndarray,
                 mfWIn: np.ndarray,
                 tDt: Optional[float] = None,
                 fNoiseStd: Optional[float] = None,
                 strName: Optional[str] = 'unnamed',
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
        super().__init__(mfW = mfWRec, tDt = tDt, fNoiseStd = fNoiseStd, strName = strName)
        self._nSizeIn = mfWIn.shape[0]

        # - Map neuron indices to neurons
        self._lHWInputNeurons = allocate_input_neurons(self.nSizeIn)
        self._lHWLayerNeurons = allocate_layer_neurons(self.nSize)

    def evolve(self,
               tsInput: Optional[TSEvent] = None,
               tDuration: Optional[float] = None,
               nNumTimeSteps: Optional[int] = None,
               ) -> TSEvent:
        """
        evolve - Evolve the layer by queueing spikes, stimulating and recording

        :param tsInput:         TSEvent input time series, containing `self.nSize` channels
        :param tDuration:       float   Desired evolution duration, in seconds
        :param nNumTimeSteps:   int     Desired evolution duration, in integer steps of `self.tDt`

        :return:                TSEvent spikes emitted by the neurons in this layer, during the evolution time
        """
        # - Compute duration for evolution
        if tDuration is None:
            if nNumTimeSteps is None:
                # - Check that we have an input time series
                assert tsInput is not None, \
                    '`tsInput` must be provided, if no evolution duration is specified.'

                # - Use the duration of the input time series
                tDuration = tsInput.tDuration

            else:
                # - Compute the evolution duration using the number of supplied time steps
                tDuration = nNumTimeSteps * self.tDt

        # - Get input events from tsInput
        # - Convert events to fpga representation
        spikeList = TSEvent_to_spike_list(tsInput, self._lHWNeurons)

        # - Send event sequence to fpga module
        DHW_dDynapse['fpgaSpikeGen'].preload_stimulus(spikeList)
        DHW_dDynapse['fpgaSpikeGen'].set_repeat_mode(False)
        DHW_dDynapse['fpgaSpikeGen'].set_base_addr(0)

        # - Define recording callback
        lEvents = []
        def func_event_callback(lTheseEvents):
            # - Append these events to list
            lEvents.append(lTheseEvents)

        # - Configure recording callback
        oFilter = EventFilter(DHW_dDynapse['model'],
                              callback_function = func_event_callback,
                              id_list = [n.get_id() for n in self._lHWLayerNeurons],
                              )

        # - Reset FPGA timestamp
        warn('FPGA timestamp not reset --- IMPLEMENT ME ---')

        # - Stimulate / record for desired duration
        DHW_dDynapse['fpgaSpikeGen'].start()
        # - wait for required time
        DHW_dDynapse['fpgaSpikeGen'].stop()

        # - Flatten list of events
        lEvents = [event for lTheseEvents in lEvents for event in lTheseEvents]

        # - Extract monitored event channels and timestamps
        vnChannels = neurons_to_channels([e.neuron for e in lEvents],
                                         self._lHWLayerNeurons,
                                         )
        vtTimeTrace = np.ndarray([e.timestamp for e in lEvents]) * 1.e-6

        # - Locate synchronisation timestamp
        nSynchEvent = ...
        vnChannels = vnChannels[nSynchEvent:]
        vtTimeTrace = vtTimeTrace[nSynchEvent:]
        vtTimeTrace -= vtTimeTrace[0]

        # - Convert recorded events into TSEvent object
        tsResponse = TSEvent(vtTimeTrace,
                             vnChannels,
                             strName = 'DynapSE spikes',
                             )

        # - Trim recorded events if necessary
        tsResponse = tsResponse.clip([0, tDuration])

        # - Shift monitored events to current layer start time
        tsResponse.delay(self.t, bInPlace = True)

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

def neurons_to_channels(lNeurons: List[DynapseNeuron],
                        lLayerNeurons: List[DynapseNeuron],
                        ) -> np.ndarray:
    """
    neurons_to_channels - Convert a list of neurons into layer channel indices

    :param lNeurons:        List[DynapseNeuron] Lx0 to match against layer neurons
    :param lLayerNeurons:   List[DynapseNeuron] Nx0 HW neurons corresponding to each channel index
    :return:                np.ndarray[int] Lx0 channel indices corresponding to each neuron in lNeurons
    """
    # - Initialise list to return
    lChannelIndices = []

    for neurTest in lNeurons:
        try:
            nChannelIndex = lLayerNeurons.index(neurTest)
        except ValueError:
            nChannelIndex = np.nan

        # - Append discovered index
        lChannelIndices.append(nChannelIndex)

    # - Convert to numpy array
    return np.array(lChannelIndices)


def compile_weights_and_configure(mfWIn: np.ndarray,
                                  mfWRec: np.ndarray,
                                  lInputNeurons: List[DynapseNeuron],
                                  lLayerNeurons: List[DynapseNeuron],
                                  ):
    """

    :param mfWIn:
    :param mfWRec:
    :param lInputNeurons:
    :param lLayerNeurons:
    :return:
    """

    # - Get a connector object
    connector = NeuronNeuronConnector.DynapseConnector()

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


def get_input_to_layer_connection_list(mfWIn: np.ndarray,
                                       lInputNeurons: List[DynapseNeuron],
                                       lLayerNeurons: List[DynapseNeuron],
                                       ) -> List:
    """
    get_input_to_layer_connection_list - Convert an input weight matrix to a list of input to layer neuron connections

    :param mfWIn:
    :param lInputNeurons:
    :param lLayerNeurons:
    :return:
    """
    # - Pre-allocate inputs lists
    lInputsToNeuron = [] * mfWIn.shape[1]



def get_recurrent_connection_list(mfWRec: np.ndarray,
                                  lLayerNeurons: List[DynapseNeuron],
                                  ) -> List:
    """
    get_recurrent_connection_list - Convert a recurrent weight matrix to a list of neuron connections

    :param mfWRec:
    :param lLayerNeurons:
    :return:
    """
    # - Pre-allocate inputs lists
    lInputsToNeuron = [] * mfWRec.shape[1]

    # - Loop over targets
    for nInput, vfConns in enumerate(mfWRec):
        # - Extract excitatory and inhibitory inputs
        vnExcInputs = np.nonzero(vfConns > 0)
        vnInhInputs = np.nonzero(vfConns < 0)

        # - Loop over inputs and add to list
        lInputsToNeuron[nInput].append()