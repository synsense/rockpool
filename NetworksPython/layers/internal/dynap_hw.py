# ----
# dynap_hw.py - Implementation of HW FF and Rec layers for DynapSE, via ctxCTL
# ----

from ..layer import Layer
from ...timeseries import TSEvent

import numpy as np
from warnings import warn
from typing import Tuple, List, Optional, Union
import time

# - Imports from ctxCTL
import CtxDynapse
import NeuronNeuronConnector
from CtxDynapse import DynapseCamType as SynapseTypes
from CtxDynapse import dynapse, DynapseFpgaSpikeGen, DynapseNeuron, VirtualNeuron, EventFilter, FpgaSpikeEvent

# - Declare a Neuron type
Neuron = Union[DynapseNeuron, VirtualNeuron]

# - Default ISI multiplier
DEF_FPGA_ISI_MULTIPLIER = 10

def init_dynapse() -> dict:
    """
    init_dynapse - Initialise and configure DynapSE interface

    :return: dict Global dictionary containing DynapSE HW models
    """
    # - Initialise HW dictionary
    dDynapse = {}

    dDynapse['model'] = CtxDynapse.model
    dDynapse['virtualModel'] = CtxDynapse.VirtualModel()
    lFPGAModules = dDynapse['model'].get_fpga_modules()

    # - Find a spike generator module
    vbIsSpikeGenModule = [isinstance(m, DynapseFpgaSpikeGen) for m in lFPGAModules]
    if not np.any(vbIsSpikeGenModule):
        # - There is no spike generator, so we can't use this Python layer on the HW
        raise ModuleNotFoundError('An `fpgaSpikeGen` module is required to use the DynapSE layer.')

    else:
        # - Get first spike generator module
        dDynapse['fpgaSpikeGen'] = lFPGAModules[np.argwhere(vbIsSpikeGenModule)[0][0]]

    # - Get all neurons
    dDynapse['vHWNeurons'] = np.asarray(dDynapse['model'].get_neurons())
    dDynapse['vVirtualNeurons'] = np.asarray(dDynapse['virtualModel'].get_neurons())

    # - Initialise neuron allocation
    dDynapse['vbFreeInputNeurons'] = np.array([True for i in range(np.size(dDynapse['vVirtualNeurons']))])
    dDynapse['vbFreeHWNeurons'] = np.array([True for i in range(np.size(dDynapse['vHWNeurons']))])

    # - Wipe configuration
    for nChip in range(4):
        dynapse.clear_cam(int(nChip))
        dynapse.clear_sram(int(nChip))
    # warn('DynapSE configuration is not wiped -- IMPLEMENT ME --')

    # - Return dictionary
    return dDynapse

def init_fpgaSpikeGen(nMultiplier: int) -> Tuple[int, float]:
    """
    init_fpgaSpikeGen - Initialise and configure FPGA spike generator

    :param nMultiplier: int Multiplier to set, in units of 11.11r ns
    :return:            nMultipler, tISIBase
    """
    DHW_dDynapse['fpgaSpikeGen'].set_repeat_mode(False)
    DHW_dDynapse['fpgaSpikeGen'].set_variable_isi(True)
    DHW_dDynapse['fpgaSpikeGen'].set_isi_multiplier(nMultiplier)
    DHW_dDynapse['fpgaSpikeGen'].set_base_addr(0)

    # - Record current multiplier and time base
    DHW_dDynapse['nISIMultiplier'] = nMultiplier
    DHW_dDynapse['tISIBase'] = nMultiplier * 11.11111111e-9

    # - Return new multiplier and time base
    return DHW_dDynapse['nISIMultiplier'], DHW_dDynapse['tISIBase']


# -- Create global dictionary, only initialise on first import of this module
global DHW_dDynapse
if 'DHW_dDynapse' not in dir():
    # - Initialise DynapSE
    DHW_dDynapse = init_dynapse()

    # - Set ISI multiplier
    init_fpgaSpikeGen(DEF_FPGA_ISI_MULTIPLIER)


def allocate_layer_neurons(nNumNeurons: int) -> DynapseNeuron:
    """
    allocate_layer_neurons - Return a list of neurons that may be used. These are guaranteed not to already be assigned.

    :param nNumNeurons: int     The number of neurons requested
    :return:            list    A list of neurons that may be used
    """
    # - Are there sufficient unallocated neurons?
    if np.sum(DHW_dDynapse['vbFreeHWNeurons']) < nNumNeurons:
        raise MemoryError('Insufficient unallocated neurons available. {}'.format(nNumNeurons) + ' requested.')

    # - Pick the first available neurons
    vnNeuronsToAllocate = np.nonzero(DHW_dDynapse['vbFreeHWNeurons'])[0][:nNumNeurons]

    # - Mark these neurons as allocated
    DHW_dDynapse['vbFreeHWNeurons'][vnNeuronsToAllocate] = False

    vnInputNeuronOverlap = vnNeuronsToAllocate[vnNeuronsToAllocate < np.size(DHW_dDynapse['vbFreeInputNeurons'])]
    DHW_dDynapse['vbFreeInputNeurons'][vnInputNeuronOverlap] = False

    # - Return these allocated neurons
    return DHW_dDynapse['vHWNeurons'][vnNeuronsToAllocate]


def allocate_input_neurons(nNumNeurons: int) -> DynapseNeuron:
    """
    allocate_input_neurons - Return a list of neurons that may be used. These are guaranteed not to already be assigned.

    :param nNumNeurons: int     The number of neurons requested
    :return:            list    A list of neurons that may be used
    """
    # - Are there sufficient unallocated neurons?
    # for k, v in DHW_dDynapse.items():
    #     print("{}: {}".format(k, v))
    if np.sum(DHW_dDynapse['vbFreeInputNeurons']) < nNumNeurons:
        raise MemoryError('Insufficient unallocated neurons available. {}'.format(nNumNeurons) + ' requested.')

    # - Pick the first available neurons
    vnNeuronsToAllocate = np.nonzero(DHW_dDynapse['vbFreeInputNeurons'])[0][:nNumNeurons]

    # - Mark these as allocated
    DHW_dDynapse['vbFreeInputNeurons'][vnNeuronsToAllocate] = False
    DHW_dDynapse['vbFreeHWNeurons'][vnNeuronsToAllocate] = False

    # - Return these neurons
    return DHW_dDynapse['vVirtualNeurons'][vnNeuronsToAllocate]


# -- Define the HW layer class for recurrent networks
class RecDynapSE(Layer):
    """
    RecDynapSE - Recurrent layer implemented on DynapSE
    """
    def __init__(self,
                 mfWIn: np.ndarray,
                 mfWRec: np.ndarray,
                 tDt: Optional[float] = None,
                 fNoiseStd: Optional[float] = None,
                 strName: Optional[str] = 'unnamed',
                 ):
        """
        RecDynapSE - Recurrent layer implemented on DynapSE

        :param mfWIn:       ndarray[int] MxN matrix of input weights
        :param mfWRec:      ndarray[int] NxN matrix of recurrent weights. Supplied in units of synaptic connection.
                                        Negative elements lead to inhibitory synapses
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

        assert mfWRec.shape[0] == mfWRec.shape[1], \
            'The recurrent weight matrix `mnWRec` must be square.'

        # - Initialise superclass
        super().__init__(mfW = np.asarray(np.round(mfWIn), 'int'), tDt = tDt, fNoiseStd = fNoiseStd, strName = strName)

        # - Configure input and input-to-recurrent connections
        if mfWIn is None:
            mfWIn = np.ones(1, self.nSize)

        # - Check weight matrices
        assert mfWIn.shape[1] == mfWRec.shape[0], \
            '`mnWIn` and `mnWRec` must have compatible shapes: `mnWIn` is MxN, `mnWRec` is NxN.'

        # - Map neuron indices to neurons
        self._lHWInputNeurons = allocate_input_neurons(self.nSizeIn)
        self._lHWRecLayerNeurons = allocate_layer_neurons(self.nSize)

        # - Store input weights
        self._mfWRec = np.asarray(np.round(mfWRec), 'int')

        # - Configure connectivity
        self._compile_weights_and_configure()

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

        # - Clip tsInput to required duration
        tsInput = tsInput.clip([self.t, self.t + tDuration])#, bInPlace = True)

        # - Convert input events to fpga spike list representation
        spikeList = TSEvent_to_spike_list(tsInput, self._lHWInputNeurons)

        # - Send event sequence to fpga module
        DHW_dDynapse['fpgaSpikeGen'].preload_stimulus(spikeList)

        # - Define recording callback
        lEvents = []
        def func_event_callback(lTheseEvents):
            # - Append these events to list
            lEvents.append(lTheseEvents)

        # - Configure recording callback
        oFilter = EventFilter(DHW_dDynapse['model'],
                              func_event_callback,
                              [n.get_id() for n in self._lHWRecLayerNeurons],
                              )

        # - Define special events callback
        lTrigger = []
        def special_event_callback(nTimestamp: int):
            lTrigger.append(nTimestamp)

        # - Configure special event callback
        oFilter.set_special_event_callback(special_event_callback)

        # - Reset FPGA timestamp
        dynapse.reset_timestamp()

        # - Stimulate / record for desired duration
        DHW_dDynapse['fpgaSpikeGen'].start()

        # - Wait until stimulation is finished
        time.sleep(tDuration + .1)

        # - Stop stimulation
        DHW_dDynapse['fpgaSpikeGen'].stop()

        # - Flatten list of events
        lEvents = [event for lTheseEvents in lEvents for event in lTheseEvents]

        # - Extract monitored event channels and timestamps
        vnChannels = neurons_to_channels(
            [e.neuron for e in lEvents], self._lHWRecLayerNeurons
        )
        vtTimeTrace = np.ndarray([e.timestamp for e in lEvents]) * 1e-6

        # - Locate synchronisation timestamp
        tStartTime = lTrigger[0] * 1e-6
        nSynchEvent = np.argwhere(vtTimeTrace >= tStartTime)[0]
        vnChannels = vnChannels[nSynchEvent:]
        vtTimeTrace = vtTimeTrace[nSynchEvent:]
        vtTimeTrace -= tStartTime

        # - Convert recorded events into TSEvent object
        tsResponse = TSEvent(vtTimeTrace,
                             vnChannels,
                             strName = 'DynapSE spikes',
                             )

        # - Trim recorded events if necessary
        tsResponse = tsResponse.clip([0, tDuration])

        # - Shift monitored events to current layer start time
        tsResponse = tsResponse.delay(self.t)#, bInPlace = True)

        # - Set layer time
        self.t += tDuration

        # - Return recorded events
        return tsResponse

    def _compile_weights_and_configure(self):
        """
        _compile_weights_and_configure - Configure DynapSE weights from the weight matrices
        """

        # - Get a connector object
        connector = NeuronNeuronConnector.DynapseConnector()

        # - Get input to layer connections
        vnPreSynE, vnPostSynE,\
            vnPreSynI, vnPostSynI = connectivity_matrix_to_prepost_lists(self._mfW)

        # - Connect input to layer

        # - Connect the excitatory neurons
        connector.add_connection_from_list(self._lHWInputNeurons[vnPreSynE],
                                           self._lHWRecLayerNeurons[vnPostSynE],
                                           [SynapseTypes.SLOW_EXC]
                                           )

        # - Connect the inhibitory neurons
        connector.add_connection_from_list(self._lHWInputNeurons[vnPreSynI],
                                           self._lHWRecLayerNeurons[vnPostSynI],
                                           [SynapseTypes.SLOW_INH]
                                           )

        # - Get layer recurrent connections
        vnPreSynE, vnPostSynE,\
            vnPreSynI, vnPostSynI = connectivity_matrix_to_prepost_lists(self._mfWRec)

        # - Connect the excitatory neurons
        connector.add_connection_from_list(self._lHWRecLayerNeurons[vnPreSynE],
                                           self._lHWRecLayerNeurons[vnPostSynE],
                                           [SynapseTypes.SLOW_EXC]
                                           )

        # - Connect the inhibitory neurons
        connector.add_connection_from_list(self._lHWRecLayerNeurons[vnPreSynI],
                                           self._lHWRecLayerNeurons[vnPostSynI],
                                           [SynapseTypes.SLOW_INH]
                                           )

def connectivity_matrix_to_prepost_lists(mnW: np.ndarray) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    connectivity_matrix_to_prepost_lists - Convert a matrix into a set of pre-post connectivity lists

    :param mnW: ndarray[int]    Matrix[pre, post] containing integer numbers of synaptic connections
    :return:    (vnPreE,
                 vnPostE,
                 vnPreI,
                 vnPostI)       Each ndarray(int), containing a single pre- and post- synaptic partner.
                                    vnPreE and vnPostE together define excitatory connections
                                    vnPreI and vnPostI together define inhibitory connections
    """
    # - Get lists of pre and post-synaptic IDs
    vnPreECompressed, vnPostECompressed = np.nonzero(mnW > 0)
    vnPreICompressed, vnPostICompressed = np.nonzero(mnW < 0)

    # - Preallocate connection lists
    vnPreE = []
    vnPostE = []
    vnPreI = []
    vnPostI = []

    # - Loop over lists, appending when necessary
    for nPre, nPost in zip(vnPreECompressed, vnPostECompressed):
        for _ in range(mnW[nPre, nPost]):
            vnPreE.append(nPre)
            vnPostE.append(nPost)

    # - Loop over lists, appending when necessary
    for nPre, nPost in zip(vnPreICompressed, vnPostICompressed):
        for _ in range(mnW[nPre, nPost]):
            vnPreI.append(nPre)
            vnPostI.append(nPost)

    # - Return augmented lists
    return vnPreE, vnPostE, vnPreI, vnPostI


def TSEvent_to_spike_list(tsSeries: TSEvent, lNeurons: List[Neuron]) -> List[FpgaSpikeEvent]:
    """
    TSEvent_to_spike_list - Convert a TSEvent object to a ctxctl spike list

    :param tsSeries:    TSEvent         Time series of events to send as input
    :param lNeurons:    List[Neuron]    List of neurons that should appear as sources of the events
    :return:
    """
    # - Check that the number of channels is the same between time series and list of neurons
    assert tsSeries.nNumChannels <= len(lNeurons), \
        '`tsSeries` contains more channels than the number of neurons in `lNeurons`.'

    # - Get events from this time series
    vtTimes, vnChannels, _ = tsSeries.find()

    # - Convert to ISIs
    vtISIs = np.diff(np.r_[self.t, vtTimes])
    vnDiscreteISIs = (np.round(vtISIs / DHW_dDynapse['tISIBase'])).astype('int')

    # - Get neuron information
    vnCoreIDs = []
    vnNeuronIDs = []
    vnChipIDs = []
    for n in lNeurons:
        vnCoreIDs.append(n.get_core_id())
        vnNeuronIDs.append(n.get_neuron_id())
        vnChipIDs.append(n.get_chip_id())

    # - Convert each event to an FpgaSpikeEvent
    def generate_fpga_event(target_chip, core_mask, neuron_id, isi) -> FpgaSpikeEvent:
        event = FpgaSpikeEvent()
        event.target_chip = target_chip
        event.core_mask = core_mask
        event.neuron_id = neuron_id
        event.isi = isi
        return event

    lEvents = [
        generate_fpga_event(vnCoreIDs[nChannel], vnChipIDs[nChannel], vnNeuronIDs[nChannel], nISI)
        for nChannel, nISI in zip(vnChannels, vnDiscreteISIs)
    ]

    # - Return a list of events
    return lEvents


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
