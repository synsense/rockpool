# ----
# dynap_hw.py - Implementation of HW FF and Rec layers for DynapSE, via ctxCTL
# ----

from ..layer import Layer
from ...timeseries import TSEvent

import numpy as np
from warnings import warn
from typing import Tuple, List, Optional, Union
import time
import threading

# - Imports from ctxCTL
import CtxDynapse
import NeuronNeuronConnector
from CtxDynapse import DynapseCamType as SynapseTypes
from CtxDynapse import dynapse, DynapseFpgaSpikeGen, DynapseNeuron, VirtualNeuron, BufferedEventFilter, FpgaSpikeEvent, PyTimer

# - Declare a Neuron type
Neuron = Union[DynapseNeuron, VirtualNeuron]

# - Default ISI multiplier
tIsiTimeStep = 2e-5
tFpgaTimeStep = 1./9.*1e-7  # 11.111...ns 
DEF_FPGA_ISI_MULTIPLIER = int(np.round(tIsiTimeStep / tFpgaTimeStep))
nMaxEventsPerBatch = int(2**19-1)
# - Default maximum numbers of time steps for a single evolution batch
#   Assuming one input event after the maximum ISI - This is the maximally possible
#   value. In practice there will be more events per time. Therefore the this value
#   does not guarantee that the complete input batch fits onto the fpga
nDefaultMaxNumTimeSteps = int(nMaxEventsPerBatch * 2**16-1)

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
        print("DynapSE: Spike generator module ready.")

    # - Get all neurons
    dDynapse['vHWNeurons'] = np.asarray(dDynapse['model'].get_neurons())
    dDynapse['vVirtualNeurons'] = np.asarray(dDynapse['virtualModel'].get_neurons())

    # - Initialise neuron allocation
    dDynapse['vbFreeVirtualNeurons'] = np.array([True for i in range(np.size(dDynapse['vVirtualNeurons']))])
    dDynapse['vbFreeHWNeurons'] = np.array([True for i in range(np.size(dDynapse['vHWNeurons']))])
    print("DynapSE: Neurons initialized.")

    # - Wipe configuration
    for nChip in range(1): #range(4):
        dynapse.clear_cam(int(nChip))
        print("DynapSE: Chip {}: CAM cleared.".format(nChip))
        dynapse.clear_sram(int(nChip))
        print("DynapSE: SRAMs cleared.")
    # - Reset neuron weights in model
    for neuron in dDynapse["vHWNeurons"]:
        viSrams = neuron.get_srams()
        for iSramIndex in range(1, 4):
            viSrams[iSramIndex].set_target_chip_id(0)
            viSrams[iSramIndex].set_virtual_core_id(0)
            viSrams[iSramIndex].set_used(False)
            viSrams[iSramIndex].set_core_mask(0)
        viCams = neuron.get_cams()
        for cam in viCams:
            cam.set_pre_neuron_id(0)
            cam.set_pre_neuron_core_id(0)
    print("Model neuron weights have been reset")

    # - Lists for recording events
    dDynapse["lEvents"] = []
    dDynapse["lTrigger"] = []       

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
    DHW_dDynapse['tISIBase'] = nMultiplier * tFpgaTimeStep

    # - Return new multiplier and time base
    return DHW_dDynapse['nISIMultiplier'], DHW_dDynapse['tISIBase']



# -- Create global dictionary, only initialise on first import of this module
global DHW_dDynapse
if 'DHW_dDynapse' not in dir():
    # - Initialise DynapSE
    DHW_dDynapse = init_dynapse()

    # - Set ISI multiplier
    init_fpgaSpikeGen(DEF_FPGA_ISI_MULTIPLIER)
    print("DynapSE prepared.")

def allocate_hw_neurons(nNumNeurons: int) -> np.ndarray:
    """
    allocate_hw_neurons - Return a list of neurons that may be used. These are guaranteed not to already be assigned.

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

    vnInputNeuronOverlap = vnNeuronsToAllocate[vnNeuronsToAllocate < np.size(DHW_dDynapse['vbFreeVirtualNeurons'])]
    DHW_dDynapse['vbFreeVirtualNeurons'][vnInputNeuronOverlap] = False

    # - Return these allocated neurons
    return DHW_dDynapse['vHWNeurons'][vnNeuronsToAllocate]


def allocate_virtual_neurons(nNumNeurons: int) -> np.ndarray:
    """
    allocate_virtual_neurons - Return a list of neurons that may be used. These are guaranteed not to already be assigned.

    :param nNumNeurons: int     The number of neurons requested
    :return:            list    A list of neurons that may be used
    """
    # - Are there sufficient unallocated neurons?
    # for k, v in DHW_dDynapse.items():
    #     print("{}: {}".format(k, v))
    if np.sum(DHW_dDynapse['vbFreeVirtualNeurons']) < nNumNeurons:
        raise MemoryError('Insufficient unallocated neurons available. {}'.format(nNumNeurons) + ' requested.')

    # - Pick the first available neurons
    vnNeuronsToAllocate = np.nonzero(DHW_dDynapse['vbFreeVirtualNeurons'])[0][:nNumNeurons]

    # - Mark these as allocated
    DHW_dDynapse['vbFreeVirtualNeurons'][vnNeuronsToAllocate] = False
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
                 vHWNeurons: Optional[np.ndarray] = None,
                 vVirtualNeurons: Optional[np.ndarray] = None,
                 tDt: Optional[float] = None,
                 fNoiseStd: Optional[float] = None,
                 tMaxBatchDur: float = 120,
                 strName: Optional[str] = 'unnamed',
                 ):
        """
        RecDynapSE - Recurrent layer implemented on DynapSE

        :param mfWIn:       ndarray[int] MxN matrix of input weights from virtual to hardware neurons
        :param mfWRec:      ndarray[int] NxN matrix of weights between hardware neurons. Supplied in units of synaptic connection.
                                         Negative elements lead to inhibitory synapses
        :param vHWNeurons:  ndarray  1D array of N hardware neurons that are to be used as layer neurons
        :param vVirtualNeurons:  ndarray  1D array of M virtual neurons that are to be used as input neurons
        :param tDt:         float   Dummy time step. Not used in layer evolution
        :param fNoiseStd    float   Dummy noise to inject. Not used in layer evolution
        :param tMaxBatchDur: float  Maximum duration of single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
                                      Cannot be larger than nDefaultMaxNumTimeSteps
        :param strName:     str     Layer name
        """
        # - Check supplied arguments
        if tDt is not None:
            nMultiplier = int(np.round(tDt / tFpgaTimeStep))
            DHW_dDynapse["fpgaSpikeGen"].set_isi_multiplier(nMultiplier)
            # - Record current multiplier and time base
            DHW_dDynapse['nISIMultiplier'] = nMultiplier
            DHW_dDynapse['tISIBase'] = nMultiplier * tFpgaTimeStep
            print("Layer `{}`: Fpga ISI multiplier set to {}".format(strName, nMultiplier))
        else:
            tDt = tIsiTimeStep

        if fNoiseStd is not None:
            warn('Caution: `fNoiseStd` is ignored during DynapSE layer evolution.')
        else:
            fNoiseStd = 0.

        assert mfWRec.shape[0] == mfWRec.shape[1], \
            'Layer `{}`: The recurrent weight matrix `mnWRec` must be square.'.format(strName)

        # - Initialise superclass
        super().__init__(mfW = np.asarray(np.round(mfWIn), 'int'), tDt = tDt, fNoiseStd = fNoiseStd, strName = strName)
        print("Layer `{}`: Superclass initialized".format(strName))

        # - Configure input and input-to-recurrent connections
        mfWIn = np.ones((1, self.nSize), int) if mfWIn is None else np.asarray(np.round(mfWIn), int)

        # - Check weight matrices
        assert mfWIn.shape[1] == mfWRec.shape[0], \
            'Layer `{}`: `mnWIn` and `mnWRec` must have compatible shapes: `mnWIn` is MxN, `mnWRec` is NxN.'.format(strName)

        assert nMaxNumTimeSteps <= nDefaultMaxNumTimeSteps, \
            "Layer `{}`: nMaxNumTimeSteps must not be greater than {}.".format(nDefaultMaxNumTimeSteps)
        
        # - Store maximum batch duration during evolution
        self.tMaxBatchDur = tMaxBatchDur

        # - Map neuron indices to neurons
        self._vVirtualNeurons = (
            allocate_virtual_neurons(self.nSizeIn) if vVirtualNeurons is None
            else vVirtualNeurons
        )
        print("Layer `{}`: Virtual neurons allocated".format(strName))
        self._vHWNeurons = (
            allocate_hw_neurons(self.nSize) if vHWNeurons is None
            else vHWNeurons
        )
        print("Layer `{}`: Layer neurons allocated".format(strName))
        # - Store recurrent weights
        self._mfWRec = np.asarray(np.round(mfWRec), int)

        # - Configure connectivity
        self._compile_weights_and_configure()

        # - Lists for recording events
        self.lEvents = []
        self.lTrigger = []
        self._bStimulusStarted = False

        print("Layer `{}` prepared.".format(self.strName))

    def _batch_data(
        self, mfInput: np.ndarray, nNumTimeSteps: int, nMaxNumTimeSteps: int = None
    ) -> (np.ndarray, int):
        """_batch_data: Generator that returns the data in batches"""
        # - Handle None for nMaxNumTimeSteps
        nMaxNumTimeSteps = (
            nNumTimeSteps if nMaxNumTimeSteps is None else nMaxNumTimeSteps
        )
        nStart = 0
        while nStart < nNumTimeSteps:
            # - Endpoint of current batch
            nEnd = min(nStart + nMaxNumTimeSteps, nNumTimeSteps)
            # - Data for current batch
            mfCurrentInput = mfInput[nStart:nEnd]
            yield mfCurrentInput, nEnd - nStart
            # - Update nStart
            nStart = nEnd

    def evolve(self,
               tsInput: Optional[TSEvent] = None,
               tDuration: Optional[float] = None,
               nNumTimeSteps: Optional[int] = None,
               bVerbose: bool = True,
               ) -> TSEvent:
        """
        evolve - Evolve the layer by queueing spikes, stimulating and recording

        :param tsInput:         TSEvent input time series, containing `self.nSize` channels
        :param tDuration:       float   Desired evolution duration, in seconds
        :param nNumTimeSteps:   int     Desired evolution duration, in integer steps of `self.tDt`
        :param bVerbose:        bool    Output information on evolution progress

        :return:                TSEvent spikes emitted by the neurons in this layer, during the evolution time
        """
        # - Compute duration for evolution
        if nNumTimeSteps is None:
            # - Determine nNumTimeSteps
            if tDuration is None:
                # - Determine tDuration
                assert (
                    tsInput is not None
                ), "Layer `{}`: One of `nNumTimeSteps`, `tsInput` or `tDuration` must be supplied".format(
                    self.strName
                )

                if tsInput.bPeriodic:
                    # - Use duration of periodic TimeSeries, if possible
                    tDuration = tsInput.tDuration

                else:
                    # - Evolve until the end of the input TImeSeries
                    tDuration = tsInput.tStop - self.t
                    assert tDuration > 0, (
                        "Layer `{}`: Cannot determine an appropriate evolution duration.".format(
                            self.strName
                        )
                        + " `tsInput` finishes before the current evolution time."
                    )
            nNumTimeSteps = int(np.floor((tDuration + fTolAbs) / self.tDt))
        else:
            assert isinstance(
                nNumTimeSteps, int
            ), "Layer `{}`: nNumTimeSteps must be of type int.".format(self.strName)
            tDuration = nNumTimeSteps * self.tDt

        # - Clip tsInput to required duration
        tsInput = tsInput.clip([self.t, self.t + tDuration])#, bInPlace = True)
        tsInput.tStartTime = self.t

        # - Convert input events to fpga spike list representation
        spikeList = TSEvent_to_spike_list(tsInput, self._vVirtualNeurons)
        if bVerbose:
            print("Layer `{}`: Prepared FGPA event list.".format(self.strName))

        # - Send event sequence to fpga module
        DHW_dDynapse['fpgaSpikeGen'].preload_stimulus(spikeList)
        if bVerbose:
            print("Layer `{}`: Stimulus preloaded.".format(self.strName))

        # -- Set up event recording callbacks
        oFilter = BufferedEventFilter(
            DHW_dDynapse['model'],
            [n.get_id() for n in self._vHWNeurons],
        )
        if bVerbose:
            print("Layer `{}`: Event filter ready.".format(self.strName))

        # - Stimulate / record for desired duration
        if bVerbose:
            print("Layer `{}`: Starting stimulation.".format(self.strName))
        DHW_dDynapse['fpgaSpikeGen'].start()
        # - Wait for tDuration + some additional time
        if bVerbose:
            print("Layer `{}`: Waiting for stimulation to end.".format(self.strName))
        time.sleep(tDuration + 0.5)
        # - Stop stimulation and clear filter to stop recording events
        DHW_dDynapse['fpgaSpikeGen'].stop()
        oFilter.clear()
        if bVerbose:
            print("Layer `{}`: Stimulation ended.".format(self.strName))
        # - Retrieve events
        self.lEvents = list(oFilter.get_events())
        self.lTrigger = list(oFilter.get_special_event_timestamps())
        if bVerbose:
            print("Layer `{}`: Recorded {} events and {} trigger events".format(self.strName, len(self.lEvents), len(self.lTrigger)))

        # - Extract monitored event channels and timestamps
        vnChannels = neurons_to_channels(
            [e.neuron for e in self.lEvents], list(self._vHWNeurons)
        )
        vtTimeTrace = np.array([e.timestamp for e in self.lEvents]) * 1e-6

        # - Locate synchronisation timestamp
        tStartTime = self.lTrigger[0] * 1e-6
        try:
            nSynchEvent = np.argwhere(vtTimeTrace >= tStartTime)[0,0]
        except IndexError:
            if bVerbose:
                print("Layer `{}`: No events recorded.".format(self.strName))
            vnChannels = []
            vtTimeTrace = []
        else:
            vnChannels = vnChannels[nSynchEvent:]
            vtTimeTrace = vtTimeTrace[nSynchEvent:]
            vtTimeTrace -= tStartTime
            if bVerbose:
                print("Layer `{}`: Extracted event data".format(self.strName))

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
        self._nTimeStep += nNumTimeSteps

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
        connector.add_connection_from_list(self._vVirtualNeurons[vnPreSynE],
                                           self._vHWNeurons[vnPostSynE],
                                           [SynapseTypes.SLOW_EXC]
                                           )
        print("Layer: `{}`: Excitatory connections from virtual neurons to layer neurons have been set.".format(self.strName))
        # - Connect the inhibitory neurons
        connector.add_connection_from_list(self._vVirtualNeurons[vnPreSynI],
                                           self._vHWNeurons[vnPostSynI],
                                           [SynapseTypes.SLOW_INH]
                                           )
        print("Layer: `{}`: Inhibitory connections from virtual neurons to layer neurons have been set.".format(self.strName))
        # - Get layer recurrent connections
        vnPreSynE, vnPostSynE,\
            vnPreSynI, vnPostSynI = connectivity_matrix_to_prepost_lists(self._mfWRec)

        # - Connect the excitatory neurons
        connector.add_connection_from_list(self._vHWNeurons[vnPreSynE],
                                           self._vHWNeurons[vnPostSynE],
                                           [SynapseTypes.SLOW_EXC]
                                           )
        print("Layer: `{}`: Excitatory connections within layer have been set.".format(self.strName))
        # - Connect the inhibitory neurons
        connector.add_connection_from_list(self._vHWNeurons[vnPreSynI],
                                           self._vHWNeurons[vnPostSynI],
                                           [SynapseTypes.SLOW_INH]
                                           )
        print("Layer: `{}`: Inhibitory connections within layer have been set.".format(self.strName))

    @property
    def mfWIn(self):
        return self._mfWIn

    @mfWIn.setter
    def mfWIn(self, mfNewW):
        self._mfWIn = np.round(self._expand_to_shape(
            mfNewW, (self.nSizeIn, self.nSize), "mfWIn", bAllowNone=False
        )).astype(int)

    @property
    def mfWRec(self):
        return self._mfWRec

    @mfWRec.setter
    def mfWRec(self, mfNewW):
        self._mfWRec = np.round(self._expand_to_shape(
            mfNewW, (self.nSize, self.nSize), "mfWRec", bAllowNone=False
        )).astype(int)

    # mfW as alias for mfWRec
    @property
    def mfW(self):
        return self.mfWRec

    @mfW.setter
    def mfW(self, mfNewW):
        self.mfWRec = mfNewW

    # _mfW as alias for _mfWRec
    @property
    def _mfW(self):
        return self._mfWRec

    @_mfW.setter
    def _mfW(self, mfNewW):
        self._mfWRec = mfNewW

    @property
    def tMaxBatchDur(self):
        return self._tMaxBatchDur

    @tMaxBatchDur.setter
    def tMaxBatchDur(self, tNewMax):
        assert (
            type(tNewMax) == int and 0 < tNewMax
        ), "Layer `{}`: nMaxNumTimeSteps must be an integer greater than 0.".format(
            self.strName
        )
        self._tMaxBatchDur = tNewMax


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
    vtISIs = np.diff(np.r_[tsSeries.tStartTime, vtTimes])
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
