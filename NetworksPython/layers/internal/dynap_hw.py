# ----
# dynap_hw.py - Implementation of HW FF and Rec layers for DynapSE, via ctxCTL
# ----

from ..layer import Layer
from ...timeseries import TSEvent

import numpy as np
from warnings import warn
from typing import Tuple, List, Optional, Union
import time

# - Programmatic imports for CtxCtl
try:
    import CtxDynapse
    import NeuronNeuronConnector

except ModuleNotFoundError:
    # - Try with RPyC
    try:
        import rpyc
        conn = rpyc.classic.connect('localhost', '1300')
        CtxDynapse = conn.modules.CtxDynapse
        NeuronNeuronConnector = conn.modules.NeuronNeuronConnector
    except:
        raise

# - Imports from ctxCTL
# import CtxDynapse
# import NeuronNeuronConnector

# - Remaining imports
SynapseTypes = CtxDynapse.DynapseCamType
dynapse = CtxDynapse.dynapse
DynapseFpgaSpikeGen = CtxDynapse.DynapseFpgaSpikeGen
DynapsePoissonGen = CtxDynapse.DynapsePoissonGen
DynapseNeuron = CtxDynapse.DynapseNeuron
VirtualNeuron = CtxDynapse.VirtualNeuron
BufferedEventFilter = CtxDynapse.BufferedEventFilter
FpgaSpikeEvent = CtxDynapse.FpgaSpikeEvent

# from CtxDynapse import DynapseCamType as SynapseTypes
# from CtxDynapse import (
#     dynapse,
#     DynapseFpgaSpikeGen,
#     DynapsePoissonGen,
#     DynapseNeuron,
#     VirtualNeuron,
#     BufferedEventFilter,
#     FpgaSpikeEvent,
# )

print("CtxDynapse modules loaded.")

# - Declare a Neuron type
# Neuron = Union[DynapseNeuron, VirtualNeuron]

# - Chip properties
# - Dimensions of single core in neurons
nWidthCore = 16
nHeightCore = 16

# - Default ISI multiplier
tIsiTimeStep = 2e-5
tFpgaTimeStep = 1. / 9. * 1e-7  # 11.111...ns
DEF_FPGA_ISI_MULTIPLIER = int(np.round(tIsiTimeStep / tFpgaTimeStep))
nFpgaEventLimit = int(2 ** 19 - 1)
# - Default maximum numbers of time steps for a single evolution batch
#   Assuming one input event after the maximum ISI - This is the maximally possible
#   value. In practice there will be more events per time. Therefore the this value
#   does not guarantee that the complete input batch fits onto the fpga
nDefaultMaxNumTimeSteps = int(nFpgaEventLimit * (2 ** 16 - 1))

# - Absolute tolerance, e.g. for comparing float values
fTolAbs = 1e-9


def assign_neurons_rectangle(nFirstNeuron: int, nNumNeurons: int, nWidth: int) -> List[int]:
    """
    assign_neurons_rectangle: return neurons that form a rectangle on the chip
                              with nFirstNeuron as upper left corner and width
                              nWidth. (Last row may not be full). Neurons
                              have to fit onto single core.
    :param nFirstNeuron:  int ID of neuron that is in the upper left corner of the rectangle
    :param nNumNeurons:   int Number of neurons in the rectangle
    :param nWidth:        int Width of the rectangle (in neurons)  
    :return lNeuronIDs:   list 1D array of IDs of neurons that from the rectangle.
    """
    nNumRows = int(np.ceil(nNumNeurons / nWidth))
    nFirstRow = int(np.floor(nFirstNeuron / nWidthCore))
    nFirstCol = nFirstNeuron % nWidthCore
    # - Make sure rectangle fits on single core
    assert (
        nFirstCol + nWidth <= nWidthCore  # not too wide
        and nFirstRow % nHeightCore + nNumRows <= nHeightCore  # not too high
    ), "Rectangle does not fit onto single core."
    lNeuronIDs = [
        (nFirstRow + nRow) * nWidthCore + (nFirstCol + nID)
        for nRow in range(nNumRows)
        for nID in range(nWidth)
    ]
    return lNeuronIDs


def init_dynapse() -> dict:
    """
    init_dynapse - Initialise and configure DynapSE interface

    :return: dict Global dictionary containing DynapSE HW models
    """

    print("Initializing DynapSE")
    # - Initialise HW dictionary
    dDynapse = {}

    dDynapse["model"] = CtxDynapse.model
    dDynapse["virtualModel"] = CtxDynapse.VirtualModel()
    lFPGAModules = dDynapse["model"].get_fpga_modules()

    # - Find a spike generator module
    vbIsSpikeGenModule = [isinstance(m, DynapseFpgaSpikeGen) for m in lFPGAModules]
    if not np.any(vbIsSpikeGenModule):
        # - There is no spike generator, so we can't use this Python layer on the HW
        raise ModuleNotFoundError(
            "DynapSE: An `fpgaSpikeGen` module is required to use the DynapSE layer."
        )

    else:
        # - Get first spike generator module
        dDynapse["fpgaSpikeGen"] = lFPGAModules[np.argwhere(vbIsSpikeGenModule)[0][0]]
        print("DynapSE: Spike generator module ready.")

    # - Find a poisson spike generator module
    vbIsPoissonGenModule = [isinstance(m, DynapsePoissonGen) for m in lFPGAModules]
    if np.any(vbIsPoissonGenModule):
        dDynapse["fpgaPoissonGen"] = lFPGAModules[np.argwhere(vbIsPoissonGenModule)[0][0]]
    else:
        raise ModuleNotFoundError(
            "DynapSE: Could not find poisson generator module (DynapsePoissonGen)."
        )

    # - Get all neurons
    dDynapse["vHWNeurons"] = np.asarray(dDynapse["model"].get_neurons())
    dDynapse["vShadowNeurons"] = np.asarray(dDynapse['model'].get_shadow_state_neurons())
    dDynapse["vVirtualNeurons"] = np.asarray(dDynapse["virtualModel"].get_neurons())

    # - Initialise neuron allocation
    dDynapse["vbFreeVirtualNeurons"] = np.ones(dDynapse["vVirtualNeurons"].size, bool)
    # Do not use virtual neuron 0
    dDynapse["vbFreeVirtualNeurons"][0] = False
    
    dDynapse["vbFreeHWNeurons"] = np.ones(dDynapse["vHWNeurons"].size, bool)
    # Do not use hardware neurons with ID 0 and core ID 0 (first of each core)
    dDynapse["vbFreeHWNeurons"][0::1024] = False
    print("DynapSE: Neurons initialized.")

    # - Get a connector object
    dDynapse["dcNeuronConnector"] = NeuronNeuronConnector.DynapseConnector()
    print("DynapSE: Neuron connector initialized")

    # - Wipe configuration
    for nChip in range(1):  # range(4):
        dynapse.clear_cam(int(nChip))
        print("DynapSE: Chip {}: CAM cleared.".format(nChip))
        dynapse.clear_sram(int(nChip))
        print("DynapSE: SRAMs cleared.")
    # - Reset neuron weights in model
    for neuron in dDynapse["vShadowNeurons"]:
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
    print("DynapSE: Model neuron weights have been reset")

    # - Return dictionary
    return dDynapse


def init_fpgaSpikeGen(nMultiplier: int) -> Tuple[int, float]:
    """
    init_fpgaSpikeGen - Initialise and configure FPGA spike generator

    :param nMultiplier: int Multiplier to set, in units of 11.11r ns
    :return:            nMultipler, tISIBase
    """
    DHW_dDynapse["fpgaSpikeGen"].set_repeat_mode(False)
    DHW_dDynapse["fpgaSpikeGen"].set_variable_isi(True)
    DHW_dDynapse["fpgaSpikeGen"].set_isi_multiplier(nMultiplier)
    DHW_dDynapse["fpgaSpikeGen"].set_base_addr(0)

    # - Record current multiplier and time base
    DHW_dDynapse["nISIMultiplier"] = nMultiplier
    DHW_dDynapse["tISIBase"] = nMultiplier * tFpgaTimeStep

    # - Return new multiplier and time base
    return DHW_dDynapse["nISIMultiplier"], DHW_dDynapse["tISIBase"]


def re_init_hardware():
    # - Initialise DynapSE
    global DHW_dDynapse
    DHW_dDynapse = init_dynapse()

    # - Set ISI multiplier
    init_fpgaSpikeGen(DEF_FPGA_ISI_MULTIPLIER)
    print("DynapSE prepared.")


# -- Create global dictionary, only initialise on first import of this module
global DHW_dDynapse
if "DHW_dDynapse" not in dir():
    re_init_hardware()

def allocate_hw_neurons(vnNeuronIDs: Union[int, np.ndarray]) -> (np.ndarray, np.ndarray):
    """
    allocate_hw_neurons - Return a list of neurons that may be used. These are guaranteed not to already be assigned.

    :param vnNeuronIDs:  int or np.ndarray    The number of neurons requested or IDs of requested neurons
    :return:             list    A list of neurons that may be used
    """

    if isinstance(vnNeuronIDs, int):
        nNumNeurons = vnNeuronIDs
        # - Are there sufficient unallocated neurons?
        if np.sum(DHW_dDynapse["vbFreeHWNeurons"]) < nNumNeurons:
            raise MemoryError(
                "Insufficient unallocated neurons available. {} requested.".format(
                    nNumNeurons
                )
            )
        # - Pick the first available neurons
        vnNeuronsToAllocate = np.nonzero(DHW_dDynapse["vbFreeHWNeurons"])[0][:nNumNeurons]

    else:
        vnNeuronsToAllocate = np.array(vnNeuronIDs).flatten()
        # - Make sure neurons are available
        if (DHW_dDynapse["vbFreeHWNeurons"][vnNeuronsToAllocate] == False).any():
            raise MemoryError(
                "{} of the requested neurons are already allocated.".format(
                    np.sum(
                        DHW_dDynapse["vbFreeHWNeurons"][vnNeuronsToAllocate] == False
                    )
                )
            )

    # - Mark these neurons as allocated
    DHW_dDynapse["vbFreeHWNeurons"][vnNeuronsToAllocate] = False

    # - Prevent allocation of virtual neurons with same ID as allocated hardware neurons
    vnInputNeuronOverlap = vnNeuronsToAllocate[
        vnNeuronsToAllocate < np.size(DHW_dDynapse["vbFreeVirtualNeurons"])
    ]
    DHW_dDynapse["vbFreeVirtualNeurons"][vnInputNeuronOverlap] = False

    # - Return these allocated neurons
    return (
        DHW_dDynapse["vHWNeurons"][vnNeuronsToAllocate],
        DHW_dDynapse["vShadowNeurons"][vnNeuronsToAllocate]
    )


def allocate_virtual_neurons(vnNeuronIDs: Union[int, np.ndarray]) -> np.ndarray:
    """
    allocate_virtual_neurons - Return a list of neurons that may be used. These are guaranteed not to already be assigned.

    :param vnNeuronIDs:  int or np.ndarray    The number of neurons requested or IDs of requested neurons
    :return:             list    A list of neurons that may be used
    """
    if isinstance(vnNeuronIDs, int):
        nNumNeurons = vnNeuronIDs
        # - Are there sufficient unallocated neurons?
        if np.sum(DHW_dDynapse["vbFreeVirtualNeurons"]) < nNumNeurons:
            raise MemoryError(
                "Insufficient unallocated neurons available. {}".format(nNumNeurons)
                + " requested."
            )
        # - Pick the first available neurons
        vnNeuronsToAllocate = np.nonzero(DHW_dDynapse["vbFreeVirtualNeurons"])[0][
            :nNumNeurons
        ]

    else:
        vnNeuronsToAllocate = np.array(vnNeuronIDs).flatten()
        # - Make sure neurons are available
        if (DHW_dDynapse["vbFreeVirtualNeurons"][vnNeuronsToAllocate] == False).any():
            raise MemoryError(
                "{} of the requested neurons are already allocated.".format(
                    np.sum(
                        DHW_dDynapse["vbFreeVirtualNeurons"][vnNeuronsToAllocate]
                        == False
                    )
                )
            )

    # - Mark these as allocated
    DHW_dDynapse["vbFreeVirtualNeurons"][vnNeuronsToAllocate] = False
    # - Prevent allocation of hardware neurons with same ID as allocated virtual neurons
    DHW_dDynapse["vbFreeHWNeurons"][vnNeuronsToAllocate] = False

    # - Return these neurons
    return DHW_dDynapse["vVirtualNeurons"][vnNeuronsToAllocate]


# -- Define the HW layer class for recurrent networks
class RecDynapSE(Layer):
    """
    RecDynapSE - Recurrent layer implemented on DynapSE
    """

    def __init__(
        self,
        mfWIn: np.ndarray,
        mfWRec: np.ndarray,
        vnLayerNeuronIDs: Optional[np.ndarray] = None,
        vnVirtualNeuronIDs: Optional[np.ndarray] = None,
        tDt: Optional[float] = None,
        fNoiseStd: Optional[float] = None,
        nMaxTrialsPerBatch: Optional[float] = None,
        tMaxBatchDur: Optional[float] = None,
        nMaxNumTimeSteps: Optional[int] = None,
        nMaxEventsPerBatch: int = nFpgaEventLimit,
        lInputCoreIDs: List[int] = [0],
        nInputChipID: int = 0,
        strName: Optional[str] = "unnamed",
    ):
        """
        RecDynapSE - Recurrent layer implemented on DynapSE

        :param mfWIn:               ndarray[int] MxN matrix of input weights from virtual to hardware neurons
        :param mfWRec:              ndarray[int] NxN matrix of weights between hardware neurons.
                                                 Supplied in units of synaptic connection. Negative elements
                                                 lead to inhibitory synapses
        :param vnLayerNeuronIDs:    ndarray  1D array of IDs of N hardware neurons that are to be used as layer neurons
        :param vnVirtualNeuronIDs:  ndarray  1D array of IDs of M virtual neurons that are to be used as input neurons
        :param tDt:                 float   Time-step.
        :param fNoiseStd:           float   Dummy noise to inject. Not used in layer evolution
        :param nMaxTrialsPerBatch:  int  Maximum number of trials (specified in input timeseries) per batch.
                                         Longer evolution periods will automatically split in smaller batches.
        :param tMaxBatchDur:        float  Maximum duration of single evolution batch.
        :param nMaxNumTimeSteps:    float  Maximum number of time steps of of single evolution batch.
        :param nMaxEventsPerBatch:  float  Maximum number of input events per evolution batch.
                                           Cannot be larger than nDefaultMaxNumTimeSteps.
        :param lInputCoreIDs:       list  IDs of the cores that contain neurons receiving external inputs.
                                          To avoid ID collisions neurons on these cores should not receive inputs
                                          from other neurons.
        :param nInputChipID:        int  ID of the chip with neurons that receive external input.
        :param strName:             str     Layer name
        """
        # - Check supplied arguments
        if tDt is not None:
            nMultiplier = int(np.round(tDt / tFpgaTimeStep))
            DHW_dDynapse["fpgaSpikeGen"].set_isi_multiplier(nMultiplier)
            # - Record current multiplier and time base
            DHW_dDynapse["nISIMultiplier"] = nMultiplier
            DHW_dDynapse["tISIBase"] = nMultiplier * tFpgaTimeStep
            print(
                "Layer `{}`: Fpga ISI multiplier set to {}".format(strName, nMultiplier)
            )
        else:
            tDt = tIsiTimeStep

        if fNoiseStd is not None:
            warn(
                "Layer `{}`: Caution: `fNoiseStd` is ignored during DynapSE layer evolution.".format(
                    strName
                )
            )
        else:
            fNoiseStd = 0.

        assert (
            mfWRec.shape[0] == mfWRec.shape[1]
        ), "Layer `{}`: The recurrent weight matrix `mnWRec` must be square.".format(
            strName
        )

        # - Initialise superclass
        super().__init__(
            mfW=np.asarray(np.round(mfWIn), "int"),
            tDt=tDt,
            fNoiseStd=fNoiseStd,
            strName=strName,
        )
        print("Layer `{}`: Superclass initialized".format(strName))

        # - Check weight matrices
        assert (
            mfWIn.shape[1] == mfWRec.shape[0]
        ), "Layer `{}`: `mnWIn` and `mnWRec` must have compatible shapes: `mnWIn` is MxN, `mnWRec` is NxN.".format(
            strName
        )

        # - Store weight matrices
        self.mfWIn = mfWIn
        self.mfWRec = mfWRec
        # - Record input core mask and chip ID
        self._nInputCoreMask = np.sum([2**nID for nID in lInputCoreIDs])
        self._nInputChipID = nInputChipID
        # - Store evolution batch size limitations
        self.nMaxTrialsPerBatch = nMaxTrialsPerBatch
        self.nMaxEventsPerBatch = nMaxEventsPerBatch
        if nMaxNumTimeSteps is not None:
            if tMaxBatchDur is not None:
                warn(
                    "Layer `{}`: Caution: If both `nMaxNumTimeSteps` and `tMaxBatchDur` are provided, only `nMaxNumTimeSteps` is considered.".format(
                        strName
                    )
                )
            self.nMaxNumTimeSteps = nMaxNumTimeSteps
        else:
            self.tMaxBatchDur = tMaxBatchDur

        # - Allocate layer neurons
        self._vHWNeurons, self._vShadowNeurons = (
            allocate_hw_neurons(self.nSize) if vnLayerNeuronIDs is None
            else allocate_hw_neurons(vnLayerNeuronIDs)
        )
        # Make sure number of neurons is correct
        assert self._vHWNeurons.size == self.nSize, (
            "Layer `{}`: `vnLayerNeuronIDs` must be of size {} or None.".format(strName, self.nSize)
        )
        print("Layer `{}`: Layer neurons allocated".format(strName))

        # - Allocate virtual neurons
        self._vVirtualNeurons = (
            allocate_virtual_neurons(self.nSizeIn) if vnVirtualNeuronIDs is None
            else allocate_virtual_neurons(vnVirtualNeuronIDs)
        )
        # Make sure number of neurons is correct
        assert self._vVirtualNeurons.size == self.nSizeIn, (
            "Layer `{}`: `vnVirtualNeuronIDs` must be of size {} or None.".format(strName, self.nSizeIn)
        )
        print("Layer `{}`: Virtual neurons allocated".format(strName))
        
        # - Store recurrent weights
        self._mfWRec = np.asarray(np.round(mfWRec), int)

        # - Configure connectivity
        self._compile_weights_and_configure()

        print("Layer `{}` prepared.".format(self.strName))

    def _batch_input_data(
        self, tsInput: TSEvent, nNumTimeSteps: int, bVerbose: bool = False
    ) -> (np.ndarray, int):
        """_batch_data: Generator that returns the data in batches"""
        # - Time points of input trace in discrete layer time base
        vnInputTimeSteps = np.floor(tsInput.vtTimeTrace / self.tDt).astype(int)
        # - Make sure evolution is within correct interval
        iStartIndex = np.searchsorted(vnInputTimeSteps, self._nTimeStep)
        iEndIndex = np.searchsorted(vnInputTimeSteps, self._nTimeStep + nNumTimeSteps)
        vnInputTimeSteps = vnInputTimeSteps[iStartIndex:iEndIndex]
        vnInputChannels = tsInput.vnChannels[iStartIndex:iEndIndex]
        # vnInputChannels = tsInput.vnChannels

        # - Check whether data for splitting by trial is available
        if hasattr(tsInput, "vtTrialStarts") and self.nMaxTrialsPerBatch is not None:
            ## -- Split by trials
            vnTrialStarts = np.floor(tsInput.vtTrialStarts / self.tDt).astype(int)
            # - Make sure only trials within evolution period are considered
            vnTrialStarts = vnTrialStarts[
                np.logical_and(
                    self._nTimeStep <= vnTrialStarts,
                    vnTrialStarts < self._nTimeStep + nNumTimeSteps,
                )
            ]
            # - Total number of trials
            nNumTrials = vnTrialStarts.size
            # - Array indices of tsInput.vtTimeTrace and tsInput.vnChannels where trials start
            viTrialStartIndices = np.searchsorted(vnInputTimeSteps, vnTrialStarts)
            # - Count number of events for each trial (np.r_ to include last trial)
            vnCumulEventsPerTrial = np.r_[viTrialStartIndices, vnInputTimeSteps.size]
            vnEventsPerTrial = np.diff(vnCumulEventsPerTrial)

            # - First trial of current batch
            iCurrentTrial = 0
            while iCurrentTrial < nNumTrials:
                # - Cumulated numbers of events per trial for coming trials
                vnCumulNextEvents = (
                    vnCumulEventsPerTrial[iCurrentTrial + 1 :]
                    - vnCumulEventsPerTrial[iCurrentTrial]
                )
                nMaxNumTrialsE = np.searchsorted(
                    vnCumulNextEvents, self.nMaxEventsPerBatch
                )
                if self.nMaxNumTimeSteps is not None:
                    # - Cumulated numbers of time steps per trial for coming trials (np.r_ for including last trial)
                    vnCumulNextTimeSteps = (
                        np.r_[vnTrialStarts[iCurrentTrial + 1 :], nNumTimeSteps]
                        - vnTrialStarts[iCurrentTrial]
                    )
                    # - Maximum number of trials before self.nMaxNumTimeSteps is exceeded
                    nMaxNumTrialsNTS = np.searchsorted(
                        vnCumulNextEvents, self.nMaxNumTimeSteps
                    )
                else:
                    nMaxNumTrialsNTS = np.inf
                # - Number of trials to be used in current batch, considering max. number of trials per batch,
                #   events per batch and (if applies) time steps per batch
                nNumTrialsBatch = min(
                    self.nMaxTrialsPerBatch, nMaxNumTrialsE, nMaxNumTrialsNTS
                )
                assert nNumTrialsBatch > 0, (
                    "Layer `{}`: Cannot continue evolution. ".format(self.strName)
                    + "Either too many timesteps or events in this trial."
                )
                # - Start and end time steps and indices (wrt vnInputTimeSteps) of this batch
                nTSStartBatch = vnTrialStarts[iCurrentTrial]
                iStartIndexBatch = viTrialStartIndices[iCurrentTrial]
                try:
                    nTSEndBatch = vnTrialStarts[iCurrentTrial + nNumTrialsBatch]
                    iEndIndexBatch = viTrialStartIndices[
                        iCurrentTrial + nNumTrialsBatch
                    ]
                except IndexError as e:
                    # - If index error occurs because final batch is included
                    if iCurrentTrial + nNumTrialsBatch == viTrialStartIndices.size:
                        iEndIndexBatch = vnInputTimeSteps.size
                        nTSEndBatch = nNumTimeSteps + self._nTimeStep
                    else:
                        raise e
                # - Generate event list for fpga
                lInputEvents = arrays_to_spike_list(
                    vnTimeSteps=vnInputTimeSteps[iStartIndexBatch:iEndIndexBatch],
                    vnChannels=vnInputChannels[iStartIndexBatch:iEndIndexBatch],
                    nTSStart=nTSStartBatch,
                    lNeurons=self._vVirtualNeurons,
                    nTargetCoreMask=self._nInputCoreMask,
                    nTargetChipID=self._nInputChipID,
                )
                nNumTimeStepsBatch = nTSEndBatch - nTSStartBatch
                if bVerbose:
                    nNumEventsBatch = iEndIndexBatch - iStartIndexBatch
                    print(
                        "Layer `{}`: Current batch input: {} s ({} timesteps), {} events".format(
                            self.strName,
                            nNumTimeStepsBatch * self.tDt,
                            nNumTimeStepsBatch,
                            nNumEventsBatch,
                        )
                    )
                yield lInputEvents, nNumTimeStepsBatch * self.tDt
                iCurrentTrial += nNumTrialsBatch
        else:
            ## -- Split by Maximum number of events and time steps
            # - Handle None for nMaxNumTimeSteps
            nMaxNumTimeSteps = (
                nNumTimeSteps
                if self.nMaxNumTimeSteps is None
                else self.nMaxNumTimeSteps
            )
            # - Time step at which current batch starts
            nTSStartBatch = self._nTimeStep
            # - Corresponding index wrt vnInputTimeSteps
            iStartIndexBatch = 0
            # - Time step after evolution ends
            nTSEndEvo = nTSStartBatch + nNumTimeSteps
            while nTSStartBatch < nTSEndEvo:
                # - Endpoint of current batch
                nTSEndBatch = min(nTSStartBatch + nMaxNumTimeSteps, nTSEndEvo)
                # - Corresponding intex wrt vnInputTimeSteps
                iEndIndexBatch = np.searchsorted(vnInputTimeSteps, nTSEndBatch)
                # - Correct if too many events are included
                if iEndIndexBatch - iStartIndexBatch > self.nMaxEventsPerBatch:
                    iEndIndexBatch = iStartIndexBatch + self.nMaxEventsPerBatch
                    nTSEndBatch = vnInputTimeSteps[iEndIndexBatch]
                # - Generate event list for fpga
                lInputEvents = arrays_to_spike_list(
                    vnTimeSteps=vnInputTimeSteps[iStartIndexBatch:iEndIndexBatch],
                    vnChannels=vnInputChannels[iStartIndexBatch:iEndIndexBatch],
                    nTSStart=nTSStartBatch,
                    lNeurons=self._vVirtualNeurons,
                    nTargetCoreMask=self._nInputCoreMask,
                    nTargetChipID=self._nInputChipID,
                )
                nNumTimeStepsBatch = nTSEndBatch - nTSStartBatch
                if bVerbose:
                    nNumEventsBatch = iEndIndexBatch - iStartIndexBatch
                    print(
                        "Layer `{}`: Current batch input: {} s ({} timesteps), {} events".format(
                            self.strName,
                            nNumTimeStepsBatch * self.tDt,
                            nNumTimeStepsBatch,
                            nNumEventsBatch,
                        )
                    )
                yield lInputEvents, nNumTimeStepsBatch * self.tDt
                nTSStartBatch = nTSEndBatch
                iStartIndexBatch = iEndIndexBatch

    def evolve(
        self,
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

        # - Lists for storing recorded data
        lTimeTrace = []
        lChannels = []

        # - Clip tsInput to required duration
        gInputGenerator = self._batch_input_data(
            tsInput.clip([self.t, self.t + tDuration]), nNumTimeSteps, bVerbose
        )
        tStartBatch = self.t
        for lCurrentEvents, tDurBatch in gInputGenerator:
            # - Send event sequence to fpga module
            DHW_dDynapse["fpgaSpikeGen"].preload_stimulus(lCurrentEvents)
            if bVerbose:
                print("Layer `{}`: Stimulus preloaded.".format(self.strName))

            # -- Set up event recording
            oFilter = BufferedEventFilter(
                DHW_dDynapse["model"], [n.get_id() for n in self._vHWNeurons]
            )
            if bVerbose:
                print("Layer `{}`: Event filter ready.".format(self.strName))

            # - Stimulate / record for desired duration
            if bVerbose:
                print("Layer `{}`: Starting stimulation.".format(self.strName))
            DHW_dDynapse["fpgaSpikeGen"].start()
            # - Wait for tDuration + some additional time
            if bVerbose:
                print(
                    "Layer `{}`: Waiting for stimulation to end.".format(self.strName)
                )
            time.sleep(tDurBatch + 0.5)
            # - Stop stimulation and clear filter to stop recording events
            DHW_dDynapse["fpgaSpikeGen"].stop()
            oFilter.clear()
            if bVerbose:
                print("Layer `{}`: Stimulation ended.".format(self.strName))

            lEventsBatch = list(oFilter.get_events())
            lTriggerBatch = list(oFilter.get_special_event_timestamps())
            if bVerbose:
                print(
                    "Layer `{}`: Recorded {} events and {} trigger events".format(
                        self.strName, len(lEventsBatch), len(lTriggerBatch)
                    )
                )

            # - Extract monitored event channels and timestamps
            vnChannelsBatch = neurons_to_channels(
                [e.neuron for e in lEventsBatch], list(self._vHWNeurons)
            )
            vtTimeTraceBatch = np.array([e.timestamp for e in lEventsBatch]) * 1e-6

            # - Locate synchronisation timestamp
            tStartTriggerBatch = lTriggerBatch[0] * 1e-6
            iStartIndex = np.searchsorted(vtTimeTraceBatch, tStartTriggerBatch)
            iEndIndex = np.searchsorted(
                vtTimeTraceBatch, tStartTriggerBatch + tDurBatch
            )
            lChannels.append(vnChannelsBatch[iStartIndex:iEndIndex])
            lTimeTrace.append(
                vtTimeTraceBatch[iStartIndex:iEndIndex]
                - tStartTriggerBatch
                + tStartBatch
            )
            if bVerbose:
                print("Layer `{}`: Extracted event data".format(self.strName))

            # - Start time of next batch
            tStartBatch += tDurBatch

        # - Flatten out lTimeTrace and lChannels
        lTimeTrace = [t for vThisTrace in lTimeTrace for t in vThisTrace]
        lChannels = [ch for vTheseChannels in lChannels for ch in vTheseChannels]
        print(np.amax(lChannels))
        # - Convert recorded events into TSEvent object
        tsResponse = TSEvent(
            lTimeTrace, lChannels, nNumChannels=self.nSize, strName="DynapSE spikes"
        )

        # - Set layer time
        self._nTimeStep += nNumTimeSteps

        if bVerbose:
            print("Layer `{}`: Evolution successful.".format(self.strName))

        return tsResponse

    def _compile_weights_and_configure(self):
        """
        _compile_weights_and_configure - Configure DynapSE weights from the weight matrices
        """

        # - Get a connector object
        connector = NeuronNeuronConnector.DynapseConnector()

        # - Get input to layer connections
        liPreSynE, liPostSynE, liPreSynI, liPostSynI = connectivity_matrix_to_prepost_lists(
            self.mfWIn
        )

        # - Connect input to layer

        # - Set excitatory input connections
        connector.add_connection_from_list(
            self._vVirtualNeurons[liPreSynE],
            self._vShadowNeurons[liPostSynE],
            [SynapseTypes.FAST_EXC],
        )
        print(
            "Layer `{}`: Excitatory connections from virtual neurons to layer neurons have been set.".format(
                self.strName
            )
        )
        # - Set inhibitory input connections
        connector.add_connection_from_list(
            self._vVirtualNeurons[liPreSynI],
            self._vShadowNeurons[liPostSynI],
            [SynapseTypes.FAST_INH],
        )
        print(
            "Layer `{}`: Inhibitory connections from virtual neurons to layer neurons have been set.".format(
                self.strName
            )
        )

        # - Infer how many input neurons there are (i.e. any neuron that receives input from a virtual neuron)
        nNumInputNeurons = max(np.amax(liPostSynE), np.amax(liPostSynI))

        ## -- Set connections wihtin hardware layer
        liPreSynE, liPostSynE, liPreSynI, liPostSynI = connectivity_matrix_to_prepost_lists(
            self.mfWRec
        )

        viPreSynE = np.array(liPreSynE)
        viPreSynI = np.array(liPreSynI)
        viPostSynE = np.array(liPostSynE)
        viPostSynI = np.array(liPostSynI)

        # - Connections from input neurons to exceitatory neurons
        # Excitatory
        connector.add_connection_from_list(
            self._vShadowNeurons[viPreSynE[viPreSynE < nNumInputNeurons]],
            self._vShadowNeurons[viPostSynE[viPreSynE < nNumInputNeurons]],
            [SynapseTypes.FAST_EXC],
        )
        print(
            "Layer `{}`: Excitatory connections from input neurons to reservoir neurons have been set.".format(
                self.strName
            )
        )
        # Inhibitory
        connector.add_connection_from_list(
            self._vShadowNeurons[viPreSynI[viPreSynI < nNumInputNeurons]],
            self._vShadowNeurons[viPostSynI[viPreSynI < nNumInputNeurons]],
            [SynapseTypes.FAST_INH],
        )
        print(
            "Layer `{}`: Inhibitory connections from input neurons to reservoir neurons have been set.".format(
                self.strName
            )
        )

        # - Connections between recurrently connected excitatory neurons and inhibitory neurons
        # Excitatory
        connector.add_connection_from_list(
            self._vShadowNeurons[viPreSynE[viPreSynE >= nNumInputNeurons]],
            self._vShadowNeurons[viPostSynE[viPreSynE >= nNumInputNeurons]],
            [SynapseTypes.SLOW_EXC],
        )
        print(
            "Layer `{}`: Excitatory connections within layer have been set.".format(
                self.strName
            )
        )
        # - Set inhibitory connections
        connector.add_connection_from_list(
            self._vShadowNeurons[viPreSynI[viPreSynI >= nNumInputNeurons]],
            self._vShadowNeurons[viPostSynI[viPreSynI >= nNumInputNeurons]],
            [SynapseTypes.FAST_INH],
        )
        print(
            "Layer `{}`: Inhibitory connections within layer have been set.".format(
                self.strName
            )
        )

        DHW_dDynapse['model'].apply_diff_state()
        print("Layer `{}`: Connections have been written to the chip.".format(self.strName))

    @property
    def cInput(self):
        return TSEvent

    @property
    def cOutput(self):
        return TSEvent

    @property
    def mfWIn(self):
        return self._mfWIn

    @mfWIn.setter
    def mfWIn(self, mfNewW):
        self._mfWIn = np.round(
            self._expand_to_shape(
                mfNewW, (self.nSizeIn, self.nSize), "mfWIn", bAllowNone=False
            )
        ).astype(int)

    @property
    def mfWRec(self):
        return self._mfWRec

    @mfWRec.setter
    def mfWRec(self, mfNewW):
        self._mfWRec = np.round(
            self._expand_to_shape(
                mfNewW, (self.nSize, self.nSize), "mfWRec", bAllowNone=False
            )
        ).astype(int)

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
    def nMaxTrialsPerBatch(self):
        return self._nMaxTrialsPerBatch

    @nMaxTrialsPerBatch.setter
    def nMaxTrialsPerBatch(self, nNewMax):
        assert nNewMax is None or (
            type(nNewMax) == int and 0 < nNewMax
        ), "Layer `{}`: nMaxTrialsPerBatch must be an integer greater than 0 or None.".format(
            self.strName
        )
        self._nMaxTrialsPerBatch = nNewMax

    @property
    def nMaxNumTimeSteps(self):
        return self._nMaxNumTimeSteps

    @nMaxNumTimeSteps.setter
    def nMaxNumTimeSteps(self, nNewMax):
        assert nNewMax is None or (
            type(nNewMax) == int and 0 < nNewMax
        ), "Layer `{}`: nMaxNumTimeSteps must be an integer greater than 0 or None.".format(
            self.strName
        )
        self._nMaxNumTimeSteps = nNewMax

    @property
    def tMaxBatchDur(self):
        return (
            None
            if self._nMaxNumTimeSteps is None
            else self._nMaxNumTimeSteps * self.tDt
        )

    @tMaxBatchDur.setter
    def tMaxBatchDur(self, tNewMax):
        assert tNewMax is None or (
            type(tNewMax) == int and 0 < tNewMax
        ), "Layer `{}`: tMaxBatchDur must be an integer greater than 0 or None.".format(
            self.strName
        )
        self._nMaxNumTimeSteps = (
            None if tNewMax is None else int(np.round(tNewMax / self.tDt))
        )

    @property
    def nMaxEventsPerBatch(self):
        return self._nMaxEventsPerBatch

    @nMaxEventsPerBatch.setter
    def nMaxEventsPerBatch(self, nNewMax):
        assert (
            type(nNewMax) == int and 0 < nNewMax <= nFpgaEventLimit
        ), "Layer `{}`: nMaxEventsPerBatch must be an integer between 0 and {}.".format(
            self.strName, nFpgaEventLimit
        )
        self._nMaxEventsPerBatch = nNewMax


def connectivity_matrix_to_prepost_lists(
    mnW: np.ndarray
) -> Tuple[List[int], List[int], List[int], List[int]]:
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
        for _ in range(np.abs(mnW[nPre, nPost])):
            vnPreI.append(nPre)
            vnPostI.append(nPost)

    # - Return augmented lists
    return vnPreE, vnPostE, vnPreI, vnPostI


def TSEvent_to_spike_list(
    tsSeries: TSEvent,
    lNeurons: List,
    nTargetCoreMask: int = 1,
    nTargetChipID: int = 0,
) -> List:
    """
    TSEvent_to_spike_list - Convert a TSEvent object to a ctxctl spike list

    :param tsSeries:        TSEvent         Time series of events to send as input
    :param lNeurons:        List[Neuron]    List of neurons that should appear as sources of the events
    :param nTargetCoreMask: int          Mask defining target cores (sum of 2**core_id)
    :param nTargetChipID:   int          ID of target chip
    :return:                list of FpgaSpikeEvent objects
    """
    # - Check that the number of channels is the same between time series and list of neurons
    assert tsSeries.nNumChannels <= len(
        lNeurons
    ), "`tsSeries` contains more channels than the number of neurons in `lNeurons`."

    # - Get events from this time series
    vtTimes, vnChannels, _ = tsSeries.find()

    # - Convert to ISIs
    try:
        tStartTime = tsSeries.tStartTime
    except AttributeError:
        tStartTime = 0
    vtISIs = np.diff(np.r_[tStartTime, vtTimes])
    vnDiscreteISIs = (np.round(vtISIs / DHW_dDynapse["tISIBase"])).astype("int")

    # - Get neuron information
    try:
        vnNeuronIDs = [n.get_id() for n in lNeurons]
    except AttributeError:
        # - If lNeurons contains virtual neurons, they do not have a get_id() method
        vnNeuronIDs = [n.get_neuron_id() for n in lNeurons]

    # - Convert each event to an FpgaSpikeEvent
    def generate_fpga_event(nTargetChipID, nTargetCoreMask, nNeuronID, nISI) -> FpgaSpikeEvent:
        event = FpgaSpikeEvent()
        event.target_chip = nTargetChipID
        event.core_mask = nTargetCoreMask
        event.neuron_id = nNeuronID
        event.isi = nISI
        return event

    lEvents = [
        generate_fpga_event(
            nTargetChipID, nTargetCoreMask, vnNeuronIDs[nChannel], nISI
        )
        for nChannel, nISI in zip(vnChannels, vnDiscreteISIs)
    ]

    # - Return a list of events
    return lEvents


def arrays_to_spike_list(
    vnTimeSteps: np.ndarray,
    vnChannels: np.ndarray,
    lNeurons: List[Neuron],
    nTSStart: int = 0,
    nTargetCoreMask: int = 1,
    nTargetChipID: int = 0,
) -> List:
    """
    arrays_to_spike_list - Convert an array of input time steps and an an array
                           of event channels to a ctxctl spike list

    :param vnTimeSteps:     np.ndarray   Event time steps (Using FPGA time base)
    :param vnChannels:      np.ndarray   Event time steps (Using FPGA time base)
    :param lNeurons:        List[Neuron] List of neurons that should appear as sources of the events
    :param nTSStart:        int          Time step at which to start
    :param nTargetCoreMask: int          Mask defining target cores (sum of 2**core_id)
    :param nTargetChipID:   int          ID of target chip
    :return:                list of FpgaSpikeEvent objects
    """
    # - Ignore data that comes before nTSStart
    vnTimeSteps = vnTimeSteps[vnTimeSteps >= nTSStart]
    vnChannels = vnChannels[vnTimeSteps >= nTSStart]

    # - Check that the number of channels is the same between time series and list of neurons
    assert np.amax(vnChannels) <= np.size(
        lNeurons
    ), "`vnChannels` contains more channels than the number of neurons in `lNeurons`."

    # - Convert to ISIs
    vnDiscreteISIs = np.diff(np.r_[nTSStart, vnTimeSteps])

    # - Get neuron information
    try:
        vnNeuronIDs = [n.get_id() for n in lNeurons]
    except AttributeError:
        # - If lNeurons contains virtual neurons, they do not have a get_id() method
        vnNeuronIDs = [n.get_neuron_id() for n in lNeurons]

    # - Convert each event to an FpgaSpikeEvent
    def generate_fpga_event(nTargetChipID, nTargetCoreMask, nNeuronID, nISI) -> FpgaSpikeEvent:
        event = FpgaSpikeEvent()
        event.target_chip = nTargetChipID
        event.core_mask = nTargetCoreMask
        event.neuron_id = nNeuronID
        event.isi = nISI
        return event

    lEvents = [
        generate_fpga_event(
            nTargetChipID, nTargetCoreMask, vnNeuronIDs[nChannel], nISI
        )
        for nChannel, nISI in zip(vnChannels, vnDiscreteISIs)
    ]

    # - Return a list of events
    return lEvents


def neurons_to_channels(
    lNeurons: List[DynapseNeuron], lLayerNeurons: List[DynapseNeuron]
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

def collect_spiking_neurons(liRecordNeuronIDs, tDuration):
    if isinstance(liRecordNeuronIDs, int):
        liRecordNeuronIDs = range(liRecordNeuronIDs)
    liRecordNeuronIDs = list(liRecordNeuronIDs)
    print("DHW: Collecting IDs of neurons that spike within the next {} seconds".format(tDuration))
    oFilter = BufferedEventFilter(DHW_dDynapse["model"], liRecordNeuronIDs)
    time.sleep(tDuration)
    oFilter.clear()
    lnNeuronIDs = sorted(set((event.neuron.get_id() for event in oFilter.get_events())))
    print("DHW: {} neurons spiked".format(len(lnNeuronIDs)))
    return lnNeuronIDs

def silence_hot_neurons(vnNeuronIDs, tDuration):
    lnHotNeurons = collect_spiking_neurons(vnNeuronIDs, tDuration=tDuration)
    # - Silence these neurons by assigning different Tau bias
    for nID in lnHotNeurons:
        dynapse.set_tau_2(0, nID)
    print("DHW: Neurons {} have been silenced".format(lnHotNeurons))