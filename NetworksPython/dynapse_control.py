# ----
# dynapse_control.py - Class to interface the DynapSE chip
# ----

### --- Imports

import numpy as np
from warnings import warn
from typing import Tuple, List, Optional, Union
import time
from .timeseries import TSEvent

# - Programmatic imports for CtxCtl
bUsing_rPyC = False
RPYC_TIMEOUT = 300

try:
    import CtxDynapse
    import NeuronNeuronConnector

except ModuleNotFoundError:
    # - Try with RPyC
    try:
        import rpyc
        conn = rpyc.classic.connect('localhost', '1300')
        conn._config["sync_request_timeout"] = RPYC_TIMEOUT
        CtxDynapse = conn.modules.CtxDynapse
        NeuronNeuronConnector = conn.modules.NeuronNeuronConnector
        print("dynapse_control: RPyC connection established.")
        bUsing_rPyC = True

    except:
        # - Raise an ImportError (so that the smart __init__.py module can skip over missing CtxCtl
        raise ImportError

# - Imports from ctxCTL
SynapseTypes = CtxDynapse.DynapseCamType
dynapse = CtxDynapse.dynapse
DynapseFpgaSpikeGen = CtxDynapse.DynapseFpgaSpikeGen
DynapsePoissonGen = CtxDynapse.DynapsePoissonGen
DynapseNeuron = CtxDynapse.DynapseNeuron
VirtualNeuron = CtxDynapse.VirtualNeuron
BufferedEventFilter = CtxDynapse.BufferedEventFilter
FpgaSpikeEvent = CtxDynapse.FpgaSpikeEvent

print("dynapse_control: CtxDynapse modules loaded.")


### --- Parameters
# - Fix (hardware)
FPGA_EVENT_LIMIT = int(2 ** 16 - 1)  # Max. number of events that can be sent to FPGA
FPGA_ISI_LIMIT = int(2 ** 16 - 1)  # Max. number of timesteps for single inter-spike interval between FPGA events
FPGA_TIMESTEP = 1. / 9. * 1e-7  # Internal clock of FPGA, 11.111...ns
CORE_DIMENSIONS = (16, 16)  # Numbers of neurons in core (rows, columns)
# - Default values, can be changed
DEF_FPGA_ISI_BASE = 2e-5  # Default timestep between events sent to FPGA
DEF_FPGA_ISI_MULTIPLIER = int(np.round(DEF_FPGA_ISI_BASE / FPGA_TIMESTEP))

# MOVE THIS TO LAYER #
# - Default maximum numbers of time steps for a single evolution batch
#   Assuming one input event after the maximum ISI - This is the maximally possible
#   value. In practice there will be more events per time. Therefore the this value
#   does not guarantee that the complete input batch fits onto the fpga
nDefaultMaxNumTimeSteps = int(FPGA_EVENT_LIMIT * FPGA_ISI_LIMIT)


### --- Utility functions
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

def rectangular_neuron_arrangement(
    nFirstNeuron: int,
    nNumNeurons: int,
    nWidth: int,
    tupCoreDimensions: tuple=CORE_DIMENSIONS,
) -> List[int]:
    """
    rectangular_neuron_arrangement: return neurons that form a rectangle on the chip
                                    with nFirstNeuron as upper left corner and width
                                    nWidth. (Last row may not be full). Neurons
                                    have to fit onto single core.
    :param nFirstNeuron:  int ID of neuron that is in the upper left corner of the rectangle
    :param nNumNeurons:   int Number of neurons in the rectangle
    :param nWidth:        int Width of the rectangle (in neurons) 
    :param tupCoreDimensions:  tuple (number of neuron rows in core, number of neurons in row)
    :return lNeuronIDs:   list 1D array of IDs of neurons that from the rectangle.
    """
    nHeightCore, nWidthCore = tupCoreDimensions
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

def generate_event_raster(
    lEvents: list,
    tDuration: float,
    vnNeuronIDs: list,
) -> np.ndarray:
    """
    generate_event_raster - Generate a boolean spike raster of a list of events with timestep 0.001ms
    :param lEvents:         list of Spike objects
    :param tDuration:       float Overall time covered by the raster
    :param vnNeuronIDs:     array-like of neuron IDs corresponding to events.
    
    :return:
        mbEventRaster   np.ndarray  - Boolean event raster
    """
    vnNeuronIDs = list(vnNeuronIDs)        
    # - Extract event timestamps and neuron IDs
    tupTimestamps, tupEventNeuronIDs = zip(*((event.timestamp, event.neuron.get_id()) for event in lEvents))
    # - Event times in microseconds
    vnEventTimes = np.array(tupTimestamps, int)
    # - Set time of first event to 0
    vnEventTimes -= vnEventTimes[0]
    # - Convert neuron IDs of event to index starting from 0
    viEventIndices = np.array([vnNeuronIDs.index(nNeuronID) for nNeuronID in tupEventNeuronIDs])
    # - Convert events to binary raster
    nTimeSteps = int(np.ceil(tDuration*1e6))
    mbEventRaster = np.zeros((nTimeSteps, np.size(viEventIndices)), bool)
    mbEventRaster[vnEventTimes, viEventIndices] = True
    return mbEventRaster

def evaluate_firing_rates(
    lEvents: list,
    tDuration: float,
    vnNeuronIDs: Optional[list] = None,
) -> (np.ndarray, float, float):
    """
    evaluate_firing_rates - Determine the neuron-wise firing rates from a
                            list of events. Calculate mean, max and min.
    :param lEvents:         list of Spike objects
    :param tDuration:       float Time over which rates are normalized
    :param vnNeuronIDs:     array-like of neuron IDs corresponding to events.
                            If None, vnNeuronIDs will consists of to the neurons
                            corresponding to the events in lEvents.

    :return:
        vfFiringRates  np.ndarray - Each neuron's firing rate
        fMeanRate      float - Average firing rate over all neurons
        fMaxRate       float - Highest firing rate of all neurons
        fMinRate       float - Lowest firing rate of all neurons            
    """
    # - Extract event timestamps and neuron IDs
    tupTimestamps, tupEventNeuronIDs = zip(*((event.timestamp, event.neuron.get_id()) for event in lEvents))
    
    # - Count events for each neuron
    viUniqueEventIDs, vnEventCounts = np.unique(tupEventNeuronIDs, return_counts=True)
    
    if vnNeuronIDs is None:
        # - lNeuronIDs as list of neurons that have spiked
        vnNeuronIDs = viUniqueEventIDs
    
    # - Neurons that have not spiked
    viNoEvents = (np.asarray(vnNeuronIDs))[np.isin(vnNeuronIDs, viUniqueEventIDs) == False]
    
    # - Count events
    viUniqueEventIDs = np.r_[viUniqueEventIDs, viNoEvents]
    vnEventCounts = np.r_[vnEventCounts, np.zeros(viNoEvents.size)]

    # - Sort event counts to same order as in vnNeuronIDs
    liUniqueEventIDs = list(viUniqueEventIDs)
    lSort = [liUniqueEventIDs.index(nNeuronID) for nNeuronID in vnNeuronIDs]
    vnEventCounts = vnEventCounts[lSort]
    vfFiringRates = vnEventCounts / tDuration

    # - Calculate mean, max and min rates
    fMeanRate = np.size(lEvents) / tDuration / len(vnNeuronIDs)
    print("Mean firing rate: {} Hz".format(fMeanRate))
    iMaxRateNeuron = np.argmax(vnEventCounts)
    fMaxRate = vfFiringRates[iMaxRateNeuron]
    print("Maximum firing rate: {} Hz (neuron {})".format(fMaxRate, vnNeuronIDs[iMaxRateNeuron]))
    iMinRateNeuron = np.argmin(vnEventCounts)
    fMinRate = vfFiringRates[iMinRateNeuron]
    print("Minimum firing rate: {} Hz (neuron {})".format(fMinRate, vnNeuronIDs[iMinRateNeuron]))

    return vfFiringRates, fMeanRate, fMaxRate, fMinRate

def teleport_function(func):
    """
    telport_function - Decorator. If using RPyC, then teleport the resulting function

    :param func:            Function to maybe teleport
    :return:                Maybe teleported function
    """
    # - Teleport if bUsing_RPyC flag is set
    if bUsing_rPyC:
        return rpyc.classic.teleport_function(conn, func)

    else:
        # - Otherwise just return the undecorated function
        return func

@teleport_function
def extract_event_data(lEvents):
    ltupEvents = [(event.timestamp, event.neuron.get_id()) for event in lEvents]
    lTimeStamps, lNeuronIDs = zip(*ltupEvents)
    return lTimeStamps, lNeuronIDs

def event_data_to_channels(
    lEvents: List, lLayerNeuronIDs: List
) -> (np.ndarray, np.ndarray):
    lTimeStamps, lNeuronIDs = extract_event_data(lEvents)
    # - Convert to numpy array and thereby fetch data from connection if using RPyC
    vTimeStamps = np.array(lTimeStamps)
    vNeuronIDs = np.array(lNeuronIDs)
    # - Convert neuron IDs to channels
    dChannels = {nID: iChannel for iChannel, nID in enumerate(lLayerNeuronIDs)}
    vChannelIndices = np.array(list(
        map(lambda nID: dChannels.get(nID, float("nan")), vNeuronIDs)
    ))
    if np.isnan(vChannelIndices).any():
        warn("dynapse_control: Some events did not match `lLayerNeuronIDs`")
    return vTimeStamps, vChannelIndices

@teleport_function
def generate_fpga_event_list(
    vnDiscreteISIs: list,
    vnNeuronIDs: list,
    nTargetCoreMask: int,
    nTargetChipID: int,
) -> list:
    """
    generate_fpga_event_list - Generate a list of FpgaSpikeEvent objects
    :param vnDiscreteISIs:  array-like  Inter-spike intervalls in Fpga time base
    :param vnNeuronIDs:     array-like  IDs of neurons corresponding to events
    :param nNeuronID:       int ID of source neuron
    :param nTargetCoreMask: int Coremask to determine target cores

    :return:
        event  list of generated FpgaSpikeEvent objects.
    """
    from CtxDynapse import FpgaSpikeEvent
    from copy import deepcopy
    
    # - Make sure objects live on required side of RPyC connection
    nTargetCoreMask = int(nTargetCoreMask)
    nTargetChipID = int(nTargetChipID)
    vnNeuronIDs = deepcopy(vnNeuronIDs)
    vnDiscreteISIs = deepcopy(vnDiscreteISIs)
    
    def generate_fpga_event(
        nNeuronID: int,
        nISI: int,
    ) -> FpgaSpikeEvent:
        """
        generate_fpga_event - Generate a single FpgaSpikeEvent objects.
        :param nNeuronID:       int ID of source neuron
        :param nISI:            int Timesteps after previous event before
                                    this event will be sent
        :return:
            event  FpgaSpikeEvent
        """
        event = FpgaSpikeEvent()
        event.target_chip = nTargetChipID
        event.core_mask = nTargetCoreMask
        event.neuron_id = nNeuronID
        event.isi = nISI
        return event
    
    # - Generate events
    print("dynapse_control: Generating event list")
    lEvents = [
        generate_fpga_event(nNeuronID, nISI)
        for nNeuronID, nISI in zip(vnNeuronIDs, vnDiscreteISIs)
    ]
    
    return lEvents

@teleport_function
def clear_chips(
    oDynapse,
    lClearChips: Optional[list]=None,
    lShadowNeurons: Optional[list] = None,
):
    """
    clear_chips - Clear the CAM and SRAM cells of chips defined in lClearCam.

    :param oDynapse:     dynapse object
    :param lClearChips:  list or None  IDs of chips where configurations
                                       should be cleared.
    """
    # - Make sure lClearChips is a list
    if lClearChips is None:
        return
    
    if isinstance(lClearChips, int):
        lClearChips = [lClearChips, ]

    from copy import deepcopy
    # - Make sure that lClearChips is on correct side of RPyC connection
    lClearChips = deepcopy(lClearChips)
    
    for nChip in lClearChips:
        print("DynapseControl: Clearing chip {}.".format(nChip))

        # - Clear CAMs
        oDynapse.clear_cam(int(nChip))
        print("\t CAMs cleared.")

        # - Clear SRAMs
        oDynapse.clear_sram(int(nChip))
        print("\t SRAMs cleared.")

        # - Reset neuron weights in model
        for neuron in lShadowNeurons[nChip*1024: (nChip+1)*1024]:
            # - Reset SRAMs for this neuron
            vSrams = neuron.get_srams()
            for iSramIndex in range(1, 4):
                vSrams[iSramIndex].set_target_chip_id(0)
                vSrams[iSramIndex].set_virtual_core_id(0)
                vSrams[iSramIndex].set_used(False)
                vSrams[iSramIndex].set_core_mask(0)

            # - Reset CAMs for this neuron
            for cam in neuron.get_cams():
                cam.set_pre_neuron_id(0)
                cam.set_pre_neuron_core_id(0)
        print("\t Model neuron weights have been reset.")

    print("DynapseControl: Chips cleared.")

@teleport_function
def get_all_neurons(
    oModel: CtxDynapse.Model,
    oVirtualModel: CtxDynapse.VirtualModel
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    get_all_neurons - Get hardware, virtual and shadow state neurons
                      from oModel and oVirtualModel and return them
                      in arrays.
    :param oModel:  CtxDynapse.Model
    :param oVirtualModel: CtxDynapse.VirtualModel
    :return:
        np.ndarray  Hardware neurons
        np.ndarray  Virtual neurons
        np.ndarray  Shadow state neurons
    """
    lHWNeurons = oModel.get_neurons()
    lVirtualNeurons = oVirtualModel.get_neurons()
    lShadowNeurons = oModel.get_shadow_state_neurons()
    print("dynapse_control: Fetched all neurons from models.")
    return lHWNeurons, lVirtualNeurons, lShadowNeurons

@teleport_function
def remove_all_connections_to(vNeurons: List, oModel, bApplyDiff: bool = True):
    """
    remove_all_connections_to - Remove all presynaptic connections
                                to neurons defined in vnNeuronIDs
    :param vnNeuronIDs:     np.ndarray IDs of neurons whose presynaptic
                                       connections should be removed
    :param oModel:          CtxDynapse.model
    :param bApplyDiff:      bool If False do not apply the changes to
                                 chip but only to shadow states of the
                                 neurons. Useful if new connections are
                                 going to be added to the given neurons.
    """
    # - Reset neuron weights in model
    for neuron in vNeurons:
        # - Reset SRAMs
        viSrams = neuron.get_srams()
        for iSramIndex in range(1, 4):
            viSrams[iSramIndex].set_target_chip_id(0)
            viSrams[iSramIndex].set_virtual_core_id(0)
            viSrams[iSramIndex].set_used(False)
            viSrams[iSramIndex].set_core_mask(0)

        # - Reset CAMs
        viCams = neuron.get_cams()
        for cam in viCams:
            cam.set_pre_neuron_id(0)
            cam.set_pre_neuron_core_id(0)

    print("DynapseControl: Shadow state neuron weights have been reset")

    if bApplyDiff:
        # - Apply changes to the connections on chip
        oModel.apply_diff_state()
        print("DynapseControl: New state has been applied to the hardware")

@teleport_function
def generate_buffered_filter(
    model, lnRecordNeuronIDs
):
    from CtxDynapse import BufferedEventFilter
    return BufferedEventFilter(model, lnRecordNeuronIDs)

@teleport_function
def 

class DynapseControl():

    _nFpgaEventLimit = FPGA_EVENT_LIMIT
    _nFpgaIsiLimit = FPGA_ISI_LIMIT
    _tFpgaTimestep = FPGA_TIMESTEP

    def __init__(self,
                 tFpgaIsiBase: float=DEF_FPGA_ISI_BASE,
                 lClearChips: Optional[list] = None,
    ):
        """
        DynapseControl - Class for interfacing DynapSE

        :param tFpgaIsiBase:    float           Time step for inter-spike intervals when sending events to FPGA
        :param lClearChips:     list or None    IDs of chips where configurations should be cleared.
        """

        print("DynapseControl: Initializing DynapSE")
       
        # - Chip model and virtual model
        self.model = CtxDynapse.model
        self.virtualModel = CtxDynapse.VirtualModel()

        # - dynapse object from CtxDynapse
        self.dynapse = dynapse

        ## -- Modules for sending input to FPGA
        lFPGAModules = self.model.get_fpga_modules()

        # - Find a spike generator module
        vbIsSpikeGenModule = [isinstance(m, DynapseFpgaSpikeGen) for m in lFPGAModules]
        if not np.any(vbIsSpikeGenModule):
            # There is no spike generator, so we can't use this Python layer on the HW
            raise ModuleNotFoundError(
                "DynapseControl: An `fpgaSpikeGen` module is required to use the DynapSE layer."
            )
        else:
            # Get first spike generator module
            self.fpgaSpikeGen = lFPGAModules[np.argwhere(vbIsSpikeGenModule)[0][0]]
            print("DynapseControl: Spike generator module ready.")

        # - Find a poisson spike generator module
        vbIsPoissonGenModule = [isinstance(m, DynapsePoissonGen) for m in lFPGAModules]
        if np.any(vbIsPoissonGenModule):
            self.fpgaPoissonGen = lFPGAModules[np.argwhere(vbIsPoissonGenModule)[0][0]]
        else:
            warn("DynapseControl: Could not find poisson generator module (DynapsePoissonGen).")

        # - Get all neurons from models
        self.lHWNeurons, self.lVirtualNeurons, self.lShadowNeurons = get_all_neurons(
            self.model, self.virtualModel
        )

        # - Initialise neuron allocation
        self.clear_neuron_assignments(True, True)

        print("DynapseControl: Neurons initialized.")

        # - Get a connector object
        self.dcNeuronConnector = NeuronNeuronConnector.DynapseConnector()
        print("DynapseControl: Neuron connector initialized")

        # - Wipe configuration
        self.clear_chips(lClearChips)

        ## -- Initialize Fpga spike generator
        self.tFpgaIsiBase = tFpgaIsiBase
        self.fpgaSpikeGen.set_repeat_mode(False)
        self.fpgaSpikeGen.set_variable_isi(True)
        self.fpgaSpikeGen.set_base_addr(0)
        print("DynapseControl: FPGA spike generator prepared.")

        print("DynapseControl ready.")

    def clear_chips(self, lClearChips: Optional[list]=None):
        """
        clear_chips - Clear the CAM and SRAM cells of chips defined in lClearCam.

        :param lClearChips:  list or None  IDs of chips where configurations
                                           should be cleared.
        """
        # - Use `clear_chips` function
        clear_chips(self.dynapse, lClearChips, self.lShadowNeurons)


    ### --- Neuron allocation and connections

    def clear_neuron_assignments(self, bHardware: bool=True, bVirtual: bool=True):
        """
        clear_neuron_assignments - Mark neurons as free again.
        :param bHardware:   Mark all hardware neurons as free (except neuron 0 of each chip)
        :param bVirtual:    Mark all virtual neurons as free (except neuron 0)
        """
        if bHardware:
            # - Hardware neurons
            self.vbFreeHWNeurons = np.ones(len(self.lHWNeurons), bool)
            # Do not use hardware neurons with ID 0 and core ID 0 (first of each core)
            self.vbFreeHWNeurons[0::1024] = False
            print("DynapseControl: {} hardware neurons available.".format(np.sum(self.vbFreeHWNeurons)))

        if bVirtual:
            # - Virtual neurons
            self.vbFreeVirtualNeurons = np.ones(len(self.lVirtualNeurons), bool)
            # Do not use virtual neuron 0
            self.vbFreeVirtualNeurons[0] = False
            print("DynapseControl: {} virtual neurons available.".format(np.sum(self.vbFreeVirtualNeurons)))        

    def allocate_hw_neurons(self, vnNeuronIDs: Union[int, np.ndarray]) -> (np.ndarray, np.ndarray):
        """
        allocate_hw_neurons - Return a list of neurons that may be used.
                              These are guaranteed not to already be assigned.

        :param vnNeuronIDs:  int or np.ndarray    The number of neurons requested or IDs of requested neurons
        :return:             list                 A list of neurons that may be used
        """

        ## -- Choose neurons
        if isinstance(vnNeuronIDs, int):
            # - Choose first available neurons
            nNumNeurons = vnNeuronIDs
            # - Are there sufficient unallocated neurons?
            if np.sum(self.vbFreeHWNeurons) < nNumNeurons:
                raise MemoryError(
                    "Insufficient unallocated neurons available. {} requested.".format(
                        nNumNeurons
                    )
                )
            else:
                # - Pick the first available neurons
                vnNeuronsToAllocate = np.nonzero(self.vbFreeHWNeurons)[0][:nNumNeurons]

        else:
            # - Choose neurons defined in vnNeuronIDs
            vnNeuronsToAllocate = np.array(vnNeuronIDs).flatten()
            # - Make sure neurons are available
            if (self.vbFreeHWNeurons[vnNeuronsToAllocate] == False).any():
                raise MemoryError(
                    "{} of the requested neurons are already allocated.".format(
                        np.sum(
                            self.vbFreeHWNeurons[vnNeuronsToAllocate] == False
                        )
                    )
                )

        # - Mark these neurons as allocated
        self.vbFreeHWNeurons[vnNeuronsToAllocate] = False

        # - Prevent allocation of virtual neurons with same ID as allocated hardware neurons
        vnInputNeuronOverlap = vnNeuronsToAllocate[
            vnNeuronsToAllocate < np.size(self.vbFreeVirtualNeurons)
        ]
        self.vbFreeVirtualNeurons[vnInputNeuronOverlap] = False

        # - Return these allocated neurons
        return (
            np.array([self.lHWNeurons[i] for i in  vnNeuronsToAllocate]),
            np.array([self.lShadowNeurons[i] for i in vnNeuronsToAllocate]),
        )

    def allocate_virtual_neurons(self, vnNeuronIDs: Union[int, np.ndarray]) -> np.ndarray:
        """
        allocate_virtual_neurons - Return a list of neurons that may be used.
                                   These are guaranteed not to already be assigned.

        :param vnNeuronIDs:  int or np.ndarray    The number of neurons requested or IDs of requested neurons
        :return:             list    A list of neurons that may be used
        """
        if isinstance(vnNeuronIDs, int):
            nNumNeurons = vnNeuronIDs
            # - Are there sufficient unallocated neurons?
            if np.sum(self.vbFreeVirtualNeurons) < nNumNeurons:
                raise MemoryError(
                    "Insufficient unallocated neurons available. {}".format(nNumNeurons)
                    + " requested."
                )
            # - Pick the first available neurons
            vnNeuronsToAllocate = np.nonzero(self.vbFreeVirtualNeurons)[0][
                :nNumNeurons
            ]

        else:
            vnNeuronsToAllocate = np.array(vnNeuronIDs).flatten()
            # - Make sure neurons are available
            if (self.vbFreeVirtualNeurons[vnNeuronsToAllocate] == False).any():
                raise MemoryError(
                    "{} of the requested neurons are already allocated.".format(
                        np.sum(
                            self.vbFreeVirtualNeurons[vnNeuronsToAllocate]
                            == False
                        )
                    )
                )

        # - Mark these as allocated
        self.vbFreeVirtualNeurons[vnNeuronsToAllocate] = False
        # - Prevent allocation of hardware neurons with same ID as allocated virtual neurons
        self.vbFreeHWNeurons[vnNeuronsToAllocate] = False

        # - Return these neurons
        return np.array([self.lVirtualNeurons[i] for i in vnNeuronsToAllocate])

    def connect_to_virtual(
        self,
        vnVirtualNeuronIDs: Union[int, np.ndarray],
        vnNeuronIDs: Union[int, np.ndarray],
        lSynapseTypes: List
    ):
        """
        conncect_to_virtual - Connect a group of hardware neurons or
                              single hardware neuron to a groupd of
                              virtual neurons (1 to 1) or to a single
                              virtual neuron.
        :param vnVirtualNeuronIDs:   np.ndarray  IDs of virtual neurons
        :param vnNeuronIDs:          np.ndarray  IDs of hardware neurons
        :param lSynapseTypes:        list        Types of the synapses
        """

        # - Handle single neurons
        if np.size(vnVirtualNeuronIDs) == 1:
            vnVirtualNeuronIDs = np.repeat(vnVirtualNeuronIDs, np.size(vnNeuronIDs))
        if np.size(vnNeuronIDs) == 1:
            vnNeuronIDs = np.repeat(vnNeuronIDs, np.size(vnVirtualNeuronIDs))
        if np.size(lSynapseTypes) == 1:
            lSynapseTypes = list(np.repeat(lSynapseTypes, np.size(vnNeuronIDs)))
        else:
            lSynapseTypes = list(lSynapseTypes)

        # - Get neurons that are to be connected
        lPreNeurons = [self.lVirtualNeurons[i] for i in vnVirtualNeuronIDs]
        lPostNeurons = [self.lShadowNeurons[i] for i in vnNeuronIDs]
        
        # - Set connections
        self.dcNeuronConnector.add_connection_from_list(
            lPreNeurons,
            lPostNeurons,
            lSynapseTypes,
        )
        print("DynapseControl: Setting up {} connections".format(np.size(vnNeuronIDs)))
        self.model.apply_diff_state()
        print("DynapseControl: Connections set")

    def set_virtual_connections_from_weights(
        self,
        mnW: np.ndarray,
        vnVirtualNeuronIDs: np.ndarray,
        vnHWNeuronIDs: np.ndarray,
        synExcitatory: CtxDynapse.DynapseCamType,
        synInhibitory: CtxDynapse.DynapseCamType,
        bApplyDiff: bool=True,
    ):
        """
        set_virtual_connections_from_weights - Set connections from virtual to hardware
                                               neurons based on discrete weight matrix
        :param mnW:                 np.ndarray  Weights for connections from
                                                virtual to layer neurons
        :param vnVirtualNeuronIDs:  np.ndarray  Virtual neuron IDs
        :param vnHWNeuronIDs:       np.ndarray  Hardware neuron IDs
        :param synExcitatory:       DynapseCamType  Excitatory synapse type
        :param synInhibitory:       DynapseCamType  Inhibitory synapse type
        :param bApplyDiff:          bool   If False, do not apply the changes to
                                           chip but only to shadow states of the
                                           neurons. Useful if new connections are
                                           going to be added to the given neurons.
        """
        
        lTimes = [time.time()]

        # - Get connection lists
        liPreSynE, liPostSynE, liPreSynI, liPostSynI = connectivity_matrix_to_prepost_lists(
            mnW
        )
        lTimes.append(time.time())

        # - Set excitatory connections
        self.dcNeuronConnector.add_connection_from_list(
            [self.lVirtualNeurons[i] for i in liPreSynE],
            [self.  lShadowNeurons[i] for i in liPostSynE],
            [synExcitatory],
        )
        print(
            "DynapseControl: Excitatory connections of type `{}`".format(
                str(synExcitatory).split('.')[1]
            )
            + " from virtual neurons to hardware neurons have been set."
        )
        # - Set inhibitory connections
        self.dcNeuronConnector.add_connection_from_list(
            [self.lVirtualNeurons[i] for i in liPreSynI],
            [self.lShadowNeurons[i] for i in liPostSynI],
            [synInhibitory],
        )
        print(
            "DynapseControl: Inhibitory connections of type `{}`".format(
                str(synInhibitory).split('.')[1]
            )
            + " from virtual neurons to hardware neurons have been set."
        )
        lTimes.append(time.time())

        if bApplyDiff:
            self.model.apply_diff_state()
            print("DynapseControl: Connections have been written to the chip.")

        lTimes.append(time.time())
        print(np.diff(lTimes)) 

    def set_connections_from_weights(
        self,
        mnW: np.ndarray,
        vnHWNeuronIDs: np.ndarray,
        synExcitatory: CtxDynapse.DynapseCamType,
        synInhibitory: CtxDynapse.DynapseCamType,
        bApplyDiff: bool=True,
    ):
        """
        set_connections_from_weights - Set connections between hardware neurons
                                       based on  discrete weight matrix
        :param mnW:                 np.ndarray  Weights for connections between
                                                hardware neurons
        :param vnHWNeuronIDs:       np.ndarray  Hardware neuron IDs
        :param synExcitatory:       DynapseCamType  Excitatory synapse type
        :param synInhibitory:       DynapseCamType  Inhibitory synapse type
        :param bApplyDiff:          bool   If False, do not apply the changes to
                                           chip but only to shadow states of the
                                           neurons. Useful if new connections are
                                           going to be added to the given neurons.
        """
        
        lTimes = [time.time()]

        ## -- Connect virtual neurons to hardware neurons
        
        # - Get virtual to hardware connections
        liPreSynE, liPostSynE, liPreSynI, liPostSynI = connectivity_matrix_to_prepost_lists(
            mnW
        )
        lTimes.append(time.time())

        # - Set excitatory input connections
        self.dcNeuronConnector.add_connection_from_list(
            [self.lShadowNeurons[i] for i in liPreSynE],
            [self.lShadowNeurons[i] for i in liPostSynE],
            [synExcitatory],
        )
        print(
            "DynapseControl: Excitatory connections of type `{}`".format(
                str(synExcitatory).split('.')[1]
            )
            + " between hardware neurons have been set."
        )
        # - Set inhibitory input connections
        self.dcNeuronConnector.add_connection_from_list(
            [self.lShadowNeurons[i] for i in liPreSynI],
            [self.lShadowNeurons[i] for i in liPostSynI],
            [synInhibitory],
        )
        print(
            "DynapseControl: Inhibitory connections of type `{}`".format(
                str(synInhibitory).split('.')[1]
            )
            + " between hardware neurons have been set."
        )
        lTimes.append(time.time())

        if bApplyDiff:
            self.model.apply_diff_state()
            print("DynapseControl: Connections have been written to the chip.")

        lTimes.append(time.time())
        print(np.diff(lTimes))    

    def remove_all_connections_to(self, vnNeuronIDs, bApplyDiff: bool=True):
        """
        remove_all_connections_to - Remove all presynaptic connections
                                    to neurons defined in vnNeuronIDs
        :param vnNeuronIDs:     np.ndarray IDs of neurons whose presynaptic
                                           connections should be removed
        :param bApplyDiff:      bool If False do not apply the changes to
                                     chip but only to shadow states of the
                                     neurons. Useful if new connections are
                                     going to be added to the given neurons.
        """     
        # - Make sure neurons vnNeuronIDs is an array
        vnNeuronIDs = np.asarray(vnNeuronIDs)

        # - Call `remove_all_connections_to` function
        remove_all_connections_to(
            [self.lShadowNeurons[i] for i in vnNeuronIDs], self.model, bApplyDiff
        )


    ### --- Stimulation and event generation

    def TSEvent_to_spike_list(
        self,
        tsSeries: TSEvent,
        vnNeuronIDs: np.ndarray,
        nTargetCoreMask: int = 1,
        nTargetChipID: int = 0,
    ) -> List:
        """
        TSEvent_to_spike_list - Convert a TSEvent object to a ctxctl spike list

        :param tsSeries:        TSEvent      Time series of events to send as input
        :param vnNeuronIDs:     ArrayLike    IDs of neurons that should appear as sources of the events
        :param nTargetCoreMask: int          Mask defining target cores (sum of 2**core_id)
        :param nTargetChipID:   int          ID of target chip
        :return:                list of FpgaSpikeEvent objects
        """
        # - Check that the number of channels is the same between time series and list of neurons
        assert tsSeries.nNumChannels <= np.size(
            vnNeuronIDs
        ), "`tsSeries` contains more channels than the number of neurons in `vnNeuronIDs`."

        # - Make sure vnNeuronIDs is iterable
        vnNeuronIDs = np.array(vnNeuronIDs)

        # - Get events from this time series
        vtTimes, vnChannels, _ = tsSeries.find()

        # - Convert to ISIs
        tStartTime = tsSeries.tStart
        vtISIs = np.diff(np.r_[tStartTime, vtTimes])
        vnDiscreteISIs = (np.round(vtISIs / self.tFpgaIsiBase)).astype("int")

        # - Convert events to an FpgaSpikeEvent
        print("dynapse_control: Generating FPGA event list from TSEvent.")
        lEvents = generate_fpga_event_list(
            list(vnDiscreteISIs),
            list(vnNeuronIDs[vnChannels]),
            int(nTargetCoreMask),  # This makes sure that no np.int64 or other non-basic type is passed
            int(nTargetChipID),
        )
        # - Return a list of events
        return lEvents

    def arrays_to_spike_list(
        self,
        vnChannels: np.ndarray,
        vnNeuronIDs: np.ndarray,
        vnTimeSteps: Optional[np.ndarray]=None,
        vtTimeTrace: Optional[np.ndarray]=None,
        nTSStart: Optional[int] = None,
        tStart: Optional[int] = 0,
        nTargetCoreMask: int = 1,
        nTargetChipID: int = 0,
    ) -> List:
        """
        arrays_to_spike_list - Convert an array of input time steps and an an array
                               of event channels to a ctxctl spike list

        :param vnChannels:      np.ndarray   Event channels
        :param vnNeuronIDs:     ArrayLike    IDs of neurons that should appear as sources of the events
        :param vnTimeSteps:     np.ndarray   Event time steps (Using FPGA time base, overwrites vtTimeTrace if not None)
        :param vtTimeTrace:     np.ndarray   Event time points (in seconds)
        :param nTSStart:        int          Time step at which to start (overwrites tStart if not None)
        :param tStart:          float        Time at which to start
        :param nTargetCoreMask: int          Mask defining target cores (sum of 2**core_id)
        :param nTargetChipID:   int          ID of target chip
        :return:                list of FpgaSpikeEvent objects
        """
        # - Process input arguments
        if vnTimeSteps is None:
            assert vtTimeTrace is not None, (
                "DynapseControl: Either `vnTimeSteps` or `vtTimeTrace` must be provided."
            )
            vnTimeSteps = np.floor(vtTimeTrace / self.tFpgaIsiBase).astype(int)
        if nTSStart is None:
            assert tStart is not None, (
                "DynapseControl: Either `nTSStart` or `tStart` must be provided."
            )
            nTSStart = int(np.floor(tStart / self.tFpgaIsiBase))

        # - Ignore data that comes before nTSStart
        vnTimeSteps = vnTimeSteps[vnTimeSteps >= nTSStart]
        vnChannels = vnChannels[vnTimeSteps >= nTSStart]

        # - Check that the number of channels is the same between time series and list of neurons
        assert np.amax(vnChannels) <= np.size(
            vnNeuronIDs
        ), "DynapseControl: `vnChannels` contains more channels than the number of neurons in `vnNeuronIDs`."

        # - Make sure vnNeuronIDs is iterable
        vnNeuronIDs = np.array(vnNeuronIDs)

        # - Convert to ISIs
        vnDiscreteISIs = np.diff(np.r_[nTSStart, vnTimeSteps])

        print("DynapseControl: Generating FPGA event list from arrays.")
        # - Convert events to an FpgaSpikeEvent
        lEvents = generate_fpga_event_list(
            list(vnDiscreteISIs),
            list(vnNeuronIDs[vnChannels]),
            int(nTargetCoreMask),  # This makes sure that no np.int64 or other non-basic type is passed
            int(nTargetChipID),
        )

        # - Return a list of events
        return lEvents

    def start_cont_stim(
        self,
        fFrequency: float,
        vnNeuronIDs: int,
        nChipID: int=0,
        nCoreMask: int=15,
    ):
        """
        start_cont_stim - Start sending events with fixed frequency.
                          FPGA repeat mode will be set True.
        :param fFrequency:  float  Frequency at which events are sent
        :param vnNeuronIDs:   int  Event neuron ID(s)
        :param nChipID:     int  Target chip ID
        :param nCoreMask:   int  Target core mask
        """
        
        # - Set FPGA to repeating mode
        self.fpgaSpikeGen.set_repeat_mode(True)

        # - Interspike interval
        tISI = 1. / fFrequency
        # - ISI in units of fpga time step
        nISIfpga = int(np.round(tISI / self.tFpgaIsiBase))

        # # - Handle integers for vnNeuronIDs
        # if isinstance(vnNeuronIDs, int):
        #     vnNeuronIDs = [vnNeuronIDs]

        # - Generate events
        # List for events to be sent to fpga
        lEvents = generate_fpga_event_list(
            list(np.array([nISIfpga])),
            list(np.array(vnNeuronIDs).reshape(1, -1)),
            int(nCoreMask),
            int(nChipID),
        )
        self.fpgaSpikeGen.preload_stimulus(lEvents)
        print("DynapseControl: Stimulus prepared with {} Hz".format(
            1. / (nISIfpga * self.tFpgaIsiBase))
        )
        
        # - Start stimulation
        self.fpgaSpikeGen.start()
        print("DynapseControl: Stimulation started")

    def stop_stim(self, bClearFilter: bool=False):
        """
        stop_stim - Stop stimulation with FGPA spke generator. 
                    FPGA repeat mode will be set False.
        :param bStopRecording:  bool  Clear buffered event filter if present.
        """
        # - Stop stimulation
        self.fpgaSpikeGen.stop()
        # - Set default FPGA settings
        self.fpgaSpikeGen.set_repeat_mode(False)
        print("DynapseControl: Stimulation stopped")
        if bClearFilter:
            self.clear_buffered_event_filter()

    def start_poisson_stim(
        self,
        vfFrequencies: Union[int, np.ndarray],
        vnNeuronIDs: Union[int, np.ndarray],
        nChipID: int = 0,
    ):
        """
        start_poisson_stim - Start generated events by poisson processes and send them.
        :param vfFrequency: int or array-like  Frequencies of poisson processes
        :param vnNeuronIDs: int or array-like  Event neuron ID(s)
        :param nChipID:     int  Target chip ID
        """
        
        # - Handle single values for frequencies and neurons
        if np.size(vnNeuronIDs) == 1:
            vnNeuronIDs = np.repeat(vnNeuronIDs, np.size(vfFrequencies))
        if np.size(vfFrequencies) == 1:
            vfFrequencies = np.repeat(vfFrequencies, np.size(vnNeuronIDs))
        else:
            assert np.size(vfFrequencies) == np.size(vnNeuronIDs), \
                "DynapseControl: Length of `vfFrequencies must be same as length of `vnNeuronIDs` or 1."

        # - Set firing rates for selected neurons
        for fFreq, nNeuronID in zip(vfFrequencies, vnNeuronIDs):
            self.fpgaPoissonGen.write_poisson_rate_hz(nNeuronID, fFreq)

        # - Set chip ID
        self.fpgaPoissonGen.set_target_chip_id(nChipID)

        print("DynapseControl: Poisson stimuli prepared for chip {}.".format(nChipID))
        
        # - Start stimulation
        self.fpgaPoissonGen.start()
        print("DynapseControl: Poisson rate stimulation started")

    def stop_poisson_stim(self):
        """
        stop_stim - Stop stimulation with FGPA poisson generator.
                    Does not stop any event recording.
        """
        self.fpgaPoissonGen.stop()
        print("DynapseControl: Poisson rate stimulation stopped")

    def reset_poisson_rates(self):
        """reset_poisson_rates - Set all firing rates of poisson generator to 0."""
        for i in range(1024):
            self.fpgaPoissonGen.write_poisson_rate_hz(i, 0)
        print("DynapseControl: Firing rates for poisson generator have been set to 0.")

    def send_pulse(
        self,
        tWidth: float=0.1,
        fFreq: float=1000,
        tRecord: float=3,
        tBuffer: float=0.5,
        nInputNeuronID: int=0,
        vnRecordNeuronIDs: Union[int, np.ndarray]=0,
        nTargetCoreMask: int=15,
        nTargetChipID: int=0,
        bPeriodic: bool=False,
        bRecord: bool=False,
        bTSEvent: bool=False,
    ) -> TSEvent:
        """
        send_pulse - Send a pulse of periodic input events to the chip.
                     Return a TSEvent wih the recorded hardware activity.
        :param tWidth:              float  Duration of the input pulse
        :param fFreq:               float  Frequency of the events that constitute the pulse
        :param tRecord:             float  Duration of the recording (including stimulus)
        :param tBuffer:             float  Record slightly longer than tRecord to
                                           make sure to catch all relevant events
        :param nInputNeuronID:      int    ID of input neuron
        :param vnRecordNeuronIDs:   array-like  ID(s) of neuron(s) to be recorded
        :param nChipID:     int  Target chip ID
        :param nCoreMask:   int  Target core mask
        :param bPeriodic:   bool    Repeat the stimulus indefinitely
        :param bRecord:     bool    Set up buffered event filter that records events
                                    from neurons defined in vnRecordNeuronIDs
        :param bTSEvent:    bool    If True and bRecord==True: output TSEvent instead of arrays of times and channels

        :return:
            if bRecord==False:  None
            elif bTSEvent:      TSEvent object of recorded data
            else:               (vtTimeTrace, vnChannels)  np.ndarrays that contain recorded data
        """
        # - Prepare input events
        # Actual input time steps
        vnTimeSteps = np.floor(np.arange(0, tWidth, 1./fFreq) / self.tFpgaIsiBase).astype(int)
        # Add dummy events at end to avoid repeated stimulation due to "2-trigger-bug"
        tISILimit = self.nFpgaIsiLimit * self.tFpgaIsiBase
        nAdd = int(np.ceil(tRecord / tISILimit))
        vnTimeSteps = np.r_[vnTimeSteps, vnTimeSteps[-1] + np.arange(1, nAdd+1)*self.nFpgaIsiLimit]
        
        lEvents = self.arrays_to_spike_list(
            vnTimeSteps=vnTimeSteps,
            vnChannels=np.repeat(nInputNeuronID, vnTimeSteps.size),
            vnNeuronIDs=range(len(self.lVirtualNeurons)),
            nTSStart=0,
            nTargetCoreMask=nTargetCoreMask,
            nTargetChipID=nTargetChipID,        
        )
        # - Do not send dummy events to any core
        for event in lEvents[-nAdd:]:
            event.core_mask = 0
        print("DynapseControl: Stimulus pulse prepared")

        # - Stimulate and return recorded data if any
        return self.send_stimulus_list(
            lEvents=lEvents,
            tDuration=tRecord,
            tBuffer=tBuffer,
            vnRecordNeuronIDs=vnRecordNeuronIDs,
            bPeriodic=bPeriodic,
            bRecord=bRecord,
            bTSEvent=bTSEvent,
        )

    def send_TSEvent(
        self,
        tsSeries,
        tRecord: Optional[float]=None,
        tBuffer: float=0.5,
        vnNeuronIDs: Optional[np.ndarray]=None,
        vnRecordNeuronIDs: Optional[np.ndarray]=None,
        nTargetCoreMask: int=15,
        nTargetChipID: int=0,
        bPeriodic=False,
        bRecord=False,
        bTSEvent=False,
    ):
        """
        send_TSEvent - Extract events from a TSEvent object and send them to FPGA.

        :param tsSeries:        TSEvent      Time series of events to send as input
        :param tRecord:         float  Duration of the recording (including stimulus)
                                       If None, use tsSeries.tDuration
        :param tBuffer:         float  Record slightly longer than tRecord to
                                       make sure to catch all relevant events
        :param vnNeuronIDs:     ArrayLike    IDs of neurons that should appear as sources of the events
                                             If None, use channels from tsSeries
        :param vnRecordNeuronIDs: ArrayLike    IDs of neurons that should be recorded (if bRecord==True)
                                               If None and bRecord==True, record neurons in vnNeuronIDs
        :param nTargetCoreMask: int          Mask defining target cores (sum of 2**core_id)
        :param nTargetChipID:   int          ID of target chip
        :param bPeriodic:       bool         Repeat the stimulus indefinitely
        :param bRecord:         bool         Set up buffered event filter that records events
                                             from neurons defined in vnNeuronIDs
        :param bTSEvent:        bool         If bRecord: output TSEvent instead of arrays of times and channels

        :return:
            if bRecord==False:  None
            elif bTSEvent:      TSEvent object of recorded data
            else:               (vtTimeTrace, vnChannels)  np.ndarrays that contain recorded data
        """
        
        # - Process input arguments
        vnNeuronIDs = (
            np.arange(tsSeries.nNumChannels) if vnNeuronIDs is None
            else np.array(vnNeuronIDs)
        )
        vnRecordNeuronIDs = vnNeuronIDs if vnRecordNeuronIDs is None else vnRecordNeuronIDs
        tRecord = tsSeries.tDuration if tRecord is None else tRecord

        # - Prepare event list
        lEvents = self.TSEvent_to_spike_list(
            tsSeries,
            vnNeuronIDs=vnNeuronIDs,
            nTargetCoreMask=nTargetCoreMask,
            nTargetChipID=nTargetChipID,        
        )
        print("DynapseControl: Stimulus prepared from TSEvent `{}`.".format(
            tsSeries.strName
        ))

        # - Stimulate and return recorded data if any
        return self.send_stimulus_list(
            lEvents=lEvents,
            tDuration=tRecord,
            tBuffer=tBuffer,
            vnRecordNeuronIDs=vnRecordNeuronIDs,
            bPeriodic=bPeriodic,
            bRecord=bRecord,
            bTSEvent=bTSEvent,
        )

    def send_arrays(
        self,
        vnChannels: np.ndarray,
        vnTimeSteps: Optional[np.ndarray]=None,
        vtTimeTrace: Optional[np.ndarray]=None,
        tRecord: Optional[float]=None,
        tBuffer: float=0.5,
        vnNeuronIDs: Optional[np.ndarray]=None,
        vnRecordNeuronIDs: Optional[np.ndarray]=None,
        nTargetCoreMask: int=15,
        nTargetChipID: int=0,
        bPeriodic=False,
        bRecord=False,
        bTSEvent=False,
    ):
        """
        send_arrays - Send events defined in timetrace and channel arrays to FPGA.

        :param vnChannels:      np.ndarray  Event channels
        :param vnTimeSeops:     np.ndarray  Event times in Fpga time base (overwrites vtTimeTrace if not None)
        :param vtTimeTrace:     np.ndarray  Event times in seconds
        :param tRecord:         float  Duration of the recording (including stimulus)
                                       If None, use vtTimeTrace[-1]
        :param tBuffer:         float  Record slightly longer than tRecord to
                                       make sure to catch all relevant events
        :param vnNeuronIDs:     ArrayLike    IDs of neurons that should appear as sources of the events
                                             If None, use channels from vnChannels
        :param vnRecordNeuronIDs: ArrayLike    IDs of neurons that should be recorded (if bRecord==True)
                                               If None and bRecord==True, record neurons in vnNeuronIDs
        :param nTargetCoreMask: int          Mask defining target cores (sum of 2**core_id)
        :param nTargetChipID:   int          ID of target chip
        :param bPeriodic:       bool         Repeat the stimulus indefinitely
        :param bRecord:         bool         Set up buffered event filter that records events
                                             from neurons defined in vnNeuronIDs
        :param bTSEvent:        bool         If bRecord: output TSEvent instead of arrays of times and channels

        :return:
            if bRecord==False:  None
            elif bTSEvent:      TSEvent object of recorded data
            else:               (vtTimeTrace, vnChannels)  np.ndarrays that contain recorded data
        """
        
        # - Process input arguments
        vnNeuronIDs = (
            np.arange(np.amax(vnChannels)) if vnNeuronIDs is None
            else np.array(vnNeuronIDs)
        )
        vnRecordNeuronIDs = vnNeuronIDs if vnRecordNeuronIDs is None else vnRecordNeuronIDs
        if tRecord is None:
            try:
                tRecord = vtTimeTrace[-1]
            except TypeError:  # vtTimeTrace is None
                try:
                    tRecord = vnTimeSteps[-1] * self.tFpgaIsiBase
                except TypeError:  # vnTimeSteps is also None
                    raise ValueError(
                        "DynapseControl: Either `vnTimeSteps` or `vtTimeTrace` has to be provided."
                    )
            print("DynapseControl: Stimulus/recording time inferred to be {} s.".format(tRecord))

        # - Prepare event list
        lEvents = self.arrays_to_spike_list(
            vtTimeTrace=vtTimeTrace,
            vnTimeSteps=vnTimeSteps,
            vnChannels=vnChannels,
            vnNeuronIDs=vnNeuronIDs,
            nTSStart=0,
            nTargetCoreMask=nTargetCoreMask,
            nTargetChipID=nTargetChipID,        
        )
        print("DynapseControl: Stimulus prepared from arrays.")

        # - Stimulate and return recorded data if any
        return self.send_stimulus_list(
            lEvents=lEvents,
            tDuration=tRecord,
            tBuffer=tBuffer,
            vnRecordNeuronIDs=vnRecordNeuronIDs,
            bPeriodic=bPeriodic,
            bRecord=bRecord,
            bTSEvent=bTSEvent,
        )

    def send_stimulus_list(
        self,
        lEvents,
        tDuration,
        tBuffer,
        vnRecordNeuronIDs: Optional[np.ndarray]=None,
        bPeriodic: bool=False,
        bRecord: bool=False,
        bTSEvent: bool=False,
    ):
        """
        send_stimulus_list - Send a list of FPGA events to hardware. Possibly record hardware events.

        :param lEvents:           list   List of FpgaSpikeEvent objects to be sent to hardware
        :param tDuration:         float  Duration of the stimulation and recording
                                         If None, record indefinitely
        :param tBuffer:           float  Record slightly longer than tDuration to
                                         make sure to catch all relevant events
        :param vnRecordNeuronIDs: ArrayLike    IDs of neurons that should be recorded (if bRecord==True)
                                               If None and bRecord==True, no neurons will be recorded
        :param bPeriodic:       bool         Repeat the stimulus indefinitely
        :param bRecord:         bool         Set up buffered event filter that records events
                                             from neurons defined in vnRecordNeuronIDs
        :param bTSEvent:        bool         If True and bRecord==True: output TSEvent instead of arrays of times and channels

        :return:
            if bRecord==False:  None
            elif bTSEvent:      TSEvent object of recorded data
            else:               (vtTimeTrace, vnChannels)  np.ndarrays that contain recorded data
        """
        # - Prepare FPGA
        self.fpgaSpikeGen.set_repeat_mode(bPeriodic)
        self.fpgaSpikeGen.preload_stimulus(lEvents)
        print("DynapseControl: Stimulus preloaded.")
        if bRecord:
            if vnRecordNeuronIDs is None:
                vnRecordNeuronIDs = []
                warn("DynapseControl: No neuron IDs specified for recording.")
            self.add_buffered_event_filter(vnRecordNeuronIDs)

        # - Stimulate
        print(
            "DynapseControl: Starting{} stimulation{}.".format(
                bPeriodic * " periodic",
                (not bPeriodic) * " for {} s".format(tDuration),
            )
        )
        self.fpgaSpikeGen.start()

        if tDuration is None:
            return

        # - Run stimulation (and record)
        tBuffer = 0 if tBuffer is None else tBuffer

        if bPeriodic:
            return

        time.sleep(tDuration + tBuffer)

        # - Stop stimulation and clear filter to stop recording events
        self.fpgaSpikeGen.stop()
        print("DynapseControl: Stimulation ended.")

        if bRecord:
            self.bufferedfilter.clear()
            if bTSEvent:
                # - Extract TSEvent from recorded data
                return self.recorded_data_to_TSEvent(vnRecordNeuronIDs, tDuration)          
            else:
                # - Extract arrays from recorded data
                return self.recorded_data_to_arrays(vnRecordNeuronIDs, tDuration)

    def recorded_data_to_TSEvent(
        self,
        vnNeuronIDs: np.ndarray,
        tRecord: float
    ) -> TSEvent:
        lEvents = self.bufferedfilter.get_events()
        lTrigger = self.bufferedfilter.get_special_event_timestamps()
        print(
            "DynapseControl: Recorded {} event(s) and {} trigger event(s)".format(len(lEvents), len(lTrigger))
        )

        # - Extract monitored event channels and timestamps
        vnTimeStamps, vnChannels = event_data_to_channels(
            lEvents, vnNeuronIDs,
        )
        vtTimeTrace = np.array(vnTimeStamps) * 1e-6
        vnChannels = np.array(vnChannels)

        # - Locate synchronisation timestamp
        tStartTrigger = lTrigger[0] * 1e-6
        iStartIndex = np.searchsorted(vtTimeTrace, tStartTrigger)
        iEndIndex = np.searchsorted(vtTimeTrace, tStartTrigger+tRecord)
        vtTimeTrace = vtTimeTrace[iStartIndex: iEndIndex] - tStartTrigger
        vnChannels = vnChannels[iStartIndex: iEndIndex]
        print("DynapseControl: Extracted event data")

        return TSEvent(
            vtTimeTrace,
            vnChannels,
            tStart=0,
            tStop=tRecord,
            nNumChannels=np.size(vnNeuronIDs)
        )

    def recorded_data_to_arrays(
        self,
        vnNeuronIDs: np.ndarray,
        tRecord: float
    ) -> TSEvent:
        lEvents = self.bufferedfilter.get_events()
        lTrigger = self.bufferedfilter.get_special_event_timestamps()
        
        print(
            "DynapseControl: Recorded {} event(s) and {} trigger event(s)".format(len(lEvents), len(lTrigger))
        )

        # - Extract monitored event channels and timestamps
        vnTimeStamps, vnChannels = event_data_to_channels(
            lEvents, vnNeuronIDs,
        )
        vtTimeTrace = np.array(vnTimeStamps) * 1e-6
        vnChannels = np.array(vnChannels)

        # - Locate synchronisation timestamp
        tStartTrigger = lTrigger[0] * 1e-6
        iStartIndex = np.searchsorted(vtTimeTrace, tStartTrigger)
        iEndIndex = np.searchsorted(vtTimeTrace, tStartTrigger+tRecord)
        vtTimeTrace = vtTimeTrace[iStartIndex: iEndIndex] - tStartTrigger
        vnChannels = vnChannels[iStartIndex: iEndIndex]
        print("DynapseControl: Extracted event data")
        return vtTimeTrace, vnChannels

    ### --- Tools for tuning and observing activities

    def add_buffered_event_filter(self, vnNeuronIDs):
        """
        add_buffered_event_filter - Add a BufferedEventFilter to record from
                                    neurons defined in vnNeuronIDs
        :param vnNeuronIDs:   array-like  IDs of neurons to be recorded
        :return:
            Reference to selfbufferedfilter, the BufferedEventFilter that has been created.
        """
        # - Convert vnNeuronIDs to list
        if isinstance(vnNeuronIDs, int):
            lnRecordNeuronIDs = list(range(vnNeuronIDs))
        else:
            lnRecordNeuronIDs = list(vnNeuronIDs)
        
        # - Does a filter already exist?
        if hasattr(self, "bufferedfilter") and self.bufferedfilter is not None:
            self.bufferedfilter.clear()
            self.bufferedfilter.add_ids(lnRecordNeuronIDs)
            print("DynapseControl: Updated existing buffered event filter.")
        else:
            self.bufferedfilter = generate_buffered_filter(self.model, lnRecordNeuronIDs)
            print("DynapseControl: Generated new buffered event filter.")
        
        return self.bufferedfilter

    def clear_buffered_event_filter(self):
        """ clear_buffered_event_filter - Clear self.bufferedfilter if it exists."""
        if hasattr(self, "_bufferedfilter") and self.bufferedfilter is not None:
            self.bufferedfilter.clear()
            print("DynapseControl: Buffered event filter cleared")
        else:
            warn("DynapseControl: No buffered event filter found.")

    def buffered_events_to_TSEvent(self, vnNeuronIDs=range(4096), tDuration=None):
        """
        buffered_events_to_TSEvent - Fetch events from self.bufferedfilter and 
                                     convert them to a TSEvent
        :param vnNeuronIDs:     
        """
        # - Fetch events from filter
        lEvents = list(TR_oFilter.get_events())
        lTrigger = list(TR_oFilter.get_special_event_timestamps())
        print(
            "DynapseControl: Recorded {} event(s) and {} trigger event(s)".format(
                len(lEvents), len(lTrigger)
            )
        )

        # - Extract monitored event channels and timestamps
        vnNeuronIDs = np.array(vnNeuronIDs)
        vnChannels = DHW.neurons_to_channels(
            [e.neuron for e in lEvents],
            [self.lHWNeurons[i] for i in vnNeuronIDs],
        )
        # - Remove events that are not from neurons defined in vnNeuronIDs
        vnChannels = vnChannels[np.isnan(vnChannels) == False]
        # - TIme trace
        vtTimeTrace = (
            np.array([e.timestamp for e in lEvents]) * 1e-6
        )[np.isnan(vnChannels) == False]

        # - Locate synchronisation timestamp
        tStartTrigger = lTrigger[0] * 1e-6
        iStartIndex = np.searchsorted(vtTimeTrace, tStartTrigger)
        iEndIndex = (
            None if tDuration is None
            else np.searchsorted(vtTimeTrace, tStartTrigger+tDuration)
        )
        vnChannels = vnChannels[iStartIndex:iEndIndex]
        vtTimeTrace = vtTimeTrace[iStartIndex:iEndIndex] - tStartTrigger
        print("DynapseControl: Extracted event data")

        return TSEvent(
            vtTimeTrace,
            vnChannels,
            tStart=0,
            tStop=tDuration,
            nNumChannels=np.size(vnNeuronIDs),
            strName="pulse_response"
        )

    def collect_spiking_neurons(self, vnNeuronIDs, tDuration):
        """
        collect_spiking_neurons - Return a list of IDs of neurons that
                                  that spike within tDuration
        :param vnNeuronIDs:   list   IDs of neurons to be observed.
        :param tDuration:     float  How long to wait for spiking neurons 
        :return  lnRecordedNeuronIDs:  list IDs of neurons that have spiked.
        """

        # - Convert vnNeuronIDs to list
        if isinstance(vnNeuronIDs, int):
            vnNeuronIDs = range(vnNeuronIDs)
        
        print("DynapseControl: Collecting IDs of neurons that spike within the next {} seconds".format(tDuration))
        
        # - Filter for recording neurons
        oFilter = self.add_buffered_event_filter(vnNeuronIDs)
        
        # - Wait and record spiking neurons
        time.sleep(tDuration)
        
        oFilter.clear()
        
        # - Sorted unique list of neurons' IDs that have spiked
        lnRecordedNeuronIDs = sorted(set((event.neuron.get_id() for event in oFilter.get_events())))
        print("DynapseControl: {} neurons spiked: {}".format(len(lnRecordedNeuronIDs), lnRecordedNeuronIDs))

        return lnRecordedNeuronIDs

    def silence_hot_neurons(self, vnNeuronIDs, tDuration):
        """
        silence_hot_neurons - Collect IDs of all neurons that spike 
                              within tDuration. Assign them different
                              time constant to silence them.
        :param vnNeuronIDs:  list   IDs of neurons to be observed.
        :param tDuration:    float  How long to wait for spiking neurons 
        """
        # - Neurons that spike within tDuration
        lnHotNeurons = self.collect_spiking_neurons(vnNeuronIDs, tDuration=tDuration)
        # - Silence these neurons by assigning different Tau bias
        for nID in lnHotNeurons:
            self.dynapse.set_tau_2(0, nID)
        print("DynapseControl: Neurons {} have been silenced".format(lnHotNeurons))

    def measure_population_firing_rates(
        self,
        llnPopulationIDs: list,
        tDuration: float,
        bVerbose=False
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        measure_population_firing_rates - Measure the mean, maximum and minimum
                                          firing rates for multiple neuron populatios
        :param llnPopulationIDs:  Array-like of array-like of neuron IDs of each pouplaiton
        :param tDuration:         float  Time over which rates are measured
        :param bVerbose:          bool   Print firing rates of each neuron for each population

        :return:
            vfMeanRates     np.ndarray - Population mean rates
            vfMaxRates     np.ndarray - Population maximum rates
            vfMinRates     np.ndarray - Population minimum rates
        """

        # - Arrays for collecting rates
        nNumPopulations = np.size(llnPopulationIDs)
        vfMeanRates = np.zeros(nNumPopulations)
        vfMaxRates = np.zeros(nNumPopulations)
        vfMinRates = np.zeros(nNumPopulations)

        for i, lnNeuronIDs in enumerate(llnPopulationIDs):
            print("DynapseControl: Population {}".format(i))
            vfFiringRates, vfMeanRates[i], vfMaxRates[i], vfMinRates[i] = self.measure_firing_rates(lnNeuronIDs, tDuration)
            if bVerbose:
                print(vfFiringRates)

        return vfMeanRates, vfMaxRates, vfMinRates

    def measure_firing_rates(
        self,
        vnNeuronIDs: Optional[Union[int, np.ndarray]],
        tDuration: float
    ) -> (np.ndarray, float, float, float):
        """
        measure_firing_rates - Measure the mean, maximum and minimum firing rate
                               for a group of neurons
        :param vnNeuronIDs:     Array-like or int  Neuron IDs to be measured
        :param tDuration:       float  Time over which rates are measured

        :return:
            vfFiringRates  np.ndarray - Each neuron's firing rate
            fMeanRate      float - Average firing rate over all neurons
            fMaxRate       float - Highest firing rate of all neurons
            fMinRate       float - Lowest firing rate of all neurons
        """
        if isinstance(vnNeuronIDs, int):
            vnNeuronIDs = [vnNeuronIDs]
        # - Filter for recording events
        oFilter = self.add_buffered_event_filter(vnNeuronIDs)
        # - Record events for tDuration
        time.sleep(tDuration)
        # - Stop recording
        oFilter.clear()
        # - Retrieve recorded events
        lEvents = list(oFilter.get_events())
        if not lEvents:
            # - Handle empty event lists
            print("DynapseControl: No events recorded")
            return np.zeros(np.size(vnNeuronIDs)), 0, 0, 0
        # - Evaluate non-empty event lists
        return evaluate_firing_rates(lEvents, tDuration, vnNeuronIDs)

    def sweep_freq_measure_rate(
        self,
        vfFreq: list=[1, 10, 20, 50, 100, 200, 500, 1000, 2000],
        tDuration: float=1,
        vnTargetNeuronIDs: Union[int,np.ndarray]=range(128),
        vnInputNeuronIDs: Union[int,np.ndarray]=1,
        nChipID: int=0,
        nCoreMask: int=15,
    ):
        """
        sweep_freq_measure_rate - Stimulate a group of neurons by sweeping
                                  over a list of input frequencies. Measure
                                  their firing rates.
        :param vfFreq:      array-like Stimulus frequencies
        :param tDuration:   float  Stimulus duration for each frequency
        :param vnTargetNeuronIDs: array-like  source neuron IDs
        :param vnInputNeuronIDs:  array-like  IDs of neurons 
        :param nChipID:     int  Target chip ID
        :param nCoreMask:   int  Target core mask

        :return:
            mfFiringRates   np.ndarray -  Matrix containing firing rates for each neuron (axis 1)
                                          for each frequency (axis 0)
            vfMeanRates     np.ndarray - Average firing rates over all neurons for each input frequency
            vfMaxRates      np.ndarray - Highest firing rates of all neurons for each input frequency
            vfMinRates      np.ndarray - Lowest firing rates of all neurons for each input frequency
        """

        # - Arrays for collecting firing rates
        mfFiringRates = np.zeros((np.size(vfFreq), np.size(vnTargetNeuronIDs)))
        vfMeanRates = np.zeros(np.size(vfFreq))
        vfMaxRates = np.zeros(np.size(vfFreq))
        vfMinRates = np.zeros(np.size(vfFreq))

        # - Sweep over frequencies
        for iTrial, fFreq in enumerate(vfFreq):
            print("DynapseControl: Stimulating with {} Hz input".format(fFreq))
            self.start_cont_stim(fFreq, vnInputNeuronIDs, bVirtual)
            mfFiringRates[iTrial, :], vfMeanRates[iTrial], vfMaxRates[iTrial], vfMinRates[iTrial] = (
                self.measure_firing_rates(vnTargetNeuronIDs, tDuration)
            )
            self.stop_stim()

        return mfFiringRates, vfMeanRates, vfMaxRates, vfMinRates

    
    ### - Load and save biases

    def load_biases(self, strFilename):
        """load_biases - Load biases from python file under path strFilename"""
        with open(strFilename) as file:
            lstrBiasCommands = file.readlines()[2:]
            save_file_model_ = self.model
            for strCommand in lstrBiasCommands:
                exec(strCommand)
        print("DynapseControl: Biases have been loaded from {}.".format(strFilename))
    
    @staticmethod
    def save_biases(strFilename):
        """save_biases - Save biases in python file under path strFilename"""
        bias_groups = CtxDynapse.model.get_bias_groups()
        with open(strFilename, "w") as save_file:
            save_file.write("import CtxDynapse\n")
            save_file.write("save_file_model_ = CtxDynapse.model\n")
            for i, bias_group in enumerate(bias_groups):
                biases = bias_group.get_biases()
                for bias in biases:
                    save_file.write(
                        "save_file_model_.get_bias_groups()[{0}].set_bias(\"{1}\", {2}, {3})\n".format(
                            i, bias.bias_name, bias.fine_value, bias.coarse_value
                        )
                    )
        print("DynapseControl: Biases have been saved under {}.".format(strFilename))

    def copy_biases(self, nSourceCoreID: int=0, vnTargetCoreIDs: Optional[List[int]]=None):
        """
        copy_biases - Copy biases from one core to one or more other cores.
        :param nSourceCoreID:   int  ID of core from which biases are copied
        :param vnTargetCoreIDs: int or array-like ID(s) of core(s) to which biases are copied
                                If None, will copy to all other neurons
        """

        if vnTargetCoreIDs is None:
            # - Copy biases to all other cores except the source core
            vnTargetCoreIDs = list(range(16))
            vnTargetCoreIDs.remove(nSourceCoreID)
        elif np.size(vnTargetCoreIDs) == 1:
            vnTargetCoreIDs = list(np.array(vnTargetCoreIDs))
        else:
            vnTargetCoreIDs = list(vnTargetCoreIDs)

        # - List of bias groups from all cores
        lBiasgroups = self.model.get_bias_groups()
        sourcebiases = lBiasgroups[nSourceCoreID].get_biases()

        # - Set biases for target cores
        for nTargetCoreID in vnTargetCoreIDs:
            for bias in sourcebiases:
                lBiasgroups[nTargetCoreID].set_bias(bias.bias_name, bias.fine_value, bias.coarse_value)
        print("DynapseControl: Biases copied from core {} to core(s) {}".format(nSourceCoreID, vnTargetCoreIDs))


    ### --- Class properties

    @property
    def synSE(self):
        return SynapseTypes.SLOW_EXC
    @property
    def synSI(self):
        return SynapseTypes.SLOW_INH
    @property
    def synFE(self):
        return SynapseTypes.FAST_EXC
    @property
    def synFI(self):
        return SynapseTypes.FAST_INH

    @property
    def nFpgaEventLimit(self):
        return self._nFpgaEventLimit

    @property
    def nFpgaIsiLimit(self):
        return self._nFpgaIsiLimit

    @property
    def tFpgaTimestep(self):
        return self._tFpgaTimestep

    @property
    def tFpgaIsiBase(self):
        return self._nFpgaIsiMultiplier * self.tFpgaTimestep

    @tFpgaIsiBase.setter
    def tFpgaIsiBase(self, tNewBase):
        if not tNewBase > self.tFpgaTimestep:
            raise ValueError(
                "DynapseControl: `tFpgaTimestep` must be at least {}".format(self.tFpgaTimestep)
            )
        else:
            self._nFpgaIsiMultiplier = int(np.floor(tNewBase / self.tFpgaTimestep))
            self.fpgaSpikeGen.set_isi_multiplier(self._nFpgaIsiMultiplier)
