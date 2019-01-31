# ----
# dynapse_control.py - Class to interface the DynapSE chip
# ----

### --- Imports

import numpy as np
from warnings import warn
from typing import Tuple, List, Optional, Union
import time
import os.path
import threading
from .timeseries import TSEvent

# - Programmatic imports for CtxCtl
bUsing_RPyC = False
bUsing_deepcopy = False
RPYC_TIMEOUT = 300

try:
    import CtxDynapse
    import NeuronNeuronConnector

except ModuleNotFoundError:
    # - Try with RPyC
    try:
        import rpyc
        conn = rpyc.classic.connect("localhost", "1300")
        print("dynapse_control: RPyC connection established through port 1300.")
    except ModuleNotFoundError:
        # - Raise an ImportError (so that the smart __init__.py module can skip over missing CtxCtl
        raise ImportError("dynapse_control: rpyc not found")
    except:
        try:
            conn = rpyc.classic.connect("localhost", "1301")
            print("dynapse_control: RPyC connection established through port 1301.")
        except:
            raise ImportError("dynapse_control: Connection not possible")
    # - Set up rpyc conneciton settings
    conn._config[
        "sync_request_timeout"
    ] = RPYC_TIMEOUT  # Set timeout to higher level
    CtxDynapse = conn.modules.CtxDynapse
    NeuronNeuronConnector = conn.modules.NeuronNeuronConnector
    bUsing_RPyC = True

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
lnChipIDs = [0]  # Chips to be used
# - Fix (hardware)
FPGA_EVENT_LIMIT = int(2 ** 16 - 1)  # Max. number of events that can be sent to FPGA
FPGA_ISI_LIMIT = int(
    2 ** 16 - 1
)  # Max. number of timesteps for single inter-spike interval between FPGA events
FPGA_TIMESTEP = 1. / 9. * 1e-7  # Internal clock of FPGA, 11.111...ns
CORE_DIMENSIONS = (16, 16)  # Numbers of neurons in core (rows, columns)
NUM_CORE_NEURONS = (
    CORE_DIMENSIONS[0] * CORE_DIMENSIONS[1]
)  # Number of neurons on one core
NUM_CHIP_CORES = 4  # Number of cores per chip
NUM_CHIPS = 4  # Number of available chips
# - Default values, can be changed
DEF_FPGA_ISI_BASE = 2e-5  # Default timestep between events sent to FPGA
DEF_FPGA_ISI_MULTIPLIER = int(np.round(DEF_FPGA_ISI_BASE / FPGA_TIMESTEP))

if bUsing_RPyC:
    # - Setup parameters on RPyC server
    conn.namespace["bUsing_RPyC"] = True
    conn.namespace["NUM_CORE_NEURONS"] = NUM_CORE_NEURONS
    conn.namespace["NUM_CHIP_CORES"] = NUM_CHIP_CORES
    conn.namespace["NUM_CHIPS"] = NUM_CHIPS
    conn.namespace["copy"] = conn.modules.copy
    conn.namespace["os"] = conn.modules.os
    conn.namespace["CtxDynapse"] = conn.modules.CtxDynapse
    conn.namespace["rpyc"] = conn.modules.rpyc

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
            vnPreE.append(nPre.item())  # avoid using np.int64 type for integers
            vnPostE.append(nPost.item())

    # - Loop over lists, appending when necessary
    for nPre, nPost in zip(vnPreICompressed, vnPostICompressed):
        for _ in range(np.abs(mnW[nPre, nPost])):
            vnPreI.append(nPre.item())
            vnPostI.append(nPost.item())

    # - Return augmented lists
    return vnPreE, vnPostE, vnPreI, vnPostI


def rectangular_neuron_arrangement(
    nFirstNeuron: int,
    nNumNeurons: int,
    nWidth: int,
    tupCoreDimensions: tuple = CORE_DIMENSIONS,
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
    lEvents: list, tDuration: float, vnNeuronIDs: list
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
    tupTimestamps, tupEventNeuronIDs = zip(
        *((event.timestamp, event.neuron.get_id()) for event in lEvents)
    )
    # - Event times in microseconds
    vnEventTimes = np.array(tupTimestamps, int)
    # - Set time of first event to 0
    vnEventTimes -= vnEventTimes[0]
    # - Convert neuron IDs of event to index starting from 0
    viEventIndices = np.array(
        [vnNeuronIDs.index(nNeuronID) for nNeuronID in tupEventNeuronIDs]
    )
    # - Convert events to binary raster
    nTimeSteps = int(np.ceil(tDuration * 1e6))
    mbEventRaster = np.zeros((nTimeSteps, np.size(viEventIndices)), bool)
    mbEventRaster[vnEventTimes, viEventIndices] = True
    return mbEventRaster


def evaluate_firing_rates(
    lEvents: list,
    tDuration: float,
    vnNeuronIDs: Optional[list] = None,
    bVerbose: bool=True,
) -> (np.ndarray, float, float, float):
    """
    evaluate_firing_rates - Determine the neuron-wise firing rates from a
                            list of events. Calculate mean, max and min.
    :param lEvents:         list of Spike objects
    :param tDuration:       float Time over which rates are normalized
    :param vnNeuronIDs:     array-like of neuron IDs corresponding to events.
                            If None, vnNeuronIDs will consists of to the neurons
                            corresponding to the events in lEvents.
    :param bVerbose:        bool  Print out information about firing rates.

    :return:
        vfFiringRates  np.ndarray - Each neuron's firing rate
        fMeanRate      float - Average firing rate over all neurons
        fMaxRate       float - Highest firing rate of all neurons
        fMinRate       float - Lowest firing rate of all neurons            
    """
    # - Extract event timestamps and neuron IDs
    tupTimestamps, tupEventNeuronIDs = extract_event_data(lEvents)
    # - Count events for each neuron
    viUniqueEventIDs, vnEventCounts = np.unique(tupEventNeuronIDs, return_counts=True)
    
    if vnNeuronIDs is None:
        # - lNeuronIDs as list of neurons that have spiked
        vnNeuronIDs = viUniqueEventIDs
    
    # - Neurons that have not spiked
    viNoEvents = (np.asarray(vnNeuronIDs))[
        np.isin(vnNeuronIDs, viUniqueEventIDs) == False
    ]
    
    # - Count events
    viUniqueEventIDs = np.r_[viUniqueEventIDs, viNoEvents]
    vnEventCounts = np.r_[vnEventCounts, np.zeros(viNoEvents.size)]
    
    # - Sort event counts to same order as in vnNeuronIDs
    liUniqueEventIDs = list(viUniqueEventIDs)
    lSort = [liUniqueEventIDs.index(nNeuronID) for nNeuronID in vnNeuronIDs]
    vnEventCounts = vnEventCounts[lSort]
    vfFiringRates = vnEventCounts / tDuration
    
    # - Calculate mean, max and min rates
    fMeanRate = np.mean(vfFiringRates)
    iMaxRateNeuron = np.argmax(vnEventCounts)
    fMaxRate = vfFiringRates[iMaxRateNeuron]
    iMinRateNeuron = np.argmin(vnEventCounts)
    fMinRate = vfFiringRates[iMinRateNeuron]
    
    if bVerbose:
        # - Print results
        print("\tMean firing rate: {} Hz".format(fMeanRate))
        print(
            "\tMaximum firing rate: {} Hz (neuron {})".format(
                fMaxRate, vnNeuronIDs[iMaxRateNeuron]
            )
        )
        print(
            "\tMinimum firing rate: {} Hz (neuron {})".format(
                fMinRate, vnNeuronIDs[iMinRateNeuron]
            )
        )
    
    return vfFiringRates, fMeanRate, fMaxRate, fMinRate


def event_data_to_channels(
    lEvents: List, lLayerNeuronIDs: List
) -> (np.ndarray, np.ndarray):
    """
    event_data_to_channels - Convert timestamps and neuron IDs from list of Events
                             to arrays with timestamps and channel indices wrt lLayerNeuronIDs
                             Fill in nan where event does not correspond to any given ID.
    :param lEvents:         list  SpikeEvent objects from BufferedEventFilter
    :param lLayerNeuronIDs: list  Neuron IDs corresponding to channels
    :return:
        vTimeStamps         np.ndarray  Extracted timestam
        vChannelIndices     np.ndarray  Extracted channel indices
    """
    tupTimeStamps, tupNeuronIDs = extract_event_data(lEvents)
    # - Convert to numpy array and thereby fetch data from connection if using RPyC
    vTimeStamps = np.array(tupTimeStamps)
    vNeuronIDs = np.array(tupNeuronIDs)
    # - Convert neuron IDs to channels
    dChannels = {nID: iChannel for iChannel, nID in enumerate(lLayerNeuronIDs)}
    vChannelIndices = np.array(
        list(map(lambda nID: dChannels.get(nID, float("nan")), vNeuronIDs))
    )
    if np.isnan(vChannelIndices).any():
        warn("dynapse_control: Some events did not match `lLayerNeuronIDs`")

    return vTimeStamps, vChannelIndices


def correct_type(obj):
    """
    correct_type - Check if an object is of a type or contains objects of a
                   type that is not supported by cortexcontrol or contains
                   and convert them where possible.
    :param obj:    object to be tested
    :return:
        object with corrected type
    """
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.complexfloating):
        return complex(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, (np.ndarray, list)):
        return [correct_type(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(correct_type(item) for item in obj)
    elif isinstance(obj, set):
        return set(correct_type(item) for item in obj)
    elif isinstance(obj, frozenset):
        return frozenset(correct_type(item) for item in obj)
    elif isinstance(obj, dict):
        return {correct_type(key): correct_type(val) for key, val in obj.items()}
    # Extend for types like bytes, unicode, char,... if necessary
    elif type(obj).__module__ not in ["builtins", "CtxDynapse", "rpyc"]:
        warn("Unrecognized type: {}".format(type(obj)))
    return obj


def correct_argument_types_and_teleport(func):
    """
    correct_argument_types_and_teleport -  Wrapper for functions that tries to
            correct argument types that are not supported by cortexcontrol and
            teleports the function via rpyc. Returns original function if 
            bUsing_RPyC == False
    :param func:  function to be teleported
    :return:      teleported function
    """

    if bUsing_RPyC:
        func = rpyc.classic.teleport_function(conn, func)

        def clean_func(*args, **kwargs):
            """Return a function with corrected argument types"""
            newargs = list(args)
            for i, argument in enumerate(args):
                newargs[i] = correct_type(argument)
            for key, val in kwargs.items():
                kwargs[key] = correct_type(val)
            return func(*newargs, **kwargs)

        return clean_func

    else:
        return func


def correct_argument_types(func):
    """
    correct_argument_types - If bUsing_RPyC is not False, try changing the 
                             arguments to a function to types that are 
                             supported by cortexcontrol
    :param func:    funciton where arguments should be corrected
    :return:        functions with possibly corrected arguments
    """
    if bUsing_RPyC:

        def clean_func(*args, **kwargs):
            """Return a function with corrected argument types"""
            newargs = list(args)
            for i, argument in enumerate(args):
                newargs[i] = correct_type(argument)
            for key, val in kwargs.items():
                kwargs[key] = correct_type(val)
            return func(*newargs, **kwargs)

        return clean_func

    else:
        return func


def teleport_function(func):
    """
    telport_function - Decorator. If using RPyC, then teleport the resulting function

    :param func:            Function to maybe teleport
    :return:                Maybe teleported function
    """
    # - Teleport if bUsing_RPyC flag is set
    if bUsing_RPyC:
        func = rpyc.classic.teleport_function(conn, func)
        return func

    else:
        # - Otherwise just return the undecorated function
        return func


def remote_function(func):
    """
    remote_function - If using RPyC then teleport the resulting function and
                      and register it in the remote namespace.

    :param func:            Function to maybe teleport
    :return:                Maybe teleported function    
    """
    # - Teleport if bUsing_RPyC flag is set
    if bUsing_RPyC:
        func = rpyc.classic.teleport_function(conn, func)
        conn.namespace[func.__name__] = func
        return func

    else:
        # - Otherwise just return the undecorated function
        return func


@remote_function
def local_arguments(func):
    def local_func(*args, **kwargs):
        for i, argument in enumerate(args):
            newargs = list(args)
            if isinstance(argument, rpyc.core.netref.BaseNetref):
                newargs[i] = copy.copy(argument)
        for key, val in kwargs.items():
            if isinstance(key, rpyc.core.netref.BaseNetref):
                del kwargs[key]
                kwargs[copy.copy(key)] = copy.copy(val)
            elif isinstance(val, rpyc.core.netref.BaseNetref):
                kwargs[key] = copy.copy(val)

        return func(*newargs, **kwargs)

    return local_func

# - Example on how to use local_arguments_rpyc decorator
@teleport_function
def _define_print_type():
    @local_arguments
    def print_type(obj):
        print(type(obj))

    return print_type

print_type = correct_argument_types(_define_print_type())  # or just print_type = _define_print_type()


@teleport_function
def extract_event_data(lEvents) -> (tuple, tuple):
    """
    extract_event_data - Extract timestamps and neuron IDs from list of recorded events.
    :param lEvents:     list  SpikeEvent objects from BufferedEventFilter
    :return:
        lTimeStamps     list  Timestamps of events
        lNeuronIDs      list  Neuron IDs of events
    """

    ltupEvents = [(event.timestamp, event.neuron.get_id()) for event in lEvents]
    try:
        tupTimeStamps, tupNeuronIDs = zip(*ltupEvents)
    except ValueError as e:
        # - Handle emptly event lists
        if len(ltupEvents) == 0:
            tupTimeStamps = ()
            tupNeuronIDs = ()
        else:
            raise e
    return tupTimeStamps, tupNeuronIDs

@remote_function
def _replace_too_large_value(nVal, nLimit: int=FPGA_ISI_LIMIT):
    """
    replace_too_large_entry - Return a list of integers <= nLimit, that sum up to nVal
    :param nVal:    int  Value to be replaced
    :param nLimit:  int  Maximum allowed value
    :return:
        lnReplace   list  Values to replace nVal
    """
    if nVal > nLimit:
        nReps = (nVal-1) // nLimit
        # - Return nReps times nLimit, then the remainder
        #   For modulus shift nVal to avoid replacing with 0 if nVal==nLimit
        return [*(nLimit for _ in range(nReps)), (nVal-1) % nLimit + 1]
    else:
        # - If clause in particular for case where nVal <= 0
        return [nVal]

@remote_function
def _auto_insert_dummies(
    lnDiscreteISIs: list,
    lnNeuronIDs: list,
    nFpgaIsiLimit: int = FPGA_ISI_LIMIT
) -> (list, list):
    """
    auto_insert_dummies - Insert dummy events where ISI limit is exceeded
    :param lnDiscreteISIs:  list  Inter-spike intervals of events
    :param lnNeuronIDs:     list  IDs of neurons corresponding to the ISIs

    :return:
        lnCorrectedISIs     list  ISIs that do not exceed limit, including dummies
        lnCorrectedIDs      list  Neuron IDs corresponding to corrected ISIs. Dummy events have None.
        lbDummy             list  Boolean indicating which events are dummy events
    """
    # - List of lists with corrected entries
    llCorrected = [_replace_too_large_value(nISI, nFpgaIsiLimit) for nISI in lnDiscreteISIs]
    # - Number of new entries for each old entry
    lnNumNew = [len(l) for l in llCorrected]
    # - List of lists with neuron IDs corresponding to ISIs. Dummy events have ID None
    llIDs = [
        [nID, *(None for _ in range(nLen - 1))]
        for nID, nLen in zip(lnNeuronIDs, lnNumNew)
    ]
    # - Flatten out lists
    lnCorrectedISIs = [nISI for l in llCorrected for nISI in l]
    lnCorrectedIDs = [nID for l in llIDs for nID in l]
    # - Count number of added dummy events (each one has None as ID)
    nNumDummies = len(tuple(filter(lambda x: x is None, lnCorrectedIDs)))
    if nNumDummies > 0:
        print("dynapse_control: Inserted {} dummy events.".format(nNumDummies))

    return lnCorrectedISIs, lnCorrectedIDs


@teleport_function
def generate_fpga_event_list(
    lnDiscreteISIs: list,
    lnNeuronIDs: list,
    nTargetCoreMask: int,
    nTargetChipID: int,
    nFpgaIsiLimit: int = FPGA_ISI_LIMIT,
    bCorrectISI: bool = True,
) -> list:
    """
    generate_fpga_event_list - Generate a list of FpgaSpikeEvent objects
    :param lnDiscreteISIs:  array-like  Inter-spike intervalls in Fpga time base
    :param lnNeuronIDs:     array-like  IDs of neurons corresponding to events
    :param nNeuronID:       int ID of source neuron
    :param nTargetCoreMask: int Coremask to determine target cores
    :nFpgaIsiLimit:         int Maximum ISI size (in time steps)
    :bCorrectISI:           bool Correct too large ISIs in lnDiscreteISIs

    :return:
        event  list of generated FpgaSpikeEvent objects.
    """

    # - Make sure objects live on required side of RPyC connection
    nTargetCoreMask = int(nTargetCoreMask)
    nTargetChipID = int(nTargetChipID)
    lnNeuronIDs = copy.copy(lnNeuronIDs)
    lnDiscreteISIs = copy.copy(lnDiscreteISIs)

    if bCorrectISI:
        lnDiscreteISIs, lnNeuronIDs = _auto_insert_dummies(
            lnDiscreteISIs, lnNeuronIDs, nFpgaIsiLimit
        )

    def generate_fpga_event(nNeuronID: int, nISI: int) -> CtxDynapse.FpgaSpikeEvent:
        """
        generate_fpga_event - Generate a single FpgaSpikeEvent objects.
        :param nNeuronID:       int ID of source neuron
        :param nISI:            int Timesteps after previous event before
                                    this event will be sent
        :return:
            event  CtxDynapse.FpgaSpikeEvent
        """
        event = CtxDynapse.FpgaSpikeEvent()
        event.target_chip = nTargetChipID
        event.core_mask = 0 if nNeuronID is None else nTargetCoreMask
        event.neuron_id = 0 if nNeuronID is None else nNeuronID
        event.isi = nISI
        return event

    # - Generate events
    print("dynapse_control: Generating event list")
    lEvents = [
        generate_fpga_event(nNeuronID, nISI)
        for nNeuronID, nISI in zip(lnNeuronIDs, lnDiscreteISIs)
    ]

    return lEvents


@teleport_function
def _generate_buffered_filter(oModel: CtxDynapse.Model, lnRecordNeuronIDs: list):
    """
    _generate_buffered_filter - Generate and return a BufferedEventFilter object that
                               records from neurons specified in lnRecordNeuronIDs.
    :param oModel:               CtxDynapse model
    :param lnRecordNeuronIDs:    list  IDs of neurons to be recorded.
    """
    return CtxDynapse.BufferedEventFilter(oModel, lnRecordNeuronIDs)


@teleport_function
def load_biases(strFilename, lnCoreIDs: Optional[Union[list, int]] = None):
    """
    load_biases - Load biases from python file under path strFilename
    :param strFilename:  str  Path to file where biases are stored.
    :param lnCoreIDs:    list, int or None  IDs of cores for which biases
                                            should be loaded. Load all if
                                            None.
    """

    def use_line(strCodeLine):
        """
        use_line - Determine whether strCodeLine should be executed considering
                   the IDs of cores in lnCoreIDs
        :param strCodeLine:  str  Line of code to be analyzed.
        :return:  bool  Whether line should be executed or not.
        """
        try:
            nCore = int(strCodeLine.split("get_bias_groups()[")[1][0])
        except IndexError:
            # - Line is not specific to core ID
            return True
        else:
            # - Return true if addressed core is in lnCoreIDs
            return nCore in lnCoreIDs

    if lnCoreIDs is None:
        lnCoreIDs = list(range(NUM_CHIPS * NUM_CHIP_CORES))
        bAllCores = True
    else:
        # - Handle integer arguments
        if isinstance(lnCoreIDs, int):
            lnCoreIDs = [lnCoreIDs]
        bAllCores = False

    with open(os.path.abspath(strFilename)) as file:
        # list of lines of code of the file. Skip import statement.
        lstrCodeLines = file.readlines()[1:]
        # Iterate over lines of file to apply biases
        for strCommand in lstrCodeLines:
            if bAllCores or use_line(strCommand):
                exec(strCommand)

    print(
        "dynapse_control: Biases have been loaded from {}.".format(
            os.path.abspath(strFilename)
        )
    )


@teleport_function
def save_biases(strFilename, lnCoreIDs: Optional[Union[list, int]] = None):
    """
    save_biases - Save biases in python file under path strFilename
    :param strFilename:  str  Path to file where biases should be saved.
    :param lnCoreIDs:    list, int or None  ID(s) of cores whose biases
                                            should be saved. If None,
                                            save all cores.
    """

    if lnCoreIDs is None:
        lnCoreIDs = list(range(NUM_CHIPS * NUM_CHIP_CORES))
    else:
        # - Handle integer arguments
        if isinstance(lnCoreIDs, int):
            lnCoreIDs = [lnCoreIDs]
        # - Include cores in filename, consider possible file endings
        lstrFilenameParts = strFilename.split(".")
        # Information to be inserted in filename
        strInsert = "_cores_" + "_".join(str(nCore) for nCore in lnCoreIDs)
        try:
            lstrFilenameParts[-2] += strInsert
        except IndexError:
            # strFilename does not contain file ending
            lstrFilenameParts[0] += strInsert
            lstrFilenameParts.append("py")
        strFilename = ".".join(lstrFilenameParts)

    lBiasGroups = CtxDynapse.model.get_bias_groups()
    # - Only save specified cores
    lBiasGroups = [lBiasGroups[i] for i in lnCoreIDs]
    with open(strFilename, "w") as file:
        file.write("import CtxDynapse\n")
        file.write("save_file_model_ = CtxDynapse.model\n")
        for nCore, bias_group in zip(lnCoreIDs, lBiasGroups):
            biases = bias_group.get_biases()
            for bias in biases:
                file.write(
                    'save_file_model_.get_bias_groups()[{0}].set_bias("{1}", {2}, {3})\n'.format(
                        nCore, bias.bias_name, bias.fine_value, bias.coarse_value
                    )
                )
    print(
        "dynapse_control: Biases have been saved under {}.".format(
            os.path.abspath(strFilename)
        )
    )


@teleport_function
def copy_biases(nSourceCoreID: int = 0, lnTargetCoreIDs: Optional[List[int]] = None):
    """
    copy_biases - Copy biases from one core to one or more other cores.
    :param nSourceCoreID:   int  ID of core from which biases are copied
    :param lnTargetCoreIDs: int or array-like ID(s) of core(s) to which biases are copied
                            If None, will copy to all other neurons
    """

    lnTargetCoreIDs = copy.copy(lnTargetCoreIDs)
    if lnTargetCoreIDs is None:
        # - Copy biases to all other cores except the source core
        lnTargetCoreIDs = list(range(16))
        lnTargetCoreIDs.remove(nSourceCoreID)
    elif isinstance(lnTargetCoreIDs, int):
        lnTargetCoreIDs = [lnTargetCoreIDs]

    # - List of bias groups from all cores
    lBiasgroups = CtxDynapse.model.get_bias_groups()
    sourcebiases = lBiasgroups[nSourceCoreID].get_biases()

    # - Set biases for target cores
    for nTargetCoreID in lnTargetCoreIDs:
        for bias in sourcebiases:
            lBiasgroups[nTargetCoreID].set_bias(
                bias.bias_name, bias.fine_value, bias.coarse_value
            )

    print(
        "dynapse_control: Biases have been copied from core {} to core(s) {}".format(
            nSourceCoreID, lnTargetCoreIDs
        )
    )


@teleport_function
def get_all_neurons(
    oModel: CtxDynapse.Model, oVirtualModel: CtxDynapse.VirtualModel
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
def _clear_chips(lnChipIDs: Optional[list] = None):
    """
    _clear_chips - Clear the physical CAM and SRAM cells of the chips defined
                   in lnChipIDs.
                   This is necessary when CtxControl is loaded (and only then)
                   to make sure that the configuration of the model neurons
                   matches the hardware.

    :param lnChipIDs:   list  IDs of chips to be cleared.
    """

    # - Make sure lnChipIDs is a list
    if lnChipIDs is None:
        return

    if isinstance(lnChipIDs, int):
        lnChipIDs = [lnChipIDs]

    # - Make sure that lnChipIDs is on correct side of RPyC connection
    lnChipIDs = copy.copy(lnChipIDs)

    for nChip in lnChipIDs:
        print("dynapse_control: Clearing chip {}.".format(nChip))

        # - Clear CAMs
        CtxDynapse.dynapse.clear_cam(int(nChip))
        print("\t CAMs cleared.")

        # - Clear SRAMs
        CtxDynapse.dynapse.clear_sram(int(nChip))
        print("\t SRAMs cleared.")

    print("dynapse_control: {} chip(s) cleared.".format(len(lnChipIDs)))


@teleport_function
def _reset_connections(
    lnCoreIDs: Optional[list] = None, bApplyDiff=True
):
    """
    _reset_connections - Reset connections going to all nerons of cores defined
                         in lnCoreIDs. Core IDs from 0 to 15.
    :param lnCoreIDs:   list  IDs of cores to be reset
    :param bApplyDiff:  bool  Apply changes to hardware. Setting False is useful
                              if new connections will be set afterwards.
    """
    # - Make sure lnCoreIDs is a list
    if lnCoreIDs is None:
        return

    if isinstance(lnCoreIDs, int):
        lnCoreIDs = [lnCoreIDs]

    # - Make sure that lnCoreIDs is on correct side of RPyC connection
    lnCoreIDs = copy.copy(lnCoreIDs)

    # - Get shadow state neurons
    lShadowNeurons = CtxDynapse.model.get_shadow_state_neurons()

    for nCore in lnCoreIDs:
        print("dynapse_control: Clearing connections of core {}.".format(nCore))

        # - Reset neuron weights in model
        for neuron in lShadowNeurons[
            nCore * NUM_CORE_NEURONS : (nCore + 1) * NUM_CORE_NEURONS
        ]:
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
    print("dynapse_control: {} core(s) cleared.".format(len(lnCoreIDs)))

    if bApplyDiff:
        # - Apply changes to the connections on chip
        CtxDynapse.model.apply_diff_state()
        print("dynapse_control: New state has been applied to the hardware")


@teleport_function
def remove_all_connections_to(
    lNeuronIDs: List, oModel: CtxDynapse.Model, bApplyDiff: bool = True
):
    """
    remove_all_connections_to - Remove all presynaptic connections
                                to neurons defined in vnNeuronIDs
    :param lNeuronIDs:      list  IDs of neurons whose presynaptic
                                      connections should be removed
    :param oModel:          CtxDynapse.model
    :param bApplyDiff:      bool If False do not apply the changes to
                                 chip but only to shadow states of the
                                 neurons. Useful if new connections are
                                 going to be added to the given neurons.
    """
    # - Make sure that lNeuronIDs is on correct side of RPyC connection
    lNeuronIDs = copy.copy(lNeuronIDs)

    # - Get shadow state neurons
    lShadowNeurons = CtxDynapse.model.get_shadow_state_neurons()

    # - Reset neuron weights in model
    for neuron in lShadowNeurons:
        # - Reset SRAMs
        viSrams = neuron.get_srams()
        for iSramIndex in range(1, 4):
            viSrams[iSramIndex].set_target_chip_id(0)
            viSrams[iSramIndex].set_virtual_core_id(0)
            viSrams[iSramIndex].set_used(False)
            viSrams[iSramIndex].set_core_mask(0)

        # - Reset CAMs
        for cam in neuron.get_cams():
            cam.set_pre_neuron_id(0)
            cam.set_pre_neuron_core_id(0)

    print("dynapse_control: Shadow state neuron weights have been reset")

    if bApplyDiff:
        # - Apply changes to the connections on chip
        oModel.apply_diff_state()
        print("dynapse_control: New state has been applied to the hardware")


@teleport_function
def set_connections(
    lPreNeuronIDs: list,
    lPostNeuronIDs: list,
    lSynapseTypes: list,
    lShadowNeurons: list,
    lVirtualNeurons: Optional[list],
    dcNeuronConnector: NeuronNeuronConnector.DynapseConnector,
):
    """
    set_connections - Set connections between pre- and post synaptic neurons from lists.
    :param lPreNeuronIDs:       list  N Presynaptic neurons
    :param lPostNeuronIDs:      list  N Postsynaptic neurons
    :param lSynapseTypes:       list  N or 1 Synapse type(s)
    :param lShadowNeurons:      list  Shadow neurons that the indices correspond to.
    :param lVirtualNeurons:     list  If None, presynaptic neurons are shadow neurons,
                                      otherwise virtual neurons from this list.
    :param dcNeuronConnector:   NeuronNeuronConnector.DynapseConnector
    """
    lPreNeuronIDs = copy.copy(lPreNeuronIDs)
    lPostNeuronIDs = copy.copy(lPostNeuronIDs)
    lSynapseTypes = copy.copy(lSynapseTypes)
    lPresynapticNeurons = lShadowNeurons if lVirtualNeurons is None else lVirtualNeurons

    dcNeuronConnector.add_connection_from_list(
        [lPresynapticNeurons[i] for i in lPreNeuronIDs],
        [lShadowNeurons[i] for i in lPostNeuronIDs],
        lSynapseTypes,
    )

    print("dynapse_control: {} connections have been set.".format(len(lPreNeuronIDs)))


@teleport_function
def _define_silence_neurons():
    @local_arguments
    def silence_neurons(lnNeuronIDs):
        """
        silence_neurons - Assign time contant tau2 to neurons definedin lnNeuronIDs
                          to make them silent.
        :param lnNeuronIDs:  list  IDs of neurons to be silenced
        """
        if isinstance(lnNeuronIDs, int):
            lnNeuronIDs = (lnNeuronIDs, )
        nNeuronsPerChip = NUM_CHIP_CORES * NUM_CORE_NEURONS
        for nID in lnNeuronIDs:
            CtxDynapse.dynapse.set_tau_2(
                nID // nNeuronsPerChip,  # Chip ID
                nID % nNeuronsPerChip,  # Neuron ID on chip
            )
        print("dynapse_control: Set {} neurons to tau 2.".format(len(lnNeuronIDs)))
    return silence_neurons
_silence_neurons = correct_argument_types(_define_silence_neurons())


@teleport_function
def _define_reset_silencing():
    @local_arguments
    def reset_silencing(lnCoreIDs):
        """
        reset_silencing - Assign time constant tau1 to all neurons on cores defined
                          in lnCoreIDs. Convenience function that does the same as
                          global _reset_silencing but also updates self._vbSilenced.
        :param lnCoreIDs:   list  IDs of cores to be reset
        """
        if isinstance(lnCoreIDs, int):
            lnCoreIDs = (lnCoreIDs, )
        for nID in lnCoreIDs:
            CtxDynapse.dynapse.reset_tau_1(
                nID // NUM_CHIP_CORES,  # Chip ID
                nID % NUM_CHIP_CORES,  # Core ID on chip
            )
        print("dynapse_control: Set neurons of cores {} to tau 1.".format(
            lnCoreIDs
        ))
    return reset_silencing
_reset_silencing = correct_argument_types(_define_reset_silencing())


# - Clear hardware configuration at startup
print("dynapse_control: Initializing hardware.")
if not bUsing_RPyC or not "bInitialized" in conn.namespace.keys():
    _clear_chips(lnChipIDs)
    conn.namespace["bInitialized"] = True
    print("dynapse_control: Hardware initialized.")
else:
    print("dynapse_control: Hardware has already been initialized.")


class DynapseControl:

    _nFpgaEventLimit = FPGA_EVENT_LIMIT
    _nFpgaIsiLimit = FPGA_ISI_LIMIT
    _tFpgaTimestep = FPGA_TIMESTEP
    _nNumCoreNeurons = NUM_CORE_NEURONS
    _nNumChipCores = NUM_CHIP_CORES
    _nNumChips = NUM_CHIPS

    def __init__(
        self,
        tFpgaIsiBase: float = DEF_FPGA_ISI_BASE,
        lnClearCores: Optional[list] = None,
    ):
        """
        DynapseControl - Class for interfacing DynapSE

        :param tFpgaIsiBase:    float           Time step for inter-spike intervals when sending events to FPGA
        :param lnClearCores:     list or None    IDs of cores where configurations should be cleared.
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
            warn(
                "DynapseControl: Could not find poisson generator module (DynapsePoissonGen)."
            )

        # - Get all neurons from models
        self.lHWNeurons, self.lVirtualNeurons, self.lShadowNeurons = get_all_neurons(
            self.model, self.virtualModel
        )

        # - Initialise neuron allocation
        self.vbFreeHWNeurons, self.vbFreeVirtualNeurons = (
            self._initial_free_neuron_lists()
        )
        
        # - Store which neurons have been assigned tau 2 (i.e. are silenced)
        self._vbSilenced = np.zeros_like(self.vbFreeHWNeurons, bool)
        # - Make sure no neuron is silenced, yet
        self.reset_silencing(range(16))

        print("DynapseControl: Neurons initialized.")
        print(
            "\t {} hardware neurons and {} virtual neurons available.".format(
                np.sum(self.vbFreeHWNeurons), np.sum(self.vbFreeHWNeurons)
            )
        )

        # - Get a connector object
        self.dcNeuronConnector = NeuronNeuronConnector.DynapseConnector()
        print("DynapseControl: Neuron connector initialized")

        # - Wipe configuration
        self.clear_connections(lnClearCores)

        ## -- Initialize Fpga spike generator
        self.tFpgaIsiBase = tFpgaIsiBase
        self.fpgaSpikeGen.set_repeat_mode(False)
        self.fpgaSpikeGen.set_variable_isi(True)
        self.fpgaSpikeGen.set_base_addr(0)
        print("DynapseControl: FPGA spike generator prepared.")

        print("DynapseControl ready.")

    @staticmethod
    def clear_connections(lnCoreIDs: Optional[list] = None):
        """
        clear_connections - Reset connections for cores defined in lnCoreIDs.

        :param lnCoreIDs:  list or None  IDs of cores where configurations
                                         should be cleared (0-15).
        """
        # - Use `_reset_connections` function
        _reset_connections(lnCoreIDs)
        print(
            "DynapseControl: Connections to cores {} have been cleared.".format(
                lnCoreIDs
            )
        )

    def silence_neurons(self, lnNeuronIDs: list):
        """
        silence_neurons - Assign time contant tau2 to neurons definedin lnNeuronIDs
                          to make them silent. Convenience function that does the
                          same as global _silence_neurons but also stores silenced
                          neurons in self._vbSilenced.
        :param lnNeuronIDs:  list  IDs of neurons to be silenced
        """
        _silence_neurons(lnNeuronIDs)
        # - Mark that neurons have been silenced
        self._vbSilenced[lnNeuronIDs] = True
        print("DynapseControl: {} neurons have been silenced.".format(len(lnNeuronIDs)))

    def reset_silencing(self, lnCoreIDs: list):
        """
        reset_silencing - Assign time constant tau1 to all neurons on cores defined
                          in lnCoreIDs. Convenience function that does the same as
                          global _reset_silencing but also updates self._vbSilenced.
        :param lnCoreIDs:   list  IDs of cores to be reset
        """
        if isinstance(lnCoreIDs, int):
            lnCoreIDs = (lnCoreIDs, )
        _reset_silencing(lnCoreIDs)
        # - Mark that neurons are not silenced anymore
        for nID in lnCoreIDs:
            self._vbSilenced[nID*self.nNumCoreNeurons: (nID+1)*self.nNumCoreNeurons] = False
        print("DynapseControl: Time constants of cores {} have been reset.".format(
            lnCoreIDs
        ))

    def reset_cores(
        self,
        lnCoreIDs: Optional[Union[list, int]] = None,
        bVirtual: bool = True,
        bSilence: bool = True,
    ):
        """
        reset_cores - Reset neuron assginments and connections for specified cores
                      and virtual neurons if requested.
        """
        # - Clear neuron assignments
        self.clear_neuron_assignments(lnCoreIDs, bVirtual)
        # - Reset connections
        self.clear_connections(lnCoreIDs)
        if bSilence:
            # - Reset neuron silencing
            self.reset_silencing(lnCoreIDs)

    def reset_all(self):
        """
        reset_all - Convenience function to reset neuron assginments and connections
                    for all cores and virtual neurons.
        """
        # - Clear neuron assignments
        self.reset_cores(range(self.nNumCores), True)

    ### --- Neuron allocation and connections

    def _initial_free_neuron_lists(self) -> (np.ndarray, np.ndarray):
        """
        _initial_free_neuron_lists - Generate initial lit of free hardware and
                                     virtual neurons as boolean arrays.
        :return:
            vbFreeHWNeurons         np.ndarray  Boolean array indicating which hardware
                                                neurons are available
            vbFreeVirtualNeurons    np.ndarray  Boolean array indicating which virtual
                                                neurons are available
        """
        # - Hardware neurons
        vbFreeHWNeurons = np.ones(len(self.lHWNeurons), bool)
        # Do not use hardware neurons with ID 0 and core ID 0 (first of each core)
        vbFreeHWNeurons[0 :: self.nNumChipNeurons] = False

        # - Virtual neurons
        vbFreeVirtualNeurons = np.ones(len(self.lVirtualNeurons), bool)
        # Do not use virtual neuron 0
        vbFreeVirtualNeurons[0] = False

        return vbFreeHWNeurons, vbFreeVirtualNeurons

    def clear_neuron_assignments(
        self, lnCoreIDs: Optional[Union[list, int]] = None, bVirtual: bool = True
    ):
        """
        clear_neuron_assignments - Mark neurons as free again.
        :param lnCoreIDs:   IDs of cores where neuron assignments should be reset.
                            If None, do not reset hardware neuron assignments.
        :param bVirtual:    Mark all virtual neurons as free (except neuron 0)
        """
        # - Original neuron availabilities
        vbFreeHWNeurons0, vbFreeVirtualNeurons0 = self._initial_free_neuron_lists()

        if lnCoreIDs is not None:
            # - Hardware neurons

            # Make lnCoreIDs iterable if integer
            if isinstance(lnCoreIDs, int):
                lnCoreIDs = [lnCoreIDs]

            for nCore in lnCoreIDs:
                iStartClear = nCore * self.nNumCoreNeurons
                iEndClear = iStartClear + self.nNumCoreNeurons
                self.vbFreeHWNeurons[iStartClear:iEndClear] = vbFreeHWNeurons0[
                    iStartClear:iEndClear
                ]
            print(
                "DynapseControl: {} hardware neurons available.".format(
                    np.sum(self.vbFreeHWNeurons)
                )
            )

        if bVirtual:
            # - Virtual neurons
            self.vbFreeVirtualNeurons = vbFreeVirtualNeurons0
            print(
                "DynapseControl: {} virtual neurons available.".format(
                    np.sum(self.vbFreeVirtualNeurons)
                )
            )

    def allocate_hw_neurons(
        self, vnNeuronIDs: Union[int, np.ndarray]
    ) -> (np.ndarray, np.ndarray):
        """
        allocate_hw_neurons - If vnNeuronIDs is ArrayLike, verify that neurons have not
                              been assigned yet and mark them as assigned and return a
                              list of corresponding hardware and shadow state neurons.
                              If vnNeuronIDs is an integer, assing first n available neurons.

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
                        np.sum(self.vbFreeHWNeurons[vnNeuronsToAllocate] == False)
                    )
                )

        # - Mark these neurons as allocated
        self.vbFreeHWNeurons[vnNeuronsToAllocate] = False

        # - Prevent allocation of virtual neurons with same (logical) ID as allocated hardware neurons
        #  IS THIS REALLY NECESSARY?
        vnInputNeuronOverlap = vnNeuronsToAllocate[
            vnNeuronsToAllocate < np.size(self.vbFreeVirtualNeurons)
        ]
        self.vbFreeVirtualNeurons[vnInputNeuronOverlap] = False

        # - Return these allocated neurons
        return (
            np.array([self.lHWNeurons[i] for i in vnNeuronsToAllocate]),
            np.array([self.lShadowNeurons[i] for i in vnNeuronsToAllocate]),
        )

    def allocate_virtual_neurons(
        self, vnNeuronIDs: Union[int, np.ndarray]
    ) -> np.ndarray:
        """
        allocate_virtual_neurons - If vnNeuronIDs is ArrayLike, verify that virtual neurons have not
                                   been assigned yet and mark them as assigned and return a
                                   list of virtual neurons neurons.
                                   If vnNeuronIDs is an integer, assing first n available virtual neurons.

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
            vnNeuronsToAllocate = np.nonzero(self.vbFreeVirtualNeurons)[0][:nNumNeurons]

        else:
            vnNeuronsToAllocate = np.array(vnNeuronIDs).flatten()
            # - Make sure neurons are available
            if (self.vbFreeVirtualNeurons[vnNeuronsToAllocate] == False).any():
                raise MemoryError(
                    "{} of the requested neurons are already allocated.".format(
                        np.sum(self.vbFreeVirtualNeurons[vnNeuronsToAllocate] == False)
                    )
                )

        # - Mark these as allocated
        self.vbFreeVirtualNeurons[vnNeuronsToAllocate] = False
        # - Prevent allocation of hardware neurons with same ID as allocated virtual neurons
        #  IS THIS REALLY NECESSARY?
        self.vbFreeHWNeurons[vnNeuronsToAllocate] = False

        # - Return these neurons
        return np.array([self.lVirtualNeurons[i] for i in vnNeuronsToAllocate])

    def connect_to_virtual(
        self,
        vnVirtualNeuronIDs: Union[int, np.ndarray],
        vnNeuronIDs: Union[int, np.ndarray],
        lSynapseTypes: List,
    ):
        """
        conncect_to_virtual - Connect a group of hardware neurons or
                              single hardware neuron to a group of
                              virtual neurons (1 to 1) or to a single
                              virtual neuron.
        :param vnVirtualNeuronIDs:   np.ndarray  IDs of virtual neurons
        :param vnNeuronIDs:          np.ndarray  IDs of hardware neurons
        :param lSynapseTypes:        list        Types of the synapses
        """

        # - Handle single neurons
        if isinstance(vnVirtualNeuronIDs, int):
            vnVirtualNeuronIDs = [vnVirtualNeuronIDs for _ in range(np.size(vnNeuronIDs))]
        if isinstance(vnNeuronIDs, int):
            vnNeuronIDs = [vnNeuronIDs for _ in range(np.size(vnVirtualNeuronIDs))]
        if np.size(lSynapseTypes) == 1:
            lSynapseTypes = list(np.repeat(lSynapseTypes, np.size(vnNeuronIDs)))
        else:
            lSynapseTypes = list(lSynapseTypes)

        # - Set connections
        set_connections(
            lPreNeuronIDs=list(vnVirtualNeuronIDs),
            lPostNeuronIDs=list(vnNeuronIDs),
            lSynapseTypes=lSynapseTypes,
            lShadowNeurons=self.lShadowNeurons,
            lVirtualNeurons=self.lVirtualNeurons,
            dcNeuronConnector=self.dcNeuronConnector,
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
        bApplyDiff: bool = True,
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

        # - Get connection lists
        liPreSynE, liPostSynE, liPreSynI, liPostSynI = connectivity_matrix_to_prepost_lists(
            mnW
        )

        # - Extract neuron IDs and remove numpy wrapper around int type
        lPreNeuronIDsExc = [int(vnVirtualNeuronIDs[i]) for i in liPreSynE]
        lPostNeuronIDsExc = [int(vnHWNeuronIDs[i]) for i in liPostSynE]
        lPreNeuronIDsInh = [int(vnVirtualNeuronIDs[i]) for i in liPreSynI]
        lPostNeuronIDsInh = [int(vnHWNeuronIDs[i]) for i in liPostSynI]

        # - Set excitatory connections
        set_connections(
            lPreNeuronIDs=lPreNeuronIDsExc,
            lPostNeuronIDs=lPostNeuronIDsExc,
            lSynapseTypes=[synExcitatory],
            lShadowNeurons=self.lShadowNeurons,
            lVirtualNeurons=self.lVirtualNeurons,
            dcNeuronConnector=self.dcNeuronConnector,
        )
        print(
            "DynapseControl: Excitatory connections of type `{}`".format(
                str(synExcitatory).split(".")[1]
            )
            + " from virtual neurons to hardware neurons have been set."
        )
        # - Set inhibitory connections
        set_connections(
            lPreNeuronIDs=lPreNeuronIDsInh,
            lPostNeuronIDs=lPostNeuronIDsInh,
            lSynapseTypes=[synInhibitory],
            lShadowNeurons=self.lShadowNeurons,
            lVirtualNeurons=self.lVirtualNeurons,
            dcNeuronConnector=self.dcNeuronConnector,
        )
        print(
            "DynapseControl: Inhibitory connections of type `{}`".format(
                str(synInhibitory).split(".")[1]
            )
            + " from virtual neurons to hardware neurons have been set."
        )

        if bApplyDiff:
            self.model.apply_diff_state()
            print("DynapseControl: Connections have been written to the chip.")

    def set_connections_from_weights(
        self,
        mnW: np.ndarray,
        vnHWNeuronIDs: np.ndarray,
        synExcitatory: CtxDynapse.DynapseCamType,
        synInhibitory: CtxDynapse.DynapseCamType,
        bApplyDiff: bool = True,
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

        ## -- Connect virtual neurons to hardware neurons

        # - Get virtual to hardware connections
        liPreSynE, liPostSynE, liPreSynI, liPostSynI = connectivity_matrix_to_prepost_lists(
            mnW
        )

        # - Extract neuron IDs and remove numpy wrapper around int type
        lPreNeuronIDsExc = [int(vnHWNeuronIDs[i]) for i in liPreSynE]
        lPostNeuronIDsExc = [int(vnHWNeuronIDs[i]) for i in liPostSynE]
        lPreNeuronIDsInh = [int(vnHWNeuronIDs[i]) for i in liPreSynI]
        lPostNeuronIDsInh = [int(vnHWNeuronIDs[i]) for i in liPostSynI]

        # - Set excitatory input connections
        set_connections(
            lPreNeuronIDs=lPreNeuronIDsExc,
            lPostNeuronIDs=lPostNeuronIDsExc,
            lSynapseTypes=[synExcitatory],
            lShadowNeurons=self.lShadowNeurons,
            lVirtualNeurons=None,
            dcNeuronConnector=self.dcNeuronConnector,
        )
        print(
            "DynapseControl: Excitatory connections of type `{}`".format(
                str(synExcitatory).split(".")[1]
            )
            + " between hardware neurons have been set."
        )
        # - Set inhibitory input connections
        set_connections(
            lPreNeuronIDs=lPreNeuronIDsInh,
            lPostNeuronIDs=lPostNeuronIDsInh,
            lSynapseTypes=[synInhibitory],
            lShadowNeurons=self.lShadowNeurons,
            lVirtualNeurons=None,
            dcNeuronConnector=self.dcNeuronConnector,
        )
        print(
            "DynapseControl: Inhibitory connections of type `{}`".format(
                str(synInhibitory).split(".")[1]
            )
            + " between hardware neurons have been set."
        )

        if bApplyDiff:
            self.model.apply_diff_state()
            print("DynapseControl: Connections have been written to the chip.")

    def remove_all_connections_to(self, vnNeuronIDs, bApplyDiff: bool = True):
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
        vnNeuronIDs = [int(nID) for nID in np.asarray(vnNeuronIDs)]

        # - Call `remove_all_connections_to` function
        remove_all_connections_to(
            vnNeuronIDs, self.model, bApplyDiff
        )

    ### --- Stimulation and event generation

    def _TSEvent_to_spike_list(
        self,
        tsSeries: TSEvent,
        vnNeuronIDs: np.ndarray,
        nTargetCoreMask: int = 1,
        nTargetChipID: int = 0,
    ) -> List:
        """
        _TSEvent_to_spike_list - Convert a TSEvent object to a ctxctl spike list

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
            # Make sure that no np.int64 or other non-native type is passed
            [int(nISI) for nISI in vnDiscreteISIs],
            [int(vnNeuronIDs[i]) for i in vnChannels],
            int(nTargetCoreMask),
            int(nTargetChipID),
        )
        # - Return a list of events
        return lEvents

    def _arrays_to_spike_list(
        self,
        vnChannels: np.ndarray,
        vnNeuronIDs: np.ndarray,
        vnTimeSteps: Optional[np.ndarray] = None,
        vtTimeTrace: Optional[np.ndarray] = None,
        nTSStart: Optional[int] = None,
        tStart: Optional[int] = 0,
        nTargetCoreMask: int = 1,
        nTargetChipID: int = 0,
    ) -> List:
        """
        _arrays_to_spike_list - Convert an array of input time steps and an an array
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
            assert (
                vtTimeTrace is not None
            ), "DynapseControl: Either `vnTimeSteps` or `vtTimeTrace` must be provided."
            vnTimeSteps = np.floor(vtTimeTrace / self.tFpgaIsiBase).astype(int)
        if nTSStart is None:
            assert (
                tStart is not None
            ), "DynapseControl: Either `nTSStart` or `tStart` must be provided."
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
            # Make sure that no np.int64 or other non-native type is passed
            [int(nISI) for nISI in vnDiscreteISIs],
            [int(vnNeuronIDs[i]) for i in vnChannels],
            int(nTargetCoreMask),
            int(nTargetChipID),
        )

        # - Return a list of events
        return lEvents

    def start_cont_stim(
        self, fFrequency: float, vnNeuronIDs: int, nChipID: int = 0, nCoreMask: int = 15
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

        # - Handle integers for vnNeuronIDs
        if isinstance(vnNeuronIDs, int):
            lnNeuronIDs = [vnNeuronIDs]
        else:
            lnNeuronIDs = [int(nID) for nID in vnNeuronIDs]

        # - Interspike interval
        tISI = 1. / fFrequency
        # - ISI in units of fpga time step
        nISIfpga = int(np.round(tISI / self.tFpgaIsiBase))
        # - ISI for neurons other than first is zero as they are simultaneous
        lnISI = [nISIfpga] + [0] * (len(lnNeuronIDs) - 1)


        # - Generate events
        # List for events to be sent to fpga
        lEvents = generate_fpga_event_list(
            lnISI, lnNeuronIDs, int(nCoreMask), int(nChipID)
        )
        self.fpgaSpikeGen.preload_stimulus(lEvents)
        print(
            "DynapseControl: Stimulus prepared with {} Hz".format(
                1. / (nISIfpga * self.tFpgaIsiBase)
            )
        )

        # - Start stimulation
        self.fpgaSpikeGen.start()
        print("DynapseControl: Stimulation started")

    def stop_stim(self, bClearFilter: bool = False):
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
        start_poisson_stim - Start generating events by poisson processes and send them.
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
            assert np.size(vfFrequencies) == np.size(
                vnNeuronIDs
            ), "DynapseControl: Length of `vfFrequencies must be same as length of `vnNeuronIDs` or 1."

        # - Set firing rates for selected neurons
        for fFreq, nNeuronID in zip(vfFrequencies, vnNeuronIDs):
            self.fpgaPoissonGen.write_poisson_rate_hz(nNeuronID, fFreq)

        # - Set chip ID
        self.fpgaPoissonGen.set_chip_id(nChipID)

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
        tWidth: float = 0.1,
        fFreq: float = 1000,
        tRecord: float = 3,
        tBuffer: float = 0.5,
        nInputNeuronID: int = 0,
        vnRecordNeuronIDs: Union[int, np.ndarray] = np.arange(1024),
        nTargetCoreMask: int = 15,
        nTargetChipID: int = 0,
        bPeriodic: bool = False,
        bRecord: bool = False,
        bTSEvent: bool = False,
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
        vnTimeSteps = np.floor(
            np.arange(0, tWidth, 1. / fFreq) / self.tFpgaIsiBase
        ).astype(int)
        # Add dummy events at end to avoid repeated stimulation due to "2-trigger-bug"
        tISILimit = self.nFpgaIsiLimit * self.tFpgaIsiBase
        nAdd = int(np.ceil(tRecord / tISILimit))
        vnTimeSteps = np.r_[
            vnTimeSteps, vnTimeSteps[-1] + np.arange(1, nAdd + 1) * self.nFpgaIsiLimit
        ]

        lEvents = self._arrays_to_spike_list(
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
        return self._send_stimulus_list(
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
        tRecord: Optional[float] = None,
        tBuffer: float = 0.5,
        vnNeuronIDs: Optional[np.ndarray] = None,
        vnRecordNeuronIDs: Optional[np.ndarray] = None,
        nTargetCoreMask: int = 15,
        nTargetChipID: int = 0,
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
            np.arange(tsSeries.nNumChannels)
            if vnNeuronIDs is None
            else np.array(vnNeuronIDs)
        )
        vnRecordNeuronIDs = (
            vnNeuronIDs if vnRecordNeuronIDs is None else vnRecordNeuronIDs
        )
        tRecord = tsSeries.tDuration if tRecord is None else tRecord

        # - Prepare event list
        lEvents = self._TSEvent_to_spike_list(
            tsSeries,
            vnNeuronIDs=vnNeuronIDs,
            nTargetCoreMask=nTargetCoreMask,
            nTargetChipID=nTargetChipID,
        )
        print(
            "DynapseControl: Stimulus prepared from TSEvent `{}`.".format(
                tsSeries.strName
            )
        )

        # - Stimulate and return recorded data if any
        return self._send_stimulus_list(
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
        vnTimeSteps: Optional[np.ndarray] = None,
        vtTimeTrace: Optional[np.ndarray] = None,
        tRecord: Optional[float] = None,
        tBuffer: float = 0.5,
        vnNeuronIDs: Optional[np.ndarray] = None,
        vnRecordNeuronIDs: Optional[np.ndarray] = None,
        nTargetCoreMask: int = 15,
        nTargetChipID: int = 0,
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
            np.arange(np.amax(vnChannels) + 1)
            if vnNeuronIDs is None
            else np.array(vnNeuronIDs)
        )
        vnRecordNeuronIDs = (
            vnNeuronIDs if vnRecordNeuronIDs is None else vnRecordNeuronIDs
        )
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
            print(
                "DynapseControl: Stimulus/recording time inferred to be {} s.".format(
                    tRecord
                )
            )

        # - Prepare event list
        lEvents = self._arrays_to_spike_list(
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
        return self._send_stimulus_list(
            lEvents=lEvents,
            tDuration=tRecord,
            tBuffer=tBuffer,
            vnRecordNeuronIDs=vnRecordNeuronIDs,
            bPeriodic=bPeriodic,
            bRecord=bRecord,
            bTSEvent=bTSEvent,
        )

    def _send_stimulus_list(
        self,
        lEvents,
        tDuration,
        tBuffer,
        vnRecordNeuronIDs: Optional[np.ndarray] = None,
        bPeriodic: bool = False,
        bRecord: bool = False,
        bTSEvent: bool = False,
    ):
        """
        _send_stimulus_list - Send a list of FPGA events to hardware. Possibly record hardware events.

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
        # - Throw an exception if event list is too long
        if len(lEvents) > self.nFpgaEventLimit:
            raise MemoryError("DynapseControl: lEvents can have at most {} elements (has {}).".format(
                self.nFpgaEventLimit, len(lEvents)
            ))

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
                bPeriodic * " periodic", (not bPeriodic) * " for {} s".format(tDuration)
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
                return self._recorded_data_to_TSEvent(vnRecordNeuronIDs, tDuration)
            else:
                # - Extract arrays from recorded data
                return self._recorded_data_to_arrays(vnRecordNeuronIDs, tDuration)

    def _recorded_data_to_TSEvent(
        self, vnNeuronIDs: np.ndarray, tRecord: float
    ) -> TSEvent:
        lEvents = self.bufferedfilter.get_events()
        lTrigger = self.bufferedfilter.get_special_event_timestamps()
        print(
            "DynapseControl: Recorded {} event(s) and {} trigger event(s)".format(
                len(lEvents), len(lTrigger)
            )
        )

        # - Extract monitored event channels and timestamps
        vnTimeStamps, vnChannels = event_data_to_channels(lEvents, vnNeuronIDs)
        vtTimeTrace = np.array(vnTimeStamps) * 1e-6
        vnChannels = np.array(vnChannels)

        # - Locate synchronisation timestamp
        vtStartTriggers = np.array(lTrigger) * 1e-6
        viStartIndices = np.searchsorted(vtTimeTrace, vtStartTriggers)
        viEndIndices = np.searchsorted(vtTimeTrace, vtStartTriggers + tRecord)
        # - Choose first trigger where start and end indices not equal. If not possible, take first trigger
        iTrigger = np.argmax((viEndIndices - viStartIndices) > 0)
        print("\t\t Using trigger event {}".format(iTrigger))
        tStartTrigger = vtStartTriggers[iTrigger]
        iStartIndex = viStartIndices[iTrigger]
        iEndIndex = viEndIndices[iTrigger]
        # - Filter time trace
        vtTimeTrace = vtTimeTrace[iStartIndex:iEndIndex] - tStartTrigger
        vnChannels = vnChannels[iStartIndex:iEndIndex]
        print("DynapseControl: Extracted event data")

        return TSEvent(
            vtTimeTrace,
            vnChannels,
            tStart=0,
            tStop=tRecord,
            nNumChannels=np.size(vnNeuronIDs),
            strName="DynapSE"
        )

    def _recorded_data_to_arrays(
        self, vnNeuronIDs: np.ndarray, tRecord: float
    ) -> TSEvent:
        lEvents = self.bufferedfilter.get_events()
        lTrigger = self.bufferedfilter.get_special_event_timestamps()

        print(
            "DynapseControl: Recorded {} event(s) and {} trigger event(s)".format(
                len(lEvents), len(lTrigger)
            )
        )

        # - Extract monitored event channels and timestamps
        vnTimeStamps, vnChannels = event_data_to_channels(lEvents, vnNeuronIDs)
        vtTimeTrace = np.array(vnTimeStamps) * 1e-6
        vnChannels = np.array(vnChannels)

        # - Locate synchronisation timestamp
        vtStartTriggers = np.array(lTrigger) * 1e-6
        viStartIndices = np.searchsorted(vtTimeTrace, vtStartTriggers)
        viEndIndices = np.searchsorted(vtTimeTrace, vtStartTriggers + tRecord)
        # - Choose first trigger where start and end indices not equal. If not possible, take first trigger
        iTrigger = np.argmax((viEndIndices - viStartIndices) > 0)
        print("\t\t Using trigger event {}".format(iTrigger))
        tStartTrigger = vtStartTriggers[iTrigger]
        iStartIndex = viStartIndices[iTrigger]
        iEndIndex = viEndIndices[iTrigger]
        # - Filter time trace
        vtTimeTrace = vtTimeTrace[iStartIndex:iEndIndex] - tStartTrigger
        vnChannels = vnChannels[iStartIndex:iEndIndex]
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
            self.bufferedfilter = _generate_buffered_filter(
                self.model, lnRecordNeuronIDs
            )
            print("DynapseControl: Generated new buffered event filter.")

        return self.bufferedfilter

    def clear_buffered_event_filter(self):
        """ clear_buffered_event_filter - Clear self.bufferedfilter if it exists."""
        if hasattr(self, "_bufferedfilter") and self.bufferedfilter is not None:
            self.bufferedfilter.clear()
            print("DynapseControl: Buffered event filter cleared")
        else:
            warn("DynapseControl: No buffered event filter found.")

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

        print(
            "DynapseControl: Collecting IDs of neurons that spike within the next {} seconds".format(
                tDuration
            )
        )

        # - Filter for recording neurons
        oFilter = self.add_buffered_event_filter(vnNeuronIDs)

        # - Wait and record spiking neurons
        time.sleep(tDuration)

        oFilter.clear()

        # - Sorted unique list of neurons' IDs that have spiked
        lnRecordedNeuronIDs = sorted(
            set((event.neuron.get_id() for event in oFilter.get_events()))
        )
        print(
            "DynapseControl: {} neurons spiked: {}".format(
                len(lnRecordedNeuronIDs), lnRecordedNeuronIDs
            )
        )

        return lnRecordedNeuronIDs

    def silence_hot_neurons(self, vnNeuronIDs: Union[list, np.ndarray], tDuration: float) -> list:
        """
        silence_hot_neurons - Collect IDs of all neurons that spike 
                              within tDuration. Assign them different
                              time constant to silence them.
        :param vnNeuronIDs:  list   IDs of neurons to be observed.
        :param tDuration:    float  How long to wait for spiking neurons 
        :return:
            lnHotNeurons    list  IDs of hot neurons that have been silenced.
        """
        # - Neurons that spike within tDuration
        lnHotNeurons = self.collect_spiking_neurons(vnNeuronIDs, tDuration=tDuration)
        # - Silence these neurons by assigning different Tau bias
        print("DynapseControl: Neurons {} will be silenced".format(lnHotNeurons))
        self.silence_neurons(lnHotNeurons)
        return lnHotNeurons

    def measure_population_firing_rates(
        self, llnPopulationIDs: list, tDuration: float, bVerbose=False
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
            vfFiringRates, vfMeanRates[i], vfMaxRates[i], vfMinRates[
                i
            ] = self.measure_firing_rates(lnNeuronIDs, tDuration)
            if bVerbose:
                print(vfFiringRates)

        return vfMeanRates, vfMaxRates, vfMinRates

    def measure_firing_rates(
        self, vnNeuronIDs: Optional[Union[int, np.ndarray]], tDuration: float
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
        lEvents = oFilter.get_events()
        if not lEvents:
            # - Handle empty event lists
            print("DynapseControl: No events recorded")
            return np.zeros(np.size(vnNeuronIDs)), 0, 0, 0
        # - Evaluate non-empty event lists
        return evaluate_firing_rates(lEvents, tDuration, vnNeuronIDs)

    def _monitor_firing_rates_inner(self, vnNeuronIDs, tInterval):
        """
        monitor_firing_rates - Continuously monitor the activity of a population
                               and periodically output the average firing rate.
        """

        self._bMonitoring = True
        # - Set up event filter
        self.add_buffered_event_filter(vnNeuronIDs)
        print("DynapseControl: Start monitoring firing rates of {} neurons.".format(
            np.size(vnNeuronIDs)
        ))
        while self._bMonitoring:
            # - Flush event stack
            __ = self.bufferedfilter.get_events()
            # - Collect events
            time.sleep(tInterval)
            lEvents = self.bufferedfilter.get_events()
            # - Process events
            if not lEvents:
                # - Handle empty event lists
                print("DynapseControl: No events recorded")
            else:
                # - Evaluate non-empty event lists
                fMeanRate = evaluate_firing_rates(lEvents, tInterval, vnNeuronIDs, bVerbose=False)[2]
                print("DynapseControl: Mean firing rate: {} Hz".format(fMeanRate))

    def monitor_firing_rates(self, vnNeuronIDs, tInterval):
        """
        monitor_firing_rates - Create a thread that continuously monitors the
                               activity of a population and periodically output
                               the average firing rate.
        """

        self.thrMonitor = threading.Thread(
            target=self._monitor_firing_rates_inner,
            kwargs={
                "vnNeuronIDs": vnNeuronIDs,
                "tInterval": tInterval,
            }
        )
        self.thrMonitor.start()

    def stop_monitor(self):
        self._bMonitoring = False
        self.bufferedfilter.clear()
        self.thrMonitor.join(timeout=5)
        del self.thrMonitor
        print("DynapseControl: Stopped monitoring.")


    def sweep_freq_measure_rate(
        self,
        vfFreq: list = [1, 10, 20, 50, 100, 200, 500, 1000, 2000],
        tDuration: float = 1,
        vnTargetNeuronIDs: Union[int, np.ndarray] = range(128),
        vnInputNeuronIDs: Union[int, np.ndarray] = 1,
        nChipID: int = 0,
        nCoreMask: int = 15,
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
            self.start_cont_stim(fFreq, vnInputNeuronIDs, nChipID, nCoreMask)
            mfFiringRates[iTrial, :], vfMeanRates[iTrial], vfMaxRates[
                iTrial
            ], vfMinRates[iTrial] = self.measure_firing_rates(
                vnTargetNeuronIDs, tDuration
            )
            self.stop_stim()

        return mfFiringRates, vfMeanRates, vfMaxRates, vfMinRates

    ### - Load and save biases

    @staticmethod
    def load_biases(strFilename, lnCoreIDs: Optional[Union[list, int]] = None):
        """
        load_biases - Load biases from python file under path strFilename.
                      Convenience function. Same as global load_biases.
        :param strFilename:  str  Path to file where biases are stored.
        :param lnCoreIDs:    list, int or None  IDs of cores for which biases
                                                should be loaded. Load all if
                                                None.
        """
        load_biases(os.path.abspath(strFilename), lnCoreIDs)
        print("DynapseControl: Biases have been loaded from {}.".format(strFilename))

    @staticmethod
    def save_biases(strFilename, lnCoreIDs: Optional[Union[list, int]] = None):
        """
        save_biases - Save biases in python file under path strFilename
                      Convenience function. Same as global save_biases.
        :param strFilename:  str  Path to file where biases should be saved.
        :param lnCoreIDs:    list, int or None  ID(s) of cores whose biases
                                                should be saved. If None,
                                                save all cores.
        """
        save_biases(os.path.abspath(strFilename), lnCoreIDs)
        print("DynapseControl: Biases have been saved under {}.".format(strFilename))

    @staticmethod
    def copy_biases(
        nSourceCoreID: int = 0, lnTargetCoreIDs: Optional[List[int]] = None
    ):
        """
        copy_biases - Copy biases from one core to one or more other cores.
                      Convenience function. Same as global copy_biases.
        :param nSourceCoreID:   int  ID of core from which biases are copied
        :param vnTargetCoreIDs: int or array-like ID(s) of core(s) to which biases are copied
                                If None, will copy to all other neurons
        """
        copy_biases(nSourceCoreID, lnTargetCoreIDs)
        print(
            "DynapseControl: Biases have been copied from core {} to core(s) {}".format(
                nSourceCoreID, lnTargetCoreIDs
            )
        )

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
    def nNumCoreNeurons(self):
        return self._nNumCoreNeurons

    @property
    def nNumChipCores(self):
        return self._nNumChipCores

    @property
    def nNumChipNeurons(self):
        return self._nNumChipCores * self._nNumCoreNeurons

    @property
    def nNumChips(self):
        return self._nNumChips

    @property
    def nNumCores(self):
        return self._nNumChips * self.nNumChipCores

    @property
    def nNumNeurons(self):
        return self._nNumChips * self.nNumChipNeurons

    @property
    def vbSilenced(self):
        return self._vbSilenced

    @property
    def vnSilencedIndices(self):
        return np.where(self._vbSilenced)[0]  

    @property
    def mnConnections(self, vnNeuronIDs):
        return self._foo
    

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
                "DynapseControl: `tFpgaTimestep` must be at least {}".format(
                    self.tFpgaTimestep
                )
            )
        else:
            self._nFpgaIsiMultiplier = int(np.floor(tNewBase / self.tFpgaTimestep))
            self.fpgaSpikeGen.set_isi_multiplier(self._nFpgaIsiMultiplier)
