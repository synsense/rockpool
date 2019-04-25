# ----
# dynapse_control.py - Module to interface cortexcontrol and the DynapSE chip
# Author: Felix Bauer, aiCTX AG, felix.bauer@ai-ctx.com
# ----

### --- Imports

import numpy as np
from warnings import warn
from typing import Tuple, List, Optional, Union
import time
import os.path
import threading

# - Global settings
_USE_RPYC = False
_USE_DEEPCOPY = False
RPYC_TIMEOUT = 300


## -- Import cortexcontrol modules or establish connection via RPyC
try:
    import CtxDynapse
    import NeuronNeuronConnector as nnconnector

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
    conn._config["sync_request_timeout"] = RPYC_TIMEOUT  # Set timeout to higher level
    CtxDynapse = conn.modules.CtxDynapse
    nn_connector = conn.modules.NeuronNeuronConnector
    _USE_RPYC = True

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
chip_ids = [0]  # Chips to be used
# - Fix (hardware)
SRAM_EVENT_LIMIT = int(2 ** 19 - 1)  # Max. number of events that can be loaded to SRAM
FPGA_EVENT_LIMIT = int(2 ** 16 - 1)  # Max. number of events that can be sent to FPGA
FPGA_ISI_LIMIT = int(
    2 ** 16 - 1
)  # Max. number of timesteps for single inter-spike interval between FPGA events
FPGA_TIMESTEP = 1.0 / 9.0 * 1e-7  # Internal clock of FPGA, 11.111...ns
CORE_DIMENSIONS = (16, 16)  # Numbers of neurons in core (rows, columns)
NUM_NEURONS_CORE = (
    CORE_DIMENSIONS[0] * CORE_DIMENSIONS[1]
)  # Number of neurons on one core
NUM_CORES_CHIP = 4  # Number of cores per chip
NUM_CHIPS = 4  # Number of available chips
# - Default values, can be changed
DEF_FPGA_ISI_BASE = 2e-5  # Default timestep between events sent to FPGA
DEF_FPGA_ISI_MULTIPLIER = int(np.round(DEF_FPGA_ISI_BASE / FPGA_TIMESTEP))

if _USE_RPYC:
    # - Setup parameters on RPyC server
    conn.namespace["_USE_RPYC"] = True
    conn.namespace["NUM_NEURONS_CORE"] = NUM_NEURONS_CORE
    conn.namespace["NUM_CORES_CHIP"] = NUM_CORES_CHIP
    conn.namespace["NUM_CHIPS"] = NUM_CHIPS
    conn.namespace["copy"] = conn.modules.copy
    conn.namespace["os"] = conn.modules.os
    conn.namespace["CtxDynapse"] = conn.modules.CtxDynapse
    conn.namespace["rpyc"] = conn.modules.rpyc

### --- Utility functions
def connectivity_matrix_to_prepost_lists(
    weights: np.ndarray
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    connectivity_matrix_to_prepost_lists - Convert a matrix into a set of pre-post connectivity lists

    :param weights: ndarray[int]    Matrix[pre, post] containing integer numbers of synaptic connections
    :return:    (presyn_ids_exc,
                 postsyn_ids_exc,
                 presyn_ids_inh,
                 postsyn_ids_inh)       Each ndarray(int), containing a single pre- and post- synaptic partner.
                                    presyn_ids_exc and postsyn_ids_exc together define excitatory connections
                                    presyn_ids_inh and postsyn_ids_inh together define inhibitory connections
    """
    # - Get lists of pre and post-synaptic IDs
    presyn_ids_exc_compressed, postsyn_ids_exc_compressed = np.nonzero(weights > 0)
    presyn_ids_inh_compressed, postsyn_ids_inh_compressed = np.nonzero(weights < 0)

    # - Preallocate connection lists
    presyn_ids_exc = []
    postsyn_ids_exc = []
    presyn_ids_inh = []
    postsyn_ids_inh = []

    # - Loop over lists, appending when necessary
    for pre_id, post_id in zip(presyn_ids_exc_compressed, postsyn_ids_exc_compressed):
        for _ in range(weights[pre_id, post_id]):
            presyn_ids_exc.append(
                pre_id.item()
            )  # avoid using np.int64 type for integers
            postsyn_ids_exc.append(post_id.item())

    # - Loop over lists, appending when necessary
    for pre_id, post_id in zip(presyn_ids_inh_compressed, postsyn_ids_inh_compressed):
        for _ in range(np.abs(weights[pre_id, post_id])):
            presyn_ids_inh.append(pre_id.item())
            postsyn_ids_inh.append(post_id.item())

    # - Return augmented lists
    return presyn_ids_exc, postsyn_ids_exc, presyn_ids_inh, postsyn_ids_inh


def rectangular_neuron_arrangement(
    first_neuron: int,
    num_neurons: int,
    width: int,
    core_dimensions: tuple = CORE_DIMENSIONS,
) -> List[int]:
    """
    rectangular_neuron_arrangement: return neurons that form a rectangle on the chip
                                    with first_neuron as upper left corner and width
                                    width. (Last row may not be full). Neurons
                                    have to fit onto single core.
    :param first_neuron:  int ID of neuron that is in the upper left corner of the rectangle
    :param num_neurons:   int Number of neurons in the rectangle
    :param width:        int Width of the rectangle (in neurons)
    :param core_dimensions:  tuple (number of neuron rows in core, number of neurons in row)
    :return neuron_ids:   list 1D array of IDs of neurons that from the rectangle.
    """
    height_core, width_core = core_dimensions
    num_rows = int(np.ceil(num_neurons / width))
    first_row = int(np.floor(first_neuron / width_core))
    first_col = first_neuron % width_core
    # - Make sure rectangle fits on single core
    if (
        first_col + width > width_core
        or first_row % height_core + num_rows > height_core
    ):
        raise ValueError("dynapse_control: Rectangle does not fit onto single core.")
    neuron_ids = [
        (first_row + rownum) * width_core + (first_col + colnum)
        for rownum in range(num_rows)
        for colnum in range(width)
    ]
    return neuron_ids


def generate_event_raster(
    events: list, duration: float, neuron_ids: list
) -> np.ndarray:
    """
    generate_event_raster - Generate a boolean spike raster of a list of events with timestep 0.001ms
    :param events:         list of Spike objects
    :param duration:       float Overall time covered by the raster
    :param neuron_ids:     array-like of neuron IDs corresponding to events.

    :return:
        event_raster   np.ndarray  - Boolean event raster
    """
    neuron_ids = list(neuron_ids)
    # - Extract event timestamps and neuron IDs
    timestamps: tuple
    event_neuron_ids: tuple
    timestamps, event_neuron_ids = zip(
        *((event.timestamp, event.neuron.get_id()) for event in events)
    )
    # - Event times in microseconds
    event_times = np.array(timestamps, int)
    # - Set time of first event to 0
    event_times -= event_times[0]
    # - Convert neuron IDs of event to index starting from 0
    event_indices = np.array(
        [neuron_ids.index(neuron_id) for neuron_id in event_neuron_ids]
    )
    # - Convert events to binary raster
    num_timesteps = int(np.ceil(duration * 1e6))
    event_raster = np.zeros((num_timesteps, np.size(event_indices)), bool)
    event_raster[event_times, event_indices] = True
    return event_raster


def evaluate_firing_rates(
    events: list,
    duration: float,
    neuron_ids: Optional[list] = None,
    verbose: bool = True,
) -> (np.ndarray, float, float, float):
    """
    evaluate_firing_rates - Determine the neuron-wise firing rates from a
                            list of events. Calculate mean, max and min.
    :param events:         list of Spike objects
    :param duration:       float Time over which rates are normalized
    :param neuron_ids:     array-like of neuron IDs corresponding to events.
                            If None, neuron_ids will consists of to the neurons
                            corresponding to the events in events.
    :param verbose:        bool  Print out information about firing rates.

    :return:
        firing_rates  np.ndarray - Each neuron's firing rate
        rate_mean      float - Average firing rate over all neurons
        rate_max       float - Highest firing rate of all neurons
        rate_min       float - Lowest firing rate of all neurons
    """
    # - Extract event timestamps and neuron IDs
    timestamps, event_neuron_ids = extract_event_data(events)
    # - Count events for each neuron
    unique_event_ids, event_counts = np.unique(event_neuron_ids, return_counts=True)

    if neuron_ids is None:
        # - neuron_ids as list of neurons that have spiked
        neuron_ids = unique_event_ids

    # - Neurons that have not spiked
    indices_noevents = (np.asarray(neuron_ids))[
        np.isin(neuron_ids, unique_event_ids) == False
    ]

    # - Count events
    unique_event_ids = np.r_[unique_event_ids, indices_noevents]
    event_counts = np.r_[event_counts, np.zeros(indices_noevents.size)]

    # - Sort event counts to same order as in neuron_ids
    unique_event_ids = list(unique_event_ids)
    sort_events = [unique_event_ids.index(neuron_id) for neuron_id in neuron_ids]
    event_counts = event_counts[sort_events]
    firing_rates = event_counts / duration

    # - Calculate mean, max and min rates
    rate_mean = np.mean(firing_rates)
    max_rate_index = np.argmax(event_counts)
    rate_max = firing_rates[max_rate_index]
    min_rate_index = np.argmin(event_counts)
    rate_min = firing_rates[min_rate_index]

    if verbose:
        # - Print results
        print("\tMean firing rate: {} Hz".format(rate_mean))
        print(
            "\tMaximum firing rate: {} Hz (neuron {})".format(
                rate_max, neuron_ids[max_rate_index]
            )
        )
        print(
            "\tMinimum firing rate: {} Hz (neuron {})".format(
                rate_min, neuron_ids[min_rate_index]
            )
        )

    return firing_rates, rate_mean, rate_max, rate_min


# @profile
def event_data_to_channels(
    events: List, layer_neuron_ids: List
) -> (np.ndarray, np.ndarray):
    """
    event_data_to_channels - Convert timestamps and neuron IDs from list of Events
                             to arrays with timestamps and channel indices wrt layer_neuron_ids
                             Fill in nan where event does not correspond to any given ID.
    :param events:         list  SpikeEvent objects from BufferedEventFilter
    :param layer_neuron_ids: list  Neuron IDs corresponding to channels
    :return:
        timestamps         np.ndarray  Extracted timestam
        channel_indices     np.ndarray  Extracted channel indices
    """
    timestamps: tuple
    neuron_ids: tuple
    timestamps, neuron_ids = extract_event_data(events)
    # - Convert to numpy array and thereby fetch data from connection if using RPyC
    timestamps = np.array(timestamps)
    neuron_ids = np.array(neuron_ids)
    # - Convert neuron IDs to channels
    channel_dict = {
        id_neuron: iChannel for iChannel, id_neuron in enumerate(layer_neuron_ids)
    }
    channel_indices = np.array(
        [channel_dict.get(id_neuron, float("nan")) for id_neuron in neuron_ids]
    )
    if np.isnan(channel_indices).any():
        warn("dynapse_control: Some events did not match `layer_neuron_ids`")

    return timestamps, channel_indices


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
    # - Check if object type is defined in remote namespace
    elif type(obj).__module__ not in ["builtins", "CtxDynapse", "rpyc"]:
        warn("Unrecognized type: {}".format(type(obj)))
    return obj


def correct_argument_types_and_teleport(func):
    """
    correct_argument_types_and_teleport -  Wrapper for functions that tries to
            correct argument types that are not supported by cortexcontrol and
            teleports the function via rpyc. Returns original function if
            _USE_RPYC == False
    :param func:  function to be teleported
    :return:      teleported function
    """

    if _USE_RPYC:
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
    correct_argument_types - If _USE_RPYC is not False, try changing the
                             arguments to a function to types that are
                             supported by cortexcontrol
    :param func:    funciton where arguments should be corrected
    :return:        functions with possibly corrected arguments
    """
    if _USE_RPYC:

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
    # - Teleport if _USE_RPYC flag is set
    if _USE_RPYC:
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
    # - Teleport if _USE_RPYC flag is set
    if _USE_RPYC:
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


# # - Example on how to use local_arguments_rpyc decorator
# @teleport_function
# def _define_print_type():
#     @local_arguments
#     def print_type(obj):
#         print(type(obj))
#     return print_type
# print_type = correct_argument_types(
#     _define_print_type()
# )  # or just print_type = _define_print_type()


@teleport_function
def extract_event_data(events) -> (tuple, tuple):
    """
    extract_event_data - Extract timestamps and neuron IDs from list of recorded events.
    :param events:     list  SpikeEvent objects from BufferedEventFilter
    :return:
        lTimeStamps     list  Timestamps of events
        neuron_ids      list  Neuron IDs of events
    """
    # Extract event timestamps and neuron IDs. Skip events with neuron None.
    event_tuples: List[Tuple] = [
        (event.timestamp, event.neuron.get_id())
        for event in events
        if isinstance(event.neuron, CtxDynapse.DynapseNeuron)
    ]
    try:
        timestamps, neuron_ids = zip(*event_tuples)
    except ValueError as e:
        # - Handle emptly event lists
        if len(event_tuples) == 0:
            timestamps = ()
            neuron_ids = ()
        else:
            raise e
    return timestamps, neuron_ids


@remote_function
def _replace_too_large_value(value, limit: int = FPGA_ISI_LIMIT):
    """
    replace_too_large_entry - Return a list of integers <= limit, that sum up to value
    :param value:    int  Value to be replaced
    :param limit:  int  Maximum allowed value
    :return:
        lnReplace   list  Values to replace value
    """
    if value > limit:
        reps = (value - 1) // limit
        # - Return reps times limit, then the remainder
        #   For modulus shift value to avoid replacing with 0 if value==limit
        return [*(limit for _ in range(reps)), (value - 1) % limit + 1]
    else:
        # - If clause in particular for case where value <= 0
        return [value]


@remote_function
def _auto_insert_dummies(
    discrete_isi_list: list, neuron_ids: list, fpga_isi_limit: int = FPGA_ISI_LIMIT
) -> (list, list):
    """
    auto_insert_dummies - Insert dummy events where ISI limit is exceeded
    :param discrete_isi_list:  list  Inter-spike intervals of events
    :param neuron_ids:     list  IDs of neurons corresponding to the ISIs

    :return:
        corrected_isi_list     list  ISIs that do not exceed limit, including dummies
        corrected_id_list      list  Neuron IDs corresponding to corrected ISIs. Dummy events have None.
        isdummy_list           list  Boolean indicating which events are dummy events
    """
    # - List of lists with corrected entries
    corrected: List[List] = [
        _replace_too_large_value(isi, fpga_isi_limit) for isi in discrete_isi_list
    ]
    # - Number of new entries for each old entry
    new_event_counts = [len(l) for l in corrected]
    # - List of lists with neuron IDs corresponding to ISIs. Dummy events have ID None
    id_lists: List[List] = [
        [id_neur, *(None for _ in range(length - 1))]
        for id_neur, length in zip(neuron_ids, new_event_counts)
    ]
    # - Flatten out lists
    corrected_isi_list = [isi for l in corrected for isi in l]
    corrected_id_list = [id_neur for l in id_lists for id_neur in l]
    # - Count number of added dummy events (each one has None as ID)
    num_dummies = len(tuple(filter(lambda x: x is None, corrected_id_list)))
    if num_dummies > 0:
        print("dynapse_control: Inserted {} dummy events.".format(num_dummies))

    return corrected_isi_list, corrected_id_list


@teleport_function
def generate_fpga_event_list(
    discrete_isi_list: list,
    neuron_ids: list,
    targetcore_mask: int,
    targetchip_id: int,
    fpga_isi_limit: int = FPGA_ISI_LIMIT,
    correct_isi: bool = True,
) -> list:
    """
    generate_fpga_event_list - Generate a list of FpgaSpikeEvent objects
    :param discrete_isi_list:  array-like  Inter-spike intervalls in Fpga time base
    :param neuron_ids:     array-like  IDs of neurons corresponding to events
    :param neuron_id:       int ID of source neuron
    :param targetcore_mask: int Coremask to determine target cores
    :fpga_isi_limit:         int Maximum ISI size (in time steps)
    :correct_isi:           bool Correct too large ISIs in discrete_isi_list

    :return:
        event  list of generated FpgaSpikeEvent objects.
    """

    # - Make sure objects live on required side of RPyC connection
    targetcore_mask = int(targetcore_mask)
    targetchip_id = int(targetchip_id)
    neuron_ids = copy.copy(neuron_ids)
    discrete_isi_list = copy.copy(discrete_isi_list)

    if correct_isi:
        discrete_isi_list, neuron_ids = _auto_insert_dummies(
            discrete_isi_list, neuron_ids, fpga_isi_limit
        )

    def generate_fpga_event(neuron_id: int, isi: int) -> CtxDynapse.FpgaSpikeEvent:
        """
        generate_fpga_event - Generate a single FpgaSpikeEvent objects.
        :param neuron_id:       int ID of source neuron
        :param isi:            int Timesteps after previous event before
                                    this event will be sent
        :return:
            event  CtxDynapse.FpgaSpikeEvent
        """
        event = CtxDynapse.FpgaSpikeEvent()
        event.target_chip = targetchip_id
        event.core_mask = 0 if neuron_id is None else targetcore_mask
        event.neuron_id = 0 if neuron_id is None else neuron_id
        event.isi = isi
        return event

    # - Generate events
    print("dynapse_control: Generating event list")
    events = [
        generate_fpga_event(neuron_id, isi)
        for neuron_id, isi in zip(neuron_ids, discrete_isi_list)
    ]
    return events


@teleport_function
def _generate_buffered_filter(model: CtxDynapse.Model, record_neuron_ids: list):
    """
    _generate_buffered_filter - Generate and return a BufferedEventFilter object that
                               records from neurons specified in record_neuron_ids.
    :param model:               CtxDynapse model
    :param record_neuron_ids:    list  IDs of neurons to be recorded.
    """
    return CtxDynapse.BufferedEventFilter(model, record_neuron_ids)


@teleport_function
def load_biases(filename: str, core_ids: Optional[Union[list, int]] = None):
    """
    load_biases - Load biases from python file under path filename
    :param filename:  str  Path to file where biases are stored.
    :param core_ids:    list, int or None  IDs of cores for which biases
                                            should be loaded. Load all if
                                            None.
    """

    def use_line(codeline: str):
        """
        use_line - Determine whether codeline should be executed considering
                   the IDs of cores in core_ids
        :param codeline:  str  Line of code to be analyzed.
        :return:  bool  Whether line should be executed or not.
        """
        try:
            core_id = int(codeline.split("get_bias_groups()[")[1][0])
        except IndexError:
            # - Line is not specific to core ID
            return True
        else:
            # - Return true if addressed core is in core_ids
            return core_id in core_ids

    if core_ids is None:
        core_ids = list(range(NUM_CHIPS * NUM_CORES_CHIP))
        load_all_cores = True
    else:
        # - Handle integer arguments
        if isinstance(core_ids, int):
            core_ids = [core_ids]
        load_all_cores = False

    with open(os.path.abspath(filename)) as file:
        # list of lines of code of the file. Skip import statement.
        codeline_list = file.readlines()[1:]
        # Iterate over lines of file to apply biases
        for command in codeline_list:
            if load_all_cores or use_line(command):
                exec(command)

    print(
        "dynapse_control: Biases have been loaded from {}.".format(
            os.path.abspath(filename)
        )
    )


@teleport_function
def save_biases(filename: str, core_ids: Optional[Union[list, int]] = None):
    """
    save_biases - Save biases in python file under path filename
    :param filename:  str  Path to file where biases should be saved.
    :param core_ids:    list, int or None  ID(s) of cores whose biases
                                            should be saved. If None,
                                            save all cores.
    """

    if core_ids is None:
        core_ids = list(range(NUM_CHIPS * NUM_CORES_CHIP))
    else:
        # - Handle integer arguments
        if isinstance(core_ids, int):
            core_ids = [core_ids]
        # - Include cores in filename, consider possible file endings
        filename_parts: List[str] = filename.split(".")
        # Information to be inserted in filename
        insertstring = "_cores_" + "_".join(str(core_id) for core_id in core_ids)
        try:
            filename_parts[-2] += insertstring
        except IndexError:
            # filename does not contain file ending
            filename_parts[0] += insertstring
            filename_parts.append("py")
        filename = ".".join(filename_parts)

    biasgroup_list = CtxDynapse.model.get_bias_groups()
    # - Only save specified cores
    biasgroup_list = [biasgroup_list[i] for i in core_ids]
    with open(filename, "w") as file:
        file.write("import CtxDynapse\n")
        file.write("save_file_model_ = CtxDynapse.model\n")
        for core_id, bias_group in zip(core_ids, biasgroup_list):
            biases = bias_group.get_biases()
            for bias in biases:
                file.write(
                    'save_file_model_.get_bias_groups()[{0}].set_bias("{1}", {2}, {3})\n'.format(
                        core_id,
                        bias.get_bias_name(),
                        bias.get_fine_value(),
                        bias.get_coarse_value(),
                    )
                )
    print(
        "dynapse_control: Biases have been saved under {}.".format(
            os.path.abspath(filename)
        )
    )


@teleport_function
def copy_biases(sourcecore_id: int = 0, targetcore_ids: Optional[List[int]] = None):
    """
    copy_biases - Copy biases from one core to one or more other cores.
    :param sourcecore_id:   int  ID of core from which biases are copied
    :param targetcore_ids: int or array-like ID(s) of core(s) to which biases are copied
                            If None, will copy to all other neurons
    """

    targetcore_ids = copy.copy(targetcore_ids)
    if targetcore_ids is None:
        # - Copy biases to all other cores except the source core
        targetcore_ids = list(range(16))
        targetcore_ids.remove(sourcecore_id)
    elif isinstance(targetcore_ids, int):
        targetcore_ids = [targetcore_ids]

    # - List of bias groups from all cores
    biasgroup_list = CtxDynapse.model.get_bias_groups()
    sourcebiases = biasgroup_list[sourcecore_id].get_biases()

    # - Set biases for target cores
    for tgtcore_id in targetcore_ids:
        for bias in sourcebiases:
            biasgroup_list[tgtcore_id].set_bias(
                bias.bias_name, bias.fine_value, bias.coarse_value
            )

    print(
        "dynapse_control: Biases have been copied from core {} to core(s) {}".format(
            sourcecore_id, targetcore_ids
        )
    )


@teleport_function
def get_all_neurons(
    model: CtxDynapse.Model, virtual_model: CtxDynapse.VirtualModel
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    get_all_neurons - Get hardware, virtual and shadow state neurons
                      from model and virtual_model and return them
                      in arrays.
    :param model:  CtxDynapse.Model
    :param virtual_model: CtxDynapse.VirtualModel
    :return:
        np.ndarray  Hardware neurons
        np.ndarray  Virtual neurons
        np.ndarray  Shadow state neurons
    """
    hw_neurons: List = model.get_neurons()
    virtual_neurons: List = virtual_model.get_neurons()
    shadow_neurons: List = model.get_shadow_state_neurons()
    print("dynapse_control: Fetched all neurons from models.")
    return hw_neurons, virtual_neurons, shadow_neurons


@teleport_function
def _clear_chips(chip_ids: Optional[list] = None):
    """
    _clear_chips - Clear the physical CAM and SRAM cells of the chips defined
                   in chip_ids.
                   This is necessary when CtxControl is loaded (and only then)
                   to make sure that the configuration of the model neurons
                   matches the hardware.

    :param chip_ids:   list  IDs of chips to be cleared.
    """

    # - Make sure chip_ids is a list
    if chip_ids is None:
        return

    if isinstance(chip_ids, int):
        chip_ids = [chip_ids]

    # - Make sure that chip_ids is on correct side of RPyC connection
    chip_ids = copy.copy(chip_ids)

    for nchip in chip_ids:
        print("dynapse_control: Clearing chip {}.".format(nchip))

        # - Clear CAMs
        CtxDynapse.dynapse.clear_cam(int(nchip))
        print("\t CAMs cleared.")

        # - Clear SRAMs
        CtxDynapse.dynapse.clear_sram(int(nchip))
        print("\t SRAMs cleared.")

    print("dynapse_control: {} chip(s) cleared.".format(len(chip_ids)))


@teleport_function
def _reset_connections(core_ids: Optional[list] = None, apply_diff=True):
    """
    _reset_connections - Reset connections going to all nerons of cores defined
                         in core_ids. Core IDs from 0 to 15.
    :param core_ids:   list  IDs of cores to be reset
    :param apply_diff:  bool  Apply changes to hardware. Setting False is useful
                              if new connections will be set afterwards.
    """
    # - Make sure core_ids is a list
    if core_ids is None:
        return

    if isinstance(core_ids, int):
        core_ids = [core_ids]

    # - Make sure that core_ids is on correct side of RPyC connection
    core_ids = copy.copy(core_ids)

    # - Get shadow state neurons
    shadow_neurons = CtxDynapse.model.get_shadow_state_neurons()

    for core_id in core_ids:
        print("dynapse_control: Clearing connections of core {}.".format(core_id))

        # - Reset neuron weights in model
        for neuron in shadow_neurons[
            core_id * NUM_NEURONS_CORE : (core_id + 1) * NUM_NEURONS_CORE
        ]:
            # - Reset SRAMs for this neuron
            srams = neuron.get_srams()
            for sram_idx in range(1, 4):
                srams[sram_idx].set_target_chip_id(0)
                srams[sram_idx].set_virtual_core_id(0)
                srams[sram_idx].set_used(False)
                srams[sram_idx].set_core_mask(0)

            # - Reset CAMs for this neuron
            for cam in neuron.get_cams():
                cam.set_pre_neuron_id(0)
                cam.set_pre_neuron_core_id(0)
        print("\t Model neuron weights have been reset.")
    print("dynapse_control: {} core(s) cleared.".format(len(core_ids)))

    if apply_diff:
        # - Apply changes to the connections on chip
        CtxDynapse.model.apply_diff_state()
        print("dynapse_control: New state has been applied to the hardware")


@teleport_function
def remove_all_connections_to(
    neuron_ids: List, model: CtxDynapse.Model, apply_diff: bool = True
):
    """
    remove_all_connections_to - Remove all presynaptic connections
                                to neurons defined in neuron_ids
    :param neuron_ids:      list  IDs of neurons whose presynaptic
                                      connections should be removed
    :param model:          CtxDynapse.model
    :param apply_diff:      bool If False do not apply the changes to
                                 chip but only to shadow states of the
                                 neurons. Useful if new connections are
                                 going to be added to the given neurons.
    """
    # - Make sure that neuron_ids is on correct side of RPyC connection
    neuron_ids = copy.copy(neuron_ids)

    # - Get shadow state neurons
    shadow_neurons = CtxDynapse.model.get_shadow_state_neurons()

    # - Reset neuron weights in model
    for neuron in shadow_neurons:
        # - Reset SRAMs
        srams = neuron.get_srams()
        for sram_idx in range(1, 4):
            srams[sram_idx].set_target_chip_id(0)
            srams[sram_idx].set_virtual_core_id(0)
            srams[sram_idx].set_used(False)
            srams[sram_idx].set_core_mask(0)

        # - Reset CAMs
        for cam in neuron.get_cams():
            cam.set_pre_neuron_id(0)
            cam.set_pre_neuron_core_id(0)

    print("dynapse_control: Shadow state neuron weights have been reset")

    if apply_diff:
        # - Apply changes to the connections on chip
        model.apply_diff_state()
        print("dynapse_control: New state has been applied to the hardware")


@teleport_function
def set_connections(
    preneuron_ids: list,
    postneuron_ids: list,
    syntypes: list,
    shadow_neurons: list,
    virtual_neurons: Optional[list],
    neuronconnector: nn_connector.DynapseConnector,
):
    """
    set_connections - Set connections between pre- and post synaptic neurons from lists.
    :param preneuron_ids:       list  N Presynaptic neurons
    :param postneuron_ids:      list  N Postsynaptic neurons
    :param syntypes:       list  N or 1 Synapse type(s)
    :param shadow_neurons:      list  Shadow neurons that the indices correspond to.
    :param virtual_neurons:     list  If None, presynaptic neurons are shadow neurons,
                                      otherwise virtual neurons from this list.
    :param neuronconnector:   nn_connector.DynapseConnector
    """
    preneuron_ids = copy.copy(preneuron_ids)
    postneuron_ids = copy.copy(postneuron_ids)
    syntypes = copy.copy(syntypes)
    presyn_neurons: List = shadow_neurons if virtual_neurons is None else virtual_neurons

    neuronconnector.add_connection_from_list(
        [presyn_neurons[i] for i in preneuron_ids],
        [shadow_neurons[i] for i in postneuron_ids],
        syntypes,
    )

    print("dynapse_control: {} connections have been set.".format(len(preneuron_ids)))


@teleport_function
def _define_silence_neurons():
    @local_arguments
    def silence_neurons(neuron_ids):
        """
        silence_neurons - Assign time contant tau2 to neurons definedin neuron_ids
                          to make them silent.
        :param neuron_ids:  list  IDs of neurons to be silenced
        """
        if isinstance(neuron_ids, int):
            neuron_ids = (neuron_ids,)
        neurons_per_chip = NUM_CORES_CHIP * NUM_NEURONS_CORE
        for id_neur in neuron_ids:
            CtxDynapse.dynapse.set_tau_2(
                id_neur // neurons_per_chip,  # Chip ID
                id_neur % neurons_per_chip,  # Neuron ID on chip
            )
        print("dynapse_control: Set {} neurons to tau 2.".format(len(neuron_ids)))

    return silence_neurons


_silence_neurons = correct_argument_types(_define_silence_neurons())


@teleport_function
def _define_reset_silencing():
    @local_arguments
    def reset_silencing(core_ids):
        """
        reset_silencing - Assign time constant tau1 to all neurons on cores defined
                          in core_ids. Convenience function that does the same as
                          global _reset_silencing but also updates self._is_silenced.
        :param core_ids:   list  IDs of cores to be reset
        """
        if isinstance(core_ids, int):
            core_ids = (core_ids,)
        for id_neur in core_ids:
            CtxDynapse.dynapse.reset_tau_1(
                id_neur // NUM_CORES_CHIP,  # Chip ID
                id_neur % NUM_CORES_CHIP,  # Core ID on chip
            )
        print("dynapse_control: Set neurons of cores {} to tau 1.".format(core_ids))

    return reset_silencing


_reset_silencing = correct_argument_types(_define_reset_silencing())


# - Clear hardware configuration at startup
print("dynapse_control: Initializing hardware.")
if not _USE_RPYC or "bInitialized" not in conn.namespace.keys():
    _clear_chips(chip_ids)
    conn.namespace["bInitialized"] = True
    print("dynapse_control: Hardware initialized.")
else:
    print("dynapse_control: Hardware has already been initialized.")


class DynapseControl:

    _sram_event_limit = SRAM_EVENT_LIMIT
    _fpga_event_limit = FPGA_EVENT_LIMIT
    _fpga_isi_limit = FPGA_ISI_LIMIT
    _fpga_timestep = FPGA_TIMESTEP
    _num_neur_core = NUM_NEURONS_CORE
    _num_cores_chip = NUM_CORES_CHIP
    _num_chips = NUM_CHIPS

    def __init__(
        self,
        fpga_isibase: float = DEF_FPGA_ISI_BASE,
        clearcores_list: Optional[list] = None,
    ):
        """
        DynapseControl - Class for interfacing DynapSE

        :param fpga_isibase:    float           Time step for inter-spike intervals when sending events to FPGA
        :param clearcores_list:     list or None    IDs of cores where configurations should be cleared.
        """

        print("DynapseControl: Initializing DynapSE")

        # - Chip model and virtual model
        self.model = CtxDynapse.model
        self.virtual_model = CtxDynapse.VirtualModel()

        # - dynapse object from CtxDynapse
        self.dynapse = dynapse

        ## -- Modules for sending input to FPGA
        fpga_modules = self.model.get_fpga_modules()

        # - Find a spike generator module
        is_spikegen: List[bool] = [
            isinstance(m, DynapseFpgaSpikeGen) for m in fpga_modules
        ]
        if not np.any(is_spikegen):
            # There is no spike generator, so we can't use this Python layer on the HW
            raise ModuleNotFoundError(
                "DynapseControl: An `fpga_spikegen` module is required to use the DynapSE layer."
            )
        else:
            # Get first spike generator module
            self.fpga_spikegen = fpga_modules[np.argwhere(is_spikegen)[0][0]]
            print("DynapseControl: Spike generator module ready.")

        # - Find a poisson spike generator module
        is_poissongen: List[bool] = [
            isinstance(m, DynapsePoissonGen) for m in fpga_modules
        ]
        if np.any(is_poissongen):
            self.fpga_poissongen = fpga_modules[np.argwhere(is_poissongen)[0][0]]
        else:
            warn(
                "DynapseControl: Could not find poisson generator module (DynapsePoissonGen)."
            )

        # - Get all neurons from models
        self.hw_neurons, self.virtual_neurons, self.shadow_neurons = get_all_neurons(
            self.model, self.virtual_model
        )

        # - Initialise neuron allocation
        self.hwneurons_isfree, self.virtualneurons_isfree = (
            self._initial_free_neuron_lists()
        )

        # - Store which neurons have been assigned tau 2 (i.e. are silenced)
        self._is_silenced = np.zeros_like(self.hwneurons_isfree, bool)
        # - Make sure no neuron is silenced, yet
        self.reset_silencing(range(16))

        print("DynapseControl: Neurons initialized.")
        print(
            "\t {} hardware neurons and {} virtual neurons available.".format(
                np.sum(self.hwneurons_isfree), np.sum(self.hwneurons_isfree)
            )
        )

        # - Get a connector object
        self.neuronconnector = nn_connector.DynapseConnector()
        print("DynapseControl: Neuron connector initialized")

        # - Wipe configuration
        self.clear_connections(clearcores_list)

        ## -- Initialize Fpga spike generator
        self.fpga_isibase = fpga_isibase
        self.fpga_spikegen.set_repeat_mode(False)
        self.fpga_spikegen.set_variable_isi(True)
        self.fpga_spikegen.set_base_addr(0)
        print("DynapseControl: FPGA spike generator prepared.")

        print("DynapseControl ready.")

    @staticmethod
    def clear_connections(core_ids: Optional[list] = None):
        """
        clear_connections - Reset connections for cores defined in core_ids.

        :param core_ids:  list or None  IDs of cores where configurations
                                         should be cleared (0-15).
        """
        # - Use `_reset_connections` function
        _reset_connections(core_ids)
        print(
            "DynapseControl: Connections to cores {} have been cleared.".format(
                core_ids
            )
        )

    def silence_neurons(self, neuron_ids: list):
        """
        silence_neurons - Assign time contant tau2 to neurons definedin neuron_ids
                          to make them silent. Convenience function that does the
                          same as global _silence_neurons but also stores silenced
                          neurons in self._is_silenced.
        :param neuron_ids:  list  IDs of neurons to be silenced
        """
        _silence_neurons(neuron_ids)
        # - Mark that neurons have been silenced
        self._is_silenced[neuron_ids] = True
        print("DynapseControl: {} neurons have been silenced.".format(len(neuron_ids)))

    def reset_silencing(self, core_ids: list):
        """
        reset_silencing - Assign time constant tau1 to all neurons on cores defined
                          in core_ids. Convenience function that does the same as
                          global _reset_silencing but also updates self._is_silenced.
        :param core_ids:   list  IDs of cores to be reset
        """
        if isinstance(core_ids, int):
            core_ids = (core_ids,)
        _reset_silencing(core_ids)
        # - Mark that neurons are not silenced anymore
        for id_neur in core_ids:
            self._is_silenced[
                id_neur * self.num_neur_core : (id_neur + 1) * self.num_neur_core
            ] = False
        print(
            "DynapseControl: Time constants of cores {} have been reset.".format(
                core_ids
            )
        )

    def reset_cores(
        self,
        core_ids: Optional[Union[list, int]] = None,
        virtual: bool = True,
        silence: bool = True,
    ):
        """
        reset_cores - Reset neuron assginments and connections for specified cores
                      and virtual neurons if requested.
        """
        # - Clear neuron assignments
        self.clear_neuron_assignments(core_ids, virtual)
        # - Reset connections
        self.clear_connections(core_ids)
        if silence:
            # - Reset neuron silencing
            self.reset_silencing(core_ids)

    def reset_all(self):
        """
        reset_all - Convenience function to reset neuron assginments and connections
                    for all cores and virtual neurons.
        """
        # - Clear neuron assignments
        self.reset_cores(range(self.num_cores), True)

    ### --- Neuron allocation and connections

    def _initial_free_neuron_lists(self) -> (np.ndarray, np.ndarray):
        """
        _initial_free_neuron_lists - Generate initial lit of free hardware and
                                     virtual neurons as boolean arrays.
        :return:
            hwneurons_isfree         np.ndarray  Boolean array indicating which hardware
                                                neurons are available
            virtualneurons_isfree    np.ndarray  Boolean array indicating which virtual
                                                neurons are available
        """
        # - Hardware neurons
        hwneurons_isfree = np.ones(len(self.hw_neurons), bool)
        # Do not use hardware neurons with ID 0 and core ID 0 (first of each core)
        hwneurons_isfree[0 :: self.num_neur_chip] = False

        # - Virtual neurons
        virtualneurons_isfree = np.ones(len(self.virtual_neurons), bool)
        # Do not use virtual neuron 0
        virtualneurons_isfree[0] = False

        return hwneurons_isfree, virtualneurons_isfree

    def clear_neuron_assignments(
        self, core_ids: Optional[Union[list, int]] = None, virtual: bool = True
    ):
        """
        clear_neuron_assignments - Mark neurons as free again.
        :param core_ids:   IDs of cores where neuron assignments should be reset.
                            If None, do not reset hardware neuron assignments.
        :param virtual:    Mark all virtual neurons as free (except neuron 0)
        """
        # - Original neuron availabilities
        freehwneurons0, freevirtualneurons0 = self._initial_free_neuron_lists()

        if core_ids is not None:
            # - Hardware neurons

            # Make core_ids iterable if integer
            if isinstance(core_ids, int):
                core_ids = [core_ids]

            for core_id in core_ids:
                start_clear_idx = core_id * self.num_neur_core
                end_clear_idx = start_clear_idx + self.num_neur_core
                self.hwneurons_isfree[start_clear_idx:end_clear_idx] = freehwneurons0[
                    start_clear_idx:end_clear_idx
                ]
            print(
                "DynapseControl: {} hardware neurons available.".format(
                    np.sum(self.hwneurons_isfree)
                )
            )

        if virtual:
            # - Virtual neurons
            self.virtualneurons_isfree = freevirtualneurons0
            print(
                "DynapseControl: {} virtual neurons available.".format(
                    np.sum(self.virtualneurons_isfree)
                )
            )

    def allocate_hw_neurons(
        self, neuron_ids: Union[int, np.ndarray]
    ) -> (np.ndarray, np.ndarray):
        """
        allocate_hw_neurons - If neuron_ids is ArrayLike, verify that neurons have not
                              been assigned yet and mark them as assigned and return a
                              list of corresponding hardware and shadow state neurons.
                              If neuron_ids is an integer, assing first n available neurons.

        :param neuron_ids:  int or np.ndarray    The number of neurons requested or IDs of requested neurons
        :return:             list                 A list of neurons that may be used
        """

        ## -- Choose neurons
        if isinstance(neuron_ids, int):
            # - Choose first available neurons
            num_neurons = neuron_ids
            # - Are there sufficient unallocated neurons?
            if np.sum(self.hwneurons_isfree) < num_neurons:
                raise ValueError(
                    "Insufficient unallocated neurons available. {} requested.".format(
                        num_neurons
                    )
                )
            else:
                # - Pick the first available neurons
                ids_neurons_to_allocate = np.nonzero(self.hwneurons_isfree)[0][
                    :num_neurons
                ]

        else:
            # - Choose neurons defined in neuron_ids
            ids_neurons_to_allocate = np.array(neuron_ids).flatten()
            # - Make sure neurons are available
            if (self.hwneurons_isfree[ids_neurons_to_allocate] == False).any():
                raise ValueError(
                    "{} of the requested neurons are already allocated.".format(
                        np.sum(self.hwneurons_isfree[ids_neurons_to_allocate] == False)
                    )
                )

        # - Mark these neurons as allocated
        self.hwneurons_isfree[ids_neurons_to_allocate] = False

        # - Prevent allocation of virtual neurons with same (logical) ID as allocated hardware neurons
        inptneur_overlap = ids_neurons_to_allocate[
            ids_neurons_to_allocate < np.size(self.virtualneurons_isfree)
        ]
        self.virtualneurons_isfree[inptneur_overlap] = False

        # - Return these allocated neurons
        return (
            np.array([self.hw_neurons[i] for i in ids_neurons_to_allocate]),
            np.array([self.shadow_neurons[i] for i in ids_neurons_to_allocate]),
        )

    def allocate_virtual_neurons(
        self, neuron_ids: Union[int, np.ndarray]
    ) -> np.ndarray:
        """
        allocate_virtual_neurons - If neuron_ids is ArrayLike, verify that virtual neurons have not
                                   been assigned yet and mark them as assigned and return a
                                   list of virtual neurons neurons.
                                   If neuron_ids is an integer, assing first n available virtual neurons.

        :param neuron_ids:  int or np.ndarray    The number of neurons requested or IDs of requested neurons
        :return:             list    A list of neurons that may be used
        """
        if isinstance(neuron_ids, int):
            num_neurons = neuron_ids
            # - Are there sufficient unallocated neurons?
            if np.sum(self.virtualneurons_isfree) < num_neurons:
                raise ValueError(
                    "Insufficient unallocated neurons available. {}".format(num_neurons)
                    + " requested."
                )
            # - Pick the first available neurons
            ids_neurons_to_allocate = np.nonzero(self.virtualneurons_isfree)[0][
                :num_neurons
            ]

        else:
            ids_neurons_to_allocate = np.array(neuron_ids).flatten()
            # - Make sure neurons are available
            if (self.virtualneurons_isfree[ids_neurons_to_allocate] == False).any():
                raise ValueError(
                    "{} of the requested neurons are already allocated.".format(
                        np.sum(
                            self.virtualneurons_isfree[ids_neurons_to_allocate] == False
                        )
                    )
                )

        # - Mark these as allocated
        self.virtualneurons_isfree[ids_neurons_to_allocate] = False
        # - Prevent allocation of hardware neurons with same ID as allocated virtual neurons
        #  IS THIS REALLY NECESSARY?
        self.hwneurons_isfree[ids_neurons_to_allocate] = False

        # - Return these neurons
        return np.array([self.virtual_neurons[i] for i in ids_neurons_to_allocate])

    def connect_to_virtual(
        self,
        virtualneuron_ids: Union[int, np.ndarray],
        neuron_ids: Union[int, np.ndarray],
        syntypes: List,
    ):
        """
        conncect_to_virtual - Connect a group of hardware neurons or
                              single hardware neuron to a group of
                              virtual neurons (1 to 1) or to a single
                              virtual neuron.
        :param virtualneuron_ids:   np.ndarray  IDs of virtual neurons
        :param neuron_ids:          np.ndarray  IDs of hardware neurons
        :param syntypes:        list        Types of the synapses
        """

        # - Handle single neurons
        if isinstance(virtualneuron_ids, int):
            virtualneuron_ids = [virtualneuron_ids for _ in range(np.size(neuron_ids))]
        if isinstance(neuron_ids, int):
            neuron_ids = [neuron_ids for _ in range(np.size(virtualneuron_ids))]
        if np.size(syntypes) == 1:
            syntypes = list(np.repeat(syntypes, np.size(neuron_ids)))
        else:
            syntypes = list(syntypes)

        # - Set connections
        set_connections(
            preneuron_ids=list(virtualneuron_ids),
            postneuron_ids=list(neuron_ids),
            syntypes=syntypes,
            shadow_neurons=self.shadow_neurons,
            virtual_neurons=self.virtual_neurons,
            neuronconnector=self.neuronconnector,
        )
        print("DynapseControl: Setting up {} connections".format(np.size(neuron_ids)))
        self.model.apply_diff_state()
        print("DynapseControl: Connections set")

    def set_virtual_connections_from_weights(
        self,
        weights: np.ndarray,
        virtualneuron_ids: np.ndarray,
        hwneuron_ids: np.ndarray,
        syn_exc: CtxDynapse.DynapseCamType,
        syn_inh: CtxDynapse.DynapseCamType,
        apply_diff: bool = True,
    ):
        """
        set_virtual_connections_from_weights - Set connections from virtual to hardware
                                               neurons based on discrete weight matrix
        :param weights:                 np.ndarray  Weights for connections from
                                                virtual to layer neurons
        :param virtualneuron_ids:  np.ndarray  Virtual neuron IDs
        :param hwneuron_ids:       np.ndarray  Hardware neuron IDs
        :param syn_exc:       DynapseCamType  Excitatory synapse type
        :param syn_inh:       DynapseCamType  Inhibitory synapse type
        :param apply_diff:          bool   If False, do not apply the changes to
                                           chip but only to shadow states of the
                                           neurons. Useful if new connections are
                                           going to be added to the given neurons.
        """

        # - Get connection lists
        presyn_exc_list, postsyn_exc_list, presyn_inh_list, postsyn_inh_list = connectivity_matrix_to_prepost_lists(
            weights.astype(int)
        )

        # - Extract neuron IDs and remove numpy wrapper around int type
        preneur_ids_exc = [int(virtualneuron_ids[i]) for i in presyn_exc_list]
        postneur_ids_exc = [int(hwneuron_ids[i]) for i in postsyn_exc_list]
        preneur_ids_inh = [int(virtualneuron_ids[i]) for i in presyn_inh_list]
        postneur_ids_inh = [int(hwneuron_ids[i]) for i in postsyn_inh_list]

        # - Set excitatory connections
        set_connections(
            preneuron_ids=preneur_ids_exc,
            postneuron_ids=postneur_ids_exc,
            syntypes=[syn_exc],
            shadow_neurons=self.shadow_neurons,
            virtual_neurons=self.virtual_neurons,
            neuronconnector=self.neuronconnector,
        )
        print(
            "DynapseControl: Excitatory connections of type `{}`".format(
                str(syn_exc).split(".")[1]
            )
            + " from virtual neurons to hardware neurons have been set."
        )
        # - Set inhibitory connections
        set_connections(
            preneuron_ids=preneur_ids_inh,
            postneuron_ids=postneur_ids_inh,
            syntypes=[syn_inh],
            shadow_neurons=self.shadow_neurons,
            virtual_neurons=self.virtual_neurons,
            neuronconnector=self.neuronconnector,
        )
        print(
            "DynapseControl: Inhibitory connections of type `{}`".format(
                str(syn_inh).split(".")[1]
            )
            + " from virtual neurons to hardware neurons have been set."
        )

        if apply_diff:
            self.model.apply_diff_state()
            print("DynapseControl: Connections have been written to the chip.")

    def set_connections_from_weights(
        self,
        weights: np.ndarray,
        hwneuron_ids: np.ndarray,
        syn_exc: CtxDynapse.DynapseCamType,
        syn_inh: CtxDynapse.DynapseCamType,
        apply_diff: bool = True,
    ):
        """
        set_connections_from_weights - Set connections between hardware neurons
                                       based on  discrete weight matrix
        :param weights:                 np.ndarray  Weights for connections between
                                                hardware neurons
        :param hwneuron_ids:       np.ndarray  Hardware neuron IDs
        :param syn_exc:       DynapseCamType  Excitatory synapse type
        :param syn_inh:       DynapseCamType  Inhibitory synapse type
        :param apply_diff:          bool   If False, do not apply the changes to
                                           chip but only to shadow states of the
                                           neurons. Useful if new connections are
                                           going to be added to the given neurons.
        """

        ## -- Connect virtual neurons to hardware neurons

        # - Get virtual to hardware connections
        presyn_exc_list, postsyn_exc_list, presyn_inh_list, postsyn_inh_list = connectivity_matrix_to_prepost_lists(
            weights.astype(int)
        )

        # - Extract neuron IDs and remove numpy wrapper around int type
        preneur_ids_exc = [int(hwneuron_ids[i]) for i in presyn_exc_list]
        postneur_ids_exc = [int(hwneuron_ids[i]) for i in postsyn_exc_list]
        preneur_ids_inh = [int(hwneuron_ids[i]) for i in presyn_inh_list]
        postneur_ids_inh = [int(hwneuron_ids[i]) for i in postsyn_inh_list]

        # - Set excitatory input connections
        set_connections(
            preneuron_ids=preneur_ids_exc,
            postneuron_ids=postneur_ids_exc,
            syntypes=[syn_exc],
            shadow_neurons=self.shadow_neurons,
            virtual_neurons=None,
            neuronconnector=self.neuronconnector,
        )
        print(
            "DynapseControl: Excitatory connections of type `{}`".format(
                str(syn_exc).split(".")[1]
            )
            + " between hardware neurons have been set."
        )
        # - Set inhibitory input connections
        set_connections(
            preneuron_ids=preneur_ids_inh,
            postneuron_ids=postneur_ids_inh,
            syntypes=[syn_inh],
            shadow_neurons=self.shadow_neurons,
            virtual_neurons=None,
            neuronconnector=self.neuronconnector,
        )
        print(
            "DynapseControl: Inhibitory connections of type `{}`".format(
                str(syn_inh).split(".")[1]
            )
            + " between hardware neurons have been set."
        )

        if apply_diff:
            self.model.apply_diff_state()
            print("DynapseControl: Connections have been written to the chip.")

    def remove_all_connections_to(self, neuron_ids, apply_diff: bool = True):
        """
        remove_all_connections_to - Remove all presynaptic connections
                                    to neurons defined in neuron_ids
        :param neuron_ids:     np.ndarray IDs of neurons whose presynaptic
                                           connections should be removed
        :param apply_diff:      bool If False do not apply the changes to
                                     chip but only to shadow states of the
                                     neurons. Useful if new connections are
                                     going to be added to the given neurons.
        """
        # - Make sure neurons neuron_ids is an array
        neuron_ids = [int(id_neur) for id_neur in np.asarray(neuron_ids)]

        # - Call `remove_all_connections_to` function
        remove_all_connections_to(neuron_ids, self.model, apply_diff)

    ### --- Stimulation and event generation

    def _arrays_to_spike_list(
        self,
        channels: np.ndarray,
        neuron_ids: np.ndarray,
        timesteps: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
        ts_start: Optional[int] = None,
        t_start: Optional[int] = 0,
        targetcore_mask: int = 1,
        targetchip_id: int = 0,
    ) -> List:
        """
        _arrays_to_spike_list - Convert an array of input time steps and an an array
                               of event channels to a ctxctl spike list

        :param channels:        np.ndarray   Event channels
        :param neuron_ids:      ArrayLike    IDs of neurons that should appear as sources of the events
        :param timesteps:       np.ndarray   Event time steps (Using FPGA time base, overwrites times if not None)
        :param times:           np.ndarray   Event time points (in seconds)
        :param ts_start:        int          Time step at which to start (overwrites t_start if not None)
        :param t_start:         float        Time at which to start
        :param targetcore_mask: int          Mask defining target cores (sum of 2**core_id)
        :param targetchip_id:   int          ID of target chip
        :return:                list of FpgaSpikeEvent objects
        """
        # - Process input arguments
        if timesteps is None:
            if times is None:
                raise ValueError(
                    "DynapseControl: Either `timesteps` or `times` must be provided."
                )
            timesteps = np.floor(times / self.fpga_isibase).astype(int)
        if ts_start is None:
            if t_start is None:
                raise ValueError(
                    "DynapseControl: Either `ts_start` or `t_start` must be provided."
                )
            ts_start = int(np.floor(t_start / self.fpga_isibase))

        # - Ignore data that comes before ts_start
        timesteps = timesteps[timesteps >= ts_start]
        channels = channels[timesteps >= ts_start]

        # - Check that the number of channels is the same between time series and list of neurons
        if np.amax(channels) > np.size(neuron_ids):
            raise ValueError(
                "DynapseControl: `channels` contains more channels than the number of neurons in `neuron_ids`."
            )

        # - Make sure neuron_ids is iterable
        neuron_ids = np.array(neuron_ids)

        # - Convert to ISIs
        discrete_isi_list = np.diff(np.r_[ts_start, timesteps])

        print("DynapseControl: Generating FPGA event list from arrays.")
        # - Convert events to an FpgaSpikeEvent
        events = generate_fpga_event_list(
            # Make sure that no np.int64 or other non-native type is passed
            [int(isi) for isi in discrete_isi_list],
            [int(neuron_ids[i]) for i in channels],
            int(targetcore_mask),
            int(targetchip_id),
        )

        # - Return a list of events
        return events

    def start_cont_stim(
        self, frequency: float, neuron_ids: int, chip_id: int = 0, coremask: int = 15
    ):
        """
        start_cont_stim - Start sending events with fixed frequency.
                          FPGA repeat mode will be set True.
        :param frequency:  float  Frequency at which events are sent
        :param neuron_ids:   int  Event neuron ID(s)
        :param chip_id:     int  Target chip ID
        :param coremask:   int  Target core mask
        """

        # - Set FPGA to repeating mode
        self.fpga_spikegen.set_repeat_mode(True)

        # - Handle integers for neuron_ids
        if isinstance(neuron_ids, int):
            neuron_ids = [neuron_ids]
        else:
            neuron_ids = [int(id_neur) for id_neur in neuron_ids]

        # - Interspike interval
        t_isi = 1.0 / frequency
        # - ISI in units of fpga time step
        fpga_isisteps = int(np.round(t_isi / self.fpga_isibase))
        # - ISI for neurons other than first is zero as they are simultaneous
        isistep_list = [fpga_isisteps] + [0] * (len(neuron_ids) - 1)

        # - Generate events
        # List for events to be sent to fpga
        events = generate_fpga_event_list(
            isistep_list, neuron_ids, int(coremask), int(chip_id)
        )
        self.fpga_spikegen.preload_stimulus(events)
        print(
            "DynapseControl: Stimulus prepared with {} Hz".format(
                1.0 / (fpga_isisteps * self.fpga_isibase)
            )
        )

        # - Start stimulation
        self.fpga_spikegen.start()
        print("DynapseControl: Stimulation started")

    def stop_stim(self, clear_filter: bool = False):
        """
        stop_stim - Stop stimulation with FGPA spke generator.
                    FPGA repeat mode will be set False.
        :param bStopRecording:  bool  Clear buffered event filter if present.
        """
        # - Stop stimulation
        self.fpga_spikegen.stop()
        # - Set default FPGA settings
        self.fpga_spikegen.set_repeat_mode(False)
        print("DynapseControl: Stimulation stopped")
        if clear_filter:
            self.clear_buffered_event_filter()

    def start_poisson_stim(
        self,
        frequencies: Union[int, np.ndarray],
        neuron_ids: Union[int, np.ndarray],
        chip_id: int = 0,
    ):
        """
        start_poisson_stim - Start generating events by poisson processes and send them.
        :param vfFrequency: int or array-like  Frequencies of poisson processes
        :param neuron_ids: int or array-like  Event neuron ID(s)
        :param chip_id:     int  Target chip ID
        """

        # - Handle single values for frequencies and neurons
        if np.size(neuron_ids) == 1:
            neuron_ids = np.repeat(neuron_ids, np.size(frequencies))
        if np.size(frequencies) == 1:
            frequencies = np.repeat(frequencies, np.size(neuron_ids))
        else:
            if np.size(frequencies) != np.size(neuron_ids):
                raise ValueError(
                    "DynapseControl: Length of `frequencies must be same as length of `neuron_ids` or 1."
                )

        # - Set firing rates for selected neurons
        for frequency, neuron_id in zip(frequencies, neuron_ids):
            self.fpga_poissongen.write_poisson_rate_hz(neuron_id, frequency)

        # - Set chip ID
        self.fpga_poissongen.set_chip_id(chip_id)

        print("DynapseControl: Poisson stimuli prepared for chip {}.".format(chip_id))

        # - Start stimulation
        self.fpga_poissongen.start()
        print("DynapseControl: Poisson rate stimulation started")

    def stop_poisson_stim(self):
        """
        stop_stim - Stop stimulation with FGPA poisson generator.
                    Does not stop any event recording.
        """
        self.fpga_poissongen.stop()
        print("DynapseControl: Poisson rate stimulation stopped")

    def reset_poisson_rates(self):
        """reset_poisson_rates - Set all firing rates of poisson generator to 0."""
        for i in range(1024):
            self.fpga_poissongen.write_poisson_rate_hz(i, 0)
        print("DynapseControl: Firing rates for poisson generator have been set to 0.")

    def send_pulse(
        self,
        width: float = 0.1,
        frequency: float = 1000,
        t_record: float = 3,
        t_buffer: float = 0.5,
        inputneur_id: int = 0,
        record_neur_ids: Union[int, np.ndarray] = np.arange(1024),
        targetcore_mask: int = 15,
        targetchip_id: int = 0,
        periodic: bool = False,
        record: bool = False,
    ) -> (np.ndarray, np.ndarray):
        """
        send_pulse - Send a pulse of periodic input events to the chip.
                     Return a arrays with times and channels of the recorded hardware activity.
        :param width:              float  Duration of the input pulse
        :param frequency:               float  Frequency of the events that constitute the pulse
        :param t_record:             float  Duration of the recording (including stimulus)
        :param t_buffer:             float  Record slightly longer than t_record to
                                           make sure to catch all relevant events
        :param inputneur_id:      int    ID of input neuron
        :param record_neur_ids:   array-like  ID(s) of neuron(s) to be recorded
        :param chip_id:     int  Target chip ID
        :param coremask:   int  Target core mask
        :param periodic:   bool    Repeat the stimulus indefinitely
        :param record:     bool    Set up buffered event filter that records events
                                    from neurons defined in record_neur_ids

        :return:
            (times, channels)  np.ndarrays that contain recorded data
        """
        # - Prepare input events
        # Actual input time steps
        timesteps = np.floor(
            np.arange(0, width, 1.0 / frequency) / self.fpga_isibase
        ).astype(int)
        # Add dummy events at end to avoid repeated stimulation due to "2-trigger-bug"
        isilimit = self.fpga_isi_limit * self.fpga_isibase
        num_add = int(np.ceil(t_record / isilimit))
        timesteps = np.r_[
            timesteps, timesteps[-1] + np.arange(1, num_add + 1) * self.fpga_isi_limit
        ]

        events = self._arrays_to_spike_list(
            timesteps=timesteps,
            channels=np.repeat(inputneur_id, timesteps.size),
            neuron_ids=range(len(self.virtual_neurons)),
            ts_start=0,
            targetcore_mask=targetcore_mask,
            targetchip_id=targetchip_id,
        )
        # - Do not send dummy events to any core
        for event in events[-num_add:]:
            event.core_mask = 0
        print("DynapseControl: Stimulus pulse prepared")

        # - Stimulate and return recorded data if any
        return self._send_stimulus_list(
            events=events,
            duration=t_record,
            t_buffer=t_buffer,
            record_neur_ids=record_neur_ids,
            periodic=periodic,
            record=record,
        )

    def send_arrays(
        self,
        channels: np.ndarray,
        timesteps: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
        t_record: Optional[float] = None,
        t_buffer: float = 0.5,
        neuron_ids: Optional[np.ndarray] = None,
        record_neur_ids: Optional[np.ndarray] = None,
        targetcore_mask: int = 15,
        targetchip_id: int = 0,
        periodic=False,
        record=False,
    ) -> (np.ndarray, np.ndarray):
        """
        send_arrays - Send events defined in arrays to FPGA.

        :param channels:      np.ndarray  Event channels
        :param vnTimeSeops:     np.ndarray  Event times in Fpga time base (overwrites times if not None)
        :param times:     np.ndarray  Event times in seconds
        :param t_record:         float  Duration of the recording (including stimulus)
                                       If None, use times[-1]
        :param t_buffer:         float  Record slightly longer than t_record to
                                       make sure to catch all relevant events
        :param neuron_ids:     ArrayLike    IDs of neurons that should appear as sources of the events
                                             If None, use channels from channels
        :param record_neur_ids: ArrayLike    IDs of neurons that should be recorded (if record==True)
                                               If None and record==True, record neurons in neuron_ids
        :param targetcore_mask: int          Mask defining target cores (sum of 2**core_id)
        :param targetchip_id:   int          ID of target chip
        :param periodic:       bool         Repeat the stimulus indefinitely
        :param record:         bool         Set up buffered event filter that records events
                                             from neurons defined in neuron_ids

        :return:
            (times, channels)  np.ndarrays that contain recorded data
        """

        # - Process input arguments
        neuron_ids = (
            np.arange(np.amax(channels) + 1)
            if neuron_ids is None
            else np.array(neuron_ids)
        )
        record_neur_ids = neuron_ids if record_neur_ids is None else record_neur_ids
        if t_record is None:
            try:
                t_record = times[-1]
            except TypeError:  # times is None
                try:
                    t_record = timesteps[-1] * self.fpga_isibase
                except TypeError:  # timesteps is also None
                    raise ValueError(
                        "DynapseControl: Either `timesteps` or `times` has to be provided."
                    )
            print(
                "DynapseControl: Stimulus/recording time inferred to be {} s.".format(
                    t_record
                )
            )

        # - Prepare event list
        events = self._arrays_to_spike_list(
            times=times,
            timesteps=timesteps,
            channels=channels,
            neuron_ids=neuron_ids,
            ts_start=0,
            targetcore_mask=targetcore_mask,
            targetchip_id=targetchip_id,
        )
        print("DynapseControl: Stimulus prepared from arrays.")
	# - Stimulate and return recorded data if any
        return self._send_stimulus_list(
            events=events,
            duration=t_record,
            t_buffer=t_buffer,
            record_neur_ids=record_neur_ids,
            periodic=periodic,
            record=record,
        )

    def _send_stimulus_list(
        self,
        events,
        duration,
        t_buffer,
        record_neur_ids: Optional[np.ndarray] = None,
        periodic: bool = False,
        record: bool = False,
    ):
        """
        _send_stimulus_list - Send a list of FPGA events to hardware. Possibly record hardware events.

        :param events:           list   List of FpgaSpikeEvent objects to be sent to hardware
        :param duration:         float  Duration of the stimulation and recording
                                         If None, record indefinitely
        :param t_buffer:           float  Record slightly longer than duration to
                                         make sure to catch all relevant events
        :param record_neur_ids: ArrayLike    IDs of neurons that should be recorded (if record==True)
                                               If None and record==True, no neurons will be recorded
        :param periodic:       bool         Repeat the stimulus indefinitely
        :param record:         bool         Set up buffered event filter that records events
                                             from neurons defined in record_neur_ids

        :return:
            (times, channels)  np.ndarrays that contain recorded data
        """
        # - Throw an exception if event list is too long
        if len(events) > self.fpga_event_limit:
            raise ValueError(
                "DynapseControl: events can have at most {} elements (has {}).".format(
                    self.fpga_event_limit, len(events)
                )
            )

        # - Prepare FPGA
        self.fpga_spikegen.set_repeat_mode(periodic)
        self.fpga_spikegen.preload_stimulus(events)
        print("DynapseControl: Stimulus preloaded.")
        if record:
            if record_neur_ids is None:
                record_neur_ids = []
                warn("DynapseControl: No neuron IDs specified for recording.")
            self.add_buffered_event_filter(record_neur_ids)

        # - Lists for storing collected events
        timestamps_full = []
        channels_full = []
        triggerevents = []

        print(
            "DynapseControl: Starting{} stimulation{}.".format(
                periodic * " periodic", (not periodic) * " for {} s".format(duration)
            )
        )
        # - Clear event filter
        self.bufferedfilter.get_events()
        self.bufferedfilter.get_special_event_timestamps()

        # Time at which stimulation stops, including buffer
        t_stop = time.time() + duration + (0.0 if t_buffer is None else t_buffer)

        # - Stimulate
        self.fpga_spikegen.start()

        if (duration is None) or periodic:
            # - Keep running indefinitely
            return

        # - Until duration is over, record events and process in quick succession
        while time.time() < t_stop:
            if record:
                # - Collect events and possibly trigger events
                triggerevents += self.bufferedfilter.get_special_event_timestamps()
                current_events = self.bufferedfilter.get_events()

                timestamps_curr, channels_curr = event_data_to_channels(
                    current_events, record_neur_ids
                )
                timestamps_full += list(timestamps_curr)
                channels_full += list(channels_curr)

        print("DynapseControl: Stimulation ended.")

        if record:
            self.bufferedfilter.clear()
            print(
                "\tRecorded {} event(s) and {} trigger event(s)".format(
                    len(timestamps_full), len(triggerevents)
                )
            )
            return self._process_extracted_events(
                timestamps=timestamps_full,
                channels=channels_full,
                triggerevents=triggerevents,
                duration=duration,
            )

    def _process_extracted_events(
        self, timestamps: List, channels: List, triggerevents: List, duration: float
    ) -> (np.array, np.array):

        # - Post-processing of collected events
        times = np.asarray(timestamps) * 1e-6
        channels = np.asarray(channels)

        if times.size == 0:
            return np.array([]), np.array([])

        # - Locate synchronisation timestamp
        triggertimes = np.array(triggerevents) * 1e-6
        start_indices = np.searchsorted(times, triggertimes)
        end_indices = np.searchsorted(times, triggertimes + duration)
        # - Choose first trigger where start and end indices not equal. If not possible, take first trigger
        try:
            trigger_id = np.argmax((end_indices - start_indices) > 0)
            print("\t\t Using trigger event {}".format(trigger_id))
        except ValueError:
            print("\t\t No Trigger found, using recording from beginning")
            idx_start = 0
            t_trigger = times[0]
            idx_end = np.searchsorted(times, times[0] + duration)
        else:
            t_trigger = triggertimes[trigger_id]
            idx_start = start_indices[trigger_id]
            idx_end = end_indices[trigger_id]
        # - Filter time trace
        times_out = times[idx_start:idx_end] - t_trigger
        channels_out = channels[idx_start:idx_end]
        print("DynapseControl: Extracted event data")

        return times_out, channels_out

    def _recorded_data_to_arrays(
        self, neuron_ids: np.ndarray, t_record: float
    ) -> (np.ndarray, np.ndarray):
        events = self.bufferedfilter.get_events()
        triggerevents = self.bufferedfilter.get_special_event_timestamps()

        print(
            "DynapseControl: Recorded {} event(s) and {} trigger event(s)".format(
                len(events), len(triggerevents)
            )
        )

        # - Extract monitored event channels and timestamps
        timestamps, channels = event_data_to_channels(events, neuron_ids)

        return self._process_extracted_events(
            timestamps=timestamps,
            channels=channels,
            triggerevents=triggerevents,
            duration=t_record,
        )

    ### --- Tools for tuning and observing activities

    def add_buffered_event_filter(self, neuron_ids):
        """
        add_buffered_event_filter - Add a BufferedEventFilter to record from
                                    neurons defined in neuron_ids
        :param neuron_ids:   array-like  IDs of neurons to be recorded
        :return:
            Reference to selfbufferedfilter, the BufferedEventFilter that has been created.
        """
        # - Convert neuron_ids to list
        if isinstance(neuron_ids, int):
            record_neuron_ids = list(range(neuron_ids))
        else:
            record_neuron_ids = list(neuron_ids)

        # - Does a filter already exist?
        if hasattr(self, "bufferedfilter") and self.bufferedfilter is not None:
            self.bufferedfilter.clear()
            self.bufferedfilter.add_ids(record_neuron_ids)
            print("DynapseControl: Updated existing buffered event filter.")
        else:
            self.bufferedfilter = _generate_buffered_filter(
                self.model, record_neuron_ids
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

    def collect_spiking_neurons(self, neuron_ids, duration):
        """
        collect_spiking_neurons - Return a list of IDs of neurons that
                                  that spike within duration
        :param neuron_ids:   list   IDs of neurons to be observed.
        :param duration:     float  How long to wait for spiking neurons
        :return  recordedneuron_ids:  list IDs of neurons that have spiked.
        """

        # - Convert neuron_ids to list
        if isinstance(neuron_ids, int):
            neuron_ids = range(neuron_ids)

        print(
            "DynapseControl: Collecting IDs of neurons that spike within the next {} seconds".format(
                duration
            )
        )

        # - Filter for recording neurons
        eventfilter = self.add_buffered_event_filter(neuron_ids)

        # - Wait and record spiking neurons
        time.sleep(duration)

        eventfilter.clear()

        # - Sorted unique list of neurons' IDs that have spiked
        recordedneuron_ids = sorted(
            set((event.neuron.get_id() for event in eventfilter.get_events()))
        )
        print(
            "DynapseControl: {} neurons spiked: {}".format(
                len(recordedneuron_ids), recordedneuron_ids
            )
        )

        return recordedneuron_ids

    def silence_hot_neurons(
        self, neuron_ids: Union[list, np.ndarray], duration: float
    ) -> list:
        """
        silence_hot_neurons - Collect IDs of all neurons that spike
                              within duration. Assign them different
                              time constant to silence them.
        :param neuron_ids:  list   IDs of neurons to be observed.
        :param duration:    float  How long to wait for spiking neurons
        :return:
            hotneuron_ids    list  IDs of hot neurons that have been silenced.
        """
        # - Neurons that spike within duration
        hotneuron_ids = self.collect_spiking_neurons(neuron_ids, duration=duration)
        # - Silence these neurons by assigning different Tau bias
        print("DynapseControl: Neurons {} will be silenced".format(hotneuron_ids))
        self.silence_neurons(hotneuron_ids)
        return hotneuron_ids

    def measure_population_firing_rates(
        self, llnPopulationIDs: list, duration: float, verbose=False
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        measure_population_firing_rates - Measure the mean, maximum and minimum
                                          firing rates for multiple neuron populatios
        :param llnPopulationIDs:  Array-like of array-like of neuron IDs of each pouplaiton
        :param duration:         float  Time over which rates are measured
        :param verbose:          bool   Print firing rates of each neuron for each population

        :return:
            rates_mean     np.ndarray - Population mean rates
            rates_max     np.ndarray - Population maximum rates
            rates_min     np.ndarray - Population minimum rates
        """

        # - Arrays for collecting rates
        nNumPopulations = np.size(llnPopulationIDs)
        rates_mean = np.zeros(nNumPopulations)
        rates_max = np.zeros(nNumPopulations)
        rates_min = np.zeros(nNumPopulations)

        for i, neuron_ids in enumerate(llnPopulationIDs):
            print("DynapseControl: Population {}".format(i))
            firing_rates, rates_mean[i], rates_max[i], rates_min[
                i
            ] = self.measure_firing_rates(neuron_ids, duration)
            if verbose:
                print(firing_rates)

        return rates_mean, rates_max, rates_min

    def measure_firing_rates(
        self, neuron_ids: Optional[Union[int, np.ndarray]], duration: float
    ) -> (np.ndarray, float, float, float):
        """
        measure_firing_rates - Measure the mean, maximum and minimum firing rate
                               for a group of neurons
        :param neuron_ids:     Array-like or int  Neuron IDs to be measured
        :param duration:       float  Time over which rates are measured

        :return:
            firing_rates  np.ndarray - Each neuron's firing rate
            rate_mean      float - Average firing rate over all neurons
            rate_max       float - Highest firing rate of all neurons
            rate_min       float - Lowest firing rate of all neurons
        """
        if isinstance(neuron_ids, int):
            neuron_ids = [neuron_ids]
        # - Filter for recording events
        eventfilter = self.add_buffered_event_filter(neuron_ids)
        # - Record events for duration
        time.sleep(duration)
        # - Stop recording
        eventfilter.clear()
        # - Retrieve recorded events
        events = eventfilter.get_events()
        if not events:
            # - Handle empty event lists
            print("DynapseControl: No events recorded")
            return np.zeros(np.size(neuron_ids)), 0, 0, 0
        # - Evaluate non-empty event lists
        return evaluate_firing_rates(events, duration, neuron_ids)

    def _monitor_firing_rates_inner(self, neuron_ids, interval):
        """
        monitor_firing_rates - Continuously monitor the activity of a population
                               and periodically output the average firing rate.
        """

        self._is_monitoring = True
        # - Set up event filter
        self.add_buffered_event_filter(neuron_ids)
        print(
            "DynapseControl: Start monitoring firing rates of {} neurons.".format(
                np.size(neuron_ids)
            )
        )
        while self._is_monitoring:
            # - Flush event stack
            self.bufferedfilter.get_events()
            # - Collect events
            time.sleep(interval)
            events = self.bufferedfilter.get_events()
            # - Process events
            if not events:
                # - Handle empty event lists
                print("DynapseControl: No events recorded")
            else:
                # - Evaluate non-empty event lists
                rate_mean = evaluate_firing_rates(
                    events, interval, neuron_ids, verbose=False
                )[2]
                print("DynapseControl: Mean firing rate: {} Hz".format(rate_mean))

    def monitor_firing_rates(self, neuron_ids, interval):
        """
        monitor_firing_rates - Create a thread that continuously monitors the
                               activity of a population and periodically output
                               the average firing rate.
        """

        self.thread_monitor = threading.Thread(
            target=self._monitor_firing_rates_inner,
            kwargs={"neuron_ids": neuron_ids, "interval": interval},
        )
        self.thread_monitor.start()

    def stop_monitor(self):
        self._is_monitoring = False
        self.bufferedfilter.clear()
        self.thread_monitor.join(timeout=5)
        del self.thread_monitor
        print("DynapseControl: Stopped monitoring.")

    def sweep_freq_measure_rate(
        self,
        frequencies: list = [1, 10, 20, 50, 100, 200, 500, 1000, 2000],
        duration: float = 1,
        targetneuron_ids: Union[int, np.ndarray] = range(128),
        inputneuron_ids: Union[int, np.ndarray] = 1,
        chip_id: int = 0,
        coremask: int = 15,
    ):
        """
        sweep_freq_measure_rate - Stimulate a group of neurons by sweeping
                                  over a list of input frequencies. Measure
                                  their firing rates.
        :param frequencies:      array-like Stimulus frequencies
        :param duration:   float  Stimulus duration for each frequency
        :param targetneuron_ids: array-like  source neuron IDs
        :param inputneuron_ids:  array-like  IDs of neurons
        :param chip_id:     int  Target chip ID
        :param coremask:   int  Target core mask

        :return:
            firingrates_2d   np.ndarray -  Matrix containing firing rates for each neuron (axis 1)
                                          for each frequency (axis 0)
            rates_mean     np.ndarray - Average firing rates over all neurons for each input frequency
            rates_max      np.ndarray - Highest firing rates of all neurons for each input frequency
            rates_min      np.ndarray - Lowest firing rates of all neurons for each input frequency
        """

        # - Arrays for collecting firing rates
        firingrates_2d = np.zeros((np.size(frequencies), np.size(targetneuron_ids)))
        rates_mean = np.zeros(np.size(frequencies))
        rates_max = np.zeros(np.size(frequencies))
        rates_min = np.zeros(np.size(frequencies))

        # - Sweep over frequencies
        for iTrial, frequency in enumerate(frequencies):
            print("DynapseControl: Stimulating with {} Hz input".format(frequency))
            self.start_cont_stim(frequency, inputneuron_ids, chip_id, coremask)
            firingrates_2d[iTrial, :], rates_mean[iTrial], rates_max[iTrial], rates_min[
                iTrial
            ] = self.measure_firing_rates(targetneuron_ids, duration)
            self.stop_stim()

        return firingrates_2d, rates_mean, rates_max, rates_min

    ### - Load and save biases

    @staticmethod
    def load_biases(filename, core_ids: Optional[Union[list, int]] = None):
        """
        load_biases - Load biases from python file under path filename.
                      Convenience function. Same as global load_biases.
        :param filename:  str  Path to file where biases are stored.
        :param core_ids:    list, int or None  IDs of cores for which biases
                                                should be loaded. Load all if
                                                None.
        """
        load_biases(os.path.abspath(filename), core_ids)
        print("DynapseControl: Biases have been loaded from {}.".format(filename))

    @staticmethod
    def save_biases(filename, core_ids: Optional[Union[list, int]] = None):
        """
        save_biases - Save biases in python file under path filename
                      Convenience function. Same as global save_biases.
        :param filename:  str  Path to file where biases should be saved.
        :param core_ids:    list, int or None  ID(s) of cores whose biases
                                                should be saved. If None,
                                                save all cores.
        """
        save_biases(os.path.abspath(filename), core_ids)
        print("DynapseControl: Biases have been saved under {}.".format(filename))

    @staticmethod
    def copy_biases(sourcecore_id: int = 0, targetcore_ids: Optional[List[int]] = None):
        """
        copy_biases - Copy biases from one core to one or more other cores.
                      Convenience function. Same as global copy_biases.
        :param sourcecore_id:   int  ID of core from which biases are copied
        :param vnTargetCoreIDs: int or array-like ID(s) of core(s) to which biases are copied
                                If None, will copy to all other neurons
        """
        copy_biases(sourcecore_id, targetcore_ids)
        print(
            "DynapseControl: Biases have been copied from core {} to core(s) {}".format(
                sourcecore_id, targetcore_ids
            )
        )

    ### --- Class properties

    @property
    def syn_exc_slow(self):
        return SynapseTypes.SLOW_EXC

    @property
    def syn_inh_slow(self):
        return SynapseTypes.SLOW_INH

    @property
    def syn_exc_fast(self):
        return SynapseTypes.FAST_EXC

    @property
    def syn_inh_fast(self):
        return SynapseTypes.FAST_INH

    @property
    def num_neur_core(self):
        return self._num_neur_core

    @property
    def num_cores_chip(self):
        return self._num_cores_chip

    @property
    def num_neur_chip(self):
        return self._num_cores_chip * self._num_neur_core

    @property
    def num_chips(self):
        return self._num_chips

    @property
    def num_cores(self):
        return self._num_chips * self.num_cores_chip

    @property
    def num_neurons(self):
        return self._num_chips * self.num_neur_chip

    @property
    def is_silenced(self):
        return self._is_silenced

    @property
    def indices_silenced(self):
        return np.where(self._is_silenced)[0]

    @property
    def sram_event_limit(self):
        return self._sram_event_limit

    @property
    def fpga_event_limit(self):
        return self._fpga_event_limit

    @property
    def fpga_isi_limit(self):
        return self._fpga_isi_limit

    @property
    def fpga_timestep(self):
        return self._fpga_timestep

    @property
    def fpga_isibase(self):
        return self._nFpgaIsiMultiplier * self.fpga_timestep

    @fpga_isibase.setter
    def fpga_isibase(self, tNewBase):
        if not tNewBase > self.fpga_timestep:
            raise ValueError(
                "DynapseControl: `fpga_timestep` must be at least {}".format(
                    self.fpga_timestep
                )
            )
        else:
            self._nFpgaIsiMultiplier = int(np.floor(tNewBase / self.fpga_timestep))
            self.fpga_spikegen.set_isi_multiplier(self._nFpgaIsiMultiplier)
