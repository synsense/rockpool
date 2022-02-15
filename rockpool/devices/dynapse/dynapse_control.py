# ----
# dynapse_control.py - Module to interface cortexcontrol and the DynapSE chip
# Author: Felix Bauer, SynSense AG, felix.bauer@synsense.ai
# ----

### --- Imports

raise ImportError("This module needs to be ported to the v2 API")

import copy
from warnings import warn
from typing import Tuple, List, Optional, Union, Iterable, Set
import time
import os
import threading

import rockpool.devices.dynapse.params as params

import numpy as np

# - Global settings
_USE_DEEPCOPY = False
RPYC_TIMEOUT = 300
# - Default values, can be changed
DEF_FPGA_ISI_BASE = 2e-5  # Default timestep between events sent to FPGA
DEF_FPGA_ISI_MULTIPLIER = int(np.round(DEF_FPGA_ISI_BASE / params.FPGA_TIMESTEP))
USE_CHIPS = []  # Chips to be initialized for use


## -- Import cortexcontrol modules or establish connection via RPyC
try:
    import CtxDynapse as ctxdynapse
    import NeuronNeuronConnector as nnconnector
    import tools
    import params

    _USE_RPYC = False
except ModuleNotFoundError:
    # - Try with RPyC
    import rpyc

    _USE_RPYC = True


def connect_rpyc(port: Union[int, str, None] = None) -> "rpyc.core.protocol.Connection":
    """
    connect_rpyc - Establish a connection through RPyC.
    :param port:  Port over which to connect. If `None`, try 1300 and then, if
                  this fails, 1301.
    :return:
        The new RPyC connection
    """
    if not _USE_RPYC:
        raise RuntimeError(
            "dynapse_control: Connection to RPyC only possible from outside cortexcontrol"
        )

    try:
        if port is None:
            try:
                port = "1300"
                connection = rpyc.classic.connect("localhost", "1300")
            except (TimeoutError, ConnectionRefusedError):
                # - Connection with port 1300 is either occupied or does not exist
                port = "1300 or 1301"  # This is just for print statements
                connection = rpyc.classic.connect("localhost", "1301")
                port = "1301"
        else:
            connection = rpyc.classic.connect("localhost", str(port))
    except TimeoutError:
        raise TimeoutError(
            f"dynapse_control: RPyC connection on port {port} seems to be in use already."
        )
    except ConnectionRefusedError:
        raise ConnectionRefusedError(
            f"dynapse_control: No available RPyC connection on port {port}."
        )
    else:
        print(f"dynapse_control: RPyC connection established through port {port}.")

    # - Set up rpyc connection settings
    connection._config["sync_request_timeout"] = RPYC_TIMEOUT  # Increase timeout limit

    return connection


def setup_rpyc_namespace(connection: "rpyc.core.protocol.Connection"):
    """
    setup_rpyc_namespace - Register certain modules and variables in the namespace
                           of cortexcontrol that will be needed for this module.
    :param connection:  RPyC connection to cortexcontrol.
    """
    # - Setup parameters on RPyC server
    connection.namespace["_USE_RPYC"] = True
    connection.namespace["copy"] = connection.modules.copy
    connection.namespace["os"] = connection.modules.os
    connection.namespace["ctxdynapse"] = connection.modules.CtxDynapse
    connection.namespace["rpyc"] = connection.modules.rpyc
    connection.namespace["tools"] = connection.modules.tools
    connection.namespace["params"] = connection.modules.params
    if "initialized_chips" not in connection.modules.tools.storage.keys():
        connection.modules.tools.store_var("initialized_chips", [])
    else:
        print(
            "dynapse_control: Already initialized chips: {}".format(
                connection.modules.tools.storage["initialized_chips"]
            )
        )
    if "initialized_neurons" not in connection.modules.tools.storage.keys():
        connection.modules.tools.store_var("initialized_neurons", [])

    print("dynapse_control: RPyC namespace complete.")


def initialize_hardware(
    use_chips: List = [],
    connection: Optional["rpyc.core.protocol.Connection"] = None,
    enforce: bool = False,
):
    """
    initialize_hardware - If chips have not been cleared since start of current
                          cortexcontrol instance, clear them.
    :param use_chips:  List with IDs of chips that should be checked.
    :param enforce:    If `True`, clear all chips in `use_chips`, no matter if
                       they have been cleared already.
    """
    if isinstance(use_chips, int):
        use_chips = [use_chips]
    else:
        # - Convert to rpyc-safe format
        use_chips = [int(chip) for chip in use_chips]
    print("dynapse_control: Initializing hardware...", end="\r")
    if not _USE_RPYC:
        tools.init_chips(use_chips)
        # - Update lists of initialized chips and neurons
        global initialized_chips, initialized_neurons
        cleared_chips = set(use_chips).difference(initialized_chips)
        initialized_chips = sorted(initialized_chips + cleared_chips)
        cleard_neurons = [
            neuron
            for chip in cleared_chips
            for neuron in range(
                params.NUM_NEURONS_CHIP * chip, params.NUM_NEURONS_CHIP * (chip + 1)
            )
        ]
        initialized_neurons = sorted(initialized_neurons + cleard_neurons)
        if not use_chips:
            print(f"dynapse_control: No chips have been cleared.")
        elif len(use_chips == 1):
            print(f"dynapse_control: Chip `{use_chips[0]}` has been cleared.")
        else:
            print(
                "dynapse_control: Chips `{}` and `{}` have been cleared.".format(
                    "`, `".join((str(chip) for chip in use_chips[:-1])), use_chips[-1]
                )
            )
    else:
        # - If using rpyc, make sure, connection is given
        if connection is None:
            raise TypeError(
                "dynapse_control: If using RPyC, `connection` argument cannot be `None`."
            )
        else:
            # - Chips that have already been initialized. If `initialized_chips`
            #   doesn't exist, assume that no chips have been initialized yet.
            initialized_chips = connection.modules.tools.storage.get(
                "initialized_chips", []
            )
            # - Find chips that are to be used and have not been initialized yet.
            if enforce:
                do_chips = use_chips
            else:
                do_chips = list(set(use_chips).difference(initialized_chips))
            already_done = list(set(use_chips).difference(do_chips))
            # - Clear those chips and add them to list of initialized chips.
            connection.modules.tools.init_chips(do_chips)
            connection.modules.tools.store_var(
                "initialized_chips",
                # `initialized_chips` needs to be copied to local environment for concatenation
                [int(ch) for ch in np.unique(copy.copy(initialized_chips) + do_chips)],
            )
            # - Also update list of initialized neurons
            initialized_neurons = connection.modules.tools.storage.get(
                "initialized_neurons", []
            )
            cleared_neurons = [
                neuron_id
                for chip_id in do_chips
                for neuron_id in range(
                    params.NUM_NEURONS_CHIP * chip_id,
                    params.NUM_NEURONS_CHIP * (chip_id + 1),
                )
            ]
            connection.modules.tools.store_var(
                "initialized_neurons",
                [
                    int(n)
                    for n in np.unique(copy.copy(initialized_neurons) + cleared_neurons)
                ],
            )
            # - Print which chips have been cleared.
            print_statement = "dynapse_control: Chips {} have been cleared.".format(
                ", ".join((str(chip) for chip in do_chips))
            )
            if already_done:
                print_statement += " Chips {} had been cleared previously.".format(
                    ", ".join((str(chip) for chip in already_done))
                )
            print(print_statement)


def setup_rpyc(
    connect: Union["rpyc.core.protocol.Connection", int, str, None] = None,
    init_chips: Union[List[int], None] = USE_CHIPS,
) -> "rpyc.core.protocol.Connection":
    """
    setup_rpyc - Connect to cortexcontrol via RPyC, add entries to cortexcontrol
                 namespace, register global variables to access objects from whithin
                 this module and clear chips.
    :param connect:  If RPyC connection, use this instead of establishing new connection
                     If int (or string that can be converted to int), connect through
                     corresponding port
                     If `None`, first try connecting through port 1300. If this fails try
                     through port 1301.
    :param init_chips:  Chips to be initialized (cleared). Chips that have already been
                        cleared since the start of conrtexcontrol will not be cleared again.
    """
    connection = (
        connect
        if isinstance(connect, rpyc.core.protocol.Connection)
        else connect_rpyc(connect)
    )
    setup_rpyc_namespace(connection)
    if init_chips is not None:
        initialize_hardware(init_chips, connection, enforce=False)
    # - Make same objects available as when working wihtin cortexcontrol
    global tools, params, ctxdynapse, nnconnector
    tools = connection.modules.tools
    params = connection.modules.params
    ctxdynapse = connection.modules.CtxDynapse
    nnconnector = connection.modules.NeuronNeuronConnector
    print("dynapse_control: RPyC connection has been setup successfully.")

    return connection


def connectivity_matrix_to_prepost_lists(
    weights: np.ndarray,
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
    core_dimensions: tuple = params.CORE_DIMENSIONS,
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
    try:
        timestamps, event_neuron_ids = tools.extract_event_data(events)
    except NameError:
        raise NameError(
            "dynapse_control: `tools` module is not defined. "
            + "Run `setup_rpyc` with your RPyC connection as argument."
        )
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

    try:
        timestamps, neuron_ids = tools.extract_event_data(events)
    except NameError:
        raise NameError(
            "dynapse_control: `tools` module is not defined. "
            + "Run `setup_rpyc` with your RPyC connection as argument."
        )
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


def correct_argument_types_and_teleport(conn, func):
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
                             arguments of a function to types that are
                             supported by cortexcontrol
    :param func:    function whose arguments should be corrected
    :return:        function with possibly corrected arguments
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


# # - Example on how to use `correct_argument_types`:
# generate_fpga_event_list = correct_argument_types(tools.generate_fpga_event_list)


def teleport_function(conn, func):
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


def remote_function(conn, func):
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


class DynapseControl:

    _sram_event_limit = params.SRAM_EVENT_LIMIT
    _fpga_event_limit = params.FPGA_EVENT_LIMIT
    _fpga_isi_limit = params.FPGA_ISI_LIMIT
    _fpga_isi_multiplier_limit = params.FPGA_ISI_LIMIT + 1
    _fpga_timestep = params.FPGA_TIMESTEP
    _num_neur_core = params.NUM_NEURONS_CORE
    _num_cores_chip = params.NUM_CORES_CHIP
    _num_chips = params.NUM_CHIPS
    _default_cam_str = params.DEF_CAM_STR

    def __init__(
        self,
        fpga_isibase: float = DEF_FPGA_ISI_BASE,
        clearcores_list: Optional[list] = None,
        rpyc_connection: Union[None, str, int, "rpyc.core.protocol.Connection"] = None,
        init_chips: Optional[List] = None,
        prevent_aliasing: bool = True,
    ):
        """
        DynapseControl - Class for interfacing DynapSE

        :param fpga_isibase:     Time step for inter-spike intervals when sending events to FPGA
        :param clearcores_list:  IDs of cores where configurations should be cleared.
                                 Corresponding chips will automatically be added to `init_chips`.
        :param rpyc_connection:  RPyC connection to cortexcontrol, port over which to connect.
                                 If `None`, try establish connection through port 1300 or, if
                                 unsuccessful, 1301.
        :param init_chips:       Chips to be cleared, if they haven't been cleared since start
                                 of cortexcontrol instance.
        :param prevent_aliasing: Throw an exception when updating connections would result
                                 in connection aliasing.
        """

        if clearcores_list is not None:
            # - Add chips corresponding to cores in `clearcores_list` to `init_chips`
            clearchips = set(
                (core_id // self.num_cores_chip for core_id in clearcores_list)
            )
            init_chips = (
                list(clearchips.union(init_chips))
                if init_chips is not None
                else list(clearchips)
            )

        # - Store settings
        self.prevent_aliasing = prevent_aliasing

        # - Store pointer to ctxdynapse and nnconnector modules
        if _USE_RPYC:
            # - Set up connection. Make sure rpyc namespace is complete and hardware initialized.
            self.rpyc_connection = setup_rpyc(rpyc_connection, init_chips=init_chips)
            self.tools = self.rpyc_connection.modules.tools
        else:
            self.rpyc_connection = None
            if init_chips:
                initialize_hardware(init_chips)

        print("DynapseControl: Initializing DynapSE")

        # - Chip model, virtual model and dynapse
        self.model = ctxdynapse.model
        self.virtual_model = ctxdynapse.VirtualModel()
        self.dynapse = ctxdynapse.dynapse

        ## -- Modules for sending input to FPGA
        fpga_modules = self.model.get_fpga_modules()

        # - Find a spike generator module
        is_spikegen: List[bool] = [
            isinstance(m, ctxdynapse.DynapseFpgaSpikeGen) for m in fpga_modules
        ]
        if not any(is_spikegen) or all(is_spikegen):
            is_spikegen: List[bool] = [
                # Trying type(.)==... because isinstance seems to confuse types when using RPyC in some cases
                type(m) == type(ctxdynapse.DynapseFpgaSpikeGen)
                for m in fpga_modules
            ]
            if not any(is_spikegen) or all(is_spikegen):
                # There is no spike generator, so we can't use this Python layer on the HW
                raise RuntimeError(
                    "DynapseControl: Could not reliably determine fpga spike generator module (DynapseFpgaSpikeGen)."
                )
        # Get first spike generator module
        self.fpga_spikegen = fpga_modules[np.argwhere(is_spikegen)[0][0]]
        print("DynapseControl: Spike generator module ready.")

        # - Find a poisson spike generator module
        is_poissongen: List[bool] = [
            isinstance(m, ctxdynapse.DynapsePoissonGen) for m in fpga_modules
        ]
        if (not any(is_poissongen)) or all(is_poissongen):
            is_poissongen: List[bool] = [
                # Doing type(.)==... because isinstance seems to confuse types when using RPyC in some cases
                type(m) == type(ctxdynapse.DynapsePoissonGen)
                for m in fpga_modules
            ]
            if not any(is_poissongen) or all(is_poissongen):
                warn(
                    "DynapseControl: Could not find poisson generator module (DynapsePoissonGen)."
                )
            else:
                self.fpga_poissongen = fpga_modules[np.argwhere(is_poissongen)[0][0]]
                print("DynapseControl: Poisson generator module ready.")
        else:
            self.fpga_poissongen = fpga_modules[np.argwhere(is_poissongen)[0][0]]
            print("DynapseControl: Poisson generator module ready.")

        # - Get all neurons from models
        (
            self.hw_neurons,
            self.virtual_neurons,
            self.shadow_neurons,
        ) = self.tools.get_all_neurons(self.model, self.virtual_model)

        # - Initialise neuron allocation
        (
            self._hwneurons_isfree,
            self._virtualneurons_isfree,
        ) = self._initial_free_neuron_lists()

        # - Store which neurons have been assigned tau 2 (i.e. are silenced)
        self._is_silenced = np.zeros_like(self._hwneurons_isfree, bool)
        # - Make sure no neuron is silenced, yet
        core_ids = [
            i
            for chip in self.initialized_chips
            for i in range(chip * self.num_cores_chip, (chip + 1) * self.num_cores_chip)
        ]
        self.reset_silencing(core_ids)

        print("DynapseControl: Neurons initialized.")
        print(
            "\t {} hardware neurons and {} virtual neurons available.".format(
                np.sum(self.hwneurons_isavailable),
                np.sum(self.virtualneurons_isavailable),
            )
        )

        # - Get a connector object
        self.connector = nnconnector.DynapseConnector()
        print("DynapseControl: Neuron connector initialized")

        # - Dict to map cam types to 0-axis of self._connections
        self._camtypes = {
            getattr(ctxdynapse.DynapseCamType, camtype): i
            for i, camtype in enumerate(params.CAMTYPES)
        }
        # - ID of default cam type to which cams are reset
        def_camtype = getattr(ctxdynapse.DynapseCamType, self._default_cam_str)
        self._default_cam_type_index = self._camtypes[def_camtype]
        # - Store SRAM information
        self._sram_connections = np.zeros((self.num_neurons, self.num_cores), bool)
        # - Store CAM information
        self._cam_connections = np.zeros(
            (len(self._camtypes), self.num_neur_chip, self.num_neurons),
            "uint8",
        )
        # - Store connectivity array
        self._connections = np.zeros(
            (len(params.CAMTYPES), self.num_neurons, self.num_neurons),
            "uint8",
        )
        # Include previously existing connections in the model
        self._update_connectivity_array(self.initialized_chips)
        print("DynapseControl: Connectivity array initialized")

        # - Wipe configuration
        self.clear_connections(clearcores_list)

        ## -- Initialize Fpga spike generator
        self.fpga_isibase = fpga_isibase
        self.fpga_spikegen.set_repeat_mode(False)
        self.fpga_spikegen.set_variable_isi(True)
        self.fpga_spikegen.set_base_addr(0)
        print("DynapseControl: FPGA spike generator prepared.")

        print("DynapseControl ready.")

    def init_chips(self, chips: Optional[List[int]] = None, enforce: bool = True):
        """
        init_chips - Clear chips with given IDs. If `enforce` is False, only clear
                           those chips that have not been cleared since the start of the
                           current `cortexcontrol` instance.
        :param chips:    List with IDs of chips that have to be cleared.
                         If `None`, do nothing.
        :param enforce:  If `False`, only clear those chips that have not been cleared
                         since the start of the current `cortexcontrol` instance.
                         Otherwise clear all given chips.
        """
        if chips is not None:
            initialize_hardware(chips, self.rpyc_connection, enforce=enforce)
            print(
                "DynapseControl: {} hardware neurons available.".format(
                    np.sum(self.hwneurons_isavailable)
                )
            )

    def clear_connections(
        self,
        core_ids: Optional[List[int]] = None,
        presynaptic: bool = True,
        postsynaptic: bool = True,
        apply_diff=True,
    ):
        """
        clear_connections -   Remove pre- and/or postsynaptic connections of all nerons
                              of cores defined in core_ids.
        :param core_ids:      IDs of cores to be reset (between 0 and 15)
        :param presynaptic:   Remove presynaptic connections to neurons on specified cores.
        :param postsynaptic:  Remove postsynaptic connections of neurons on specified cores.
        :param apply_diff:    Apply changes to hardware. Setting False is useful
                              if new connections will be set afterwards.
        """
        if core_ids is None:
            return

        # - Handle non-list arguments for core_ids
        if not isinstance(core_ids, list):
            try:
                # - Make sure core_ids is in rpyc-compatible format
                core_ids = [int(idc) for idc in core_ids]
            except TypeError:
                if isinstance(core_ids, int):
                    core_ids = [core_ids]
                else:
                    raise TypeError(
                        "DynapseControl: `core_ids` should be of type `int` or `List[int]`."
                    )
        # - Use `reset_connections` function
        self.tools.reset_connections(core_ids, presynaptic, postsynaptic, apply_diff)
        print(
            "DynapseControl: Connections to cores {} have been cleared.".format(
                np.array(core_ids, int)
            )
        )

        neuron_ids = [
            i
            for cid in core_ids
            for i in range(cid * self.num_neur_core, (cid + 1) * self.num_neur_core)
        ]
        if postsynaptic:
            # - Reset internal representation of SRAM cells
            self._reset_srams(neuron_ids)
        if presynaptic:
            # - Reset internal representation of CAM cells
            self._reset_cams(neuron_ids)
        self._connections = self._extract_connections_from_memory(
            self._sram_connections, self._cam_connections
        )

    def silence_neurons(self, neuron_ids: list):
        """
        silence_neurons - Assign time contant tau2 to neurons definedin neuron_ids
                          to make them silent. Convenience function that does the
                          same as global _silence_neurons but also stores silenced
                          neurons in self._is_silenced.
        :param neuron_ids:  list  IDs of neurons to be silenced
        """
        self.tools.silence_neurons(neuron_ids)
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
        if core_ids is not None:
            core_ids = np.array(core_ids)
            # - Only reset cores on chips that have been initialized
            on_init_chip = np.isin(
                core_ids // self.num_cores_chip, self.initialized_chips
            )
            if not on_init_chip.all():
                warn(
                    "DynapseControl: Cores {} are on chips that have not been initialized.".format(
                        core_ids[on_init_chip == False]
                    )
                )
            core_ids = [int(i_core) for i_core in core_ids[on_init_chip]]
        self.tools.reset_silencing(core_ids)
        # - Mark that neurons are not silenced anymore
        for id_neur in core_ids:
            self._is_silenced[
                id_neur * self.num_neur_core : (id_neur + 1) * self.num_neur_core
            ] = False
        print(
            "DynapseControl: Time constants of cores {} have been reset.".format(
                np.array(core_ids, int)
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
        if core_ids is not None:
            core_ids = np.array(core_ids)
            # - Only reset cores on chips that have been initialized
            on_init_chip = np.isin(
                core_ids // self.num_cores_chip, self.initialized_chips
            )
            if not on_init_chip.all():
                warn(
                    "DynapseControl: Cores {} are on chips that have not been initialized.".format(
                        core_ids[on_init_chip == False]
                    )
                )
            core_ids = core_ids[on_init_chip]
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
        # - Cores on initialized chips
        core_ids = [
            i
            for chip in self.initialized_chips
            for i in range(chip * self.num_cores_chip, (chip + 1) * self.num_cores_chip)
        ]
        # - Clear neuron assignments
        self.reset_cores(core_ids, True)

    ### --- Neuron allocation and connections

    def _initial_free_neuron_lists(self) -> (np.ndarray, np.ndarray):
        """
        _initial_free_neuron_lists - Generate initial lit of free hardware and
                                     virtual neurons as boolean arrays.
        :return:
            _hwneurons_isfree         np.ndarray  Boolean array indicating which hardware
                                                neurons are available
            _virtualneurons_isfree    np.ndarray  Boolean array indicating which virtual
                                                neurons are available
        """
        # - Hardware neurons
        _hwneurons_isfree = np.ones(len(self.hw_neurons), bool)
        # Do not use hardware neurons with ID 0 and core ID 0 (first of each core)
        _hwneurons_isfree[0 :: self.num_neur_chip] = False

        # - Virtual neurons
        _virtualneurons_isfree = np.ones(len(self.virtual_neurons), bool)
        # Do not use virtual neuron 0
        _virtualneurons_isfree[0] = False

        return _hwneurons_isfree, _virtualneurons_isfree

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
                self._hwneurons_isfree[start_clear_idx:end_clear_idx] = freehwneurons0[
                    start_clear_idx:end_clear_idx
                ]
            print(
                "DynapseControl: {} hardware neurons available.".format(
                    np.sum(self.hwneurons_isavailable)
                )
            )

        if virtual:
            # - Virtual neurons
            self._virtualneurons_isfree = freevirtualneurons0
            print(
                "DynapseControl: {} virtual neurons available.".format(
                    np.sum(self.virtualneurons_isavailable)
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
            num_available_neurons = np.sum(self.hwneurons_isavailable)
            if num_available_neurons < num_neurons:
                num_missing_neurs = num_neurons - num_available_neurons
                # - Check if by clearing chips, enough neurons can be made available
                num_missing_chips = int(np.ceil(num_missing_neurs / self.num_neur_chip))
                num_unused_chips = self.num_chips - len(self.initialized_chips)
                if num_unused_chips >= num_missing_chips:
                    print(
                        "DynapseControl: Not sufficient neurons available. Initializing "
                        + " chips to make more neurons available."
                    )
                    # - Initialize chips so that enough neurons are available
                    all_chips: Set[int] = set(range(self.num_chips))
                    unused_chips: Set[int] = all_chips.difference(
                        self.initialized_chips
                    )
                    self.init_chips(list(unused_chips)[:num_missing_chips])
                else:
                    raise ValueError(
                        "Insufficient unallocated neurons available "
                        + f"({num_available_neurons}, requested: {num_neurons})."
                    )
            # - Pick the first available neurons
            ids_neurons_to_allocate = np.nonzero(self.hwneurons_isavailable)[0][
                :num_neurons
            ]

        else:
            # - Choose neurons defined in neuron_ids
            ids_neurons_to_allocate = np.array(neuron_ids).flatten()
            # - Make sure neurons have not been allocated already
            already_allocated = self._hwneurons_isfree[ids_neurons_to_allocate] == False
            if already_allocated.any():
                raise ValueError(
                    "Some of the requested neurons have already been allocated: {}".format(
                        ids_neurons_to_allocate[already_allocated]
                    )
                )
            else:
                # - Check whether any of the requested neurons is on a not initialized chip
                not_initialized = (
                    np.isin(ids_neurons_to_allocate, self.initialized_neurons) == False
                )
                if not_initialized.any():
                    print(
                        "DynapseControl: For some of the requested neurons, "
                        + "chips need to be prepared."
                    )
                    missing_neurons = ids_neurons_to_allocate[not_initialized]
                    missing_chips = np.unique(missing_neurons // self.num_neur_chip)
                    self.init_chips(missing_chips)

        # - Mark these neurons as allocated
        self._hwneurons_isfree[ids_neurons_to_allocate] = False

        # - Prevent allocation of virtual neurons with same (logical) ID as allocated hardware neurons
        inptneur_overlap = ids_neurons_to_allocate[
            ids_neurons_to_allocate < np.size(self._virtualneurons_isfree)
        ]
        self._virtualneurons_isfree[inptneur_overlap] = False

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
            if np.sum(self.virtualneurons_isavailable) < num_neurons:
                raise ValueError(
                    "Insufficient unallocated neurons available. {}".format(num_neurons)
                    + " requested."
                )
            # - Pick the first available neurons
            ids_neurons_to_allocate = np.nonzero(self.virtualneurons_isavailable)[0][
                :num_neurons
            ]

        else:
            ids_neurons_to_allocate = np.array(neuron_ids).flatten()
            # - Make sure neurons are available
            num_unavailable_neurs = np.sum(
                self.virtualneurons_isavailable[ids_neurons_to_allocate] == False
            )
            if num_unavailable_neurs > 0:
                raise ValueError(
                    "{} of the requested neurons are already allocated.".format(
                        num_unavailable_neurs
                    )
                )

        # - Mark these as allocated
        self._virtualneurons_isfree[ids_neurons_to_allocate] = False
        # - Prevent allocation of hardware neurons with same ID as allocated virtual neurons
        #  IS THIS REALLY NECESSARY?
        self._hwneurons_isfree[ids_neurons_to_allocate] = False

        # - Return these neurons
        return np.array([self.virtual_neurons[i] for i in ids_neurons_to_allocate])

    def _update_memory_cells(self, consider_chips: Optional[List] = None):
        """
        _update_memory_cells - Extract cam and sram data from shadow neurons and
                               store it in arrays
        :param consider_chips:  List-like with Chips to be updated. If `None`,
                                update for initialized chips.
        """
        # - Determine which chips are to be considered for update
        if consider_chips is None:
            consider_chips = self.initialized_chips
        else:
            try:
                consider_chips: List[int] = [int(chip_id) for chip_id in consider_chips]
            except TypeError:
                consider_chips: Tuple = (int(consider_chips),)

        if len(consider_chips) == 0:
            return

        # - Get connection information for specified chips
        connection_info = tools.get_connection_info(consider_chips)
        neuron_ids, targetcore_lists, inputid_lists, camtype_lists = connection_info

        # - Reset SRAM and CAM info for considered neurons
        self._sram_connections[neuron_ids, :] = False
        self._cam_connections[:, :, neuron_ids] = 0

        # - Update SRAM info
        for pre, post in zip(neuron_ids, targetcore_lists):
            self._sram_connections[pre, post] = True
        # - Update CAM info
        for id_post, ids_pre, syntypes in zip(neuron_ids, inputid_lists, camtype_lists):
            for id_pre, type_pre in zip(ids_pre, syntypes):
                self._cam_connections[type_pre, id_pre, id_post] += 1

    def _extract_connections_from_memory(self, sram, cam):

        # - Expand SRAM info
        # Expand from cores to number of neurons
        sram_conns = np.repeat(sram, self.num_neur_core, axis=1)
        # Repeat for each connection type
        sram_conns = np.repeat((sram_conns,), len(self._camtypes), axis=0)

        # - Expand CAM info
        cam_conns = np.concatenate(self.num_chips * (cam,), axis=1)

        return cam_conns * sram_conns

    def _update_connectivity_array(self, consider_chips: Optional[List] = None):

        # - Update sram and cam information for considered chips
        self._update_memory_cells(consider_chips)

        self._connections = self._extract_connections_from_memory(
            self._sram_connections, self._cam_connections
        )

    def add_connections_to_virtual(
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
        self.tools.set_connections(
            preneuron_ids=list(virtualneuron_ids),
            postneuron_ids=list(neuron_ids),
            syntypes=syntypes,
            shadow_neurons=self.shadow_neurons,
            virtual_neurons=self.virtual_neurons,
            connector=self.connector,
        )
        print("DynapseControl: Setting up {} connections".format(np.size(neuron_ids)))
        self.model.apply_diff_state()
        print("DynapseControl: Connections set")

        # - Update internal representation of CAM cells
        for syntype, pre_id, post_id in zip(syntypes, virtualneuron_ids, neuron_ids):
            self._cam_connections[self._camtypes[syntype], pre_id, post_id] += 1
            # - Reduce number of cam cells that are set to default
            self._cam_connections[self._default_cam_type_index, 0, post_id] -= 1

    def set_connections_from_weights(
        self,
        weights: np.ndarray,
        neuron_ids: np.ndarray,
        neuron_ids_post: Optional[np.ndarray] = None,
        syn_exc: Optional["ctxdynapse.DynapseCamType"] = None,
        syn_inh: Optional["ctxdynapse.DynapseCamType"] = None,
        virtual_pre: bool = False,
        apply_diff: bool = True,
        prevent_aliasing: Optional[bool] = None,
    ):
        """
        set_connections_from_weights - Set connections between two neuron
                                       populations based on discrete weight
                                       matrix. Previous connections to the postsyn.
                                       neurons are removed. If 'neuron_ids_post'
                                       is not provided, the connections are
                                       recurrent. Connect to virtual neurons by
                                       setting `virtual_pre` True.

        :param weights:     Weights for connections between neuron populations
        :param neuron_ids:  Neuron IDs of presynaptic population.
        :param neuron_ids_post:  Neuron IDs of presynaptic population. If `None`,
        :param syn_exc:     Excitatory synapse type. Default: syn_exc_fast
        :param syn_inh:     Inhibitory synapse type. Default: syn_exc_fast
                                 use 'neuron_ids' (i.e. recurrent connections).
        :param virtual_pre:  If `True`, presynaptic neurons are virtual. In this
                             case, `neuron_ids_post` cannot be `None`.
        :param apply_diff:  If False, do not apply the changes to chip but only
                            to shadow states of the neurons. Useful if new
                            connections are going to be added to the given neurons.
        :param prevent_aliasing:  Throw an exception if setting connections would
                                  result in connection aliasing. If `None` use
                                  instance setting (`self.prevent_aliasing`).
        """
        # - Remove existing connections
        ids_remove = neuron_ids if neuron_ids_post is None else neuron_ids_post
        self.remove_all_connections_to(ids_remove, apply_diff=False)

        # - Set new connections
        try:
            self.add_connections_from_weights(
                weights=weights,
                neuron_ids=neuron_ids,
                syn_exc=syn_exc,
                syn_inh=syn_inh,
                neuron_ids_post=neuron_ids_post,
                virtual_pre=virtual_pre,
                apply_diff=apply_diff,
                prevent_aliasing=prevent_aliasing,
            )
        except ValueError as e:
            raise ValueError(
                str(e)
                + f"Note that all all connections to neurons {ids_remove} have been "
                + "removed in this process!"
            )

    def add_connections_from_weights(
        self,
        weights: np.ndarray,
        neuron_ids: np.ndarray,
        neuron_ids_post: Optional[np.ndarray] = None,
        syn_exc: Optional["ctxdynapse.DynapseCamType"] = None,
        syn_inh: Optional["ctxdynapse.DynapseCamType"] = None,
        virtual_pre: bool = False,
        apply_diff: bool = True,
        prevent_aliasing: Optional[bool] = None,
    ):
        """
        add_connections_from_weights - Add connections between two neuron
                                       populations based on discrete weight
                                       matrix. If 'neuron_ids_post' is not
                                       provided, the connections are recurrent.
                                       Connect to virtual neurons by setting
                                       `virtual_pre` True.

        :param weights:     Weights for connections between neuron populations
        :param neuron_ids:  Neuron IDs of presynaptic population.
        :param syn_exc:     Excitatory synapse type. Default: syn_exc_fast
        :param syn_inh:     Inhibitory synapse type. Default: syn_exc_fast
        :param neuron_ids_post:  Neuron IDs of presynaptic population. If `None`,
                                 use 'neuron_ids' (i.e. recurrent connections).
        :param virtual_pre:  If `True`, presynaptic neurons are virtual. In this
                             case, `neuron_ids_post` cannot be `None`.
        :param apply_diff:  If False, do not apply the changes to chip but only
                            to shadow states of the neurons. Useful if new
                            connections are going to be added to the given neurons.
        :param prevent_aliasing:  Throw an exception if setting connections would
                                  result in connection aliasing. If `None` use
                                  instance setting (`self.prevent_aliasing`).
        """

        # - Resolve synapse types
        syn_exc = self.syn_exc_fast if syn_exc is None else syn_exc
        syn_inh = self.syn_inh_fast if syn_inh is None else syn_inh

        ## -- Connect virtual neurons to hardware neurons
        weights = np.atleast_2d(weights).astype("int16")

        # - Get virtual to hardware connections
        (
            presyn_exc_list,
            postsyn_exc_list,
            presyn_inh_list,
            postsyn_inh_list,
        ) = connectivity_matrix_to_prepost_lists(weights)

        if neuron_ids_post is None:
            if virtual_pre:
                # - Cannot connect virtual neurons to themselves.
                raise ValueError(
                    "DynapseControl: For setting virtual connections, `neuron_ids_post`"
                    + " cannot be `None`."
                )
            # - Pre- and postsynaptic populations are the same (recurrent connections)
            neuron_ids_post = neuron_ids

        # - Extract neuron IDs and remove numpy wrapper around int type
        preneur_ids_exc = [int(neuron_ids[i]) for i in presyn_exc_list]
        postneur_ids_exc = [int(neuron_ids_post[i]) for i in postsyn_exc_list]
        preneur_ids_inh = [int(neuron_ids[i]) for i in presyn_inh_list]
        postneur_ids_inh = [int(neuron_ids_post[i]) for i in postsyn_inh_list]

        # - Update representations of SRAMs
        sram_connections = self._sram_connections.copy()
        if not virtual_pre:
            postneur_cores_exc = np.asarray(postneur_ids_exc) // self.num_neur_core
            postneur_cores_inh = np.asarray(postneur_ids_inh) // self.num_neur_core
            if len(preneur_ids_exc) > 0:
                sram_connections[preneur_ids_exc, postneur_cores_exc] = True
            if len(preneur_ids_inh) > 0:
                sram_connections[preneur_ids_inh, postneur_cores_inh] = True
        # - For CAMs perform loop because connections may be repeated
        cam_connections = self._cam_connections.copy()
        preneur_ids_chip_exc = np.asarray(preneur_ids_exc) % self.num_neur_chip
        for id_pre, id_post in zip(preneur_ids_chip_exc, postneur_ids_exc):
            cam_connections[self._camtypes[syn_exc], id_pre, id_post] += 1
            # - Reduce number of cam cells that are set to default
            cam_connections[self._default_cam_type_index, 0, id_post] -= 1
        preneur_ids_chip_inh = np.asarray(preneur_ids_inh) % self.num_neur_chip
        for id_pre, id_post in zip(preneur_ids_chip_inh, postneur_ids_inh):
            cam_connections[self._camtypes[syn_inh], id_pre, id_post] += 1
            # - Reduce number of cam cells that are set to default
            cam_connections[self._default_cam_type_index, 0, id_post] -= 1

        # - Make sure no aliasing occurs
        target_connections = self.connections.copy()
        idcs_pre, idcs_post = np.meshgrid(neuron_ids, neuron_ids_post, indexing="ij")
        conns_exc = np.clip(weights, 0, None).astype("uint8")
        conns_inh = np.abs(np.clip(weights, None, 0)).astype("uint8")
        if not virtual_pre:
            target_connections[
                self._camtypes[syn_exc], idcs_pre, idcs_post
            ] += conns_exc
            target_connections[
                self._camtypes[syn_inh], idcs_pre, idcs_post
            ] += conns_inh
        new_connections = self._extract_connections_from_memory(
            sram_connections, cam_connections
        )

        if (new_connections != target_connections).any():
            affected_ids = np.where(new_connections != target_connections)
            affected_pairs = np.vstack(affected_ids).T
            aliasing_warning = (
                "DynapseControl: Setting the provided connections will result in "
                + "connection aliasing for the following neuron pairs: "
                + ", ".join(str(tuple(pair[1:])) for pair in affected_pairs)
            )
            if prevent_aliasing is None:
                prevent_aliasing = self.prevent_aliasing
            if prevent_aliasing:
                raise ValueError(aliasing_warning)
            else:
                warn(aliasing_warning)

        # - Set excitatory input connections
        self.tools.set_connections(
            preneuron_ids=preneur_ids_exc,
            postneuron_ids=postneur_ids_exc,
            syntypes=[syn_exc],
            shadow_neurons=self.shadow_neurons,
            virtual_neurons=self.virtual_neurons if virtual_pre else None,
            connector=self.connector,
        )
        print(
            f"DynapseControl: Excitatory connections of type `{str(syn_exc).split('.')[1]}`"
            + f" between {virtual_pre * 'virtual and '}hardware neurons have been set."
        )

        # - Set inhibitory input connections
        self.tools.set_connections(
            preneuron_ids=preneur_ids_inh,
            postneuron_ids=postneur_ids_inh,
            syntypes=[syn_inh],
            shadow_neurons=self.shadow_neurons,
            virtual_neurons=self.virtual_neurons if virtual_pre else None,
            connector=self.connector,
        )
        print(
            f"DynapseControl: Inhibitory connections of type `{str(syn_inh).split('.')[1]}`"
            + f" between {virtual_pre * 'virtual and '}hardware neurons have been set."
        )

        if apply_diff:
            self.model.apply_diff_state()
            print("DynapseControl: Connections have been written to the chip.")

        # - Apply updates to connection and memory representations
        #   (only after connections habe been updated successfully)
        self._sram_connections = sram_connections
        self._cam_connections = cam_connections
        self._connections = new_connections

    def _reset_cams(self, neuron_ids: Iterable[int]):
        """
        Reset internal representation of CAM cells for given neurons to default
        NOTE that connections matrix needs to be updated after this! (`_extract_connections_from_memory`)
        :param neuron_ids:  List-like with IDs of neurons whose CAMs should be reset.
        """
        # - Clear all connections for given neurons
        self._cam_connections[:, :, neuron_ids] = 0
        # - Default setting for cams is slow_exc to preneuron 0
        self._cam_connections[self._default_cam_type_index, 0, neuron_ids] = 64

    def _reset_srams(self, neuron_ids: Iterable[int]):
        """
        Reset internal representation of SRAM cells for given neurons to default
        NOTE that connections matrix needs to be updated after this! (`_extract_connections_from_memory`)
        :param neuron_ids:  List-like with IDs of neurons whose SRAMs should be reset.
        """
        # - Clear all SRAM cells
        self._sram_connections[neuron_ids, :] = False

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
        self.tools.remove_all_connections_to(neuron_ids, self.model, apply_diff)
        # - Reset internal representation of CAM cells
        self._reset_cams(neuron_ids)
        # - Update internal connection representation
        self._connections = self._extract_connections_from_memory(
            self._sram_connections, self._cam_connections
        )

    def remove_all_connections_from(self, neuron_ids, apply_diff: bool = True):
        """
        remove_all_connections_to - Remove all postsynaptic connections
                                    from neurons defined in neuron_ids
        :param neuron_ids:     np.ndarray IDs of neurons whose postsynaptic
                                          connections should be removed
        :param apply_diff:      bool If False do not apply the changes to
                                     chip but only to shadow states of the
                                     neurons. Useful if new connections are
                                     going to be added to the given neurons.
        """
        # - Make sure neurons neuron_ids is an array
        neuron_ids = [int(id_neur) for id_neur in np.asarray(neuron_ids)]

        # - Call `remove_all_connections_to` function
        self.tools.remove_all_connections_from(neuron_ids, self.model, apply_diff)
        # - Reset internal representation of SRAM cells
        self._reset_srams(neuron_ids)
        # - Update internal connection representation
        self._connections = self._extract_connections_from_memory(
            self._sram_connections, self._cam_connections
        )

    def get_connections(
        self,
        pre_ids: Iterable[int],
        post_ids: Optional[Iterable[int]] = None,
        syn_types: Union[Iterable[int], int, None] = None,
        virtual_pre: bool = False,
    ) -> np.ndarray:
        """
        get_connections - Return connections between specific populations.
        :param pre_ids:   IDs of presynaptic neurons.
        :param post_ids:  IDs of postsynaptic neurons. If `None`, and `virtual_pre` is
                          `False`, use the same as `pre_ids`.
        :param syn_types: If not `None` must be an integer or iterable of integers
                          specifying which synapse types are to be returned. If `None`
                          return connections for all synapse types. Indices 0, 1, 2, and 3
                          correspond to types `fast_exc`, `slow_exc`, `fast_inh`, and
                          `slow_inh`, respectively.
        :param virtual_pre:  If `True`, return connections to virtual (external) neurons.

        :return:
            3D-connectivity matrix for selected neurons. 1st axis corresponds to synapse type.
            The order of synapse types is `fast_exc`, `slow_exc`, `fast_inh`, and `slow_inh`,
            respectively. 2nd and 3rd axis correspond to pre- and postsynaptic neurons.
        """
        if post_ids is None:
            if virtual_pre:
                raise ValueError(
                    "DynapseControl: For virtual connections need to define both "
                    + "`pre_ids` and `post_ids`."
                )
            else:
                post_ids = pre_ids
        if syn_types is None:
            syn_types = np.arange(len(self._camtypes))
        idx_pre, idx_post = np.meshgrid(pre_ids, post_ids, indexing="ij")

        try:
            # - Increase index dimensionality
            idx_pre = np.stack(len(syn_types) * (idx_pre,))
        except TypeError:
            # - If syn_types has no len, try treating it as integer
            syn_types = int(syn_types)
        else:
            idx_post = np.stack(len(syn_types) * (idx_post,))
            syn_types = np.stack(len(pre_ids) * (syn_types,), axis=-1)
            syn_types = np.stack(len(post_ids) * (syn_types,), axis=-1)
        if virtual_pre:
            return self.connections_virtual[syn_types, idx_pre, idx_post]
        else:
            return self.connections[syn_types, idx_pre, idx_post]

    ### --- Stimulation, event generation and recording

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
        # t0 = time.time()
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
        channels = channels.astype(int)

        # - Check that the number of channels is the same between time series and list of neurons
        if np.amax(channels) > np.size(neuron_ids):
            raise ValueError(
                "DynapseControl: `channels` contains more channels than the number of neurons in `neuron_ids`."
            )
        # - Make sure neuron_ids is iterable
        neuron_ids = np.array(neuron_ids)

        # - Convert to ISIs
        discrete_isi_list = np.diff(np.r_[ts_start, timesteps])

        # print(time.time() - t0)
        # t0 = time.time()
        print("DynapseControl: Generating FPGA event list from arrays.")
        # - Convert events to an FpgaSpikeEvent
        events = self.tools.generate_fpga_event_list(
            # Make sure that no np.int64 or other non-native type is passed
            [int(isi) for isi in discrete_isi_list],
            [int(neuron_ids[i]) for i in channels],
            int(targetcore_mask),
            int(targetchip_id),
        )
        # print(time.time() - t0)

        # - Return a list of events
        return events

    def start_cont_stim(
        self,
        frequency: float,
        neuron_ids: Union[int, Iterable],
        chip_id: int = 0,
        coremask: int = 15,
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
        events = self.tools.generate_fpga_event_list(
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
        if not hasattr(self, "fpga_poissongen"):
            raise RuntimeError("DynapseControl: No poissong generator available.")

        if not hasattr(self, "fpga_poissongen"):
            raise RuntimeError("DynapseControl: No poissong generator available.")

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
        if not hasattr(self, "fpga_poissongen"):
            raise RuntimeError("DynapseControl: No poissong generator available.")

        self.fpga_poissongen.stop()
        print("DynapseControl: Poisson rate stimulation stopped")

    def reset_poisson_rates(self):
        """reset_poisson_rates - Set all firing rates of poisson generator to 0."""
        if not hasattr(self, "fpga_poissongen"):
            raise RuntimeError("DynapseControl: No poissong generator available.")

        for i in range(1024):
            self.fpga_poissongen.write_poisson_rate_hz(i, 0)
        print("DynapseControl: Firing rates for poisson generator have been set to 0.")

    def send_pulse(
        self,
        width: float = 0.1,
        frequency: float = 1000,
        t_record: float = 3,
        t_buffer: float = 0.5,
        virtual_neur_id: int = 0,
        record_neur_ids: Union[int, np.ndarray] = np.arange(1024),
        targetcore_mask: int = 15,
        targetchip_id: int = 0,
        periodic: bool = False,
        record: bool = False,
        inputneur_id=None,
    ) -> (np.ndarray, np.ndarray):
        """
        send_pulse - Send a pulse of periodic input events to the chip.
                     Return a arrays with times and channels of the recorded hardware activity.
        :param width:              float  Duration of the input pulse
        :param frequency:               float  Frequency of the events that constitute the pulse
        :param t_record:             float  Duration of the recording (including stimulus)
        :param t_buffer:             float  Record slightly longer than t_record to
                                           make sure to catch all relevant events
        :param virtual_neur_id:      int    ID of neuron that appears as source of pulse
        :param record_neur_ids:   array-like  ID(s) of neuron(s) to be recorded
        :param chip_id:     int  Target chip ID
        :param coremask:   int  Target core mask
        :param periodic:   bool    Repeat the stimulus indefinitely
        :param record:     bool    Set up buffered event filter that records events
                                    from neurons defined in record_neur_ids

        :return:
            (times, channels)  np.ndarrays that contain recorded data
        """

        if inputneur_id is not None:
            warn(
                "DynapseControlExtd: The argument `inputneur_id` has been "
                + "renamed to 'virtual_neur_id`. The old name will not be "
                + "supported anymore in future versions."
            )
            if virtual_neur_id is None:
                virtual_neur_id = inputneur_id
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
            channels=np.repeat(virtual_neur_id, timesteps.size),
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
        virtual_neur_ids: Optional[np.ndarray] = None,
        record_neur_ids: Optional[np.ndarray] = None,
        targetcore_mask: int = 15,
        targetchip_id: int = 0,
        periodic: bool = False,
        record: bool = False,
        fastmode: bool = False,
        neuron_ids=None,
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
        :param virtual_neur_ids:     ArrayLike    IDs of neurons that should appear as sources of the events
                                             If None, use channels from channels
        :param record_neur_ids: ArrayLike    IDs of neurons that should be recorded (if record==True)
                                               If None and record==True, record neurons in virtual_neur_ids
        :param targetcore_mask: int          Mask defining target cores (sum of 2**core_id)
        :param targetchip_id:   int          ID of target chip
        :param periodic:       bool         Repeat the stimulus indefinitely
        :param record:         bool         Set up buffered event filter that records events
                                             from neurons defined in record_neur_ids
        :param fastmode:        bool        Skip generation of event buffers. Must be generated in advance!
                                            (saves around 0.3 s)

        :return:
            (times, channels)  np.ndarrays that contain recorded data
        """

        if neuron_ids is not None:
            warn(
                "DynapseControlExtd: The argument `neuron_ids` has been "
                + "renamed to 'virtual_neur_ids`. The old name will not be "
                + "supported anymore in future versions."
            )
            if virtual_neur_ids is None:
                virtual_neur_ids = neuron_ids

        # - Process input arguments
        virtual_neur_ids = (
            np.arange(np.amax(channels) + 1)
            if virtual_neur_ids is None
            else np.array(virtual_neur_ids, int)
        )
        record_neur_ids = (
            virtual_neur_ids if record_neur_ids is None else record_neur_ids
        )
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
            neuron_ids=virtual_neur_ids,
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
            fastmode=fastmode,
        )

    def record(
        self,
        neuron_ids: Union[np.ndarray, List[int], int],
        duration: Optional[float] = None,
    ) -> (np.array, np.array):
        """
        record - Record spiking activity of given neurons. Either record for
                 given duration or until `self.stop_recording` is called
        :param neuron_ids:  Array-like with IDs of neurons that should be recorded
        :param duration:    Recording duration in seconds. If None, will record
                            until `self.stop_recording` is called.
        :return:
            If `duration` is `None`:    `None`
            else:                       Arrays with times and channels of recorded events.
        """
        return self._send_stimulus_list(
            events=[],
            duration=duration,
            t_buffer=0,
            record_neur_ids=neuron_ids,
            periodic=False,
            record=True,
        )

    def stop_recording(self, since_trigger: bool = False) -> (np.ndarray, np.ndarray):
        """
        stop_recording - Stop recording and return recorded events as arrays.
        :param since_trigger:  If True, only use events recorded after first
                               trigger event in buffer.
        :return:
            Arrays with times and channels of recorded events.
        """
        try:
            self.bufferedfilter.clear()
        except AttributeError:
            warn("DynapseControl: No recording has been started.")
            return np.array([]), np.array([])
        else:
            return self._recorded_data_to_arrays(None, None, since_trigger)

    def _send_stimulus_list(
        self,
        events,
        duration,
        t_buffer,
        record_neur_ids: Optional[np.ndarray] = None,
        periodic: bool = False,
        record: bool = False,
        fastmode: bool = False,
    ) -> Union[None, Tuple[np.ndarray, np.ndarray]]:
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
        :param fastmode:        bool        Skip generation of event buffers. Must be generated in advance!
                                            (saves around 0.3 s)

        :return:
            (times, channels)  np.ndarrays that contain recorded data
        """
        # t0 = time.time()
        if events:
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
            # print(time.time() - t0)
            # t0 = time.time()
            print("DynapseControl: Stimulus preloaded.")

        if record and not fastmode:
            if record_neur_ids is None:
                record_neur_ids = []
                warn("DynapseControl: No neuron IDs specified for recording.")
            self.add_buffered_event_filter(record_neur_ids)
        # print(time.time() - t0)
        # t0 = time.time()

        # - Clear event filter
        self.bufferedfilter.get_events()
        self.bufferedfilter.get_special_event_timestamps()

        if events:
            # - Stimulate
            self.fpga_spikegen.start()
            print(
                "DynapseControl: Started{} stimulation{}.".format(
                    periodic * " periodic",
                    (not periodic) * " for {} s".format(duration),
                )
            )

        if (duration is None) or periodic:
            # - Keep running indefinitely
            return

        else:
            # Time at which stimulation/recording stops, including buffer
            t_wait = duration + (0.0 if t_buffer is None else t_buffer)
            t_stop = time.time() + t_wait

            if record:
                return self._record_and_process(
                    t_stop=t_stop,
                    record_neur_ids=record_neur_ids,
                    duration=duration,
                    fastmode=fastmode,
                )
            else:
                time.sleep(t_wait)
                print("DynapseControl: Stimulation ended.")

    def _record_and_process(
        self,
        t_stop: float,
        record_neur_ids: np.ndarray,
        duration: float,
        fastmode: bool,
    ) -> (np.array, np.array):
        """
        _record_and_process - Until system time reaches t_stop, record events
                              and process in quick succession. Afterwards do
                              final processing of event times and channels and
                              return them.
        :param t_stop:  System time at which to stop recording.
        :record_neur_ids:  IDs of neurons from which to record
        :duration:         Recording duration without buffer
        :fastmode:         If `True`, don't clear buffered filter after recording.

        :return:
            1D-array with event times (relative to start of recording)
            1D-array with corresponding event channels
        """
        # -
        # - Lists for storing collected events
        timestamps_full = []
        channels_full = []
        triggerevents = []

        # Set go_on to 2 to enforce another run of the loop after time is over.
        # Otherwise, if last iteration takes too long, events may be lost.
        go_on = 2
        while go_on:
            # - Collect events and possibly trigger events
            triggerevents += self.bufferedfilter.get_special_event_timestamps()
            current_events = self.bufferedfilter.get_events()

            timestamps_curr, channels_curr = event_data_to_channels(
                current_events, record_neur_ids
            )
            timestamps_full += list(timestamps_curr)
            channels_full += list(channels_curr)
            go_on -= int(time.time() >= t_stop)

        if not fastmode:
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

        return timestamps_full, channels_full, triggerevents

    def _process_extracted_events(
        self,
        timestamps: List,
        channels: List,
        triggerevents: List,
        duration: Union[float, None],
    ) -> (np.array, np.array):

        # - Post-processing of collected events
        times = np.asarray(timestamps) * 1e-6
        channels = np.asarray(channels)

        if times.size == 0:
            return np.array([]), np.array([])

        # - Locate synchronisation timestamp
        triggertimes = np.array(triggerevents) * 1e-6
        start_indices = np.searchsorted(times, triggertimes)
        if duration is None:
            end_indices = [times.size] * triggertimes.size
        else:
            end_indices = np.searchsorted(times, triggertimes + duration)
        # - Choose first trigger where start and end indices not equal. If not possible, take first trigger
        try:
            trigger_id = np.argmax((end_indices - start_indices) > 0)
            print("\t\t Using trigger event {}".format(trigger_id))
        except ValueError:
            print("\t\t No Trigger found, using recording from beginning")
            idx_start = 0
            t_trigger = times[0]
            if duration is None:
                idx_end = times.size
            else:
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
        self,
        neuron_ids: Union[np.ndarray, None],
        t_record: Union[float, None],
        since_trigger: bool = True,
    ) -> (np.ndarray, np.ndarray):
        # - Get events from buffered filter
        events = self.bufferedfilter.get_events()
        triggerevents = self.bufferedfilter.get_special_event_timestamps()
        if not since_trigger:
            # - Pass empty trigger list to enforce using all events
            #   Still call `get_special_event_timestamps` to clear buffer
            triggerevents = []

        print(
            "DynapseControl: Recorded {} event(s) and {} trigger event(s)".format(
                len(events), len(triggerevents)
            )
        )

        # - Extract monitored event channels and timestamps
        neuron_ids = range(self.num_neurons) if neuron_ids is None else neuron_ids
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
            self.bufferedfilter = self.tools.generate_buffered_filter(
                self.model, record_neuron_ids
            )
            print("DynapseControl: Generated new buffered event filter.")

        return self.bufferedfilter

    def clear_buffered_event_filter(self):
        """clear_buffered_event_filter - Clear self.bufferedfilter if it exists."""
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
            (
                firing_rates,
                rates_mean[i],
                rates_max[i],
                rates_min[i],
            ) = self.measure_firing_rates(neuron_ids, duration)
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
        if not _USE_RPYC:
            raise RuntimeError(
                "DynapseControl: This method can only be called when using RPyC."
            )

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
            (
                firingrates_2d[iTrial, :],
                rates_mean[iTrial],
                rates_max[iTrial],
                rates_min[iTrial],
            ) = self.measure_firing_rates(targetneuron_ids, duration)
            self.stop_stim()

        return firingrates_2d, rates_mean, rates_max, rates_min

    def close(self):
        self.rpyc_connection.close()
        print("DynapseControl: RPyC connection closed.")

    ### - Load and save biases

    @staticmethod
    def load_biases(filename, core_ids: Optional[Union[list, int]] = None):
        """
        load_biases - Load biases from python file under path filename.
                      Convenience function. Same load_biases from tools.
        :param filename:  str  Path to file where biases are stored.
        :param core_ids:    list, int or None  IDs of cores for which biases
                                                should be loaded. Load all if
                                                None.
        """
        tools.load_biases(os.path.abspath(filename), core_ids)
        print("DynapseControl: Biases have been loaded from {}.".format(filename))

    @staticmethod
    def save_biases(filename, core_ids: Optional[Union[list, int]] = None):
        """
        save_biases - Save biases in python file under path filename
                      Convenience function. Same as save_biases from tools.
        :param filename:  str  Path to file where biases should be saved.
        :param core_ids:    list, int or None  ID(s) of cores whose biases
                                                should be saved. If None,
                                                save all cores.
        """
        tools.save_biases(os.path.abspath(filename), core_ids)
        print("DynapseControl: Biases have been saved under {}.".format(filename))

    @staticmethod
    def copy_biases(sourcecore_id: int = 0, targetcore_ids: Optional[List[int]] = None):
        """
        copy_biases - Copy biases from one core to one or more other cores.
                      Convenience function. Same as copy_biases from tools.
        :param sourcecore_id:   int  ID of core from which biases are copied
        :param vnTargetCoreIDs: int or array-like ID(s) of core(s) to which biases are copied
                                If None, will copy to all other neurons
        """
        tools.copy_biases(sourcecore_id, targetcore_ids)
        print(
            "DynapseControl: Biases have been copied from core {} to core(s) {}".format(
                sourcecore_id, targetcore_ids
            )
        )

    ### --- Class properties

    @property
    def syn_exc_slow(self):
        return ctxdynapse.DynapseCamType.SLOW_EXC

    @property
    def syn_inh_slow(self):
        return ctxdynapse.DynapseCamType.SLOW_INH

    @property
    def syn_exc_fast(self):
        return ctxdynapse.DynapseCamType.FAST_EXC

    @property
    def syn_inh_fast(self):
        return ctxdynapse.DynapseCamType.FAST_INH

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
    def max_dt(self):
        return self.fpga_timestep * self._fpga_isi_multiplier_limit

    @property
    def fpga_isibase(self):
        return self._fpga_isi_multiplier * self.fpga_timestep

    @fpga_isibase.setter
    def fpga_isibase(self, tNewBase):
        if not self.max_dt > tNewBase > self.fpga_timestep:
            raise ValueError(
                "DynapseControl: `fpga_timestep` must be between "
                + f"{self.fpga_timestep} and {self.max_dt}."
            )
        else:
            multiplier = int(np.floor(tNewBase / self.fpga_timestep))
            self._fpga_isi_multiplier = multiplier
            self.fpga_spikegen.set_isi_multiplier(self._fpga_isi_multiplier)

    @property
    def initialized_chips(self):
        if not _USE_RPYC:
            global initialized_chips
            return initialized_chips
        else:
            return self.rpyc_connection.modules.tools.storage.get(
                "initialized_chips", []
            )

    @property
    def initialized_neurons(self):
        if not _USE_RPYC:
            global initialized_neurons
            return initialized_neurons
        else:
            return self.rpyc_connection.modules.tools.storage.get(
                "initialized_neurons", []
            )

    @property
    def hwneurons_isavailable(self) -> np.ndarray:
        hwneurons_isinitialized = np.zeros(len(self.hw_neurons), bool)
        if self.initialized_neurons:
            hwneurons_isinitialized[self.initialized_neurons] = True
        return np.logical_and(self._hwneurons_isfree, hwneurons_isinitialized)

    @property
    def virtualneurons_isavailable(self) -> np.ndarray:
        return self._virtualneurons_isfree

    @property
    def prevent_aliasing(self):
        return self._prevent_aliasing

    @prevent_aliasing.setter
    def prevent_aliasing(self, setting):
        self._prevent_aliasing = bool(setting)

    @property
    def connections(self):
        return self._connections

    @property
    def connections_virtual(self):
        return self._cam_connections


if not _USE_RPYC:
    initialize_hardware(USE_CHIPS)
