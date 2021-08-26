"""
Utilities for working with the Pollen HDK.

Ideally you should not need to use these utility functions. You should try using :py:class:`.PollenSamna` and :py:class:`.PollenCim` for high-level interfaces to Pollen.

See Also:
    The tutorials in :ref:`/devices/pollen-overview.ipynb` and :ref:`/devices/torch-training-spiking-for-pollen.ipynb`.

"""

# - Check that Samna is installed
from importlib import util

if util.find_spec("samna") is None:
    raise ModuleNotFoundError(
        "'samna' not found. Modules that rely on Samna will not be available."
    )

# - `samna` imports
import samna
from samna.pollen.configuration import PollenConfiguration

# - Other imports
from warnings import warn
import time
import numpy as np
from pathlib import Path
from os import makedirs
import json

# - Typing and useful proxy types
from typing import Any, List, Iterable, Optional, NamedTuple, Union, Tuple

PollenDaughterBoard = Any
SamnaDeviceNode = Any
PollenReadBuffer = samna.BufferSinkNode_pollen_event_output_event
PollenNeuronStateBuffer = samna.pollen.NeuronStateSinkNode


class PollenState(NamedTuple):
    """
    ``NamedTuple`` that encapsulates a recorded Pollen HDK state
    """

    Nhidden: int
    """ The number of hidden-layer neurons """

    Nout: int
    """ The number of output layer neurons """

    V_mem_hid: np.array
    """ Membrane potential of hidden neurons """

    I_syn_hid: np.array
    """ Synaptic current 1 of hidden neurons """

    V_mem_out: np.array
    """ Membrane potential of output neurons """

    I_syn_out: np.array
    """ Synaptic current of output neurons """

    I_syn2_hid: np.array
    """ Synaptic current 2 of hidden neurons """

    Spikes_hid: np.array
    """ Spikes from hidden layer neurons """

    Spikes_out: np.array
    """ Spikes from output layer neurons """


def find_pollen_boards(device_node: SamnaDeviceNode) -> List[PollenDaughterBoard]:
    """
    Search for and return a list of Pollen HDK daughterboards

    Iterate over devices and search for Pollen HDK daughterboards. Return a list of available Pollen daughterboards, or an empty list if none are found.

    Notes:
        This function will open any unopened devices on the device node.

    Args:
        device_node (SamnaDeviceNode): An opened Samna device node

    Returns: List[PollenDaughterBoard]: A (possibly empty) list of Pollen HDK daughterboards.
    """
    # - Get a list of unopened devices
    unopened_devices = device_node.DeviceController.get_unopened_devices()

    # - Open the devices
    for d in unopened_devices:
        n = 0
        try:
            device_node.DeviceController.open_device(d, f"board{n}")
        except:
            pass

    # - Get a list of opened devices
    device_list = device_node.DeviceController.get_opened_devices()

    # - Search for a pollen dev kit
    pollen_hdk_list = [
        getattr(device_node, d.name)
        for d in device_list
        if d.device_info.device_type_name == "PollenDevKit"
    ]

    # - Search for pollen boards
    pollen_daughterboard_list = []
    for d in pollen_hdk_list:
        daughterboard = d.get_daughter_board(0)
        if "PollenDaughterBoard" in str(type(daughterboard)):
            pollen_daughterboard_list.append(daughterboard)

    return pollen_daughterboard_list


def new_pollen_read_buffer(
    daughterboard: PollenDaughterBoard,
) -> PollenReadBuffer:
    """
    Create and connect a new buffer to read from a Pollen HDK

    Args:
        daughterboard (PollenDaughterBoard):

    Returns:
        samna.BufferSinkNode_pollen_event_output_event: Output buffer receiving events from Pollen HDK
    """
    # - Register a buffer to read events from Pollen
    buffer = PollenReadBuffer()

    # - Get the device model
    model = daughterboard.get_model()

    # - Get Pollen output event source node
    source_node = model.get_source_node()

    # - Add the buffer as a destination for the Pollen output events
    success = source_node.add_destination(buffer.get_input_channel())
    assert success, "Error connecting the new buffer."

    # - Return the buffer
    return buffer


def new_pollen_state_monitor_buffer(
    daughterboard: PollenDaughterBoard,
) -> PollenNeuronStateBuffer:
    """
    Create a new buffer for monitoring neuron and synapse state and connect it

    Args:
        daughterboard (PollenDaughterBoard): A Pollen HDK to configure

    Returns:
        PollenNeuronStateBuffer: A connected neuron / synapse state monitor buffer
    """
    # - Register a new buffer to receive neuron and synapse state
    buffer = PollenNeuronStateBuffer()

    # - Get the device model
    model = daughterboard.get_model()

    # - Get Pollen output event source node
    source_node = model.get_source_node()

    # - Add the buffer as a destination for the Pollen output events
    success = source_node.add_destination(buffer.get_input_channel())
    assert success, "Error connecting the new buffer."

    # - Return the buffer
    return buffer


def blocking_read(
    buffer: PollenReadBuffer,
    count: Optional[int] = None,
    target_timestamp: Optional[int] = None,
    timeout: Optional[float] = None,
) -> List:
    """
    Perform a blocking read on a buffer, optionally waiting for a certain count, a target timestamp, or imposing a timeout

    Args:
        buffer (PollenReadBuffer): A buffer to read from
        count (Optional[int]): The count of required events. Default: ``None``, just wait for any data.
        target_timestamp (Optional[int]): The desired final timestamp. Read until this timestamp is returned in an event. Default: ``None``, don't wait until a particular timestamp is read.
        timeout (Optional[float]): The time in seconds to wait for a result. Default: ``None``, no timeout: block until a read is made.

    Returns:
        List: A list of read events
    """
    all_events = []

    # - Read at least a certain number of events
    continue_read = True
    start_time = time.time()
    while continue_read:
        # - Perform a read and save events
        events = buffer.get_events()
        all_events.extend(events)

        # - Check if we reached the desired timestamp
        if target_timestamp:
            timestamps = [
                e.timestamp
                for e in events
                if hasattr(e, "timestamp") and e.timestamp is not None
            ]

            if timestamps:
                reached_timestamp = timestamps[-1] >= target_timestamp
                continue_read &= ~reached_timestamp

        # - Check timeout
        if timeout:
            continue_read &= (time.time() - start_time) <= timeout

        # - Check number of events read
        if count:
            continue_read &= len(all_events) < count

    # - Perform one final read for good measure
    all_events.extend(buffer.get_events())

    # - Return read events
    return all_events


def initialise_pollen_hdk(daughterboard: PollenDaughterBoard) -> None:
    """
    Initialise the Pollen HDK

    Args:
        daughterboard (PollenDaughterBoard): A Pollen daughterboard to initialise
    """
    # - Always need to advance one time-step to initialise
    advance_time_step(daughterboard)


def write_register(
    daughterboard: PollenDaughterBoard,
    register: int,
    data: int = 0,
) -> None:
    """
    Write data to a register on a Pollen HDK

    Args:
        daughterboard (PollenDaughterboard): A pollen HDK daughterboard to write to
        register (int): The address of the register to write to
        data (int): The data to write. Default: 0x0
    """
    wwv_ev = samna.pollen.event.WriteRegisterValue()
    wwv_ev.address = register
    wwv_ev.data = data
    daughterboard.get_model().write([wwv_ev])


def read_register(
    daughterboard: PollenDaughterBoard,
    buffer: PollenReadBuffer,
    address: int,
) -> List[int]:
    """
    Read the contents of a register

    Args:
        daughterboard (PollenDaughterBoard):
        buffer (samna.BufferSinkNode_pollen_event_output_event):
        address (int): The register address to read

    Returns:
        List[int]: A list of events returned from the read
    """
    # - Set up a register read
    rrv_ev = samna.pollen.event.ReadRegisterValue()
    rrv_ev.address = address

    # - Request read
    daughterboard.get_model().write([rrv_ev])

    # - Wait for data and read it
    events = blocking_read(buffer, count=1, timeout=1.0)

    # - Filter returned events for the desired address
    ev_filt = [e for e in events if hasattr(e, "address") and e.address == address]

    # - If we didn't get the required register read, try again by recursion
    if ev_filt == []:
        return read_register(daughterboard, buffer, address)
    else:
        return [e.data for e in ev_filt]


def read_memory(
    daughterboard: PollenDaughterBoard,
    buffer: PollenReadBuffer,
    start_address: int,
    count: int = 1,
) -> List[int]:
    """
    Read a block of memory from a Pollen HDK

    Args:
        daughterboard (PollenDaughterboard):
        buffer (samna.BufferSinkNode_pollen_event_output_event): A connected output buffer to use in reading
        start_address (int): The base address to start reading from
        count (int): The number of elements to read

    Returns:
        List[int]: A list of values read from memory
    """
    # - Set up a memory read
    read_events_list = []

    # - Insert an extra read to avoid zero data
    rmv_ev = samna.pollen.event.ReadMemoryValue()
    rmv_ev.address = start_address
    read_events_list.append(rmv_ev)

    for elem in range(count):
        rmv_ev = samna.pollen.event.ReadMemoryValue()
        rmv_ev.address = start_address + elem
        read_events_list.append(rmv_ev)

    # - Clear buffer
    buffer.get_events()

    # - Request read
    daughterboard.get_model().write(read_events_list)

    # - Read data
    events = blocking_read(buffer, count=count + 1)

    # - Filter returned events for the desired addresses
    return [
        e.data
        for e in events[1:]
        if e.address >= start_address and e.address < start_address + count
    ]


def generate_read_memory_events(
    start_address: int,
    count: int = 1,
) -> List[Any]:
    """
    Build a list of events that cause Pollen memory to be read

    This function is designed to be used with `decode_memory_read_events`.

    See Also:
        Use the `read_memory` function for a more convenient high-level API.

    Args:
        start_address (int): The starting address of the memory read
        count (int): The number of memory elements to read. Default: ``1``, read a single memory address.

    Returns:
        List: A list of events to send to a Pollen HDK
    """
    # - Set up a memory read
    read_events_list = []

    # - Insert an extra read to avoid zero data
    rmv_ev = samna.pollen.event.ReadMemoryValue()
    rmv_ev.address = start_address
    read_events_list.append(rmv_ev)

    for elem in range(count):
        rmv_ev = samna.pollen.event.ReadMemoryValue()
        rmv_ev.address = start_address + elem
        read_events_list.append(rmv_ev)

    return read_events_list


def decode_memory_read_events(
    events: List[Any],
    start_address: int,
    count: int = 1,
) -> List:
    """
    Decode a list of events containing memory reads from a Pollen HDK

    This is a low-level function designed to be used in conjuction with :py:func:`.generate_read_memory_events`.

    See Also:
        Use the :py:func:`read_memory` function for a more convenient high-level API.

    Args:
        events (List): A list of events read from a Pollen HDK
        start_address (int): The starting address for the memory read
        count (int): The number of contiguous memory elements that were read

    Returns:
        List: A list of memory entries extracted from the list of events, in address order
    """
    # - Initialise returned data list
    return_data = [[]] * count

    # - Filter returned events for the desired addresses
    for e in events:
        if e.address >= start_address and e.address < start_address + count:
            return_data[e.address - start_address] = e.data

    # - Return read data
    return return_data


def verify_pollen_version(
    daughterboard: PollenDaughterBoard,
    buffer: PollenReadBuffer,
    timeout: float = 1.0,
) -> bool:
    """
    Verify that the provided daughterbaord returns the correct version ID for Pollen

    Args:
        daughterboard (PollenDaughterBoard): A daughter-board object to test
        buffer (samna.BufferSinkNode_pollen_event_output_event): A read buffer
        timeout (float): Timeout for checking in seconds

    Returns:
        bool: ``True`` iff the version ID is correct for Pollen
    """
    # - Clear the read buffer
    buffer.get_events()

    # - Read the version register
    daughterboard.get_model().write([samna.pollen.event.ReadVersion()])

    # - Read events until timeout
    filtered_events = []
    t_end = time.time() + timeout
    while (len(filtered_events) == 0) and (time.time() < t_end):
        events = buffer.get_events()
        filtered_events = [
            e for e in events if isinstance(e, samna.pollen.event.Version)
        ]

    return (
        (len(filtered_events) > 0)
        and (filtered_events[0].major == 1)
        and (filtered_events[0].minor == 0)
    )


def write_memory(
    daughterboard: PollenDaughterBoard,
    start_address: int,
    count: Optional[int] = None,
    data: Optional[Iterable] = None,
    chunk_size: int = 65535,
) -> None:
    """
    Write data to Pollen memory

    Args:
        daughterboard (PollenDaughterBoard): A Pollen daughterboard to write to
        start_address (int): The base address to start writing from
        count (int): The number of entries to write. Default: ``len(data)``
        data (Iterable): A list of data to write to memory. Default: Write zeros.
        chunk_size (int): Chunk size to write. Default: 2000. Only needed on OS X, it seems?
    """
    # - How many entries should we write?
    if count is None and data is None:
        raise ValueError("Either `count` or `data` must be provided as arguments.")

    if count is not None and data is not None and count != len(data):
        warn(
            "Length of `data` and `count` do not match. Only `count` entries will be written."
        )

    if count is None:
        count = len(data)

    # - Set up a list of write events
    write_event_list = []
    for elem in range(count):
        wmv_ev = samna.pollen.event.WriteMemoryValue()
        wmv_ev.address = start_address + elem

        if data is not None:
            wmv_ev.data = data[elem]

        write_event_list.append(wmv_ev)

    # - Write the list of data events
    written = 0
    while written < len(write_event_list):
        daughterboard.get_model().write(
            write_event_list[written : (written + chunk_size)]
        )
        written += chunk_size
        time.sleep(0.01)


def zero_memory(
    daughterboard: PollenDaughterBoard,
) -> None:
    """
    Clear all Pollen memory

    This function writes zeros to all memory banks on a Pollen HDK.

    Args:
        daughterboard (PollenDaughterboard): The Pollen HDK to zero memory on
    """
    # - Define the memory banks
    memory_table = {
        "iwtram": (0x0100, 16000),
        "iwt2ram": (0x3F80, 16000),
        "nscram": (0x7E00, 1008),
        "rsc2ram": (0x81F0, 1000),
        "nmpram": (0x85D8, 1008),
        "ndsram": (0x89C8, 1008),
        "nds2ram": (0x8DB8, 1000),
        "ndmram": (0x91A0, 1008),
        "nthram": (0x9590, 1008),
        "rcram": (0x9980, 1000),
        "raram": (0x9D68, 1000),
        "rspkram": (0xA150, 1000),
        "refocram": (0xA538, 1000),
        "rforam": (0xA920, 32000),
        "rwtram": (0x12620, 32000),
        "rwt2ram": (0x1A320, 32000),
        "owtram": (0x22020, 8000),
    }

    # - Zero each bank in turn
    for bank in memory_table.values():
        write_memory(daughterboard, *bank)


def reset_neuron_synapse_state(daughterboard: PollenDaughterBoard) -> None:
    """
    Reset the neuron and synapse state on a Pollen HDK

    Args:
        daughterboard (PollenDaughterboard): The Pollen HDK daughterboard to reset
    """
    # - `get_configuration()` is not yet compatible with `apply_configuration()`
    config = daughterboard.get_model().get_configuration()

    # - Reset via configuration
    config.clear_network_state = True
    apply_configuration(daughterboard, config)


def apply_configuration(
    daughterboard: PollenDaughterBoard, config: PollenConfiguration
) -> None:
    """
    Apply a configuration to the Pollen HDK

    Args:
        daughterboard (PollenDaughterboard): The Pollen HDK to write the configuration to
        config (PollenConfiguration): A configuration for Pollen
    """
    # - Ideal -- just write the configuration using samna
    daughterboard.get_model().apply_configuration(config)


def read_neuron_synapse_state(
    daughterboard: PollenDaughterBoard,
    buffer: PollenReadBuffer,
    Nhidden: int = 1000,
    Nout: int = 8,
) -> PollenState:
    """
    Read and return the current neuron and synaptic state of neurons

    Args:
        daughterboard (PollenDaughterboard): The Pollen HDK to query
        buffer (PollenReadBuffer): A read buffer connected to the Pollen HDK
        Nhidden (int): Number of hidden neurons to read. Default: ``1000`` (all neurons).
        Nout (int): Number of output neurons to read. Default: ``8`` (all neurons).

    Returns:
        :py:class:`.PollenState`: The recorded state as a ``NamedTuple``. Contains keys ``V_mem_hid``,  ``V_mem_out``, ``I_syn_hid``, ``I_syn_out``, ``I_syn2_hid``, ``Nhidden``, ``Nout``. This state has **no time axis**; the first axis is the neuron ID.

    """
    # - Define the memory bank addresses
    memory_table = {
        "nscram": 0x7E00,
        "rsc2ram": 0x81F0,
        "nmpram": 0x85D8,
        "rspkram": 0xA150,
    }

    # - Read synaptic currents
    Isyn = read_memory(
        daughterboard,
        buffer,
        memory_table["nscram"],
        Nhidden + Nout + num_buffer_neurons(Nhidden),
    )

    # - Read synaptic currents 2
    Isyn2 = read_memory(daughterboard, buffer, memory_table["rsc2ram"], Nhidden)

    # - Read membrane potential
    Vmem = read_memory(
        daughterboard,
        buffer,
        memory_table["nmpram"],
        Nhidden + Nout + num_buffer_neurons(Nhidden),
    )

    # - Read reservoir spikes
    Spikes = read_memory(daughterboard, buffer, memory_table["rspkram"], Nhidden)

    # - Return the state
    return PollenState(
        Nhidden,
        Nout,
        np.array(Vmem[:Nhidden], "int16"),
        np.array(Isyn[:Nhidden], "int16"),
        np.array(Vmem[-Nout:], "int16"),
        np.array(Isyn[-Nout:], "int16"),
        np.array(Isyn2, "int16"),
        np.array(Spikes, "bool"),
        read_output_events(daughterboard, buffer),
    )


def read_accel_mode_data(
    monitor_buffer: PollenNeuronStateBuffer,
    Nhidden: int,
    Nout: int,
) -> PollenState:
    """
    Read accelerated simulation mode data from a Pollen HDK

    Args:
        monitor_buffer (PollenNeuronStateBuffer): A connected `PollenNeuronStateBuffer` to read from
        Nhidden (int): The number of hidden neurons to monitor
        Nout (int): The number of output neurons to monitor

    Returns:
        PollenState: The encapsulated state read from the Pollen device
    """
    # - Read data from neuron state buffer
    vmem_ts = np.array(monitor_buffer.get_reservoir_v_mem(), "int16").T
    isyn_ts = np.array(monitor_buffer.get_reservoir_i_syn(), "int16").T
    isyn2_ts = np.array(monitor_buffer.get_reservoir_i_syn2(), "int16").T
    spikes_ts = np.array(monitor_buffer.get_reservoir_spike(), "int8").T
    spikes_out_ts = np.array(monitor_buffer.get_output_spike(), "int8").T

    # - Separate hidden and output neurons
    isyn_out_ts = isyn_ts[:, -Nout:] if len(isyn_ts) > 0 else None
    isyn_ts = isyn_ts[:, :Nhidden] if len(isyn_ts) > 0 else None
    vmem_out_ts = vmem_ts[:, -Nout:] if len(vmem_ts) > 0 else None
    vmem_ts = vmem_ts[:, :Nhidden] if len(vmem_ts) > 0 else None

    # - Return as a PollenState object
    return PollenState(
        Nhidden,
        Nout,
        vmem_ts,
        isyn_ts,
        vmem_out_ts,
        isyn_out_ts,
        isyn2_ts,
        spikes_ts,
        spikes_out_ts,
    )


def decode_accel_mode_data(
    events: List[Any], Nhidden: int = 1000, Nout: int = 8
) -> Tuple[PollenState, np.ndarray]:
    """
    Decode events from accelerated-time operation of the Pollen HDK

    Warnings:
        ``Nhidden`` and ``Nout`` must be defined correctly for the network deployed to the Pollen HDK, for this function to operate as expected.

        This function must be called with the *full* list of events from a simulation. Otherwise the data returned will be incomplete. This function will not operate as expected if provided with incomplete data.

        You can use the ``target_timstamp`` argument to `.blocking_read` to ensure that you have read events up to the desired final timestep.

    Args:
        events (List[Any]): A list of events produced during an accelerated-mode simulation on a Pollen HDK
        Nhidden (int): The number of defined hidden-layer neurons. Default: ``1000``, expect to read the state of every neuron.
        Nout (int): The number of defined output-layer neurons. Default: ``8``, expect to read the state of every neuron.

    Returns:
        (`.PollenState`, np.ndarray): A `.NamedTuple` containing the decoded state resulting from the simulation, and an array of timestamps for each state entry over time
    """

    # - Define the memory banks
    memory_table = {
        "nscram": (0x7E00, 1008),
        "rsc2ram": (0x81F0, 1000),
        "nmpram": (0x85D8, 1008),
        "rspkram": (0xA150, 1000),
    }

    # - Range checking lambda
    address_in_range = (
        lambda address, start, count: address >= start and address < start + count
    )

    # - Initialise return data lists
    vmem_out_ts = []
    times = []
    vmem_ts = [np.zeros(Nhidden + Nout + num_buffer_neurons(Nhidden), "int16")]
    isyn_ts = [np.ones(Nhidden + Nout + num_buffer_neurons(Nhidden), "int16")]
    isyn2_ts = [np.zeros(Nhidden, "int16")]
    spikes_ts = [np.zeros(Nhidden, "bool")]
    spikes_out_ts = [np.zeros(Nout, "bool")]

    # - Loop over events and decode
    for e in events:
        # - Handle an output spike event
        if isinstance(e, samna.pollen.event.Spike):
            # - Save this output event
            spikes_out_ts[e.timestamp - 1][e.neuron_id] = True

        # - Handle a memory value read event
        if isinstance(e, samna.pollen.event.MemoryValue):
            # - Find out which memory block this event corresponds to
            memory_block = [
                block
                for (block, (start, count)) in memory_table.items()
                if address_in_range(e.address, start, count)
            ]

            # - Store the returned values
            if memory_block:
                if "nmpram" in memory_block:
                    # - Neuron membrane potentials
                    vmem_ts[-1][e.address - memory_table["nmpram"][0]] = e.data

                elif "nscram" in memory_block:
                    # - Neuron synaptic currents
                    isyn_ts[-1][e.address - memory_table["nscram"][0]] = e.data

                elif "rsc2ram" in memory_block:
                    # - Neuron synapse 2 currents
                    isyn2_ts[-1][e.address - memory_table["rsc2ram"][0]] = e.data

                elif "rspkram" in memory_block:
                    # - Reservoir spike events
                    spikes_ts[-1][e.address - memory_table["rspkram"][0]] = e.data

        # - Handle the readout event, which signals the *end* of a time step
        if isinstance(e, samna.pollen.event.Readout):
            # - Advance the timestep counter
            timestep = e.timestamp
            times.append(timestep)

            # - Append new empty arrays to state lists
            vmem_ts.append(
                np.zeros(Nhidden + Nout + num_buffer_neurons(Nhidden), "int16")
            )
            isyn_ts.append(
                np.ones(Nhidden + Nout + num_buffer_neurons(Nhidden), "int16")
            )
            isyn2_ts.append(np.zeros(Nhidden, "int16"))
            spikes_ts.append(np.zeros(Nhidden, "bool"))
            spikes_out_ts.append(np.zeros(Nout, "bool"))

    # - Convert data to numpy arrays
    vmem_out_ts = np.array(vmem_out_ts, "int16")
    times = np.array(times)

    # - Trim arrays that end up with one too many elements
    vmem_ts = np.array(vmem_ts[:-1], "int16")
    isyn_ts = np.array(isyn_ts[:-1], "int16")
    isyn2_ts = np.array(isyn2_ts[:-1], "int16")
    spikes_ts = np.array(spikes_ts[:-1], "bool")
    spikes_out_ts = np.array(spikes_out_ts[:-1], "bool")

    # - Extract output state and trim reservoir state
    isyn_out_ts = isyn_ts[:, -Nout:]
    isyn_ts = isyn_ts[:, :Nhidden]
    vmem_out_ts = vmem_ts[:, -Nout:]
    vmem_ts = vmem_ts[:, :Nhidden]

    return (
        PollenState(
            Nhidden,
            Nout,
            vmem_ts,
            isyn_ts,
            vmem_out_ts,
            isyn_out_ts,
            isyn2_ts,
            spikes_ts,
            spikes_out_ts,
        ),
        times,
    )


def is_pollen_ready(
    daughterboard: PollenDaughterBoard, buffer: PollenReadBuffer
) -> None:
    """
    Query a Pollen HDK to see if it is ready for a time-step

    Args:
        daughterboard (PollenDaughterboard): The Pollen HDK to query
        buffer (PollenReadBuffer): A buffer to use while reading

    Returns: ``True`` iff the Pollen HDK has finished all processing
    """
    return read_register(daughterboard, buffer, 0x10)[-1] & (1 << 16) is not 0


def advance_time_step(daughterboard: PollenDaughterBoard) -> None:
    """
    Take a single manual time-step on a Pollen HDK

    Args:
        daughterboard (PollenDaughterboard): The Pollen HDK to access
    """
    e = samna.pollen.event.TriggerProcessing()
    daughterboard.get_model().write([e])


def reset_input_spikes(daughterboard: PollenDaughterBoard) -> None:
    """
    Reset the input spike registers on a Pollen HDK

    Args:
        daughterboard (PollenDaughterboard): The Pollen HDK to access
    """
    for register in range(4):
        write_register(daughterboard, 0x0C + register)


def send_immediate_input_spikes(
    daughterboard: PollenDaughterBoard, spike_counts: Iterable[int]
) -> None:
    """
    Send input events with no timestamp to a Pollen HDK

    Args:
        daughterboard (PollenDaughterboard): A Pollen HDK to send events to
        spike_counts (Iterable[int]): An Iterable containing one slot per input channel. Each entry indicates how many events should be sent to the corresponding input channel.
    """
    # - Encode input events
    events_list = []
    for input_channel, event in enumerate(spike_counts):
        if event:
            for _ in range(event):
                s_event = samna.pollen.event.Spike()
                s_event.neuron = input_channel
                events_list.append(s_event)

    # - Send input spikes for this time-step
    daughterboard.get_model().write(events_list)


def read_output_events(
    daughterboard: PollenDaughterBoard, buffer: PollenReadBuffer
) -> np.ndarray:
    """
    Read the spike flags from the output neurons on a Pollen HDK

    Args:
        daughterboard (PollenDaughterBoard): The Pollen HDK to query
        buffer (PollenReadBuffer): A read buffer to use

    Returns:
        np.ndarray: A boolean array of output event flags
    """
    # - Read the status register
    status = read_register(daughterboard, buffer, 0x10)

    # - Convert to neuron events and return
    string = bin(status[-1])[-8:]
    return np.array([bool(int(e)) for e in string[::-1]], "bool")


def print_debug_ram(
    daughterboard: PollenDaughterBoard,
    buffer: PollenReadBuffer,
    Nin: int = 10,
    Nhidden: int = 10,
    Nout: int = 2,
) -> None:
    """
    Print memory contents for debugging purposes

    Args:
        daughterboard (PollenDaughterboard): A Pollen daughterboard to debug
        buffer (PollenReadBuffer): A connected Pollen read buffer to use when reading memory
        Nin (int): Number of input neurons to display. Default: ``10``.
        Nhidden (int): Number of hidden neurons to display. Default: ``10``.
    """
    print("iwtram", read_memory(daughterboard, buffer, 0x100, Nin * Nhidden))
    print("iwt2ram", read_memory(daughterboard, buffer, 0x3F80, Nin * Nhidden))

    print("nscram", read_memory(daughterboard, buffer, 0x7E00, Nhidden + Nout))
    print("rsc2ram", read_memory(daughterboard, buffer, 0x81F0, Nhidden))
    print("nmpram", read_memory(daughterboard, buffer, 0x85D8, Nhidden + Nout))

    print("ndsram", read_memory(daughterboard, buffer, 0x89C8, Nhidden + Nout))
    print("rds2ram", read_memory(daughterboard, buffer, 0x8DB8, Nhidden))
    print("ndmram", read_memory(daughterboard, buffer, 0x91A0, Nhidden + Nout))

    print("nthram", read_memory(daughterboard, buffer, 0x9590, Nhidden + Nout))
    print("rcram", read_memory(daughterboard, buffer, 0x9980, Nhidden + Nout))
    print("raram", read_memory(daughterboard, buffer, 0x9D68, Nhidden + Nout))

    print("rspkram", read_memory(daughterboard, buffer, 0xA150, Nhidden))

    print("refocram", read_memory(daughterboard, buffer, 0xA538, Nhidden))
    print("rforam", read_memory(daughterboard, buffer, 0xA920, Nhidden))

    print("rwtram", read_memory(daughterboard, buffer, 0x12620, Nhidden), "...")
    print("rwt2ram", read_memory(daughterboard, buffer, 0x1A320, Nhidden), "...")

    print("owtram", read_memory(daughterboard, buffer, 0x22020, Nhidden * Nout))


def print_debug_registers(
    daughterboard: PollenDaughterBoard, buffer: PollenReadBuffer
) -> None:
    """
    Print register contents for debugging purposes

    Args:
        daughterboard (PollenDaughterBoard): A Pollen daughterboard to debug
        buffer (PollenReadBuffer): A connected Pollen read buffer to use in reading registers
    """
    print("ctrl1", bin(read_register(daughterboard, buffer, 0x1)[0]))
    print("ctrl2", hex(read_register(daughterboard, buffer, 0x2)[0]))
    print("dbg_ctrl1", bin(read_register(daughterboard, buffer, 0x18)[0]))
    print("pwrctrl1", bin(read_register(daughterboard, buffer, 0x04)[0]))
    print("pwrctrl2", bin(read_register(daughterboard, buffer, 0x05)[0]))
    print("pwrctrl3", bin(read_register(daughterboard, buffer, 0x06)[0]))
    print("pwrctrl4", bin(read_register(daughterboard, buffer, 0x07)[0]))
    print("ispkreg00", bin(read_register(daughterboard, buffer, 0x0C)[0]))
    print("ispkreg01", bin(read_register(daughterboard, buffer, 0x0D)[0]))
    print("ispkreg10", bin(read_register(daughterboard, buffer, 0x0E)[0]))
    print("ispkreg11", bin(read_register(daughterboard, buffer, 0x0F)[0]))
    print("stat", bin(read_register(daughterboard, buffer, 0x10)[0]))
    print("int", bin(read_register(daughterboard, buffer, 0x11)[0]))
    print("omp_stat0", bin(read_register(daughterboard, buffer, 0x12)[0]))
    print("omp_stat1", bin(read_register(daughterboard, buffer, 0x13)[0]))
    print("omp_stat2", bin(read_register(daughterboard, buffer, 0x14)[0]))
    print("omp_stat3", bin(read_register(daughterboard, buffer, 0x15)[0]))


def num_buffer_neurons(Nhidden: int) -> int:
    """
    Number of buffer neurons required for this network on Pollen 1

    Args:
        Nhidden (int): Number of hidden layer neurons

    Returns:
        int: The number of buffer neurons
    """
    Nbuffer = 1 if Nhidden % 2 == 1 else 2
    return Nbuffer


def get_current_timestamp(
    daughterboard: PollenDaughterBoard, buffer: PollenReadBuffer
) -> int:
    """
    Retrieve the current timestamp on a Pollen HDK

    Args:
        daughterboard (PollenDaughterBoard): A Pollen HDK
        buffer (PollenReadBuffer): A connected read buffer for the pollen HDK

    Returns:
        int: The current timestamp on the Pollen HDK
    """

    # - Clear read buffer
    buffer.get_events()

    # - Trigger a readout event on Pollen
    e = samna.pollen.event.TriggerReadout()
    daughterboard.get_model().write([e])

    # - Wait for the readout event to be sent back, and extract the timestamp
    timestamp = None
    while timestamp is None:
        readout_events = [
            e
            for e in blocking_read(buffer, timeout=1.0)
            if isinstance(e, samna.pollen.event.Readout)
        ]
        if readout_events:
            timestamp = readout_events[0].timestamp

    # - Return the timestamp
    return timestamp


def select_accel_time_mode(
    daughterboard: PollenDaughterBoard,
    config: PollenConfiguration,
    state_monitor_buffer: PollenNeuronStateBuffer,
    monitor_Nhidden: Optional[int] = 0,
    monitor_Noutput: Optional[int] = 0,
) -> None:
    """
    Switch on accelerated-time mode on a Pollen daughterboard, and configure network monitoring

    Notes:
        Use :py:func:`new_pollen_state_monitor_buffer` to generate a buffer to monitor neuron and synapse state.

    Args:
        daughterboard (PollenDaughterBoard): A Pollen daughterboard to configure
        config (PollenConfiguration): The desired Pollen configuration to use
        state_monitor_buffer (PollenNeuronStateBuffer): A connect neuron state monitor buffer
        monitor_Nhidden (Optional[int]): The number of hidden neurons for which to monitor state during evolution. Default: ``0``, don't monitor any hidden neurons.
        monitor_Noutput (Optional[int]): The number of output neurons for which to monitor state during evolution. Default: ``0``, don't monitor any output neurons.
    """
    # - Select accelerated time mode
    config.operation_mode = samna.pollen.OperationMode.AcceleratedTime

    # - Configure reading out of neuron state during evolution
    if monitor_Nhidden + monitor_Noutput > 0:
        config.debug.monitor_neuron_i_syn = samna.pollen.configuration.NeuronRange(
            0, monitor_Nhidden + monitor_Noutput
        )
        config.debug.monitor_neuron_i_syn2 = samna.pollen.configuration.NeuronRange(
            0, monitor_Nhidden
        )
        config.debug.monitor_neuron_spike = samna.pollen.configuration.NeuronRange(
            0, monitor_Nhidden
        )
        config.debug.monitor_neuron_v_mem = samna.pollen.configuration.NeuronRange(
            0, monitor_Nhidden + monitor_Noutput
        )

    # - Write the configuration to the chip and configure the buffer accordingly
    state_monitor_buffer.set_configuration(config)
    apply_configuration(daughterboard, config)


def select_single_step_time_mode(
    daughterboard: PollenDaughterBoard, config: PollenConfiguration
) -> None:
    """
    Switch on single-step model on a Pollen daughterboard

    Args:
        daughterboard (PollenBaughterBoard): The Pollen HDK to configure
        config (PollenConfiguration): The desired Pollen configuration to use
    """
    # - Write the configuration
    config.operation_mode = samna.pollen.OperationMode.Manual
    apply_configuration(daughterboard, config)


def to_hex(n: int, digits: int) -> str:
    """
    Output a consistent-length hex string encoding a number

    Args:
        n (int): Number to export
        digits (int): Number of digits to produce

    Returns:
        str: HEx-encoded string, with ``digits`` digits
    """
    return "%s" % ("0000%x" % (n & 0xFFFFFFFF))[-digits:]


def export_config(
    path: Union[str, Path],
    config: PollenConfiguration,
    dt: float = None,
) -> None:
    """
    Export a network configuration to text files

    Args:
        path (Union[str, path]): Directory to write data
        config (PollenConfiguration): A Pollen configuraiton to export
        dt (float): The time step of the simulation
    """
    # - Check base path
    path = Path(path)
    if not path.exists():
        makedirs(path)

    # - Generate a PollenCim module from the config
    from rockpool.devices.pollen import PollenCim

    cim = PollenCim.from_config(config, dt=dt)
    model = cim._pollen_layer

    inp_size = len(model.synapses_in)
    num_neurons = len(model.synapses_rec)

    # transfer input synapses to matrix
    num_targets = num_neurons
    mat = np.zeros((2, inp_size, num_targets), dtype=int)
    for pre, syns in enumerate(model.synapses_in):
        for syn in syns:
            mat[syn.target_synapse_id, pre, syn.target_neuron_id] = syn.weight

    # iwtram and iwt2ram (input neurons of synapse IDs 0 and 1)
    for ram, mat_syn in zip(("iwt", "iwt2"), mat):
        # save to file
        print(f"Writing {ram}ram.ini", end="\r")
        with open(path / f"{ram}ram.ini", "w+") as f:
            for pre, line in enumerate(mat_syn):
                f.write(f"// {ram} for IN{pre} \n")
                for post, weight in enumerate(line):
                    f.write(to_hex(weight, 2))
                    f.write("\n")

    # create matrix for recurrent weights (slightly different convention than for input)
    mat = np.zeros((num_neurons, num_neurons, 2), dtype=int)
    for pre, syns in enumerate(model.synapses_rec):
        for syn in syns:
            mat[pre, syn.target_neuron_id, syn.target_synapse_id] = syn.weight

    # rwtram (recurrent neurons of synapse IDs 0)
    print("Writing rwtram.ini", end="\r")
    with open(path / "rwtram.ini", "w+") as f:
        for pre, line in enumerate(mat):
            f.write(f"// rwt of RSN{pre} \n")
            for syns in line:
                if np.any(syns != 0):
                    weight = syns[0]
                    f.write(to_hex(weight, 2))
                    f.write("\n")

    # rwtram2 (recurrent neurons of synapse IDs 1)
    print("Writing rwt2ram.ini", end="\r")
    with open(path / "rwt2ram.ini", "w+") as f:
        for pre, line in enumerate(mat):
            f.write(f"// rwt2 of RSN{pre} \n")
            for syns in line:
                if np.any(syns != 0):
                    weight = syns[1]
                    f.write(to_hex(weight, 2))
                    f.write("\n")

    # rforam (recurrent fanout, or target ids)
    print("Writing rforam.ini", end="\r")
    with open(path / "rforam.ini", "w+") as f:
        for pre, line in enumerate(mat):
            f.write(f"// rfo of RSN{pre} \n")
            for post, syns in enumerate(line):
                if np.any(syns != 0):
                    f.write(to_hex(post, 3))
                    f.write("\n")

    # refocram (recurrent effective fanout, or number of targets)
    print("Writing refocram.ini", end="\r")
    with open(path / "refocram.ini", "w+") as f:
        for pre, line in enumerate(mat):
            count = 0
            for post, syns in enumerate(line):
                if np.any(syns != 0):
                    count += 1
            f.write(to_hex(count, 2))
            f.write("\n")

    # owtram (output weights)
    # transfer output synapses to matrix
    post_ids_out = [
        [syn.target_neuron_id for syn in l_syn if syn.target_synapse_id == 0]
        for l_syn in model.synapses_out
    ]
    weights_out = [
        [syn.weight for syn in l_syn if syn.target_synapse_id == 0]
        for l_syn in model.synapses_out
    ]
    try:
        readout_size = max(max(post_ids_out)) + 1
    except:
        readout_size = 0

    size_total = readout_size + num_neurons
    mat = np.zeros((num_neurons, readout_size), dtype=int)
    for pre, (post, weight) in enumerate(zip(post_ids_out, weights_out)):
        mat[pre, post] = weight

    # save to file
    print("Writing owtram.ini", end="\r")
    with open(path / "owtram.ini", "w+") as f:
        for pre, line in enumerate(mat):
            f.write(f"// owt for RSN{pre} \n")
            for post, weight in enumerate(line):
                f.write(to_hex(weight, 2))
                f.write("\n")

    # ndmram (membrane time constants)
    mat = np.zeros(size_total, dtype=int)
    mat[:num_neurons] = [n.v_mem_decay for n in config.reservoir.neurons]
    mat[num_neurons:size_total] = [n.v_mem_decay for n in config.readout.neurons]

    # save to file
    print("Writing ndmram.ini", end="\r")
    with open(path / "ndmram.ini", "w+") as f:
        for pre, dash in enumerate(mat):
            f.write(to_hex(dash, 1))
            f.write("\n")

    # ndsram (synaptic time constants, ID=0)
    mat = np.zeros(size_total, dtype=int)
    mat[:num_neurons] = [n.i_syn_decay for n in config.reservoir.neurons]
    mat[num_neurons:size_total] = [n.i_syn_decay for n in config.readout.neurons]

    # save to file
    print("Writing ndsram.ini", end="\r")
    with open(path / "ndsram.ini", "w+") as f:
        for pre, dash in enumerate(mat):
            f.write(to_hex(dash, 1))
            f.write("\n")

    # nds2ram (synaptic time constants, ID=1)
    if config.synapse2_enable:
        mat = [n.i_syn2_decay for n in config.reservoir.neurons]
    else:
        mat = np.zeros(num_neurons, int)

    # save to file
    print("Writing nds2ram.ini", end="\r")
    with open(path / "nds2ram.ini", "w+") as f:
        for pre, dash in enumerate(mat):
            f.write(to_hex(dash, 1))
            f.write("\n")

    # nthram (thresholds)
    thresholds = [n.threshold for n in config.reservoir.neurons] + [
        n.threshold for n in config.readout.neurons
    ]

    # save to file
    print("Writing nthram.ini", end="\r")
    with open(path / "nthram.ini", "w+") as f:
        for pre, th in enumerate(thresholds):
            f.write(to_hex(th, 4))
            f.write("\n")

    # raram and rcram (aliases)
    mat = np.zeros(num_neurons, dtype=int) - 1
    is_source = np.zeros(num_neurons, dtype=int)
    is_target = np.zeros(num_neurons, dtype=int)
    num_sources = np.zeros(num_neurons, dtype=int)
    for i, aliases in enumerate(model.aliases):
        if len(aliases) > 0:
            mat[i] = aliases[0]
            is_source[i] = 1
            is_target[aliases[0]] += 1

    # save to file
    print("Writing raram.ini", end="\r")
    with open(path / "raram.ini", "w+") as f:
        for pre, alias in enumerate(mat):
            f.write(to_hex(alias, 3))
            f.write("\n")

    # save to file
    print("Writing rcram.ini", end="\r")
    with open(path / "rcram.ini", "w+") as f:
        for pre, issource in enumerate(is_source):
            # print(
            #     pre,
            #     "->",
            #     mat[pre],
            #     ":",
            #     is_target[mat[pre]],
            #     issource,
            #     is_target[pre],
            #     ((is_target[mat[pre]] > 1) << 2)
            #     + (issource << 1)
            #     + (is_target[pre] > 0),
            # )
            f.write(
                to_hex(
                    ((is_target[mat[pre]] > 1) << 2)
                    + (issource << 1)
                    + (is_target[pre] > 0),
                    1,
                )
            )
            f.write("\n")

    # basic config
    print("Writing basic_config.json", end="\r")
    with open(path / "basic_config.json", "w+") as f:
        conf = {}

        # number of neurons
        conf["IN"] = len(model.synapses_in)
        conf["RSN"] = len(model.synapses_rec)

        # determine output size by getting the largest target neuron id
        syns = np.hstack(model.synapses_out)
        conf["ON"] = int(np.max([s.target_neuron_id for s in syns]) + 1)

        # bit shift values
        conf["IWBS"] = model.weight_shift_out
        conf["RWBS"] = model.weight_shift_rec
        conf["OWBS"] = model.weight_shift_out

        # expansion neurons
        # if num_expansion is not None:
        #    conf["IEN"] = num_expansion

        # dt
        conf["time_resolution_wrap"] = config.time_resolution_wrap
        conf["DT"] = cim.dt

        # number of synapses
        n_syns = 1
        syns_in = np.hstack(model.synapses_in)
        if np.any(np.array([s.target_synapse_id for s in syns_in]) == 1):
            n_syns = 2
        syns_rec = np.hstack(model.synapses_rec)
        if np.any(np.array([s.target_synapse_id for s in syns_rec]) == 1):
            n_syns = 2

        conf["N_SYNS"] = n_syns

        # aliasing
        if max([len(a) for a in model.aliases]) > 0:
            conf["RA"] = True
        else:
            conf["RA"] = False

        json.dump(conf, f)


def export_frozen_state(
    path: Union[str, Path], config: PollenConfiguration, state: PollenState
) -> None:
    """
    Export a single frozen state of a Pollen network

    This function will produce a series of RAM initialisation files containing a Pollen state

    Args:
        path (Path): The directory to export the state to
        config (PollenConfiguration): The configuration of the Pollen network
        state (PollenState): A single time-step state of a Pollen network to export
    """
    # - Make `path` a path
    path = Path(path)
    if not path.exists():
        makedirs(path)

    # - Check that we have a single time point
    for k, v in zip(state._fields, state):
        if k in ["Nhidden", "Nout"]:
            continue

        assert (v.shape[0] == 1) or (
            np.ndim(v) == 1
        ), "`state` must define a single time point"

    # - Determine network size
    inp_size = np.shape(config.input.weights)[0]
    num_neurons = np.shape(config.reservoir.weights)[1]
    readout_size = np.shape(config.readout.weights)[1]
    size_total = num_neurons + readout_size

    T = 1

    # rspkram
    mat = np.zeros((T, num_neurons), dtype=int)
    spks = np.array(np.atleast_2d(state.Spikes_hid)).astype(int)

    if len(spks) > 0:
        mat[:, : spks.shape[1]] = spks

        print("Writing rspkram.ini", end="\r")
        for t, spks in enumerate(mat):
            with open(path / f"rspkram.ini", "w+") as f:
                for val in spks:
                    f.write(to_hex(val, 2))
                    f.write("\n")

    # ospkram
    mat = np.zeros((T, readout_size), dtype=int)
    spks = np.array(np.atleast_2d(state.Spikes_out)).astype(int)

    if len(spks) > 0:
        mat[:, : spks.shape[1]] = spks
        print("Writing ospkram.ini", end="\r")
        for t, spks in enumerate(mat):
            with open(path / f"ospkram.ini", "w+") as f:
                for val in spks:
                    f.write(to_hex(val, 2))
                    f.write("\n")

    # nscram
    mat = np.zeros((T, size_total), dtype=int)
    isyns = np.array(np.atleast_2d(state.I_syn_hid)).astype(int)
    mat[:, : isyns.shape[1]] = isyns
    isyns_out = np.array(np.atleast_2d(state.I_syn_out)).astype(int)
    mat[:, num_neurons : num_neurons + isyns_out.shape[1]] = isyns_out

    print("Writing nscram.ini", end="\r")
    for t, vals in enumerate(mat):
        with open(path / f"nscram.ini", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 4))
                f.write("\n")

    # nsc2ram
    if not hasattr(state, "I_syn2_hid"):
        mat = np.zeros((0, num_neurons), int)
    else:
        mat = np.zeros((T, num_neurons), dtype=int)
        isyns2 = np.array(np.atleast_2d(state.I_syn2_hid)).astype(int)
        mat[:, : isyns2.shape[1]] = isyns2

    print("Writing nsc2ram.ini", end="\r")
    for t, vals in enumerate(mat):
        with open(path / f"nsc2ram.ini", "w+") as f:
            for val in vals:
                f.write(to_hex(val, 4))
                f.write("\n")

    # nmpram
    mat = np.zeros((T, size_total), dtype=int)
    vmems = np.array(np.atleast_2d(state.V_mem_hid)).astype(int)
    mat[:, : vmems.shape[1]] = vmems
    vmems_out = np.array(np.atleast_2d(state.V_mem_out)).astype(int)
    mat[:, num_neurons : num_neurons + vmems_out.shape[1]] = vmems_out
    print("Writing nmpram.ini", end="\r")
    for t, vals in enumerate(mat):
        with open(path / f"nmpram.ini", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 4))
                f.write("\n")


def export_temporal_state(
    path: Union[Path, str],
    config: PollenConfiguration,
    inp_spks: np.ndarray,
    state: PollenState,
) -> None:
    """
    Export the state of a Pollen network over time

    This function will produce a series of RAM files, per time-step, containing the recorded state evolution of a Pollen network.

    Args:
        path (Path): The directory to export the state to
        config (PollenConfiguration): The configuration of the Pollen network
        inp_spks (np.ndarray): The input spikes for this simulation
        state (PollenState): A temporal state of a Pollen network to export
    """
    # - Make `path` a path
    path = Path(path)

    # - Determine network size
    inp_size = np.shape(config.input.weights)[0]
    num_neurons = np.shape(config.reservoir.weights)[1]
    readout_size = np.shape(config.readout.weights)[1]
    size_total = num_neurons + readout_size

    # rspkram
    mat = np.zeros((np.shape(state.Spikes_hid)[0], num_neurons), dtype=int)
    spks = np.array(state.Spikes_hid).astype(int)

    if len(spks) > 0:
        mat[:, : spks.shape[1]] = spks

        path_spkr = path / "spk_res"
        if not path_spkr.exists():
            makedirs(path_spkr)

        print("Writing rspkram files in spk_res", end="\r")
        for t, spks in enumerate(mat):
            with open(path_spkr / f"rspkram_{t}.txt", "w+") as f:
                for val in spks:
                    f.write(to_hex(val, 2))
                    f.write("\n")

    # ospkram
    mat = np.zeros((np.shape(state.Spikes_out)[0], readout_size), dtype=int)
    spks = np.array(state.Spikes_out).astype(int)

    if len(spks) > 0:
        mat[:, : spks.shape[1]] = spks

        path_spko = path / "spk_out"
        if not path_spko.exists():
            makedirs(path_spko)

        print("Writing ospkram files in spk_out", end="\r")
        for t, spks in enumerate(mat):
            with open(path_spko / f"ospkram_{t}.txt", "w+") as f:
                for val in spks:
                    f.write(to_hex(val, 2))
                    f.write("\n")

    # nscram
    mat = np.zeros((np.shape(state.I_syn_hid)[0], size_total), dtype=int)
    isyns = np.array(state.I_syn_hid).astype(int)
    mat[:, : isyns.shape[1]] = isyns
    isyns_out = np.array(state.I_syn_out).astype(int)
    mat[:, num_neurons : num_neurons + isyns_out.shape[1]] = isyns_out

    path_isyn = path / "isyn"
    if not path_isyn.exists():
        makedirs(path_isyn)

    print("Writing nscram files in isyn", end="\r")
    for t, vals in enumerate(mat):
        with open(path_isyn / f"nscram_{t}.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 4))
                f.write("\n")

    # nsc2ram
    if not hasattr(state, "I_syn2_hid"):
        mat = np.zeros((0, num_neurons), int)
    else:
        mat = np.zeros((np.shape(state.I_syn2_hid)[0], num_neurons), dtype=int)
        isyns2 = np.array(state.I_syn2_hid).astype(int)
        mat[:, : isyns2.shape[1]] = isyns2

    path_isyn2 = path / "isyn2"
    if not path_isyn2.exists():
        makedirs(path_isyn2)

    print("Writing nsc2ram files in isyn2", end="\r")
    for t, vals in enumerate(mat):
        with open(path_isyn2 / f"nsc2ram_{t}.txt", "w+") as f:
            for val in vals:
                f.write(to_hex(val, 4))
                f.write("\n")

    # nmpram
    mat = np.zeros((np.shape(state.V_mem_hid)[0], size_total), dtype=int)
    vmems = np.array(state.V_mem_hid).astype(int)
    mat[:, : vmems.shape[1]] = vmems
    vmems_out = np.array(state.V_mem_out).astype(int)
    mat[:, num_neurons : num_neurons + vmems_out.shape[1]] = vmems_out

    path_vmem = path / "vmem"
    if not path_vmem.exists():
        makedirs(path_vmem)

    print("Writing nmpram files in vmem", end="\r")
    for t, vals in enumerate(mat):
        with open(path_vmem / f"nmpram_{t}.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 4))
                f.write("\n")

    if inp_spks is not None:
        # input spikes
        path_spki = path / "spk_in"
        if not path_spki.exists():
            makedirs(path_spki)

        print("Writing inp_spks.txt", end="\r")
        with open(path_spki / "inp_spks.txt", "w+") as f:
            idle = -1
            for t, chans in enumerate(inp_spks):
                idle += 1
                if not np.all(chans == 0):
                    f.write(f"// time step {t}\n")
                    if idle > 0:
                        f.write(f"idle {idle}\n")
                    idle = 0
                    for chan, num_spikes in enumerate(chans):
                        for _ in range(num_spikes):
                            f.write(f"wr IN{to_hex(chan, 1)}\n")