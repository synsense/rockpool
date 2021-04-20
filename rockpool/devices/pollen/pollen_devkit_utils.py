"""
Utilities for working with the Pollen HDK
"""

# - Check that Samna is installed
from importlib import util

if util.find_spec("samna") is None:
    raise ModuleNotFoundError(
        "'samna' not found. Modules that rely on Samna will not be available."
    )

import samna
from samna.pollen.configuration import PollenConfiguration

from warnings import warn
import time

import numpy as np

from typing import Any, List, Iterable, Optional, NamedTuple

PollenDaughterBoard = Any
SamnaDeviceNode = Any
PollenReadBuffer = samna.BufferSinkNode_pollen_event_output_event


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

    # - Open the devices and start the reader/writer
    for d in unopened_devices:
        device_node.DeviceController.open_device(d, "board")
        device_node.board.start_reader_writer()

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


def new_pollen_output_buffer(
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
    output_buf = PollenReadBuffer()

    # - Get the device model
    model = daughterboard.get_model()

    # - Get Pollen output event source node
    source_node = model.get_source_node()

    # - Add the buffer as a destination for the Pollen output events
    success = source_node.add_destination(output_buf.get_input_channel())
    assert success, "Error connecting the new buffer."

    # - Return the buffer
    return output_buf


def blocking_read(
    buffer: PollenReadBuffer,
    count: int = 1,
    timeout: float = None,
) -> List:
    """
    Perform a blocking read on a buffer, optionally waiting for a certain count

    Args:
        buffer (PollenReadBuffer): A buffer to read from
        count (Optional[int]): The count of required events. Default: ``1``, just wait for any data.
        timeout (Optional[float]): The time in seconds to wait for a result. Default: ``None``, no timeout.

    Returns: List: A list of read events
    """
    all_events = []

    # - Read at least a certain number of events
    is_timeout = False
    start_time = time.time()
    while len(all_events) < count and not is_timeout:
        events = buffer.get_buf()
        all_events.extend(events)
        if timeout is not None:
            is_timeout = (time.time() - start_time) > timeout

    return all_events


def initialise_pollen_hdk(daughterboard: PollenDaughterBoard) -> None:
    """
    Initialise the Pollen HDK

    Args:
        daughterboard (PollenDaughterBoard): A Pollen daughterboard to initialise
    """
    # - Power on pollen
    def set_pollen_power(io, value):
        io.write_config(0x0052, value)

    def set_spi_clock(io):
        io.write_config(0x0001, 0x0000)
        io.write_config(0x0002, 0x0009)
        io.write_config(0x0003, 0x0001)

    def set_sAer_clock(io):
        io.write_config(0x0010, 0x0000)
        io.write_config(0x0011, 0x0009)
        io.write_config(0x0012, 0x0001)

    # - Configure power for Pollen
    io = daughterboard.get_io_module()
    set_pollen_power(io, 3)

    # - Strobe Pollen reset
    io.deassert_reset()
    io.assert_reset()

    set_spi_clock(io)
    set_sAer_clock(io)


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
    daughterboard.get_io_module().write([wwv_ev])


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
        address (int):

    Returns: List[int]: A list of events returned from the read
    """
    # - Set up a register read
    rrv_ev = samna.pollen.event.ReadRegisterValue()
    rrv_ev.address = address

    # - Clear buffer
    buffer.get_buf()

    # - Request read
    daughterboard.get_io_module().write([rrv_ev])

    # - Wait for data and read it
    events = blocking_read(buffer, count=1, timeout=1.0)

    # - Filter returned events for the desired address
    ev_filt = [e for e in events if e.address == address]

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

    Returns: List[int]: A list of values read from memory
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
    buffer.get_buf()

    # - Request read
    daughterboard.get_io_module().write(read_events_list)

    # - Read data
    events = blocking_read(buffer, count)

    # - Filter returned events for the desired addresses
    return [
        e.data
        for e in events[1:]
        if e.address >= start_address and e.address < start_address + count
    ]


def verify_pollen_version(
    daughterboard: PollenDaughterBoard,
    buffer: PollenReadBuffer,
) -> bool:
    """
    Verify that the provided daughterbaord returns the correct version ID for Pollen

    Args:
        daughterboard (PollenDaughterBoard): A daughter-board object to test
        buffer (samna.BufferSinkNode_pollen_event_output_event): A read buffer

    Returns: bool: ``True`` iff the version ID is correct for Pollen
    """
    # - Read from the version register
    version = read_register(daughterboard, buffer, 0x0)

    # - Search for the correct version value
    return any([v == 65536 for v in version])


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
        daughterboard.get_io_module().write(
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
        daughterboard (PollenDaughterboard): The daughterboard to zero
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


def reset_neuron_synapse_state(
    daughterboard: PollenDaughterBoard, Nhidden: int = 1000, Nout: int = 8
) -> None:
    """
    Reset the neuron and synapse state on a Pollen HDK

    Args:
        daughterboard (PollenDaughterboard): The Pollen HDK daughterboard to reset
        Nhidden (int): The number of hidden neurons. Default: 1000 (reset all neurons)
        Nout (int): THe number of output neurons. Default: 8 (reset all neurons)
    """
    # - Define the state memory banks, along with the required number of elements to zero
    memory_table = {
        "nscram": (0x7E00, Nhidden + Nout),
        "rsc2ram": (0x81F0, Nhidden),
        "nmpram": (0x85D8, Nhidden + Nout),
        "rspkram": (0xA150, Nhidden),
    }

    # - Zero each bank in turn
    for bank in memory_table.values():
        write_memory(daughterboard, *bank)


def apply_configuration(
    daughterboard: PollenDaughterBoard, config: PollenConfiguration
) -> None:
    """
    Apply a configuration to the Pollen HDK

    Args:
        daughterboard (PollenDaughterboard): The Pollen HDK to write the configuration to
        config (PollenConfiguration): A configuration for Pollen
    """
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
        buffer (PollenReadBuffer):
        Nhidden (int): Number of hidden neurons to read. Default: ``1000`` (all neurons).
        Nout (int): Number of output neurons to read. Defualt: ``8`` (all neurons).

    Returns: :py:class:`.PollenState`: The recorded state as a ``NamedTuple``. Contains keys ``V_mem_hid``,  ``V_mem_out``, ``I_syn_hid``, ``I_syn_out``, ``I_syn2_hid``, ``Nhidden``, ``Nout``

    """
    # ADD RESERVOIR SPIKE RAM TO THIS READING

    # - Define the memory bank addresses
    memory_table = {
        "nscram": 0x7E00,
        "rsc2ram": 0x81F0,
        "nmpram": 0x85D8,
        "rspkram": 0xA150,
    }

    # - Read synaptic currents
    Isyn = read_memory(daughterboard, buffer, memory_table["nscram"], Nhidden + Nout)

    # - Read synaptic currents 2
    Isyn2 = read_memory(daughterboard, buffer, memory_table["rsc2ram"], Nhidden)

    # - Read membrane potential
    Vmem = read_memory(daughterboard, buffer, memory_table["nmpram"], Nhidden + Nout)

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
    write_register(daughterboard, 0x09, 0x10)


def reset_input_spikes(daughterboard: PollenDaughterBoard) -> None:
    """
    Reset the input spike registers on a Pollen HDK

    Args:
        daughterboard (PollenDaughterboard): The Pollen HDK to access
    """
    for register in range(4):
        write_register(daughterboard, 0x0C + register)
