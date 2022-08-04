"""
Low-level device kit utilities for the SYNS61201 Xylo-A2 HDK
"""

from rockpool.utilities.backend_management import backend_available

if not backend_available("samna"):
    raise ModuleNotFoundError(
        "`samna` not found. The Xylo HDK requires `samna` for interfacing."
    )

import samna

# - Other imports
from warnings import warn
import time
import numpy as np
from pathlib import Path
from os import makedirs
import json

# - Typing and useful proxy types
from typing import Any, List, Iterable, Optional, NamedTuple, Union, Tuple

Xylo2ReadBuffer = samna.BasicSinkNode_xylo_core2_event_output_event
Xylo2WriteBuffer = samna.BasicSourceNode_xylo_core2_event_input_event
Xylo2NeuronStateBuffer = samna.xyloCore2.NeuronStateSinkNode

AFE2ReadBuffer = samna.BasicSinkNode_afe2_event_output_event
AFE2WriteBuffer = samna.BasicSourceNode_afe2_event_input_event
AFE2OutputEvent = samna.afe2.event.Spike

from samna.xyloCore2.configuration import XyloConfiguration

XyloA2HDK = Any


class XyloState(NamedTuple):
    """
    `.NamedTuple` that encapsulates a recorded Xylo HDK state
    """

    Nin: int
    """ int: The number of input-layer neurons """

    Nhidden: int
    """ int: The number of hidden-layer neurons """

    Nout: int
    """ int: The number of output layer neurons """

    V_mem_hid: np.ndarray
    """ np.ndarray: Membrane potential of hidden neurons ``(Nhidden,)``"""

    I_syn_hid: np.ndarray
    """ np.ndarray: Synaptic current 1 of hidden neurons ``(Nhidden,)``"""

    V_mem_out: np.ndarray
    """ np.ndarray: Membrane potential of output neurons ``(Nhidden,)``"""

    I_syn_out: np.ndarray
    """ np.ndarray: Synaptic current of output neurons ``(Nout,)``"""

    I_syn2_hid: np.ndarray
    """ np.ndarray: Synaptic current 2 of hidden neurons ``(Nhidden,)``"""

    Spikes_hid: np.ndarray
    """ np.ndarray: Spikes from hidden layer neurons ``(Nhidden,)``"""

    Spikes_out: np.ndarray
    """ np.ndarray: Spikes from output layer neurons ``(Nout,)``"""


def find_xylo_a2_boards() -> List[XyloA2HDK]:
    """
    Search for and return a list of Xylo AFE V2 HDKs

    Iterate over devices and search for Xylo AFE V2 HDK nodes. Return a list of available AFE HDKs, or an empty list if none are found.

    Returns:
        List[AFEHDK]: A (possibly empty) list of AFE HDK nodes.
    """
    # - Get a list of devices
    device_list = samna.device.get_all_devices()

    # - Search for a xylo dev kit
    afev2_hdk_list = [
        samna.device.open_device(d)
        for d in device_list
        if d.device_type_name == "XyloA2TestBoard"
    ]

    return afev2_hdk_list


def read_afe2_register(
    read_buffer: AFE2ReadBuffer,
    write_buffer: AFE2WriteBuffer,
    address: int,
    timeout: float = 2.0,
) -> List[int]:
    """
    Read the contents of a register

    Args:
        read_buffer (AFE2ReadBuffer): A connected read buffer to the XYlo HDK
        write_buffer (AFE2WriteBuffer): A connected write buffer to the Xylo HDK
        address (int): The register address to read
        timeout (float): A timeout in seconds

    Returns:
        List[int]: A list of events returned from the read
    """
    # - Set up a register read
    rrv_ev = samna.afe2.event.ReadRegisterValue()
    rrv_ev.address = address

    # - Request read
    write_buffer.write([rrv_ev])

    # - Wait for data and read it
    start_t = time.time()
    continue_read = True
    while continue_read:
        # - Read from the buffer
        events = read_buffer.get_events()

        # - Filter returned events for the desired address
        ev_filt = [e for e in events if hasattr(e, "address") and e.address == address]

        # - Should we continue the read?
        continue_read &= len(ev_filt) == 0
        continue_read &= (time.time() - start_t) < timeout

    # - If we didn't get the required register read, raise an error
    if len(ev_filt) == 0:
        raise TimeoutError(f"Timeout after {timeout}s when reading register {address}.")

    # - Return adta
    return [e.data for e in ev_filt]


def write_afe2_register(
    write_buffer: AFE2WriteBuffer, register: int, data: int = 0
) -> None:
    """
    Write data to a register on a Xylo AFE2 HDK

    Args:
        write_buffer (AFE2WriteBuffer): A connected write buffer to the desintation Xylo AFE2 HDK
        register (int): The address of the register to write to
        data (int): The data to write. Default: 0x0
    """
    wwv_ev = samna.afe2.event.WriteRegisterValue()
    wwv_ev.address = register
    wwv_ev.data = data
    write_buffer.write([wwv_ev])


def read_afe2_events_blocking(
    afe2hdk: XyloA2HDK,
    write_buffer: AFE2WriteBuffer,
    afe_read_buf: AFE2ReadBuffer,
    duration: float,
) -> (np.ndarray, np.ndarray):
    """
    Perform a blocking read of AFE2 audio spike events for a desired duration

    Args:
        afe2hdk (AFE2HDK): A device node for an AFE2 HDK
        write_buffer (AFE2WriteBuffer): A connected write buffer to an AFE2 HDK
        afe_read_buf (AFE2ReadBuffer): A connected read buffer from an AFE2 HDK
        duration (float): The desired duration to record from, in seconds

    Returns:
        (np.ndarray, np.ndarray) timestamps, channels
        timestamps: A list of event timestamps, in seconds from the start of recording
        channels: A list of event channels, corresponding to the event timestamps
    """
    # - Get AFE handler
    afe_handler = afe2hdk.get_io_module().get_afe_handler()

    # - Enable AER monitor mode for AFE2
    write_afe2_register(write_buffer, 0x45, 0x30000)
    # time.sleep(0.1)

    # - Clear events buffer
    afe_read_buf.get_events()

    # - Trigger recording for desired duration
    afe2hdk.get_stop_watch().set_enable_value(True)
    afe2hdk.get_stop_watch().reset()
    afe_handler.enable_event_monitor(True)
    time.sleep(duration)
    afe_handler.enable_event_monitor(False)
    time.sleep(0.1)

    # write_spi(0x45,0)
    # time.sleep(0.5)

    # - Read and filter events
    events = afe_read_buf.get_events()
    events = [
        (e.timestamp, e.channel)
        for e in events
        if isinstance(e, samna.afe2.event.Spike) and e.timestamp <= duration * 1e6
    ]

    # - Sort events by time
    if len(events) > 0:
        events = np.stack(events)
        index_array = np.argsort(events[:, 0])

        # - Convert to vectors of timestamps, channels
        timestamps = events[index_array, 0]
        channels = events[index_array, 1]
    else:
        timestamps = np.zeros(0)
        channels = np.zeros(0)

    # - Return timestamps in seconds and channels
    return timestamps * 1e-6, channels


def apply_afe2_default_config(afe2hdk: XyloA2HDK) -> None:
    """
    Configure an AFE2 HDK, including self-calibration

    Args:
        afe_write_buf (AFE2WriteBuffer): A connected AFE2 write buffer
    """
    c = samna.afe2.configuration.AfeConfiguration()

    c.analog_top.enable = True
    c.debug.enable_event_monitor = False

    c.analog_top.bpf.bias = 2
    c.analog_top.fwr.bias = 6

    c.analog_top.lna.ci_tune = 5
    c.analog_top.lna.cf_tune = 5

    c.analog_top.bpf.scale = True

    afe2hdk.get_afe_model().apply_configuration(c)

    time.sleep(45)

    c.aer_2_saer.calibration.mode = 1
    c.aer_2_saer.calibration.reset = True

    c.aer_2_saer.calibration.afe_stable_time = 0x80
    c.aer_2_saer.calibration.leak_timing_window = 0x2625A0

    c.aer_2_saer.calibration.leak_td = 0x030D4
    c.aer_2_saer.calibration.leak_target_spike_number = 2

    afe2hdk.get_afe_model().apply_configuration(c)


def read_afe2_module_version(
    afe_read_buf: AFE2ReadBuffer, afe_write_buf: AFE2WriteBuffer
) -> (int, int):
    """
    Return the version and revision numbers for a connected AFE2 HDK

    Args:
        afe_read_buf (AFE2ReadBuffer): A connected AFE2 read buffer
        afe_write_buf (AFE2WriteBuffer): A connected AFE2 write buffer

    Returns:
        (int, int): version, revision numbers of the connected chip
    """
    # - Read the version register
    version_revision = read_afe2_register(afe_read_buf, afe_write_buf, 0x0)[0]

    # - Separate version and revision
    version = (version_revision & 0xFFFF0000) >> 16
    revision = version_revision & 0x0000FFFF

    return version, revision


def new_xylo_read_buffer(
    hdk: XyloA2HDK,
) -> Xylo2ReadBuffer:
    """
    Create and connect a new buffer to read from a Xylo HDK

    Args:
        hdk (XyloDaughterBoard):

    Returns:
        samna.BasicSinkNode_xylo_event_output_event: Output buffer receiving events from Xylo HDK
    """
    # - Register a buffer to read events from Xylo
    buffer = Xylo2ReadBuffer()

    # - Get the device model
    model = hdk.get_xylo_model()

    # - Get Xylo output event source node
    source_node = model.get_source_node()

    # - Add the buffer as a destination for the Xylo output events
    graph = samna.graph.EventFilterGraph()
    graph.sequential([source_node, buffer])

    # - Return the buffer
    return buffer


def new_xylo_write_buffer(hdk: XyloA2HDK) -> Xylo2WriteBuffer:
    """
    Create a new buffer for writing events to a Xylo HDK

    Args:
        hdk (XyloDaughterBoard): A Xylo HDK to create a new buffer for

    Returns:
        XyloWriteBuffer: A connected event write buffer
    """
    buffer = Xylo2WriteBuffer()
    sink = hdk.get_xylo_model().get_sink_node()
    graph = samna.graph.EventFilterGraph()
    graph.sequential([buffer, sink])

    return buffer


def new_xylo_state_monitor_buffer(
    hdk: XyloA2HDK,
) -> Xylo2NeuronStateBuffer:
    """
    Create a new buffer for monitoring neuron and synapse state and connect it

    Args:
        hdk (XyloDaughterBoard): A Xylo HDK to configure

    Returns:
        XyloNeuronStateBuffer: A connected neuron / synapse state monitor buffer
    """
    # - Register a new buffer to receive neuron and synapse state
    buffer = Xylo2NeuronStateBuffer()

    # - Get the device model
    model = hdk.get_xylo_model()

    # - Get Xylo output event source node
    source_node = model.get_source_node()

    # - Add the buffer as a destination for the Xylo output events
    graph = samna.graph.EventFilterGraph()
    graph.sequential([source_node, buffer])

    # - Return the buffer
    return buffer


def blocking_read(
    read_buffer: Xylo2ReadBuffer,
    target_timestamp: Optional[int] = None,
    count: Optional[int] = None,
    timeout: Optional[float] = None,
) -> (List, bool):
    """
    Perform a blocking read on a buffer, optionally waiting for a certain count, a target timestamp, or imposing a timeout

    You should not provide `count` and `target_timestamp` together.

    Args:
        read_buffer (XyloReadBuffer): A buffer to read from
        target_timestamp (Optional[int]): The desired final timestamp. Read until this timestamp is returned in an event. Default: ``None``, don't wait until a particular timestamp is read.
        count (Optional[int]): The count of required events. Default: ``None``, just wait for any data.
        timeout (Optional[float]): The time in seconds to wait for a result. Default: ``None``, no timeout: block until a read is made.

    Returns:
        (List, bool): `event_list`, `is_timeout`
        `event_list` is a list of events read from the HDK. `is_timeout` is a boolean flag indicating that the read resulted in a timeout
    """
    all_events = []

    # - Read at least a certain number of events
    continue_read = True
    is_timeout = False
    start_time = time.time()
    while continue_read:
        # - Perform a read and save events
        time.sleep(0.1)
        events = read_buffer.get_events()
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
            is_timeout = (time.time() - start_time) > timeout
            continue_read &= not is_timeout

        # - Check number of events read
        if count:
            continue_read &= len(all_events) < count

        # - Continue reading if no events have been read
        if not target_timestamp and not count:
            continue_read &= len(all_events) == 0

    # - Perform one final read for good measure
    all_events.extend(read_buffer.get_events())

    # - Return read events
    return all_events, is_timeout


def gen_clear_input_registers_events() -> List:
    """
    Create events to clear the input event registers
    """
    events = []
    for addr in [0x11, 0x12, 0x13, 0x14]:
        event = samna.xyloCore2.event.WriteRegisterValue()
        event.address = addr
        events.append(event)

    return events


def initialise_xylo_hdk(write_buffer: Xylo2WriteBuffer) -> None:
    """
    Initialise the Xylo HDK

    Args:
        write_buffer (XyloWriteBuffer): A write buffer connected to a Xylo HDK to initialise
    """
    # - Always need to advance one time-step to initialise
    advance_time_step(write_buffer)


def write_register(
    write_buffer: Xylo2WriteBuffer,
    register: int,
    data: int = 0,
) -> None:
    """
    Write data to a register on a Xylo HDK

    Args:
        write_buffer (XyloWriteBuffer): A connected write buffer to the desintation Xylo HDK
        register (int): The address of the register to write to
        data (int): The data to write. Default: 0x0
    """
    wwv_ev = samna.xyloCore2.event.WriteRegisterValue()
    wwv_ev.address = register
    wwv_ev.data = data
    write_buffer.write([wwv_ev])


def read_register(
    read_buffer: Xylo2ReadBuffer,
    write_buffer: Xylo2WriteBuffer,
    address: int,
    timeout: float = 2.0,
) -> List[int]:
    """
    Read the contents of a register

    Args:
        read_buffer (XyloReadBuffer): A connected read buffer to the XYlo HDK
        write_buffer (XyloWriteBuffer): A connected write buffer to the Xylo HDK
        address (int): The register address to read
        timeout (float): A timeout in seconds

    Returns:
        List[int]: A list of events returned from the read
    """
    # - Set up a register read
    rrv_ev = samna.xyloCore2.event.ReadRegisterValue()
    rrv_ev.address = address

    # - Request read
    write_buffer.write([rrv_ev])

    # - Wait for data and read it
    start_t = time.time()
    continue_read = True
    while continue_read:
        # - Read from the buffer
        events = read_buffer.get_events()

        # - Filter returned events for the desired address
        ev_filt = [e for e in events if hasattr(e, "address") and e.address == address]

        # - Should we continue the read?
        continue_read &= len(ev_filt) == 0
        continue_read &= (time.time() - start_t) < timeout

    # - If we didn't get the required register read, raise an error
    if len(ev_filt) == 0:
        raise TimeoutError(f"Timeout after {timeout}s when reading register {address}.")

    # - Return adta
    return [e.data for e in ev_filt]


def read_memory(
    read_buffer: Xylo2ReadBuffer,
    write_buffer: Xylo2WriteBuffer,
    start_address: int,
    count: int = 1,
    read_timeout: float = 2.0,
) -> List[int]:
    """
    Read a block of memory from a Xylo HDK

    Args:
        read_buffer (XyloReadBuffer): A connected read buffer to the desired Xylo HDK
        write_buffer (XyloWriteBuffer): A connected write buffer to the desired Xylo HDK
        start_address (int): The base address to start reading from
        count (int): The number of elements to read

    Returns:
        List[int]: A list of values read from memory
    """
    # - Set up a memory read
    read_events_list = []

    # - Insert an extra read to avoid zero data
    rmv_ev = samna.xyloCore2.event.ReadMemoryValue()
    rmv_ev.address = start_address
    read_events_list.append(rmv_ev)

    for elem in range(count):
        rmv_ev = samna.xyloCore2.event.ReadMemoryValue()
        rmv_ev.address = start_address + elem
        read_events_list.append(rmv_ev)

    # - Clear buffer
    time.sleep(0.1)
    read_buffer.get_events()

    # - Request read
    write_buffer.write(read_events_list)

    # def write_spi(address, data):
    #     ev = samna.xyloCore2.event.WriteRegisterValue()
    #     ev.address = address
    #     ev.data = data
    #     events = [ev]
    #     write_buffer.write(events)
    # write_spi(0x09, 0x10)
    # time.sleep(0.01)

    # - Read data
    events, is_timeout = blocking_read(
        read_buffer, count=count + 1, timeout=read_timeout
    )
    if is_timeout:
        raise TimeoutError(
            f"Memory read timed out after {read_timeout} s. Reading @{start_address}+{count}."
        )

    # - Filter returned events for the desired addresses
    return [
        e.data
        for e in events[1:]
        if hasattr(e, "address")
        and e.address >= start_address
        and e.address < start_address + count
    ]


def generate_read_memory_events(
    start_address: int,
    count: int = 1,
) -> List[Any]:
    """
    Build a list of events that cause Xylo memory to be read

    This function is designed to be used with `decode_memory_read_events`.

    See Also:
        Use the `read_memory` function for a more convenient high-level API.

    Args:
        start_address (int): The starting address of the memory read
        count (int): The number of memory elements to read. Default: ``1``, read a single memory address.

    Returns:
        List: A list of events to send to a Xylo HDK
    """
    # - Set up a memory read
    read_events_list = []

    # - Insert an extra read to avoid zero data
    rmv_ev = samna.xyloCore2.event.ReadMemoryValue()
    rmv_ev.address = start_address
    read_events_list.append(rmv_ev)

    for elem in range(count):
        rmv_ev = samna.xyloCore2.event.ReadMemoryValue()
        rmv_ev.address = start_address + elem
        read_events_list.append(rmv_ev)

    return read_events_list


def decode_memory_read_events(
    events: List[Any],
    start_address: int,
    count: int = 1,
) -> List:
    """
    Decode a list of events containing memory reads from a Xylo HDK

    This is a low-level function designed to be used in conjuction with :py:func:`.generate_read_memory_events`.

    See Also:
        Use the :py:func:`read_memory` function for a more convenient high-level API.

    Args:
        events (List): A list of events read from a Xylo HDK
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


def verify_xylo_version(
    read_buffer: Xylo2ReadBuffer,
    write_buffer: Xylo2WriteBuffer,
    timeout: float = 1.0,
) -> bool:
    """
    Verify that the provided daughterbaord returns the correct version ID for Xylo

    Args:
        read_buffer (Xylo2ReadBuffer): A read buffer connected to the Xylo HDK
        write_buffer (Xylo2WriteBuffer): A write buffer connected to the Xylo HDK
        timeout (float): Timeout for checking in seconds

    Returns:
        bool: ``True`` iff the version ID is correct for Xylo V2
    """
    # - Clear the read buffer
    read_buffer.get_events()

    # - Read the version register
    write_buffer.write([samna.xyloCore2.event.ReadVersion()])

    # - Read events until timeout
    filtered_events = []
    t_end = time.time() + timeout
    while len(filtered_events) == 0:
        events = read_buffer.get_events()
        filtered_events = [
            e for e in events if isinstance(e, samna.xyloCore2.event.Version)
        ]

        # - Check timeout
        if time.time() > t_end:
            raise TimeoutError(f"Checking version timed out after {timeout}s.")

    return (
        (len(filtered_events) > 0)
        and (filtered_events[0].major == 1)
        and (filtered_events[0].minor == 1)
    )


def write_memory(
    write_buffer: Xylo2WriteBuffer,
    start_address: int,
    count: Optional[int] = None,
    data: Optional[Iterable] = None,
    chunk_size: int = 65535,
) -> None:
    """
    Write data to Xylo memory

    Args:
        write_buffer (XyloWriteBuffer): A write buffer connected to the desired Xylo HDK
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
        wmv_ev = samna.xyloCore2.event.WriteMemoryValue()
        wmv_ev.address = start_address + elem

        if data is not None:
            wmv_ev.data = data[elem]

        write_event_list.append(wmv_ev)

    # - Write the list of data events
    written = 0
    while written < len(write_event_list):
        write_buffer.write(write_event_list[written : (written + chunk_size)])
        written += chunk_size
        time.sleep(0.01)


def reset_neuron_synapse_state(
    hdk: XyloA2HDK,
    read_buffer: Xylo2ReadBuffer,
    write_buffer: Xylo2WriteBuffer,
) -> None:
    """
    Reset the neuron and synapse state on a Xylo HDK

    Args:
        hdk (XyloHDK): The Xylo HDK hdk to reset
        read_buffer (XyloReadBuffer): A read buffer connected to the Xylo HDK to reset
        write_buffer (XyloWriteBuffer): A write buffer connected to the Xylo HDK to reset
    """
    # - Get the current configuration
    config = hdk.get_xylo_model().get_configuration()

    # - Reset via configuration
    config.clear_network_state = True
    apply_configuration(hdk, config, read_buffer, write_buffer)


def apply_configuration(
    hdk: XyloA2HDK,
    config: XyloConfiguration,
    read_buffer: Xylo2ReadBuffer,
    write_buffer: Xylo2WriteBuffer,
) -> None:
    """
    Apply a configuration to the Xylo HDK

    Args:
        hdk (XyloHDK): The Xylo HDK to write the configuration to
        config (XyloConfiguration): A configuration for Xylo
        read_buffer (XyloReadBuffer): A connected read buffer for the Xylo HDK
        write_buffer (XyloWriteBuffer): A connected write buffer for the Xylo HDK
    """
    # - WORKAROUND: Manually enable debug clock
    config.debug.isyn_clock_enable = True
    config.debug.ra_clock_enable = True
    config.debug.hm_clock_enable = True
    config.debug.bias_clock_enable = True
    config.debug.isyn2_clock_enable = True

    config.debug.ram_power_enable = True

    # - Ideal -- just write the configuration using samna
    hdk.get_xylo_model().apply_configuration(config)

    # # - WORKAROUND: Design bug, where aliasing is not computed correctly
    # rcram = read_memory(read_buffer, write_buffer, 0x9980, 1000)
    # for i in range(1000):
    #     if rcram[i] == 2:
    #         rcram[i] = 3
    # write_memory(write_buffer, 0x9980, 1000, rcram)


def zero_memory(
    write_buffer: Xylo2WriteBuffer,
) -> None:
    """
    Clear all Xylo memory

    This function writes zeros to all memory banks on a Xylo HDK.

    Args:
        write_buffer (XyloWriteBuffer): A write buffer connected to the desired Xylo HDK
    """
    # - Define the memory banks
    memory_table = {
        "iwtram": (0x0100, 16000),
        "iwt2ram": (0x3F80, 16000),
        "nscram": (0x7E00, 1008),
        "rsc2ram": (0x81F0, 1000),
        "nmpram": (0x85D8, 1008),
        "ndsram": (0x89C8, 1008),
        "rds2ram": (0x8DB8, 1000),
        "ndmram": (0x91A0, 1008),
        "nbram": (0x9590, 1008),
        "nthram": (0x9980, 1008),
        "rcram": (0x9D70, 1000),
        "raram": (0xA158, 1000),
        "rspkram": (0xA540, 1000),
        "refocram": (0xA928, 1000),
        "rforam": (0xAD10, 32000),
        "rwtram": (0x12A10, 32000),
        "rwt2ram": (0x1A710, 32000),
        "owtram": (0x22410, 8000),
    }

    # - Zero each bank in turn
    for bank in memory_table.values():
        write_memory(write_buffer, *bank)


def read_neuron_synapse_state(
    read_buffer: Xylo2ReadBuffer,
    write_buffer: Xylo2WriteBuffer,
    Nin: int = 16,
    Nhidden: int = 1000,
    Nout: int = 8,
    record: bool = False,
    readout_mode: str = "Spike",
) -> XyloState:
    """
    Read and return the current neuron and synaptic state of neurons

    Args:
        read_buffer (XyloReadBuffer): A read buffer connected to the Xylo HDK
        write_buffer (XyloWriteBuffer): A write buffer connected to the Xylo HDK
        Nin (int): Number of input neurons to read. Default: ``16`` (all neurons).
        Nhidden (int): Number of hidden neurons to read. Default: ``1000`` (all neurons).
        Nout (int): Number of output neurons to read. Default: ``8`` (all neurons).

    Returns:
        :py:class:`.XyloState`: The recorded state as a ``NamedTuple``. Contains keys ``V_mem_hid``,  ``V_mem_out``, ``I_syn_hid``, ``I_syn_out``, ``I_syn2_hid``, ``Nhidden``, ``Nout``. This state has **no time axis**; the first axis is the neuron ID.

    """
    # - Define the memory bank addresses
    memory_table = {
        "nscram": 0x7E00,
        "rsc2ram": 0x81F0,
        "nmpram": 0x85D8,
        "rspkram": 0xA540,
    }
    Vmem, Isyn, Isyn2, Spikes = None, None, None, None

    if record:

        # - Read synaptic currents
        Isyn = read_memory(
            read_buffer,
            write_buffer,
            memory_table["nscram"],
            Nhidden + Nout,
        )

        # - Read synaptic currents 2
        Isyn2 = read_memory(read_buffer, write_buffer, memory_table["rsc2ram"], Nhidden)

        # - Read membrane potential
        Vmem = read_memory(
            read_buffer,
            write_buffer,
            memory_table["nmpram"],
            Nhidden + Nout,
        )

        # - Read reservoir spikes
        Spikes = read_memory(
            read_buffer, write_buffer, memory_table["rspkram"], Nhidden
        )
    else:

        if readout_mode == "Isyn":
            Isyn = read_memory(
                read_buffer,
                write_buffer,
                memory_table["nscram"] + Nhidden,
                Nout,
            )

        elif readout_mode == "Vmem":
            Vmem = read_memory(
                read_buffer,
                write_buffer,
                memory_table["nmpram"] + Nhidden,
                Nout,
            )

    # - Return the state
    return XyloState(
        Nin,
        Nhidden,
        Nout,
        np.array(Vmem[:Nhidden], "int16") if Vmem is not None else None,
        np.array(Isyn[:Nhidden], "int16") if Isyn is not None else None,
        np.array(Vmem[-Nout:], "int16") if Vmem is not None else None,
        np.array(Isyn[-Nout:], "int16") if Isyn is not None else None,
        np.array(Isyn2, "int16") if Isyn2 is not None else None,
        np.array(Spikes, "bool") if Spikes is not None else None,
        read_output_events(read_buffer, write_buffer)[:Nout],
    )


def is_xylo_ready(read_buffer: Xylo2ReadBuffer, write_buffer: Xylo2WriteBuffer) -> bool:
    """
    Query a Xylo HDK to see if it is ready for a time-step

    Args:
        read_buffer (XyloReadBuffer): A buffer to use while reading
        write_buffer (XyloWriteBuffer): A buffer to use while writing

    Returns: ``True`` iff the Xylo HDK has finished all processing
    """
    return read_register(read_buffer, write_buffer, 0x15)[-1] & (1 << 16) != 0


def advance_time_step(write_buffer: Xylo2WriteBuffer) -> None:
    """
    Take a single manual time-step on a Xylo HDK

    Args:
        write_buffer (XyloWriteBuffer): A write buffer connected to the Xylo HDK
    """
    e = samna.xyloCore2.event.TriggerProcessing()
    write_buffer.write([e])


def reset_input_spikes(write_buffer: Xylo2WriteBuffer) -> None:
    """
    Reset the input spike registers on a Xylo HDK

    Args:
        write_buffer (XyloWriteBuffer): A write buffer connected to the Xylo HDK to access
    """
    for register in range(4):
        write_register(write_buffer, 0x11 + register)


def send_immediate_input_spikes(
    write_buffer: Xylo2WriteBuffer,
    spike_counts: Iterable[int],
) -> None:
    """
    Send input events with no timestamp to a Xylo HDK

    Args:
        write_buffer (XyloWriteBuffer): A write buffer connected to a Xylo HDK
        spike_counts (Iterable[int]): An Iterable containing one slot per input channel. Each entry indicates how many events should be sent to the corresponding input channel.
    """
    # - Encode input events
    events_list = []
    for input_channel, event in enumerate(spike_counts):
        if event:
            for _ in range(int(event)):
                s_event = samna.xyloCore2.event.Spike()
                s_event.neuron_id = input_channel
                events_list.append(s_event)

    # - Send input spikes for this time-step
    write_buffer.write(events_list)


def read_output_events(
    read_buffer: Xylo2ReadBuffer, write_buffer: Xylo2WriteBuffer
) -> np.ndarray:
    """
    Read the spike flags from the output neurons on a Xylo HDK

    Args:
        read_buffer (XyloReadBuffer): A read buffer to use
        write_buffer (XyloWriteBuffer): A write buffer to use

    Returns:
        np.ndarray: A boolean array of output event flags
    """
    # - Read the status register
    status = read_register(read_buffer, write_buffer, 0x15)

    # - Convert to neuron events and return
    string = format(int(status[-1]), "0>32b")[-8:]
    return np.array([bool(int(e)) for e in string[::-1]], "bool")


def num_buffer_neurons(Nhidden: int) -> int:
    """
    Number of buffer neurons required for this network on Xylo 1

    Args:
        Nhidden (int): Number of hidden layer neurons

    Returns:
        int: The number of buffer neurons
    """
    Nbuffer = 2 if Nhidden % 2 == 1 else 1
    return Nbuffer


def get_current_timestamp(
    read_buffer: Xylo2ReadBuffer,
    write_buffer: Xylo2WriteBuffer,
    timeout: float = 3.0,
) -> int:
    """
    Retrieve the current timestamp on a Xylo HDK

    Args:
        read_buffer (XyloReadBuffer): A connected read buffer for the xylo HDK
        write_buffer (XyloWriteBuffer): A connected write buffer for the Xylo HDK
        timeout (float): A timeout for reading

    Returns:
        int: The current timestamp on the Xylo HDK
    """

    # - Clear read buffer
    read_buffer.get_events()

    # - Trigger a readout event on Xylo
    e = samna.xyloCore2.event.TriggerReadout()
    write_buffer.write([e])

    # - Wait for the readout event to be sent back, and extract the timestamp
    timestamp = None
    continue_read = True
    start_t = time.time()
    while continue_read:
        readout_events = read_buffer.get_events()
        ev_filt = [
            e for e in readout_events if isinstance(e, samna.xyloCore2.event.Readout)
        ]
        if ev_filt:
            timestamp = ev_filt[0].timestamp
            continue_read = False
        else:
            # - Check timeout
            continue_read &= (time.time() - start_t) < timeout

    if timestamp is None:
        raise TimeoutError(f"Timeout after {timeout}s when reading current timestamp.")

    # - Return the timestamp
    return timestamp


def configure_accel_time_mode(
    config: XyloConfiguration,
    state_monitor_buffer: Xylo2NeuronStateBuffer,
    monitor_Nhidden: Optional[int] = 0,
    monitor_Noutput: Optional[int] = 0,
    readout="Spike",
    record=False,
) -> (XyloConfiguration, Xylo2NeuronStateBuffer):
    """
    Switch on accelerated-time mode on a Xylo hdk, and configure network monitoring

    Notes:
        Use :py:func:`new_xylo_state_monitor_buffer` to generate a buffer to monitor neuron and synapse state.

    Args:
        config (XyloConfiguration): The desired Xylo configuration to use
        state_monitor_buffer (XyloNeuronStateBuffer): A connected neuron state monitor buffer
        monitor_Nhidden (Optional[int]): The number of hidden neurons for which to monitor state during evolution. Default: ``0``, don't monitor any hidden neurons.
        monitor_Noutput (Optional[int]): The number of output neurons for which to monitor state during evolution. Default: ``0``, don't monitor any output neurons.
        readout: The readout out mode for which to output neuron states. Default: ``Spike''.
        record (bool): Iff ``True``, record state during evolution. Default: ``False``, do not record state.

    Returns:
        (XyloConfiguration, XyloNeuronStateBuffer): `config` and `monitor_buffer`
    """
    assert readout in ["Isyn", "Vmem", "Spike"], f"{readout} is not supported."

    # - Select accelerated time mode
    config.operation_mode = samna.xyloCore2.OperationMode.AcceleratedTime

    config.debug.monitor_neuron_i_syn = None
    config.debug.monitor_neuron_i_syn2 = None
    config.debug.monitor_neuron_spike = None
    config.debug.monitor_neuron_v_mem = None

    if record:
        config.debug.monitor_neuron_i_syn = samna.xyloCore2.configuration.NeuronRange(
            0, monitor_Nhidden + monitor_Noutput
        )
        config.debug.monitor_neuron_i_syn2 = samna.xyloCore2.configuration.NeuronRange(
            0, monitor_Nhidden
        )
        config.debug.monitor_neuron_spike = samna.xyloCore2.configuration.NeuronRange(
            0, monitor_Nhidden
        )
        config.debug.monitor_neuron_v_mem = samna.xyloCore2.configuration.NeuronRange(
            0, monitor_Nhidden + monitor_Noutput
        )

    else:
        if readout == "Isyn":
            config.debug.monitor_neuron_i_syn = (
                samna.xyloCore2.configuration.NeuronRange(
                    monitor_Nhidden, monitor_Nhidden + monitor_Noutput
                )
            )
        elif readout == "Vmem":
            config.debug.monitor_neuron_v_mem = (
                samna.xyloCore2.configuration.NeuronRange(
                    monitor_Nhidden, monitor_Nhidden + monitor_Noutput
                )
            )

    # - Configure the monitor buffer
    state_monitor_buffer.set_configuration(config)

    # - Return the configuration and buffer
    return config, state_monitor_buffer


def config_hibernation_mode(config: XyloConfiguration) -> XyloConfiguration:
    """
    Switch on hibernaton mode on a Xylo hdk

    Args:
        config (XyloConfiguration): The desired Xylo configuration to use
    """
    config.enable_hibernation_mode = True
    return config


def configure_single_step_time_mode(
    config: XyloConfiguration,
) -> XyloConfiguration:
    """
    Switch on single-step model on a Xylo hdk

    Args:
        hdk (XyloBaughterBoard): The Xylo HDK to configure
        config (XyloConfiguration): The desired Xylo configuration to use
    """
    # - Write the configuration
    config.operation_mode = samna.xyloCore2.OperationMode.Manual
    return config


def to_hex(n: int, digits: int) -> str:
    """
    Output a consistent-length hex string encoding a number

    Args:
        n (int): Number to export
        digits (int): Number of digits to produce

    Returns:
        str: Hex-encoded string, with ``digits`` digits
    """
    return "%s" % ("0000%x" % (n & 0xFFFFFFFF))[-digits:]


def read_accel_mode_data(
    monitor_buffer: Xylo2NeuronStateBuffer,
    Nin: int,
    Nhidden: int,
    Nout: int,
) -> XyloState:
    """
    Read accelerated simulation mode data from a Xylo HDK

    Args:
        monitor_buffer (XyloNeuronStateBuffer): A connected `XyloNeuronStateBuffer` to read from
        Nin (int): Number of input neurons to read. Default: ``16`` (all neurons).
        Nhidden (int): The number of hidden neurons to monitor
        Nout (int): The number of output neurons to monitor

    Returns:
        XyloState: The encapsulated state read from the Xylo device
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

    # - Return as a XyloState object
    return XyloState(
        Nin,
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
    events: List[Any], Nin: int = 16, Nhidden: int = 1000, Nout: int = 8
) -> Tuple[XyloState, np.ndarray]:
    """
    Decode events from accelerated-time operation of the Xylo HDK

    Warnings:
        ``Nin``, ``Nhidden`` and ``Nout`` must be defined correctly for the network deployed to the Xylo HDK, for this function to operate as expected.

        This function must be called with the *full* list of events from a simulation. Otherwise the data returned will be incomplete. This function will not operate as expected if provided with incomplete data.

        You can use the ``target_timstamp`` argument to `.blocking_read` to ensure that you have read events up to the desired final timestep.

    Args:
        events (List[Any]): A list of events produced during an accelerated-mode simulation on a Xylo HDK
        Nhidden (int): The number of defined hidden-layer neurons. Default: ``1000``, expect to read the state of every neuron.
        Nout (int): The number of defined output-layer neurons. Default: ``8``, expect to read the state of every neuron.

    Returns:
        (`.XyloState`, np.ndarray): A `.NamedTuple` containing the decoded state resulting from the simulation, and an array of timestamps for each state entry over time
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
        if isinstance(e, samna.xyloCore2.event.Spike):
            # - Save this output event
            spikes_out_ts[e.timestamp - 1][e.neuron_id] = True

        # - Handle a memory value read event
        if isinstance(e, samna.xyloCore2.event.MemoryValue):
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
        if isinstance(e, samna.xyloCore2.event.Readout):
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
        XyloState(
            Nin,
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


class XyloAllRam(NamedTuple):
    """
    ``NamedTuple`` that encapsulates a recorded Xylo HDK state
    """

    # - state Ram
    Nin: int
    """ int: The number of input-layer neurons """

    Nhidden: int
    """ int: The number of hidden-layer neurons """

    Nout: int
    """ int: The number of output layer neurons """

    V_mem_hid: np.ndarray
    """ np.ndarray: Membrane potential of hidden neurons ``(Nhidden,)``"""

    I_syn_hid: np.ndarray
    """ np.ndarray: Synaptic current 1 of hidden neurons ``(Nhidden,)``"""

    V_mem_out: np.ndarray
    """ np.ndarray: Membrane potential of output neurons ``(Nhidden,)``"""

    I_syn_out: np.ndarray
    """ np.ndarray: Synaptic current of output neurons ``(Nout,)``"""

    I_syn2_hid: np.ndarray
    """ np.ndarray: Synaptic current 2 of hidden neurons ``(Nhidden,)``"""

    Spikes_hid: np.ndarray
    """ np.ndarray: Spikes from hidden layer neurons ``(Nhidden,)``"""

    Spikes_out: np.ndarray
    """ np.ndarray: Spikes from output layer neurons ``(Nout,)``"""

    # - config RAM
    IWTRAM_state: np.ndarray
    """ np.ndarray: Contents of IWTRAM """

    IWT2RAM_state: np.ndarray
    """ np.ndarray: Contents of IWT2RAM """

    NDSRAM_state: np.ndarray
    """ np.ndarray: Contents of NDSRAM """

    RDS2RAM_state: np.ndarray
    """ np.ndarray: Contents of RDS2RAM """

    NDMRAM_state: np.ndarray
    """ np.ndarray: Contents of NMDRAM """

    NTHRAM_state: np.ndarray
    """ np.ndarray: Contents of NTHRAM """

    RCRAM_state: np.ndarray
    """ np.ndarray: Contents of RCRAM """

    RARAM_state: np.ndarray
    """ np.ndarray: Contents of RARAM """

    REFOCRAM_state: np.ndarray
    """ np.ndarray: Contents of REFOCRAM """

    RFORAM_state: np.ndarray
    """ np.ndarray: Contents of RFORAM """

    RWTRAM_state: np.ndarray
    """ np.ndarray: Contents of RWTRAM """

    RWT2RAM_state: np.ndarray
    """ np.ndarray: Contents of RWT2RAM """

    OWTRAM_state: np.ndarray
    """ np.ndarray: Contents of OWTRAM """


def read_allram_state(
    read_buffer: Xylo2ReadBuffer,
    write_buffer: Xylo2WriteBuffer,
    Nin: int = 16,
    Nhidden: int = 1000,
    Nout: int = 8,
) -> XyloAllRam:
    """
    Read and return the all ram in each step as a state

    Args:
        read_buffer (XyloReadBuffer): A read buffer connected to the Xylo HDK
        write_buffer (XyloWriteBuffer): A write buffer connected to the Xylo HDK

    Returns:
        :py:class:`.XyloState`: The recorded state as a ``NamedTuple``. Contains keys ``V_mem_hid``,  ``V_mem_out``, ``I_syn_hid``, ``I_syn_out``, ``I_syn2_hid``, ``Nhidden``, ``Nout``. This state has **no time axis**; the first axis is the neuron ID.

    """
    # - Define the memory bank addresses
    memory_table = {
        "nscram": 0x7E00,
        "rsc2ram": 0x81F0,
        "nmpram": 0x85D8,
        "rspkram": 0xA150,
        "IWTRAM": 0x00100,
        "IWT2RAM": 0x03F80,
        "NDSRAM": 0x089C8,
        "RDS2RAM": 0x08DB8,
        "NDMRAM": 0x091A0,
        "NTHRAM": 0x09590,
        "RCRAM": 0x09980,
        "RARAM": 0x09D68,
        "REFOCRAM": 0x0A538,
        "RFORAM": 0x0A920,
        "RWTRAM": 0x12620,
        "RWT2RAM": 0x1A320,
        "OWTRAM": 0x22020,
    }

    # - Read synaptic currents
    Isyn = read_memory(
        read_buffer,
        write_buffer,
        memory_table["nscram"],
        Nhidden + Nout + num_buffer_neurons(Nhidden),
    )

    # - Read synaptic currents 2
    Isyn2 = read_memory(read_buffer, write_buffer, memory_table["rsc2ram"], Nhidden)

    # - Read membrane potential
    Vmem = read_memory(
        read_buffer,
        write_buffer,
        memory_table["nmpram"],
        Nhidden + Nout + num_buffer_neurons(Nhidden),
    )

    # - Read reservoir spikes
    Spikes = read_memory(read_buffer, write_buffer, memory_table["rspkram"], Nhidden)

    # - Read config RAM including buffer neuron(s)
    input_weight_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["IWTRAM"],
        Nin * (Nhidden + num_buffer_neurons(Nhidden)),
    )

    input_weight_2ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["IWT2RAM"],
        Nin * (Nhidden + num_buffer_neurons(Nhidden)),
    )

    neuron_dash_syn_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["NDSRAM"],
        Nhidden + Nout + num_buffer_neurons(Nhidden),
    )

    reservoir_dash_syn_2ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["RDS2RAM"],
        Nhidden + num_buffer_neurons(Nhidden),
    )

    neuron_dash_mem_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["NDMRAM"],
        Nhidden + Nout + num_buffer_neurons(Nhidden),
    )

    neuron_threshold_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["NTHRAM"],
        Nhidden + Nout + num_buffer_neurons(Nhidden),
    )

    reservoir_config_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["RCRAM"],
        Nhidden + num_buffer_neurons(Nhidden),
    )

    reservoir_aliasing_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["RARAM"],
        Nhidden + num_buffer_neurons(Nhidden),
    )

    reservoir_effective_fanout_count_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["REFOCRAM"],
        # Nhidden + num_buffer_neurons(Nhidden), --> dummy neuron
        Nhidden,
    )

    recurrent_fanout_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["RFORAM"],
        np.sum(np.array(reservoir_effective_fanout_count_ram, "int16")),
    )

    recurrent_weight_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["RWTRAM"],
        np.sum(np.array(reservoir_effective_fanout_count_ram, "int16")),
    )

    recurrent_weight_2ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["RWT2RAM"],
        np.sum(np.array(reservoir_effective_fanout_count_ram, "int16")),
    )

    output_weight_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["OWTRAM"],
        Nout * (Nhidden + num_buffer_neurons(Nhidden)),
    )

    # - Return the all ram state
    return XyloAllRam(
        Nin,
        Nhidden,
        Nout,
        # - state RAM
        np.array(Vmem[:Nhidden], "int16"),
        np.array(Isyn[:Nhidden], "int16"),
        np.array(Vmem[-Nout:], "int16"),
        np.array(Isyn[-Nout:], "int16"),
        np.array(Isyn2, "int16"),
        np.array(Spikes, "int16"),
        read_output_events(read_buffer, write_buffer)[:Nout],
        # - config RAM
        np.array(input_weight_ram, "int16"),
        np.array(input_weight_2ram, "int16"),
        np.array(neuron_dash_syn_ram, "int16"),
        np.array(reservoir_dash_syn_2ram, "int16"),
        np.array(neuron_dash_mem_ram, "int16"),
        np.array(neuron_threshold_ram, "int16"),
        np.array(reservoir_config_ram, "int16"),
        np.array(reservoir_aliasing_ram, "int16"),
        np.array(reservoir_effective_fanout_count_ram, "int16"),
        np.array(recurrent_fanout_ram, "int16"),
        np.array(recurrent_weight_ram, "int16"),
        np.array(recurrent_weight_2ram, "int16"),
        np.array(output_weight_ram, "int16"),
    )


def export_registers(
    read_buffer: Xylo2ReadBuffer,
    write_buffer: Xylo2WriteBuffer,
    file,
) -> None:
    """
    Print register contents for debugging purposes

    Args:
        read_buffer (XyloReadBuffer): A connected Xylo read buffer to use in reading registers
        write_buffer (XyloWriteBuffer): A write buffer connected to a Xylo HDK
        file: a file to save the registers
    """

    with open(file, "w+") as f:
        f.write("ctrl1 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x1)[0]))
        f.write("\n")

        f.write("ctrl2 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x2)[0]))
        f.write("\n")

        f.write("ctrl3 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x3)[0]))
        f.write("\n")

        f.write("pwrctrl1 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x04)[0]))
        f.write("\n")

        f.write("pwrctrl2 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x05)[0]))
        f.write("\n")

        f.write("pwrctrl3 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x06)[0]))
        f.write("\n")

        f.write("pwrctrl4 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x07)[0]))
        f.write("\n")

        f.write("ie ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x08)[0]))
        f.write("\n")

        f.write("ctrl4 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x09)[0]))
        f.write("\n")

        f.write("baddr ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x0A)[0]))
        f.write("\n")

        f.write("blen ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x0B)[0]))
        f.write("\n")

        f.write("ispkreg00 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x0C)[0]))
        f.write("\n")

        f.write("ispkreg01 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x0D)[0]))
        f.write("\n")

        f.write("ispkreg10 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x0E)[0]))
        f.write("\n")

        f.write("ispkreg11 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x0F)[0]))
        f.write("\n")

        f.write("stat ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x10)[0]))
        f.write("\n")

        f.write("int ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x11)[0]))
        f.write("\n")

        f.write("omp_stat0 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x12)[0]))
        f.write("\n")

        f.write("omp_stat1 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x13)[0]))
        f.write("\n")

        f.write("omp_stat2 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x14)[0]))
        f.write("\n")

        f.write("omp_stat3 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x15)[0]))
        f.write("\n")

        f.write("monsel0 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x16)[0]))
        f.write("\n")

        f.write("monsel1 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x17)[0]))
        f.write("\n")

        f.write("dbg_ctrl1 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x18)[0]))
        f.write("\n")

        f.write("dbg_stat1 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x19)[0]))
        f.write("\n")

        f.write("tr_cntr_stat ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x1A)[0]))
        f.write("\n")


def set_power_measure(
    hdk: XyloA2HDK,
    frequency: Optional[float] = 5.0,
):
    """
    Initialize power consumption measure on a hdk

    Args:
        hdk (XyloHDK): The Xylo HDK to be measured
        frequency (float): The frequency of power measurement. Default: 5.0
    """
    power = hdk.get_power_monitor()
    buf = samna.BasicSinkNode_unifirm_modules_events_measurement()
    graph = samna.graph.EventFilterGraph()
    graph.sequential([power.get_source_node(), buf])
    power.start_auto_power_measurement(frequency)
    return buf, power
