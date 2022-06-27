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
    c.aer_2_saer.calibration.leak_timing_window = 0x98968

    c.aer_2_saer.calibration.leak_td = 6250
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


def new_xylo_read_buffer(hdk: XyloA2HDK,) -> Xylo2ReadBuffer:
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
    model = hdk.get_model()

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
    sink = hdk.get_model().get_sink_node()
    graph = samna.graph.EventFilterGraph()
    graph.sequential([buffer, sink])

    return buffer


def new_xylo_state_monitor_buffer(hdk: XyloA2HDK,) -> Xylo2NeuronStateBuffer:
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
    model = hdk.get_model()

    # - Get Xylo output event source node
    source_node = model.get_source_node()

    # - Add the buffer as a destination for the Xylo output events
    success = source_node.add_destination(buffer.get_input_channel())
    assert success, "Error connecting the new buffer."

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


def initialise_xylo_hdk(write_buffer: Xylo2WriteBuffer) -> None:
    """
    Initialise the Xylo HDK

    Args:
        write_buffer (XyloWriteBuffer): A write buffer connected to a Xylo HDK to initialise
    """
    # - Always need to advance one time-step to initialise
    advance_time_step(write_buffer)


def write_register(
    write_buffer: Xylo2WriteBuffer, register: int, data: int = 0,
) -> None:
    """
    Write data to a register on a Xylo HDK

    Args:
        write_buffer (XyloWriteBuffer): A connected write buffer to the desintation Xylo HDK
        register (int): The address of the register to write to
        data (int): The data to write. Default: 0x0
    """
    wwv_ev = samna.xylo.event.WriteRegisterValue()
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
    rrv_ev = samna.xylo.event.ReadRegisterValue()
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
    rmv_ev = samna.xylo.event.ReadMemoryValue()
    rmv_ev.address = start_address
    read_events_list.append(rmv_ev)

    for elem in range(count):
        rmv_ev = samna.xylo.event.ReadMemoryValue()
        rmv_ev.address = start_address + elem
        read_events_list.append(rmv_ev)

    # - Clear buffer
    read_buffer.get_events()

    # - Request read
    write_buffer.write(read_events_list)

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


def generate_read_memory_events(start_address: int, count: int = 1,) -> List[Any]:
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
    rmv_ev = samna.xylo.event.ReadMemoryValue()
    rmv_ev.address = start_address
    read_events_list.append(rmv_ev)

    for elem in range(count):
        rmv_ev = samna.xylo.event.ReadMemoryValue()
        rmv_ev.address = start_address + elem
        read_events_list.append(rmv_ev)

    return read_events_list


def decode_memory_read_events(
    events: List[Any], start_address: int, count: int = 1,
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
    read_buffer: Xylo2ReadBuffer, write_buffer: Xylo2WriteBuffer, timeout: float = 1.0,
) -> bool:
    """
    Verify that the provided daughterbaord returns the correct version ID for Xylo

    Args:
        read_buffer (XyloReadBuffer): A read buffer connected to the Xylo HDK
        write_buffer (XyloWriteBuffer): A write buffer connected to the Xylo HDK
        timeout (float): Timeout for checking in seconds

    Returns:
        bool: ``True`` iff the version ID is correct for Xylo
    """
    # - Clear the read buffer
    read_buffer.get_events()

    # - Read the version register
    write_buffer.write([samna.xylo.event.ReadVersion()])

    # - Read events until timeout
    filtered_events = []
    t_end = time.time() + timeout
    while len(filtered_events) == 0:
        events = read_buffer.get_events()
        filtered_events = [e for e in events if isinstance(e, samna.xylo.event.Version)]

        # - Check timeout
        if time.time() > t_end:
            raise TimeoutError(f"Checking version timed out after {timeout}s.")

    return (
        (len(filtered_events) > 0)
        and (filtered_events[0].major == 1)
        and (filtered_events[0].minor == 0)
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
        wmv_ev = samna.xylo.event.WriteMemoryValue()
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
