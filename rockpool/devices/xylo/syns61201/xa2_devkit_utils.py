"""
Low-level device kit utilities for the SYNS61201 Xylo-A2 HDK
"""

import enum
import samna

# - Other imports
from warnings import warn
import time
import numpy as np
from bitstruct import pack_dict, unpack_dict


# - Typing and useful proxy types
from typing import Any, List, Iterable, Optional, NamedTuple, Union, Tuple

Xylo2ReadBuffer = samna.BasicSinkNode_xylo_core2_event_output_event
Xylo2WriteBuffer = samna.BasicSourceNode_xylo_core2_event_input_event
Xylo2NeuronStateBuffer = samna.xyloCore2.NeuronStateSinkNode

AFE2ReadBuffer = samna.BasicSinkNode_afe2_event_output_event
AFE2WriteBuffer = samna.BasicSourceNode_afe2_event_input_event
AFE2OutputEvent = samna.afe2.event.Spike

from samna.xyloCore2.configuration import XyloConfiguration
from samna.afe2.configuration import AfeConfiguration

XyloA2HDK = Any


class Xylo2Registers(enum.IntEnum):
    """All registers on the Xylo2 core"""

    VERSION = 0
    CTRL1 = 1
    CTRL2 = 2
    TR_WRAP = 3
    HM_TR_WRAP = 4
    PWR_CTRL1 = 5
    PWR_CTRL2 = 6
    PWR_CTRL3 = 7
    PWR_CTRL4 = 8
    ISO_CTRL1 = 9
    ISO_CTRL2 = 10
    ISO_CTRL3 = 11
    ISO_CTRL4 = 12
    IE = 13
    WO = 14
    BADDR = 15
    BLEN = 16
    ISPKREG0L = 17
    ISPKREG0H = 18
    ISPKREG1L = 19
    ISPKREG1H = 20
    STAT = 21
    INT = 22
    OMP_STAT0 = 23
    OMP_STAT1 = 24
    OMP_STAT2 = 25
    OMP_STAT3 = 26
    MONSEL = 27
    MON_GRP_SEL = 28
    DBG_CTRL1 = 29
    TRAM_CTRL = 30
    HRAM_CTRL = 31
    DBG_STAT1 = 32
    TR_CNTR_STAT = 33


Xylo2RegistersStruct = {
    "CTRL1": (
        "u4u2b1b1b1b1b1b1u2b1b1b1u3b1u3b1u3b1b1b1b1",
        [
            "RAM_WU_ST",
            "_",
            "RST_PS",
            "RST_PE",
            "HM_EN",
            "ALWAYS_UPDATE_OPM_STAT",
            "DIRECT_FETCH_STATE_RAM",
            "KEEP_INT",
            "_",
            "RAM_ACTIVE",
            "MEM_CLK_ON",
            "_",
            "OWBS",
            "_",
            "RWBS",
            "_",
            "IWBS",
            "BIAS_EN",
            "ALIAS_EN",
            "ISYN2_EN",
            "MAN",
        ],
    )
}


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
    read_buffer.get_events()
    write_buffer.write([rrv_ev])

    # - Wait for data and read it
    start_t = time.time()
    continue_read = True
    while continue_read:
        # - Read from the buffer
        events = read_buffer.get_events()

        # - Filter returned events for the desired address
        ev_filt = [
            e
            for e in events
            if hasattr(e, "address")
            and e.address == address
            and isinstance(e, samna.afe2.event.RegisterValue)
        ]

        # - Should we continue the read?
        continue_read &= len(ev_filt) == 0
        continue_read &= (time.time() - start_t) < timeout

    # - If we didn't get the required register read, raise an error
    if len(ev_filt) == 0:
        raise TimeoutError(f"Timeout after {timeout}s when reading register {address}.")

    # - Return data
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
) -> Tuple[np.ndarray, np.ndarray]:
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


def apply_afe2_default_config(
    afe2hdk: XyloA2HDK,
    config: AfeConfiguration,
    bpf_bias: int = 2,
    fwr_bias: int = 6,
    lna_ci_tune: int = 0,
    lna_cf_tune: int = 5,
    afe_stable_time: int = 0x80,
    leak_timing_window: int = int(25e5),
    leak_td: int = 0x030D4,
    leak_target_spike_number: int = 1,
    *args,
    **kwargs,
) -> AfeConfiguration:
    """
    Configure an AFE2 HDK, including self-calibration

    Args:
        afe2hdk (XyloA2HDK): A connected AFE2 HDK device
        config (AfeConfiguration): A configuration for AFE
        bpf_bias (int): master gm cell bias selected for the band pass filter
        fwr_bias (int): master gm cell bias selected for the full wave rectifier which rectifies the output of the band pass filter bank
        lna_ci_tune (int): extra 0.25p cap witch low active config cap for the low noise amplifier that amplifies the input with given gain
        lna_cf_tune (int): extra 1p cap witch low active config cap for the low noise amplifier that amplifies the input with given gain
        afe_stable_time (int): stable time for AFE after power on
        leak_timing_window (int): The timing window setting for leakage calibration
        leak_td (int): warm-up time window threshold for leakage calibration
        leak_target_spike_number (int): target spike number for leakage calibration
    """

    config.analog_top.enable = True
    config.debug.enable_event_monitor = False

    config.analog_top.bpf.bias = int(bpf_bias) if bpf_bias is not None else 2
    config.analog_top.fwr.bias = int(fwr_bias) if fwr_bias is not None else 6

    config.analog_top.lna.ci_tune = int(lna_ci_tune) if lna_ci_tune is not None else 0
    config.analog_top.lna.cf_tune = int(lna_cf_tune) if lna_cf_tune is not None else 5

    config.analog_top.bpf.scale = True

    afe2hdk.get_afe_model().apply_configuration(config)

    time.sleep(45)

    config.aer_2_saer.calibration.mode = 1
    config.aer_2_saer.calibration.reset = True

    config.aer_2_saer.calibration.afe_stable_time = (
        int(afe_stable_time) if afe_stable_time is not None else 0x80
    )
    config.aer_2_saer.calibration.leak_timing_window = (
        int(leak_timing_window) if leak_timing_window is not None else 25e5
    )

    config.aer_2_saer.calibration.leak_td = (
        int(leak_td) if leak_td is not None else 0x030D4
    )
    config.aer_2_saer.calibration.leak_target_spike_number = (
        int(leak_target_spike_number) if leak_target_spike_number is not None else 1
    )
    return config


def read_afe2_module_version(
    afe_read_buf: AFE2ReadBuffer, afe_write_buf: AFE2WriteBuffer
) -> Tuple[int, int]:
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
) -> Tuple[List, bool]:
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
    register: Union[int, Xylo2Registers],
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
    register: Union[int, Xylo2Registers],
    timeout: float = 2.0,
) -> List[int]:
    """
    Read the contents of a register

    Args:
        read_buffer (XyloReadBuffer): A connected read buffer to the XYlo HDK
        write_buffer (XyloWriteBuffer): A connected write buffer to the Xylo HDK
        address (Union[int, Xylo2Registers]): The register address to read
        timeout (float): A timeout in seconds

    Returns:
        List[int]: A list of events returned from the read
    """
    # - Set up a register read
    rrv_ev = samna.xyloCore2.event.ReadRegisterValue()
    rrv_ev.address = register

    # - Request read
    write_buffer.write([rrv_ev])

    # - Wait for data and read it
    start_t = time.time()
    continue_read = True
    while continue_read:
        # - Read from the buffer
        events = read_buffer.get_events()

        # - Filter returned events for the desired address
        ev_filt = [e for e in events if hasattr(e, "address") and e.address == register]

        # - Should we continue the read?
        continue_read &= len(ev_filt) == 0
        continue_read &= (time.time() - start_t) < timeout

    # - If we didn't get the required register read, raise an error
    if len(ev_filt) == 0:
        raise TimeoutError(
            f"Timeout after {timeout}s when reading register {register}."
        )

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
    *_,
    **__,
) -> None:
    """
    Apply a configuration to the Xylo HDK

    Args:
        hdk (XyloHDK): The Xylo HDK to write the configuration to
        config (XyloConfiguration): A configuration for Xylo
    """
    # - Ideal -- just write the configuration using samna
    hdk.get_xylo_model().apply_configuration(config)


def read_neuron_synapse_state(
    read_buffer: Xylo2ReadBuffer,
    write_buffer: Xylo2WriteBuffer,
    Nin: int = 16,
    Nhidden: int = 1000,
    Nout: int = 8,
    synapse2_enable: bool = False,
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
        synapse2_enable (bool): Synapse 2 is used and should be recoded.
        record (bool): Enable recording of all state
        readout_mode (str): Specify the "readout" mode for the network. This must be one of ``['Spike', 'Isyn', 'Vmem']``. Default: ``Spike``; read output spikes

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
        Isyn2 = (
            read_memory(read_buffer, write_buffer, memory_table["rsc2ram"], Nhidden)
            if synapse2_enable
            else None
        )

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
        np.array(Spikes, "int16") if Spikes is not None else None,
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
) -> Tuple[XyloConfiguration, Xylo2NeuronStateBuffer]:
    """
    Switch on accelerated-time mode on a Xylo hdk, and configure network monitoring

    Notes:
        Use :py:func:`new_xylo_state_monitor_buffer` to generate a buffer to monitor neuron and synapse state.

    Args:
        config (XyloConfiguration): The desired Xylo configuration to use
        state_monitor_buffer (XyloNeuronStateBuffer): A connected neuron state monitor buffer
        monitor_Nhidden (Optional[int]): The number of hidden neurons for which to monitor state during evolution. Default: ``0``, don't monitor any hidden neurons.
        monitor_Noutput (Optional[int]): The number of output neurons for which to monitor state during evolution. Default: ``0``, don't monitor any output neurons.
        readout: The readout out mode for which to output neuron states. Default: ``Spike''. Must be one of ``['Isyn', 'Vmem', 'Spike']``.
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
        config.debug.monitor_neuron_i_syn2 = (
            samna.xyloCore2.configuration.NeuronRange(0, monitor_Nhidden)
            if config.synapse2_enable
            else None
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


def config_hibernation_mode(
    config: XyloConfiguration, hibernation_mode: bool
) -> XyloConfiguration:
    """
    Switch on hibernaton mode on a Xylo hdk

    Args:
        config (XyloConfiguration): The desired Xylo configuration to use
    """
    config.enable_hibernation_mode = hibernation_mode
    return config


def configure_single_step_time_mode(
    config: XyloConfiguration,
) -> XyloConfiguration:
    """
    Switch on single-step mode on a Xylo hdk

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
    synapse2_enable: bool = False,
) -> XyloState:
    """
    Read accelerated simulation mode data from a Xylo HDK

    Args:
        monitor_buffer (XyloNeuronStateBuffer): A connected `XyloNeuronStateBuffer` to read from
        Nin (int): Number of input neurons to read. Default: ``16`` (all neurons).
        Nhidden (int): The number of hidden neurons to monitor
        Nout (int): The number of output neurons to monitor
        synapse2_enable (bool): Synapse 2 is used and should be monitored. Default: ``False``, don't monitor synapse 2

    Returns:
        XyloState: The encapsulated state read from the Xylo device
    """
    # - Read data from neuron state buffer
    vmem_ts = np.array(monitor_buffer.get_reservoir_v_mem(), "int16").T
    isyn_ts = np.array(monitor_buffer.get_reservoir_i_syn(), "int16").T
    isyn2_ts = (
        np.array(monitor_buffer.get_reservoir_i_syn2(), "int16").T
        if synapse2_enable
        else None
    )
    spikes_ts = np.array(monitor_buffer.get_reservoir_spike(), "int8").T
    spikes_out_ts = np.array(monitor_buffer.get_output_spike(), "int8").T

    # - Separate hidden and output neurons
    isyn_out_ts = isyn_ts[:, -Nout:] if len(isyn_ts) > 0 else None
    isyn_ts = isyn_ts[:, :Nhidden] if len(isyn_ts) > 0 else None
    vmem_out_ts = vmem_ts[:, -Nout:] if len(vmem_ts) > 0 else None
    vmem_ts = vmem_ts[:, :Nhidden] if len(vmem_ts) > 0 else None

    # - Return as a XyloState object
    return XyloState(
        Nin=Nin,
        Nhidden=Nhidden,
        Nout=Nout,
        V_mem_hid=vmem_ts,
        I_syn_hid=isyn_ts,
        V_mem_out=vmem_out_ts,
        I_syn_out=isyn_out_ts,
        I_syn2_hid=isyn2_ts,
        Spikes_hid=spikes_ts,
        Spikes_out=spikes_out_ts,
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

    Warning:
        Ensure that all RAM clocks are forced to be on when using this function, or you risk getting garbage data back for all memory addresses.

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
    Spikes = read_memory(read_buffer, write_buffer, memory_table["rspkram"], Nhidden)

    # - Read config RAM including buffer neuron(s)
    input_weight_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["IWTRAM"],
        Nin * Nhidden,
    )

    input_weight_2ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["IWT2RAM"],
        Nin * Nhidden,
    )

    neuron_dash_syn_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["NDSRAM"],
        Nhidden + Nout,
    )

    reservoir_dash_syn_2ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["RDS2RAM"],
        Nhidden,
    )

    neuron_dash_mem_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["NDMRAM"],
        Nhidden + Nout,
    )

    neuron_threshold_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["NTHRAM"],
        Nhidden + Nout,
    )

    reservoir_config_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["RCRAM"],
        Nhidden,
    )

    reservoir_aliasing_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["RARAM"],
        Nhidden,
    )

    reservoir_effective_fanout_count_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["REFOCRAM"],
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
        Nout * Nhidden,
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
) -> Tuple[
    samna.BasicSinkNode_unifirm_modules_events_measurement,
    samna.boards.common.power.PowerMonitor,
]:
    """
    Initialize power consumption measure on a hdk

    Args:
        hdk (XyloHDK): The Xylo HDK to be measured
        frequency (float): The frequency of power measurement. Default: 5.0

    Returns:
        power_buf: Event buffer to read power monitoring events from
        power_monitor: The power monitoring object
    """
    power_monitor = hdk.get_power_monitor()
    power_buf = samna.BasicSinkNode_unifirm_modules_events_measurement()
    graph = samna.graph.EventFilterGraph()
    graph.sequential([power_monitor.get_source_node(), power_buf])
    power_monitor.start_auto_power_measurement(frequency)
    return power_buf, power_monitor


def config_afe_channel_thresholds(
    config: AfeConfiguration, count: int = 4
) -> AfeConfiguration:
    """
    Change the AFE event count to throw 1 spikes out of 1 to the Xylo core

    Args:
        config (AfeConfiguration): A configuration for AFE
        count (int): The value of counter threshold
    """

    config.aer_2_saer.channel_thresholds = [count for _ in range(16)]
    return config


def config_lna_amplification(
    config: AfeConfiguration, level: str = "low"
) -> AfeConfiguration:
    """
    Adjust acoustic gain of LDA module

    Args:
        config (AfeConfiguration): A configuration for AFE
        level (str): gain level, which should be in ``['low', 'mid', and 'high']``. Default low level is without gain.
    """

    assert level in ["low", "mid", "high"]
    if level == "mid":
        config.analog_top.lna.ci_tune = 0x7
        config.analog_top.lna.cf_tune = 0xF
    elif level == "high":
        config.analog_top.lna.ci_tune = 0x0
        config.analog_top.lna.cf_tune = 0xF
    return config


def config_basic_mode(
    config: XyloConfiguration,
) -> XyloConfiguration:
    """
    Set the Xylo HDK to manual mode before configure to real-time mode

    Args:
        config (XyloConfiguration): A configuration for Xylo

    Return:
        updated Xylo configuration
    """
    warn("This devkit utils function should be replaced.")
    config.operation_mode = samna.xyloCore2.OperationMode.Manual
    config.debug.always_update_omp_stat = True
    config.clear_network_state = True
    return config


def _auto_mode(
    io,
    read_buffer: Xylo2ReadBuffer,
    write_buffer: Xylo2ReadBuffer,
    write_afe_buffer: AFE2WriteBuffer,
    dt: float,
    main_clk_rate: int,
    hibernation_mode: bool,
) -> None:
    """
    Set the Xylo HDK to real-time mode

    Args:
        io: io module for Xylo
        read_buffer (Xylo2ReadBuffer): A read buffer connected to the Xylo
        write_buffer (Xylo2ReadBuffer): A write buffer connected to the Xylo
        write_afe_buffer (AFE2WriteBuffer): A write buffer connected to the AFE
        dt (float): the time resolution for calculation
        main_clk_rate (int): main clock rate of Xylo
        hibernation_mode (bool): the hibernation mode to run. Iff True, the chip will output events only if it receives inputs
    """
    warn("This devkit utils function should be replaced.")

    # Set Xylo core reading frequency
    write_register(write_buffer, 0x03, int(main_clk_rate * dt))
    write_register(write_buffer, 0x04, int(main_clk_rate * dt))

    # Clear input events
    reset_input_spikes(write_buffer)

    # Set Xylo core to auto mode
    ctrl1 = read_register(read_buffer, write_buffer, 0x01)[0]
    ctrl1 &= 0xFFFFFFFE
    write_register(write_buffer, 0x01, ctrl1)

    # Set hibernation mode
    if hibernation_mode:
        write_afe2_register(write_afe_buffer, 0x25, 0x12)
    write_register(write_buffer, 0x0E, 0x10)

    time.sleep(0.1)

    # Set FPGA to auto mode
    io.get_xylo_handler().set_operation_mode(samna.xyloCore2.OperationMode.RealTime)


def config_AFE_hibernation(config: AfeConfiguration) -> AfeConfiguration:
    """
    Switch on hibernation mode on AFE

    Args:
        config (AfeConfiguration): A configuration for AFE
    """

    # timing window for active status
    config.aer_2_saer.hibernation.active_status_detection_timing_window = 0xE848
    # timing window for deactive status
    config.aer_2_saer.hibernation.deactive_status_detection_timing_window = 0x1312D0
    # spike threshold for active and deactive status
    config.aer_2_saer.hibernation.active_mode_spike_threshold = 0x9
    config.aer_2_saer.hibernation.deactive_mode_spike_threshold = 0x30

    return config


def DivisiveNormalization(
    config: AfeConfiguration,
    s: int = 5,
    p: int = 0,
    iaf_bias: int = 10,
    *args,
    **kwargs,
) -> AfeConfiguration:
    """
    The normalized Signal is:
    PostNF(t) = preNF(t) * (2**p) /(iaf_bias + M(t))
    M(t) = 2**(-s) * E(t) + (1-2**(-s)) * M(t-1), where E(t) is the input signal, M(t) estimates the background noise level

    Args:
        config (AfeConfiguration): A configuration for AFE
        s (int): The parameter which indirectly affects the normalized window length
        p (int): The precision coefficient
        iaf_bias (int): The bias parameter
    """

    config.aer_2_saer.dn.enable = 1
    config.aer_2_saer.dn.iaf_bias = int(iaf_bias) if iaf_bias is not None else 10
    config.aer_2_saer.dn.p_param = int(p) if p is not None else 0
    config.aer_2_saer.dn.s_param = int(s) if s is not None else 5

    return config


def pretty_print_register(
    register: Union[int, Xylo2Registers],
    value: Optional[int] = None,
    read_buffer: Optional[Xylo2ReadBuffer] = None,
    write_buffer: Optional[Xylo2WriteBuffer] = None,
) -> str:
    """
    Pretty-print a Xylo2 register value

    Args:
        register (Union[int, Xylo2Registers]): A Xylo2 register address to print
        value (Optional[int]): A previously-read value
        read_buffer (Xylo2ReadBuffer): A read buffer to use, if reading fresh value
        write_buffer (Xylo2WriteBuffer): A write buffer to use

    Returns:
        str: The string representation of the register value
    """
    # - Read the register
    value = (
        value
        if value is not None
        else read_register(read_buffer, write_buffer, register)[0]
    )

    try:
        import bitstring

        # - 'reverse' is needed so that MSB is bit 0
        value = bitstring.BitArray(bin(value))
        value.reverse()
        value.append("uint:32=0")

        if register == Xylo2Registers.CTRL1:
            trigger_mode = "manual" if value[0] else "auto"
            isyn2_en = "enabled" if value[1] else "disabled"
            alias_en = "enabled" if value[2] else "disabled"
            bias_en = "enabled" if value[3] else "disabled"
            inp_w_bitshift = value[4:7].uint
            res_w_bitshhift = value[8:11].uint
            out_w_bitshift = value[12:15].uint
            mem_shell_clk = "on" if value[16] else "off"
            all_ram_mode = "active" if value[17] else "inactive"
            interrupt_mode = "keep" if value[20] else "clear"
            direct_fetch_state_ram = (
                "directly fetch state RAMs without SPI RAM init"
                if value[21]
                else "SPI RAM init required"
            )
            update_OMP_STATE = "always" if value[22] else "only on output neuron event"
            hibernation = "enabled" if value[23] else "disabled"
            rst_pe = "enabled" if value[24] else "disabled"
            rst_ps = "enabled" if value[25] else "disabled"
            ram_wu_st = value[28:32].uint

            str = f"Trigger mode: {trigger_mode}\n"
            str += f"2nd synapse: {isyn2_en}\n"
            str += f"Aliases: {alias_en}\n"
            str += f"Biases: {bias_en}\n"
            str += f"Input weight bitshift: {inp_w_bitshift}\n"
            str += f"Hidden weight bitshift: {res_w_bitshhift}\n"
            str += f"Output weight bitshift: {out_w_bitshift}\n"
            str += f"Memory shell clock manual switch on: {mem_shell_clk}\n"
            str += f"Force all RAM to be active: {all_ram_mode}\n"
            str += (
                f"Output interrupt clearing mode for each time-step: {interrupt_mode}\n"
            )
            str += f"State RAM fetch mode: {direct_fetch_state_ram}\n"
            str += f"When to update OMP_STAT register: {update_OMP_STATE}\n"
            str += f"Hibernation mode: {hibernation}\n"
            str += f"RST pad pull enable: {rst_pe}\n"
            str += f"RST pad pull select: {rst_ps}\n"
            str += f"RAM wake-up settling time: {ram_wu_st}ms\n"

            return str
    except:
        pass

    return f"Register {Xylo2Registers(register).name}: {value.bin()} {value.hex()}"


def set_xylo_core_clock_freq(device: XyloA2HDK, desired_freq_MHz: float) -> float:
    """
    Set the inference core clock frequency used by Xylo

    Args:
        device (XyloA2HDK): A Xylo2 device to configure
        desired_freq_MHz (float): The desired Xylo2 core clock frequency in MHz

    Returns:
        float: The obtained Xylo2 core clock frequency in MHz
    """
    # - Determine wait period and actual obtianed clock frequency
    wait_period = int(round(100 / desired_freq_MHz) / 2 - 1)
    actual_freq = 100 / (2 * (wait_period + 1))

    saer_freq = int(actual_freq * 1e6 / 4)
    spi_freq = int(actual_freq * 1e6 / 8)

    # - Configure device
    device.get_io_module().write_config(0x0021, wait_period)

    # - Set SAER clock frequency to 1/4 master clock freq
    # device.get_io_module().set_saer_clk_rate(saer_freq)

    # - Set SPI clock frequency to 1/16 master clock freq
    # device.get_io_module().set_spi_clk_rate(spi_freq)

    # - Return actual obtained clock freq.
    return actual_freq


def print_debug_registers(
    read_buffer: Xylo2ReadBuffer,
    write_buffer: Xylo2WriteBuffer,
) -> None:
    """
    Print register contents of a Xylo HDK for debugging purposes

    Args:
        write_buffer (XyloWriteBuffer): A connected write buffer to a Xylo HDK
        read_buffer (XyloReadBuffer): A connected Xylo read buffer to use when reading registers
    """
    print(
        "ctrl1", hex(read_register(read_buffer, write_buffer, Xylo2Registers.CTRL1)[0])
    )
    print(
        "ctrl2", hex(read_register(read_buffer, write_buffer, Xylo2Registers.CTRL2)[0])
    )
    print(
        "tr_wrap",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.TR_WRAP)[0]),
    )
    print(
        "hm_tr_wrap",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.HM_TR_WRAP)[0]),
    )
    print(
        "pwrctrl1",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.PWR_CTRL1)[0]),
    )
    print(
        "pwrctrl2",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.PWR_CTRL2)[0]),
    )
    print(
        "pwrctrl3",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.PWR_CTRL3)[0]),
    )
    print(
        "pwrctrl4",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.PWR_CTRL4)[0]),
    )
    print(
        "iso_ctrl1",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.ISO_CTRL1)[0]),
    )
    print(
        "iso_ctrl2",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.ISO_CTRL2)[0]),
    )
    print(
        "iso_ctrl3",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.ISO_CTRL3)[0]),
    )
    print(
        "iso_ctrl4",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.ISO_CTRL4)[0]),
    )
    print("ie", hex(read_register(read_buffer, write_buffer, Xylo2Registers.IE)[0]))
    print("wo", hex(read_register(read_buffer, write_buffer, Xylo2Registers.WO)[0]))
    print(
        "baddr", hex(read_register(read_buffer, write_buffer, Xylo2Registers.BADDR)[0])
    )
    print("blen", hex(read_register(read_buffer, write_buffer, Xylo2Registers.BLEN)[0]))
    print(
        "ispkreg0L",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.ISPKREG0L)[0]),
    )
    print(
        "ispkreg0H",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.ISPKREG0H)[0]),
    )
    print(
        "ispkreg1L",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.ISPKREG1L)[0]),
    )
    print(
        "ispkreg1H",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.ISPKREG1H)[0]),
    )
    print("stat", hex(read_register(read_buffer, write_buffer, Xylo2Registers.STAT)[0]))
    print("int", hex(read_register(read_buffer, write_buffer, Xylo2Registers.INT)[0]))
    print(
        "omp_stat0",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.OMP_STAT0)[0]),
    )
    print(
        "omp_stat1",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.OMP_STAT1)[0]),
    )
    print(
        "omp_stat2",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.OMP_STAT2)[0]),
    )
    print(
        "omp_stat3",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.OMP_STAT3)[0]),
    )
    print(
        "monsel",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.MONSEL)[0]),
    )
    print(
        "mon_grp_sel",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.MON_GRP_SEL)[0]),
    )
    print(
        "dbg_ctrl1",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.DBG_CTRL1)[0]),
    )
    print(
        "tram_ctrl",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.TRAM_CTRL)[0]),
    )
    print(
        "hram_ctrl",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.HRAM_CTRL)[0]),
    )
    print(
        "dbg_stat1",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.DBG_STAT1)[0]),
    )
    print(
        "tr_cntr_stat",
        hex(read_register(read_buffer, write_buffer, Xylo2Registers.TR_CNTR_STAT)[0]),
    )


def set_ram_access(
    ram_access_enable: bool,
    read_buffer: Xylo2ReadBuffer,
    write_buffer: Xylo2WriteBuffer,
) -> None:
    """
    Enable or disable access to RAM on the Xylo chip. To access any internal RAM values over SPI, RAM access must be enabled.
    """
    ctrl1 = read_register(read_buffer, write_buffer, Xylo2Registers.CTRL1)[0]

    dict_ctrl1 = unpack_dict(*Xylo2RegistersStruct["CTRL1"], ctrl1.to_bytes(4, "big"))

    if ram_access_enable:
        # - Write CTRL1.RAM_ACTIVE = 1
        dict_ctrl1["RAM_ACTIVE"] = True
        write_register(
            write_buffer,
            Xylo2Registers.CTRL1,
            int.from_bytes(
                pack_dict(*Xylo2RegistersStruct["CTRL1"], dict_ctrl1), "big"
            ),
        )

        # - Write CTRL1.MEM_CLK_ON = 1
        dict_ctrl1["MEM_CLK_ON"] = True
        write_register(
            write_buffer,
            Xylo2Registers.CTRL1,
            int.from_bytes(
                pack_dict(*Xylo2RegistersStruct["CTRL1"], dict_ctrl1), "big"
            ),
        )
    else:
        # - Write CTRL1.MEM_CLK_ON = 0
        dict_ctrl1["MEM_CLK_ON"] = False
        write_register(
            write_buffer,
            Xylo2Registers.CTRL1,
            int.from_bytes(
                pack_dict(*Xylo2RegistersStruct["CTRL1"], dict_ctrl1), "big"
            ),
        )

        # - Write CTRL1.RAM_ACTIVE = 0
        dict_ctrl1["RAM_ACTIVE"] = False
        write_register(
            write_buffer,
            Xylo2Registers.CTRL1,
            int.from_bytes(
                pack_dict(*Xylo2RegistersStruct["CTRL1"], dict_ctrl1), "big"
            ),
        )


def read_all_afe2_register(
    read_buffer: AFE2ReadBuffer,
    write_buffer: AFE2WriteBuffer,
):
    """
    Read all AFE register values

    Args:
        read_buffer (AFE2ReadBuffer): A connected read buffer to the XYlo HDK
        write_buffer (AFE2WriteBuffer): A connected write buffer to the Xylo HDK

    """
    for address in range(0x49):
        data = read_afe2_register(read_buffer, write_buffer, address)[0]
        print("read afe register ", hex(address), hex(data))


def read_all_xylo_register(
    read_buffer: Xylo2ReadBuffer,
    write_buffer: Xylo2WriteBuffer,
):
    """
    Read all xylo register values

    Args:
        read_buffer (Xylo2ReadBuffer): A connected read buffer to the XYlo HDK
        write_buffer (Xylo2WriteBuffer): A connected write buffer to the Xylo HDK

    """
    for address in range(0x33):
        data = read_register(read_buffer, write_buffer, address)[0]
        print("read xylo register ", hex(address), hex(data))
