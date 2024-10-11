"""
Low-level device kit utilities for the Xylo™Audio 3 HDK
"""

import samna
from samna.xyloAudio3.configuration import XyloConfiguration

# - Other imports
import time
import numpy as np

from warnings import warn

# - Useful constants for XA3
from . import ram, reg


# - Typing and useful proxy types
from typing import List, Optional, NamedTuple, Tuple, Union, Iterable

XyloAudio3ReadBuffer = samna.BasicSinkNode_xylo_audio3_event_output_event
XyloAudio3WriteBuffer = samna.BasicSourceNode_xylo_audio3_event_input_event
ReadoutEvent = samna.xyloAudio3.event.Readout
SpikeEvent = samna.xyloAudio3.event.Spike
XyloAudio3HDK = samna.xyloAudio3.XyloAudio3TestBoard


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


def find_xylo_a3_boards() -> List[XyloAudio3HDK]:
    """
    Search for and return a list of XyloAudio 3 HDKs

    Iterate over devices and search for XyloAudio 3 HDK nodes. Return a list of available XyloAudio 3 HDKs, or an empty list if none are found.

    Returns:
        List[XyloAudio3HDK]: A (possibly empty) list of XyloAudio 3 HDK nodes.
    """

    # - Get a list of devices
    device_list = samna.device.get_all_devices()

    # - Search for a xylo dev kit
    audio3_hdk_list = [
        samna.device.open_device(d)
        for d in device_list
        if d.device_type_name == "XyloAudio3TestBoard"
    ]

    return audio3_hdk_list


def new_xylo_read_buffer(
    hdk: XyloAudio3HDK,
) -> XyloAudio3ReadBuffer:
    """
    Create and connect a new buffer to read from a Xylo HDK

    Args:
        hdk (XyloAudio3HDK): A Xylo HDK to create a new buffer for

    Returns:
        XyloAudio3ReadBuffer: A connected event read buffer
    """
    return samna.graph.sink_from(hdk.get_model_source_node())


def new_xylo_write_buffer(hdk: XyloAudio3HDK) -> XyloAudio3WriteBuffer:
    """
    Create a new buffer for writing events to a Xylo HDK

    Args:
        hdk (XyloAudio3HDK): A Xylo HDK to create a new buffer for

    Returns:
        XyloAudio3WriteBuffer: A connected event write buffer
    """
    return samna.graph.source_to(hdk.get_model_sink_node())


def new_xylo_state_monitor_buffer(
    hdk: XyloAudio3HDK,
) -> ReadoutEvent:
    """
    Create a new buffer for monitoring neuron and synapse state and connect it

    Args:
        hdk (XyloDaughterBoard): A Xylo HDK to configure

    Returns:
        XyloNeuronStateBuffer: A connected neuron / synapse state monitor buffer
    """
    # - Get the device model
    model = hdk.get_model()
    # - Get Xylo output event source node
    source_node = model.get_source_node()
    graph = samna.graph.EventFilterGraph()

    _, etf, state_buf = graph.sequential(
        [source_node, "XyloAudio3OutputEventTypeFilter", samna.graph.JitSink()]
    )
    etf.set_desired_type("xyloAudio3::event::Readout")
    graph.start()

    # - Return the buffer
    return state_buf, graph


def update_register_field(
    read_buffer: XyloAudio3ReadBuffer,
    write_buffer: XyloAudio3WriteBuffer,
    addr: int,
    lsb_pos: int,
    msb_pos: int,
    val: int,
):
    """
    Update a register field

    Args:
        write_buffer (XyloAudio3WriteBuffer): A write buffer connected to a Xylo HDK
    """
    data = read_register(read_buffer, write_buffer, addr)[0]
    data_h = data >> (msb_pos + 1)
    data_l = data & (2**lsb_pos - 1)
    data = (data_h << (msb_pos + 1)) + (val << lsb_pos) + data_l
    write_register(write_buffer, addr, data)


def get_current_timestep(
    read_buffer: XyloAudio3ReadBuffer,
    write_buffer: XyloAudio3WriteBuffer,
    timeout: float = 3.0,
) -> int:
    """
    Retrieve the current timestep on a Xylo HDK

    Args:
        read_buffer (XyloAudio3ReadBuffer): A connected read buffer for the xylo HDK
        write_buffer (XyloAudio3WriteBuffer): A connected write buffer for the Xylo HDK
        timeout (float): A timeout for reading

    Returns:
        int: The current timestep on the Xylo HDK
    """

    # - Clear read buffer
    read_buffer.get_events()

    # - Wait for the readout event to be sent back, and extract the timestep
    timestep = None
    continue_read = True
    start_t = time.time()

    # - Trigger a readout event on Xylo
    e = samna.xyloAudio3.event.TriggerProcessing()
    e.target_timestep = int(start_t)
    write_buffer.write([e])

    while continue_read:
        readout_events = read_buffer.get_n_events(1, 3000)

        # TODO: how to access Spike events for XyloAudio 3 instead of ReadoutEvents
        # the condition for getting the timestep was defined previously on the filtered list
        # ev_filt = [
        #     e for e in readout_events if isinstance(e, samna.xyloAudio3.event.Spike)
        # ]
        if readout_events:
            timestep = readout_events[0].timestep
            continue_read = False
        else:
            # - Check timeout
            continue_read &= (time.time() - start_t) < timeout

    if timestep is None:
        raise TimeoutError(f"Timeout after {timeout}s when reading current timestep.")

    # - Return the timestep
    return timestep


def advance_time_step(write_buffer: XyloAudio3WriteBuffer) -> None:
    """
    Take a single manual time-step on a Xylo HDK

    Args:
        write_buffer (XyloAudio3WriteBuffer): A write buffer connected to the Xylo HDK
    """
    write_buffer.write([samna.xyloAudio3.event.TriggerProcessing()])


def is_xylo_ready(
    read_buffer: XyloAudio3ReadBuffer, write_buffer: XyloAudio3WriteBuffer
) -> bool:
    """
    Read the status register on a XyloAudio 3 device to determine if it is finished with processing

    Args:
        read_buffer (XyloAudio3ReadBuffer): A read buffer connected to a XyloAudio 3 HDK
        write_buffer (XyloAudio3WriteBuffer): A write buffer connected to a XyloAudio 3 HDK

    Returns:
        bool: If ``True``, the Xylo device is finished with processing
    """
    # - Clear the buffer
    read_buffer.get_events()
    stat2 = read_register(read_buffer, write_buffer, reg.stat2)[0]
    return stat2 & (1 << reg.stat2__pd__pos)


def set_power_measure(
    hdk: XyloAudio3HDK,
    frequency: Optional[float] = 5.0,
) -> Tuple[
    samna.BasicSinkNode_unifirm_modules_events_measurement,
    samna.boards.common.power.PowerMonitor,
]:
    """
    Initialize power consumption measure on a hdk

    Args:
        hdk (XyloAudio3HDK): The Xylo HDK to be measured
        frequency (float): The frequency in Hz of power measurement. Default: 5.0

    Returns:
        power_buf: Event buffer to read power monitoring events from
        power_monitor: The power monitoring object
    """
    power_monitor = hdk.get_power_monitor()
    power_source = power_monitor.get_source_node()
    power_buf = samna.graph.sink_from(power_source)
    stopwatch = hdk.get_stop_watch()
    # Start the stopwatch to enable time-stamped power sampling
    stopwatch.start()
    # Start sampling power on all channels at a rate of frequency in Hz.
    power_monitor.start_auto_power_measurement(frequency)

    return power_buf, power_monitor


def apply_configuration(
    hdk: XyloAudio3HDK,
    config: XyloConfiguration,
    *_,
    **__,
) -> None:
    """
    Apply a configuration to the Xylo HDK

    Args:
        hdk (XyloAudio3HDK): The Xylo HDK to write the configuration to
        config (XyloConfiguration): A configuration for Xylo
    """
    # - Ideal -- just write the configuration using samna
    hdk.get_model().apply_configuration(config)


def configure_single_step_time_mode(
    config: XyloConfiguration,
) -> XyloConfiguration:
    """
    Switch on single-step mode on a XyloAudio 3 HDK

    Args:
        config (XyloConfiguration): The desired Xylo configuration to use
    """
    # - Write the configuration
    config.operation_mode = samna.xyloAudio3.OperationMode.Manual
    return config


def reset_input_spikes(write_buffer: XyloAudio3WriteBuffer) -> None:
    """
    Reset the input spike registers on a XyloAudio 3 HDK

    Args:
        write_buffer (XyloAudio3WriteBuffer): A write buffer connected to the Xylo HDK to access
    """
    for register in [reg.ispkreg0l, reg.ispkreg0h, reg.ispkreg1l, reg.ispkreg1h]:
        write_register(write_buffer, register)


def send_immediate_input_spikes(
    write_buffer: XyloAudio3WriteBuffer,
    spike_counts: Iterable[int],
) -> None:
    """
    Send a list of immediate input events to a XyloAudio 3 HDK in manual mode

    Args:
        write_buffer (XyloAudio3WriteBuffer): A write buffer connected to the Xylo HDK to access
        spike_counts (Iterable[int]): An Iterable containing one slot per input channel. Each entry indicates how many events should be sent to the corresponding input channel.
    """
    # - Encode input events
    events_list = []
    for input_channel, event in enumerate(spike_counts):
        if event:
            for _ in range(int(event)):
                events_list.append(
                    samna.xyloAudio3.event.Spike(neuron_id=input_channel)
                )

    # - Send input spikes for this time-step
    write_buffer.write(events_list)


def read_output_events(
    read_buffer: XyloAudio3ReadBuffer,
    write_buffer: XyloAudio3WriteBuffer,
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
    status = read_register(read_buffer, write_buffer, reg.stat1)

    # - Convert to neuron events and return
    string = format(int(status[0]), "0>32b")
    return np.array([bool(int(e)) for e in string[::-1]], "bool")


def blocking_read(
    read_buffer: XyloAudio3ReadBuffer,
    target_timestep: Optional[int] = None,
    count: Optional[int] = None,
    timeout: Optional[float] = None,
) -> Tuple[List, bool]:
    """
    Perform a blocking read on a buffer, optionally waiting for a certain count, a target timestep, or imposing a timeout


    You should not provide `count` and `target_timestep` together.


    Args:
        read_buffer (XyloAudio3ReadBuffer): A buffer to read from
        target_timestep (Optional[int]): The desired final timestep. Read until this timestep is returned in an event. Default: ``None``, don't wait until a particular timestep is read.
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

        # - Check if we reached the desired timestep
        if target_timestep:
            timesteps = [
                e.timestep
                for e in events
                if hasattr(e, "timestep") and e.timestep is not None
            ]

            if timesteps:
                reached_timestep = timesteps[-1] >= target_timestep
                continue_read &= ~reached_timestep

        # - Check timeout
        if timeout:
            is_timeout = (time.time() - start_time) > timeout
            continue_read &= not is_timeout

        # - Check number of events read
        if count:
            continue_read &= len(all_events) < count

        # - Continue reading if no events have been read
        if not target_timestep and not count:
            continue_read &= len(all_events) == 0

    # - Perform one final read for good measure
    all_events.extend(read_buffer.get_events())

    # - Return read events
    return all_events, is_timeout


def read_register(
    read_buffer: XyloAudio3ReadBuffer,
    write_buffer: XyloAudio3WriteBuffer,
    address: int,
    timeout: float = 2.0,
) -> List[int]:
    """
    Read the contents of a register

    Args:
        read_buffer (XyloAudio3ReadBuffer): A connected read buffer to the XYlo HDK
        write_buffer (XyloAudio3WriteBuffer): A connected write buffer to the Xylo HDK
        address (int): The register address to read
        timeout (float): A timeout in seconds

    Returns:
        List[int]: A list of events returned from the read
    """
    # - Clear buffer
    read_buffer.get_events()

    # - Set up a register read
    write_buffer.write([samna.xyloAudio3.event.ReadRegisterValue(address=address)])
    events = read_buffer.get_n_events(1, 3000)
    assert len(events) == 1
    return [events[0].data]


def decode_accel_mode_data(
    readout_events: List[ReadoutEvent],
    Nin: int,
    Nhidden_monitor: int,
    Nout_monitor: int,
    Nout: int,
    T_start: int,
    T_end: int,
) -> XyloState:
    """
    Read accelerated simulation mode data from a Xylo HDK

    Args:
        readout_events (List[ReadoutEvent]): A list of `ReadoutEvent`s recorded from XyloAudio 3
        Nin (int): The number of input channels for the configured network
        Nhidden_monitor (int): The number of hidden neurons to monitor
        Nout_monitor (int): The number of output neurons to monitor
        Nout (int): The number of output neurons in total
        T_start (int): Initial timestep
        T_end (int): Final timestep

    Returns:
        XyloState: The encapsulated state read from the Xylo device
    """
    # - Initialise lists for recording state
    T_count = T_end - T_start + 1
    vmem_ts = np.zeros((T_count, Nhidden_monitor), np.int16)
    isyn_ts = np.zeros((T_count, Nhidden_monitor), np.int16)
    isyn2_ts = np.zeros((T_count, Nhidden_monitor), np.int16)
    vmem_out_ts = np.zeros((T_count, Nout), np.int16)
    isyn_out_ts = np.zeros((T_count, Nout_monitor), np.int16)
    spikes_ts = np.zeros((T_count, Nhidden_monitor), np.int8)
    output_ts = np.zeros((T_count, Nout), np.int8)

    # - Loop over time steps
    for ev in readout_events:
        timestep = ev.timestep - T_start
        vmems = ev.neuron_v_mems
        isyns = ev.neuron_i_syns
        # FIXME - syn2 needs to be added

        if Nhidden_monitor != 0:
            vmem_ts[timestep, 0:Nhidden_monitor] = vmems[0:Nhidden_monitor]
            isyn_ts[timestep, 0:Nhidden_monitor] = isyns[0:Nhidden_monitor]
            isyn2_ts[timestep, 0:Nhidden_monitor] = isyns[0:Nhidden_monitor]
            spikes_ts[timestep] = ev.hidden_spikes

        if Nhidden_monitor != 0 or Nout_monitor != 0:
            isyn_out_ts[timestep, 0:Nout] = isyns[
                Nhidden_monitor : Nhidden_monitor + Nout_monitor
            ]

        vmem_out_ts[timestep, 0:Nout] = ev.output_v_mems
        output_ts[timestep] = ev.output_spikes

    # - Return as a XyloState object
    return XyloState(
        Nin=Nin,
        Nhidden=Nhidden_monitor,
        Nout=Nout,
        V_mem_hid=vmem_ts,
        I_syn_hid=isyn_ts,
        V_mem_out=vmem_out_ts,
        I_syn_out=isyn_out_ts,
        I_syn2_hid=isyn2_ts,
        Spikes_hid=spikes_ts,
        Spikes_out=output_ts,
    )


def write_register(
    write_buffer: XyloAudio3WriteBuffer, register: int, data: int = 0
) -> None:
    """
    Write data to a register on a XyloAudio 3 HDK

    Args:
        write_buffer (XyloAudio3WriteBuffer): A connected write buffer to the destination XyloAudio 3 HDK
        register (int): The address of the register to write to
        data (int): The data to write. Default: 0x0
    """
    write_buffer.write(
        [samna.xyloAudio3.event.WriteRegisterValue(address=register, data=data)]
    )


def write_memory(
    write_buffer: XyloAudio3WriteBuffer,
    start_address: int,
    data: Union[List[int], int] = 0,
) -> None:
    """
    Write data to memory on a XyloAudio 3 HDK

    This function will write a list of data to sequential locations in the XyloAudio 3 memory

    Args:
        write_buffer (XyloAudio3WriteBuffer): A connected write buffer to the target XyloAudio 3 HDK
        start_address (int): The start memory address to write to
        data (Union[List[int], int]): A list of integers, or single integer, to write to memory
    """
    # - Convert a single integer to a list
    if isinstance(data, int):
        data = [data]

    # - Generate write events
    write_events = [
        samna.xyloAudio3.event.WriteMemoryValue(addr, d)
        for (addr, d) in zip(range(start_address, start_address + len(data)), data)
    ]

    # - Send the memory write events
    write_buffer.write(write_events)


def read_memory(
    read_buffer: XyloAudio3ReadBuffer,
    write_buffer: XyloAudio3WriteBuffer,
    start_address: int,
    length: int = 1,
    read_timeout: float = 2.0,
) -> List[int]:
    """
    Read memory from a XyloAudio 3 device

    Args:
        read_buffer (XyloAudio3ReadBuffer): A read buffer connected to a XyloAudio 3 HDK
        write_buffer (XyloAudio3WriteBuffer): A write buffer connected to a XyloAudio 3 HDK
        start_address (int): The memory address to begin reading from
        length (int): The number of memory addresses to read. Default: ``1``
        read_timeout (float): The duration in seconds to wait for the read events. Default: ``2.0`` seconds

    Returns:
        List[int]: A list of the memory values read from the Xylo device
    """
    # - Generate read events
    read_events = [
        samna.xyloAudio3.event.ReadMemoryValue(addr)
        for addr in range(start_address, start_address + length)
    ]

    # - Clear buffer
    read_buffer.get_events()

    # - Request read
    write_buffer.write(read_events)

    # - Read data
    events, is_timeout = blocking_read(read_buffer, count=length, timeout=read_timeout)
    if is_timeout:
        raise TimeoutError(
            f"Memory read timed out after {read_timeout} s. Reading @{start_address}+{length}."
        )

    # - Filter returned events for the desired addresses
    return [
        e.data
        for e in events[1:]
        if hasattr(e, "address")
        and e.address >= start_address
        and e.address < start_address + length
    ]


def read_input_spikes(
    read_buffer: XyloAudio3ReadBuffer,
    write_buffer: XyloAudio3WriteBuffer,
    debug: bool = False,
) -> np.array:
    """
    Read the input spike register from a Xylo device

    Args:
        read_buffer (XyloAudio3ReadBuffer): A read buffer connected to a XyloAudio 3 HDK
        write_buffer (XyloAudio3WriteBuffer): A write buffer connected to a XyloAudio 3 HDK
        debug (bool): If ``True``, print debug information. Default: ``False``

    Returns:
        np.ndarray: An integer array containing the event counts in the current active input spike register
    """
    # - Read input spike register pointer
    dbg_stat1 = read_register(read_buffer, write_buffer, reg.dbg_stat1)[0]
    ispk_reg_ptr = bool((dbg_stat1 & 0b10000000000000000) >> 15)

    # - Read correct input spike register
    if not ispk_reg_ptr:
        ispkreg = read_register(
            read_buffer, write_buffer, reg.ispkreg0h
        ) + read_register(read_buffer, write_buffer, reg.ispkreg0l)

        print("ISPKREG0", ispkreg) if debug else None
        ispkreg = format(
            read_register(read_buffer, write_buffer, reg.ispkreg0h)[0], "0>8X"
        ) + format(read_register(read_buffer, write_buffer, reg.ispkreg0l)[0], "0>8X")
    else:
        ispkreg = read_register(
            read_buffer, write_buffer, reg.ispkreg1h
        ) + read_register(read_buffer, write_buffer, reg.ispkreg1l)

        print("ISPKREG1", ispkreg) if debug else None

        ispkreg = format(
            read_register(read_buffer, write_buffer, reg.ispkreg1h)[0], "0>8X"
        ) + format(read_register(read_buffer, write_buffer, reg.ispkreg1l)[0], "0>8X")

    # - Return input event counts as integer array
    return np.array([int(e, 16) for e in ispkreg[::-1]])


def read_neuron_synapse_state(
    read_buffer: XyloAudio3ReadBuffer,
    write_buffer: XyloAudio3WriteBuffer,
    Nin: int,
    Nhidden: int,
    Nout: int,
) -> XyloState:
    """
    Read and return the current neuron and synaptic state of neurons

    Args:
        read_buffer (XyloReadBuffer): A read buffer connected to the Xylo HDK
        write_buffer (XyloWriteBuffer): A write buffer connected to the Xylo HDK
        Nhidden (int): Number of hidden neurons to read. Default: ``1000`` (all neurons).
        Nout (int): Number of output neurons to read. Default: ``8`` (all neurons).

    Returns:
        :py:class:`.XyloState`: The recorded state as a ``NamedTuple``. Contains keys ``V_mem_hid``,  ``V_mem_out``, ``I_syn_hid``, ``I_syn_out``, ``I_syn2_hid``, ``Nhidden``, ``Nout``. This state has **no time axis**; the first axis is the neuron ID.

    """
    # - Read synaptic currents
    Isyn = read_memory(
        read_buffer,
        write_buffer,
        ram.NSCRAM,
        Nhidden + Nout,
    )

    # - Read synaptic currents 2
    Isyn2 = read_memory(read_buffer, write_buffer, ram.HSC2RAM, Nhidden)

    # - Read membrane potential
    Vmem = read_memory(
        read_buffer,
        write_buffer,
        ram.NMPRAM,
        Nhidden + Nout,
    )

    # - Read reservoir spikes
    Spikes = read_memory(read_buffer, write_buffer, ram.HSPKRAM, Nhidden)

    # - Return the state
    return XyloState(
        Nin,
        Nhidden,
        Nout,
        np.array(Vmem[:Nhidden], "int16"),
        np.array(Isyn[:Nhidden], "int16"),
        np.array(Vmem[-Nout:], "int16"),
        np.array(Isyn[-Nout:], "int16"),
        np.array(Isyn2, "int16"),
        np.array(Spikes, "bool"),
        read_output_events(read_buffer, write_buffer)[:Nout],
    )


def enable_ram_access(device: XyloAudio3HDK, enabled: bool) -> None:
    """
    Enable or disable RAM access for a Xylo device

    Args:
        device (XyloAudio3HDK): A connected XyloAudio 3 device
        enabled (bool): If ``True``, enable memory access. If ``False``, disable memory access.
    """
    if enabled:
        device.get_model().open_ram_access()
    else:
        device.get_model().close_ram_access()


def decode_realtime_mode_data(
    readout_events: List[SpikeEvent],
    Nout: int,
    T_start: int,
    T_end: int,
) -> np.ndarray:
    """
    Read realtime simulation mode data from a Xylo HDK

    Args:
        readout_events (List[ReadoutEvent]): A list of `ReadoutEvent`s recorded from XyloAudio 3
        Nout (int): The number of output neurons to monitor
        T_start (int): Initial timestep
        T_end (int): Final timestep

    Returns:
        Tuple[np.ndarray, np.ndarray]: (`vmem_out_ts`, `output_ts`) The membrane potential and output event trains from Xylo
    """
    # - Initialise lists for recording state
    T_count = T_end - T_start + 1
    neuronId = np.zeros((int(T_count), int(Nout)), np.int16)

    # - Loop over time steps
    for ev in readout_events:
        if type(ev) is SpikeEvent:
            timestep = ev.timestep - T_start
            if timestep >= 0:
                neuronId[timestep, 0:Nout] = ev.neuronId

    # - Return Vmem and spikes
    return neuronId
