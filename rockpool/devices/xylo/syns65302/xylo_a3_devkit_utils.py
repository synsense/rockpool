"""
Low-level device kit utilities for the Xylo Audio 3 HDK
"""

import samna
from samna.xyloAudio3.configuration import XyloConfiguration

# - Other imports
import time
import numpy as np

from warnings import warn

# - Useful constants for XA3
from . import ram, reg, constants


# - Typing and useful proxy types
from typing import List, Optional, NamedTuple, Tuple, Union, Iterable

XyloAudio3ReadBuffer = samna.BasicSinkNode_xylo_audio3_event_output_event
XyloAudio3WriteBuffer = samna.BasicSourceNode_xylo_audio3_event_input_event
ReadoutEvent = samna.xyloAudio3.event.Readout
SpikeEvent = samna.xyloAudio3.event.Spike
XyloAudio3HDK = samna.xyloAudio3Boards.XyloAudio3TestBoard
# Xylo2NeuronStateBuffer = samna.xyloCore2.NeuronStateSinkNode


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
    Search for and return a list of Xylo A3 HDKs

    Iterate over devices and search for Xylo A3 HDK nodes. Return a list of available A3 HDKs, or an empty list if none are found.

    Returns:
        List[XyloAudio3HDK]: A (possibly empty) list of Xylo A3 HDK nodes.
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

    # - Register a new buffer to receive neuron and synapse state
    # buffer = Xylo2NeuronStateBuffer()
    # buffer = ReadoutEvent()

    # - Add the buffer as a destination for the Xylo output events
    # graph = samna.graph.EventFilterGraph()
    # print()
    # print('graph', graph)
    # print('source node', source_node)
    # print('buffer', buffer)
    # graph.sequential([source_node, buffer])

    # - Return the buffer
    return state_buf, graph


def write_register_dict(write_buffer: XyloAudio3WriteBuffer, register_config) -> None:
    for register, data in register_config.items():
        if hasattr(reg, register):
            write_register(write_buffer, getattr(reg, register), data)
        else:
            raise ValueError(
                f"Register `{register}` not found for Xylo A3. Skipping this register."
            )


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


# To remove after test with PDM
def xylo_config_clk(
    read_buffer: XyloAudio3ReadBuffer,
    write_buffer: XyloAudio3WriteBuffer,
    clk_div: int = 1,
    debug: bool = False,
) -> None:
    """
    Configure the clock divider registers on the Xylo XA3 HDK

    Args:
        write_buffer (XyloAudio3WriteBuffer): A write buffer connected to a Xylo HDK
    """
    update_register_field(
        read_buffer,
        write_buffer,
        reg.clk_div,
        reg.clk_div__sdm__pos_lsb,
        reg.clk_div__sdm__pos_msb,
        max((clk_div >> 1) - 1, 0),
    )
    (
        print("clk_div:  0x" + format(read_register(reg.clk_div), "_X"))
        if debug >= 1
        else None
    )
    update_register_field(
        read_buffer,
        write_buffer,
        reg.clk_ctrl,
        reg.clk_ctrl__sdm__pos,
        reg.clk_ctrl__sdm__pos,
        1,
    )
    if debug:
        print("clk_ctrl: 0x" + format(read_register(reg.clk_ctrl), "_X"))


def initialise_xylo_hdk(
    hdk: XyloAudio3HDK,
    sleep_time: float = 5e-3,
) -> None:
    """
    Initialise the Xylo Audio 3 HDK

    Args:
        hdk (XyloAudio3HDK): A connected Xylo HDK
        write_buffer (XyloAudio3WriteBuffer): A write buffer connected to a Xylo HDK to initialise
    """
    ioc = hdk.get_io_control_module()
    io = hdk.get_io_module()

    ioc.write_config(0, 0x0)  # power off
    time.sleep(sleep_time)
    io.write_config(0x0008, 0)  # main clock disable
    time.sleep(sleep_time)
    ioc.write_config(0, 0x1B)  # power on
    time.sleep(sleep_time)
    io.write_config(0x0008, 1)  # main clock enable
    time.sleep(sleep_time)


def fpga_enable_pdm_interface(
    hdk: XyloAudio3HDK,
    pdm_clock_edge: bool = False,
    pdm_driving_direction: bool = False,
) -> None:
    """
    Configure the PDM input interface on a Xylo A3 HDK

    This function configures the FPGA to generate the PDM clock, and configures the FPGA and Xylo A3 chip with a common setup for PDM clock edge triggering and driving direction.

    Args:
        hdk (XyloAudio3HDK): A connected Xylo Audio 3 HDK
        read_buffer (XyloAudio3ReadBuffer):
        write_bufer (XyloAudio3WriteBuffer):
        pdm_clock_edge (bool): Which edge of the PDM clock to use to trigger Xylo clock. ``False``: rising edge; ``True`` (falling edge, default).
        pdm_driving_direction (bool): Which direction is the Xylo PDM clock pin? ``False``: Xylo PDM_CLK pin in slave mode, driven externally (default); ``True``: PDM_CLK driven by Xylo A3 chip.
    """
    io = hdk.get_io_module()

    # set PDM clock
    io.write_config(0x0027, 0)  # pdm clock msw
    io.write_config(0x0028, 19)  # pdm clock lsw
    if pdm_driving_direction:
        io.write_config(0x0029, 0)  # FPGA pdm clock generation disabled
    else:
        io.write_config(0x0029, 1)  # FPGA pdm clock generation enabled
    io.write_config(0x0026, 2)  # select: use pdm interface

    # bit 0: PDM_CLK edge (0: FPGA drives PDM_DATA at falling edge, 1: FPGA drives PDM_DATA at rising edge)
    # bit 1: PDM_CLK dir  (0: FPGA->Xylo, 1: Xylo->FPGA)
    pdm_config = pdm_clock_edge + pdm_driving_direction << 1
    io.write_config(0x002A, pdm_config)

    # FPGA drive PDM_DATA
    io.write_config(0x0012, 1)
    # PDM port write enable
    io.write_config(0x0013, 1)


def fpga_pdm_clk_enable(hdk: XyloAudio3HDK) -> None:
    io = hdk.get_io_module()
    io.write_config(0x0029, 1)  # pdm clock enable


def fpga_pdm_clk_disable(hdk: XyloAudio3HDK) -> None:
    io = hdk.get_io_module()
    io.write_config(0x0029, 0)  # pdm clock disable


def send_pdm_datas(write_buffer: XyloAudio3WriteBuffer, datas, debug=0) -> None:
    # read_important_register()
    print(f"send pdm datas: {datas}") if debug >= 2 else None
    events = []
    for n in datas:
        n = n.strip()
        # print(f"n: \${n}\$") if debug>=1 else None
        ev = samna.xyloAudio3.event.AFESample()
        ev.data = int(n)
        events.append(ev)
    write_buffer.write(events)


def enable_saer_input(hdk: XyloAudio3HDK) -> None:
    io = hdk.get_io_module()

    # set SAER clock
    io.write_config(0x0020, 0)  # saer clock msw
    io.write_config(0x0021, 3)  # saer clock lsw
    io.write_config(0x0022, 1)  # saer clock enable
    io.write_config(0x0026, 0)  # stif_select: saer

    # FPGA drive PDM_DATA pin (for SAER input)
    io.write_config(0x0012, 1)
    # pdm port write enable
    io.write_config(0x0013, 1)


def enable_real_time_mode(hdk: XyloAudio3HDK) -> None:
    io = hdk.get_io_module()

    # FPGA drive PDM_DATA pin (for SAER input)
    io.write_config(0x0012, 0)
    # set real time mode
    io.write_config(0x31, 2)


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

        # TODO: how to access Spike events for XyloA3 instead of ReadoutEvents
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
        frequency (float): The frequency of power measurement. Default: 5.0

    Returns:
        power_buf: Event buffer to read power monitoring events from
        power_monitor: The power monitoring object
    """
    power_monitor = hdk.get_power_monitor()
    power_source = power_monitor.get_source_node()
    power_buf = samna.graph.sink_from(power_source)
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
    Switch on single-step mode on a Xylo Audio 3 HDK

    Args:
        config (XyloConfiguration): The desired Xylo configuration to use
    """
    # - Write the configuration
    config.operation_mode = samna.xyloAudio3.OperationMode.Manual
    return config


def reset_input_spikes(write_buffer: XyloAudio3WriteBuffer) -> None:
    """
    Reset the input spike registers on a Xylo A3 HDK

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
    Send a list of immediate input events to a Xylo A3 HDK in manual mode

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
    # - Set up a register read
    write_buffer.write([samna.xyloAudio3.event.ReadRegisterValue(address=address)])
    events = read_buffer.get_n_events(1, 3000)
    assert len(events) == 1
    return [events[0].data]

    # # - Wait for data and read it
    # start_t = time.time()
    # continue_read = True
    # while continue_read:
    #     # - Read from the buffer
    #     events = read_buffer.get_events()

    #     # - Filter returned events for the desired address
    #     ev_filt = [e for e in events if hasattr(e, "address") and e.address == address]

    #     # - Should we continue the read?
    #     continue_read &= len(ev_filt) == 0
    #     continue_read &= (time.time() - start_t) < timeout

    # # - If we didn't get the required register read, raise an error
    # if len(ev_filt) == 0:
    #     raise TimeoutError(f"Timeout after {timeout}s when reading register {address}.")

    # # - Return data
    # return [e.data for e in ev_filt]


def write_register(
    write_buffer: XyloAudio3WriteBuffer, register: int, data: int = 0
) -> None:
    """
    Write data to a register on a Xylo A3 HDK

    Args:
        write_buffer (XyloAudio3WriteBuffer): A connected write buffer to the destination Xylo A3 HDK
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
    Write data to memory on a Xylo A3 HDK

    This function will write a list of data to sequential locations in the Xylo A3 memory

    Args:
        write_buffer (XyloAudio3WriteBuffer): A connected write buffer to the target Xylo A3 HDK
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
    if enabled:
        device.get_model().open_ram_access()
    else:
        device.get_model().close_ram_access()


def decode_ctrl1(value: int) -> dict:
    return {
        "ram_wu_st": (value >> 28) & 0b1111,
        "i2c_noise_filter_cyc": (value >> 24) & 0b111,
        "hm_en": bool((value >> reg.ctrl1__hm_en__pos) & 0b1),
        "always_update_omp_stat": bool(
            (value >> reg.ctrl1__always_update_omp_stat__pos) & 0b1
        ),
        "keep_int": bool((value >> reg.ctrl1__keep_int__pos) & 0b1),
        "ram_active": bool((value >> reg.ctrl1__ram_active__pos) & 0b1),
        "mem_clk_on": bool((value >> reg.ctrl1__mem_clk_on__pos) & 0b1),
        "owbs": (value >> reg.ctrl1__owbs__pos) & 0b111,
        "hwbs": (value >> reg.ctrl1__hwbs__pos) & 0b111,
        "iwbs": (value >> reg.ctrl1__iwbs__pos) & 0b111,
        "bias_en": bool((value >> reg.ctrl1__bias_en__pos) & 0b1),
        "alias_en": bool((value >> reg.ctrl1__alias_en__pos) & 0b1),
        "isyn2_en": bool((value >> reg.ctrl1__isyn2_en__pos) & 0b1),
        "man": bool((value >> reg.ctrl1__man__pos) & 0b1),
    }


def decode_dbg_ctrl1(value: int) -> dict:
    return {
        "dbg_en": bool((value >> 31) & 0b1),
        "mon_en": bool((value >> reg.dbg_ctrl1__mon_en__pos) & 0b1),
        "dbg_sta_upd_en": bool((value >> reg.dbg_ctrl1__dbg_sta_upd_en__pos) & 0b1),
        "spk2saer_en": bool((value >> 26) & 0b1),
        "cntr_stat_src_sel": (value >> 25) & 0b1,
        "all_ram_on": bool((value >> 24) & 0b1),
        "ds": (value >> 23) & 0b1,
        "ls": (value >> 22) & 0b1,
        "ram_auto_slp_ctrl": (value >> 21) & 0b1,
        "ram_man_slp_ctrl": (value >> 20) & 0b1,
        "stif_sel": (value >> 19) & 0b1,
        "spk_src_sel": (value >> 18) & 0b1,
        "ispkreg_sel_rd": (value >> 17) & 0b1,
        "ispkreg_sel_wr": (value >> 16) & 0b1,
        "monsig_reg": (value >> 8) & 0xF,
        "hm_clk_on": (value >> 4) & 0b1,
        "bias_clk_on": (value >> 3) & 0b1,
        "alias_clk_on": (value >> 2) & 0b1,
        "isyn2_clk_on": (value >> 1) & 0b1,
        "isyn_clk_on": bool((value >> 0) & 0b1),
    }


def decode_adc_ctrl(value: int) -> dict:
    return {
        "en": bool((value >> 0) & 0b1),
        "convert": bool((value >> 1) & 0b1),
        "en_data_p": bool((value >> 2) & 0b1),
        "en_data_s": bool((value >> 3) & 0b1),
        "speed": (value >> 4) & 0b11,
    }


def decode_dfe_ctrl(value: int) -> dict:
    return {
        "bpf_en": bool((value >> 0) & 0b1),
        "iaf_en": bool((value >> 1) & 0b1),
        "dn_en": bool((value >> 2) & 0b1),
        "hm_en": bool((value >> 3) & 0b1),
        "mic_if_sel": bool((value >> 4) & 0b1),
        "global_thr": bool((value >> 5) & 0b1),
        "pdm_clk_dir": bool((value >> 8) & 0b1),
        "pdm_clk_edge": bool((value >> 9) & 0b1),
        "bfi_en": bool((value >> 10) & 0b1),
        "adc_bit_en": (value >> 16) & 0b11_1111_1111_1111,
    }


def decode_agc_ctrl1(value: int) -> dict:
    return {
        "gs_diact": (value >> 0) & 0b1,
        "pga_gain_bypass": (value >> 1) & 0b1,
        "aaf_os_mode": (value >> 4) & 0b11,
        "pga_gain_idx_cfg": (value >> 8) & 0b1_1111,
        "rise_avg_bitshift": (value >> 16) & 0b1_1111,
        "fall_avg_bitshift": (value >> 24) & 0b1_1111,
    }


def decode_agc_ctrl2(value: int) -> dict:
    return {
        "reli_max_hystr": (value >> 0) & 0b11_1111_1111,
        "avg_bitshift": (value >> 16) & 0b1_1111,
        "num_bits_gain_fraction": (value >> 24) & 0b1111,
        "pgiv16": (value >> 28) & 0b111,
    }


def decode_agc_ctrl3(value: int) -> dict:
    return {
        "max_num_sample": (value >> 0) & 0xFF_FFFF,
        "ghri_h": (value >> 28) & 0b11,
    }


def read_all_register(read_buffer, write_buffer):
    for address in range(reg.cntr_stat + 1):
        data = read_register(read_buffer, write_buffer, address)[0]
        print("read register ", address, hex(data))


def decode_realtime_mode_data(
    readout_events: List[SpikeEvent],
    Nout: int,
    T_start: int,
    T_end: int,
) -> np.ndarray:
    """
    Read realtime simulation mode data from a Xylo HDK

    Args:
        readout_events (List[ReadoutEvent]): A list of `ReadoutEvent`s recorded from Xylo Audio3
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
