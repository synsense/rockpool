"""
Low-level device kit utilities for the SYNS61201 Xylo-A2 HDK
"""

import enum
from ipaddress import IPv4Network
from rockpool.utilities.backend_management import backend_available

if not backend_available("samna"):
    raise ModuleNotFoundError(
        "`samna` not found. The Xylo HDK requires `samna` for interfacing."
    )

import samna
from samna.xyloImu.configuration import XyloConfiguration

# - Other imports
from warnings import warn
import time
import numpy as np
from pathlib import Path
from os import makedirs
import json
from bitstruct import pack_dict, unpack_dict


# - Typing and useful proxy types
from typing import Any, List, Iterable, Optional, NamedTuple, Union, Tuple

XyloIMUReadBuffer = samna.BasicSinkNode_xylo_imu_event_output_event
XyloIMUWriteBuffer = samna.BasicSourceNode_xylo_imu_event_input_event
XyloIMUNeuronStateBuffer = samna.xyloImu.NeuronStateSinkNode

XyloIMUHDK = Any


def find_xylo_imu_boards() -> List[XyloIMUHDK]:
    """
    Search for and return a list of Xylo IMU HDKs

    Iterate over devices and search for Xylo IMU HDK nodes. Return a list of available IMU HDKs, or an empty list if none are found.

    Returns:
        List[XyloIMUHDK]: A (possibly empty) list of Xylo IMU HDK nodes.
    """

    # - Get a list of devices
    device_list = samna.device.get_all_devices()

    # - Search for a xylo dev kit
    imu_hdk_list = [
        samna.device.open_device(d)
        for d in device_list
        if d.device_type_name == "XyloImuTestBoard"
    ]

    return imu_hdk_list


def new_xylo_read_buffer(
    hdk: XyloIMUHDK,
) -> XyloIMUReadBuffer:
    """
    Create and connect a new buffer to read from a Xylo HDK

    Args:
        hdk (XyloIMUHDK): A Xylo HDK to create a new buffer for

    Returns:
        XyloIMUReadBuffer: A connected event read buffer
    """
    # - Register a buffer to read events from Xylo
    buffer = XyloIMUReadBuffer()

    # - Get the device model
    model = hdk.get_model()

    # - Get Xylo output event source node
    source_node = model.get_source_node()

    # - Add the buffer as a destination for the Xylo output events
    graph = samna.graph.EventFilterGraph()
    graph.sequential([source_node, buffer])

    # - Return the buffer
    return buffer


def new_xylo_write_buffer(hdk: XyloIMUHDK) -> XyloIMUWriteBuffer:
    """
    Create a new buffer for writing events to a Xylo HDK

    Args:
        hdk (XyloIMUHDK): A Xylo HDK to create a new buffer for

    Returns:
        XyloIMUWriteBuffer: A connected event write buffer
    """
    buffer = XyloIMUWriteBuffer()
    sink = hdk.get_model().get_sink_node()
    graph = samna.graph.EventFilterGraph()
    graph.sequential([buffer, sink])

    return buffer


def new_xylo_state_monitor_buffer(
    hdk: XyloIMUHDK,
) -> XyloIMUNeuronStateBuffer:
    """
    Create a new buffer for monitoring neuron and synapse state and connect it

    Args:
        hdk (XyloIMUHDK): A Xylo HDK to configure

    Returns:
        XyloNeuronStateBuffer: A connected neuron / synapse state monitor buffer
    """
    # - Register a new buffer to receive neuron and synapse state
    buffer = XyloIMUNeuronStateBuffer()

    # - Get the device model
    model = hdk.get_model()

    # - Get Xylo output event source node
    source_node = model.get_source_node()

    # - Add the buffer as a destination for the Xylo output events
    graph = samna.graph.EventFilterGraph()
    graph.sequential([source_node, buffer])

    # - Return the buffer
    return buffer


def gen_clear_input_registers_events() -> List:
    """
    Create events to clear the input event registers
    """
    events = []
    for addr in [0x11, 0x12, 0x13, 0x14]:
        event = samna.xyloImu.event.WriteRegisterValue()
        event.address = addr
        events.append(event)

    return events


def initialise_xylo_hdk(write_buffer: XyloIMUWriteBuffer) -> None:
    """
    Initialise the Xylo HDK

    Args:
        write_buffer (XyloIMUWriteBuffer): A write buffer connected to a Xylo HDK to initialise
    """
    # - Always need to advance one time-step to initialise
    advance_time_step(write_buffer)


def advance_time_step(write_buffer: XyloIMUWriteBuffer) -> None:
    """
    Take a single manual time-step on a Xylo HDK

    Args:
        write_buffer (XyloIMUWriteBuffer): A write buffer connected to the Xylo HDK
    """
    e = samna.xyloImu.event.TriggerProcessing()
    write_buffer.write([e])


def set_power_measure(
    hdk: XyloIMUHDK,
    frequency: Optional[float] = 5.0,
) -> Tuple[
    samna.BasicSinkNode_unifirm_modules_events_measurement,
    samna.boards.common.power.PowerMonitor,
]:
    """
    Initialize power consumption measure on a hdk

    Args:
        hdk (XyloIMUHDK): The Xylo HDK to be measured
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


def apply_configuration(
    hdk: XyloIMUHDK,
    config: XyloConfiguration,
    *_,
    **__,
) -> None:
    """
    Apply a configuration to the Xylo HDK

    Args:
        hdk (XyloIMUHDK): The Xylo HDK to write the configuration to
        config (XyloConfiguration): A configuration for Xylo
    """
    # - Ideal -- just write the configuration using samna
    hdk.get_model().apply_configuration(config)

    # Set imu sensor enable to open the port for geting spikes from imu sensor
    hdk.set_imu_sensor_enable(True)


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

    Spikes_hid: np.ndarray
    """ np.ndarray: Spikes from hidden layer neurons ``(Nhidden,)``"""

    Spikes_out: np.ndarray
    """ np.ndarray: Spikes from output layer neurons ``(Nout,)``"""


def configure_accel_time_mode(
    config: XyloConfiguration,
    state_monitor_buffer: XyloIMUNeuronStateBuffer,
    monitor_Nhidden: Optional[int] = 0,
    monitor_Noutput: Optional[int] = 0,
    readout="Spike",
    record=False,
) -> Tuple[XyloConfiguration, XyloIMUNeuronStateBuffer]:
    """
    Switch on accelerated-time mode on a Xylo hdk, and configure network monitoring

    Notes:
        Use :py:func:`new_xylo_state_monitor_buffer` to generate a buffer to monitor neuron and synapse state.

    Args:
        config (XyloConfiguration): The desired Xylo configuration to use
        state_monitor_buffer (XyloIMUNeuronStateBuffer): A connected neuron state monitor buffer
        monitor_Nhidden (Optional[int]): The number of hidden neurons for which to monitor state during evolution. Default: ``0``, don't monitor any hidden neurons.
        monitor_Noutput (Optional[int]): The number of output neurons for which to monitor state during evolution. Default: ``0``, don't monitor any output neurons.
        readout: The readout out mode for which to output neuron states. Default: ``Spike''. Must be one of ``['Vmem', 'Spike', 'Isyn']``.
        record (bool): Iff ``True``, record state during evolution. Default: ``False``, do not record state.

    Returns:
        (XyloConfiguration, XyloIMUNeuronStateBuffer): `config` and `monitor_buffer`
    """
    assert readout in ["Vmem", "Spike", "Isyn"], f"{readout} is not supported."

    # - Select accelerated time mode, and general configuration
    config.operation_mode = samna.xyloImu.OperationMode.AcceleratedTime
    config.imu_if_input_enable = False
    config.debug.always_update_omp_stat = True

    # Configurations set for state memory reading
    config.debug.isyn_clock_enable = True
    config.debug.ra_clock_enable = True
    config.debug.bias_clock_enable = True
    config.debug.hm_clock_enable = True
    config.debug.ram_power_enable = True

    config.debug.monitor_neuron_spike = None
    config.debug.monitor_neuron_v_mem = None

    if record:
        config.debug.monitor_neuron_spike = samna.xyloImu.configuration.NeuronRange(
            0, monitor_Nhidden
        )
        config.debug.monitor_neuron_v_mem = samna.xyloImu.configuration.NeuronRange(
            0, monitor_Nhidden + monitor_Noutput
        )
        config.debug.monitor_neuron_i_syn = samna.xyloImu.configuration.NeuronRange(
            0, monitor_Nhidden + monitor_Noutput
        )

    else:
        if readout == "Isyn":
            config.debug.monitor_neuron_i_syn = samna.xyloImu.configuration.NeuronRange(
                monitor_Nhidden, monitor_Nhidden + monitor_Noutput
            )
        elif readout == "Vmem":
            config.debug.monitor_neuron_v_mem = samna.xyloImu.configuration.NeuronRange(
                monitor_Nhidden, monitor_Nhidden + monitor_Noutput
            )

    # - Configure the monitor buffer
    state_monitor_buffer.set_configuration(config)

    # - Return the configuration and buffer
    return config, state_monitor_buffer


def read_accel_mode_data(
    monitor_buffer: XyloIMUNeuronStateBuffer,
    Nin: int,
    Nhidden: int,
    Nout: int,
    Nt: int,
) -> XyloState:
    """
    Read accelerated simulation mode data from a Xylo HDK

    Args:
        monitor_buffer (XyloIMUNeuronStateBuffer): A connected `XyloIMUNeuronStateBuffer` to read from
        Nin (int): Number of input neurons to read. Default: ``16`` (all neurons).
        Nhidden (int): The number of hidden neurons to monitor
        Nout (int): The number of output neurons to monitor

    Returns:
        XyloState: The encapsulated state read from the Xylo device
    """
    # - Leave time for auto-triggering
    time.sleep(0.1 * Nt)
    # - Read data from neuron state buffer
    vmem_ts = np.array(monitor_buffer.get_reservoir_v_mem(), "int16").T
    isyn_ts = np.array(monitor_buffer.get_reservoir_i_syn(), "int16").T
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
        Spikes_hid=spikes_ts,
        Spikes_out=spikes_out_ts,
    )


def gen_clear_input_registers_events() -> List:
    """
    Create events to clear the input event registers
    """
    events = []
    for addr in [0x47, 0x48, 0x49, 0x4A]:
        event = samna.xyloImu.event.WriteRegisterValue()
        event.address = addr
        events.append(event)

    return events


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


def get_current_timestamp(
    read_buffer: XyloIMUReadBuffer,
    write_buffer: XyloIMUWriteBuffer,
    timeout: float = 3.0,
) -> int:
    """
    Retrieve the current timestamp on a Xylo HDK

    Args:
        read_buffer (XyloIMUReadBuffer): A connected read buffer for the xylo HDK
        write_buffer (XyloIMUWriteBuffer): A connected write buffer for the Xylo HDK
        timeout (float): A timeout for reading

    Returns:
        int: The current timestamp on the Xylo HDK
    """

    # - Clear read buffer
    read_buffer.get_events()

    # - Trigger a readout event on Xylo
    e = samna.xyloImu.event.TriggerReadout()
    write_buffer.write([e])

    # - Wait for the readout event to be sent back, and extract the timestamp
    timestamp = None
    continue_read = True
    start_t = time.time()
    while continue_read:
        readout_events = read_buffer.get_events()
        ev_filt = [
            e for e in readout_events if isinstance(e, samna.xyloImu.event.Readout)
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


def config_auto_mode(
    config: XyloConfiguration,
    dt: float,
    main_clk_rate: int,
    io,
) -> XyloConfiguration:
    """
    Set the Xylo HDK to real-time mode

    Args:
        config (XyloConfiguration): A configuration for Xylo IMU
        dt (float): The simulation time-step to use for this Module
        main_clk_rate(int): The main clock rate of Xylo
        io: The io module of Xylo

    Return:
        updated Xylo configuration
    """
    io.write_config(0x11, 0)
    config.operation_mode = samna.xyloImu.OperationMode.RealTime
    config.bias_enable = True
    config.hidden.aliasing = True
    config.debug.always_update_omp_stat = True
    config.imu_if_input_enable = True
    config.debug.imu_if_clk_enable = True
    config.time_resolution_wrap = int(dt * main_clk_rate)
    config.debug.imu_if_clock_freq_div = 0x169

    return config


def config_if_module(
    config: XyloConfiguration,
    num_avg_bitshift: int = 6,
    select_iaf_output: bool = False,
    sampling_period: int = 256,
    filter_a1_list: list = [
        -64700,
        -64458,
        -64330,
        -64138,
        -63884,
        -63566,
        -63185,
        -62743,
        -62238,
        -61672,
        -61045,
        -60357,
        -59611,
        -58805,
        -57941,
    ],
    filter_a2_list: list = [0x00007CBF] + [0x00007C0A] * 14,
    scale_values: list = [8] * 15,
    Bb_list: list = [6] * 15,
    B_wf_list: list = [8] * 15,
    B_af_list: list = [9] * 15,
    iaf_threshold_values: list = [0x000007D0] * 15,
    *args,
    **kwargs,
) -> XyloConfiguration:
    """
    Configure the imu interface module

    Args:
        config (XyloConfiguration): a configuration for Xylo IMU
        num_avg_bitshift (int): number of bitshifts used in the low-pass filter implementation
        select_iaf_output (bool): if True, select the IAF neuron spike encoder; else, select the scale spike encoder
        sampling_period (int): sampling period
        filter_a1_list (list): list of a1 tap values
        filter_a2_list (list): list of a2 tap values
        scale_values (list): list of number of right-bit-shifts needed for down-scaling the input signal
        Bb_list (list): list of bits needed for scaling b0
        B_wf_list (list): list of bits needed for fractional part of the filter output
        B_af_list (list): list of bits needed for encoding the fractional parts of taps
        iaf_threshold_values (list): list of threshold values of IAF neurons

    Return:
        updated Xylo configuration
    """

    # IMU interface hyperparameters
    config.input_interface.enable = True
    config.input_interface.estimator_k_setting = num_avg_bitshift  # num_avg_bitshift
    config.input_interface.select_iaf_output = (
        select_iaf_output  # True if use IAF encoding
    )
    config.input_interface.update_matrix_threshold = (
        sampling_period - 1
    )  # sampling_period
    config.input_interface.delay_threshold = 1
    config.input_interface.bpf_bb_values = Bb_list
    config.input_interface.bpf_bwf_values = B_wf_list
    config.input_interface.bpf_baf_values = B_af_list
    config.input_interface.bpf_a1_values = [i & 0x1FFFF for i in filter_a1_list]
    config.input_interface.bpf_a2_values = filter_a2_list
    config.input_interface.scale_values = scale_values  # num_scale_bits
    config.input_interface.iaf_threshold_values = iaf_threshold_values

    return config


def read_imu_register(
    read_buffer: XyloIMUReadBuffer,
    write_buffer: XyloIMUWriteBuffer,
    address: int,
    timeout: float = 2.0,
) -> List[int]:
    """
    Read the contents of a register

    Args:
        read_buffer (XyloIMUReadBuffer): A connected read buffer to the XYlo HDK
        write_buffer (XyloIMUWriteBuffer): A connected write buffer to the Xylo HDK
        address (int): The register address to read
        timeout (float): A timeout in seconds

    Returns:
        List[int]: A list of events returned from the read
    """
    # - Set up a register read
    rrv_ev = samna.xyloImu.event.ReadRegisterValue()
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

    # - Return data
    return [e.data for e in ev_filt]


def write_imu_register(
    write_buffer: XyloIMUWriteBuffer, register: int, data: int = 0
) -> None:
    """
    Write data to a register on a Xylo IMU HDK

    Args:
        write_buffer (XyloIMUWriteBuffer): A connected write buffer to the destination Xylo IMU HDK
        register (int): The address of the register to write to
        data (int): The data to write. Default: 0x0
    """
    wwv_ev = samna.xyloImu.event.WriteRegisterValue()
    wwv_ev.address = register
    wwv_ev.data = data
    write_buffer.write([wwv_ev])
