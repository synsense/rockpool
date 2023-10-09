"""
Low-level device kit utilities for the Xylo-IMU HDK
"""

import samna
from samna.xyloImu.configuration import XyloConfiguration

# - Other imports
import time
import numpy as np

# - Typing and useful proxy types
from typing import List, Optional, NamedTuple, Tuple

XyloIMUReadBuffer = samna.BasicSinkNode_xylo_imu_event_output_event
XyloIMUWriteBuffer = samna.BasicSourceNode_xylo_imu_event_input_event
IMUSensorReadBuffer = samna.DeviceSinkNode_unifirm_modules_mc3632_input_event
IMUSensorWriteBuffer = samna.DeviceSourceNode_unifirm_modules_mc3632_output_event

ReadoutEvent = samna.xyloImu.event.Readout

XyloIMUHDK = samna.xyloImuBoards.XyloImuTestBoard
IMUSensorHDK = samna.unifirm.modules.mc3632.Mc3632


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
    return samna.graph.sink_from(hdk.get_model_source_node())


def new_xylo_write_buffer(hdk: XyloIMUHDK) -> XyloIMUWriteBuffer:
    """
    Create a new buffer for writing events to a Xylo HDK

    Args:
        hdk (XyloIMUHDK): A Xylo HDK to create a new buffer for

    Returns:
        XyloIMUWriteBuffer: A connected event write buffer
    """
    return samna.graph.source_to(hdk.get_model_sink_node())


def initialise_imu_sensor(
    hdk: XyloIMUHDK, frequency: int = 200
) -> Tuple[
    IMUSensorReadBuffer,
    IMUSensorWriteBuffer,
    IMUSensorReadBuffer,
    IMUSensorHDK,
    samna.graph.EventFilterGraph,
]:
    """
    Initialise the MC3632 IMU sensor HDK

    Args:
        hdk (XyloIMUHDK): A Xylo IMU device containing an IMU sensor to initialise
    """

    # - set XyloIMUHDK to read data mode and get IMU sensor device
    hdk.get_stop_watch().set_enable_value(True)
    time.sleep(0.1)
    mc = hdk.get_mc3632()

    # - Register read buffer to read data from IMU sensor
    read_buffer = samna.graph.sink_from(mc.get_source_node())
    write_buffer = samna.graph.source_to(mc.get_sink_node())

    # - Build an acceleration event filter
    graph = samna.graph.EventFilterGraph()
    _, etf0, accel_buf = graph.sequential(
        [mc.get_source_node(), "Mc3632OutputEventTypeFilter", samna.graph.JitSink()]
    )
    etf0.set_desired_type("events::Acceleration")
    graph.start()

    # - Configure the imu densor device
    mc.auto_read_enable(False)
    if not mc.setup():
        raise ConnectionError("Could not connect to the MC3632 device.")

    # - Initialise auto reading of sensor values
    mc.set_auto_read_freq(int(frequency))
    mc.auto_read_enable(True)

    # - Return the buffer and the IMU sensor
    return read_buffer, write_buffer, accel_buf, mc, graph


def initialise_xylo_hdk(write_buffer: XyloIMUWriteBuffer) -> None:
    """
    Initialise the Xylo IMU HDK

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
    power_source = power_monitor.get_source_node()
    power_buf = samna.graph.sink_from(power_source)
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
    hdk: XyloIMUHDK,
    config: XyloConfiguration,
    Nout: int = 0,
    monitor_Nhidden: int = 0,
    monitor_Noutput: int = 0,
    readout="Spike",
    record=False,
) -> Tuple[XyloConfiguration]:
    """
    Switch on accelerated-time mode on a Xylo hdk, and configure network monitoring

    Notes:
        Use :py:func:`new_xylo_state_monitor_buffer` to generate a buffer to monitor neuron and synapse state.

    Args:
        config (XyloConfiguration): The desired Xylo configuration to use
        Nout (int): Number of output neurons in total
        monitor_Nhidden (Optional[int]): The number of hidden neurons for which to monitor state during evolution. Default: ``0``, don't monitor any hidden neurons.
        monitor_Noutput (Optional[int]): The number of output neurons for which to monitor state during evolution. Default: ``0``, don't monitor any output neurons.
        readout: The readout out mode for which to output neuron states. Default: ``Spike''. Must be one of ``['Vmem', 'Spike', 'Isyn']``.
        record (bool): Iff ``True``, record state during evolution. Default: ``False``, do not record state.

    Returns:
        XyloConfiguration: `config`
    """

    # Set imu sensor enable to open the port for geting spikes from imu sensor
    hdk.enable_manual_input_acceleration(True)

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
                monitor_Nhidden, monitor_Nhidden + Nout
            )
        elif readout == "Vmem":
            config.debug.monitor_neuron_v_mem = samna.xyloImu.configuration.NeuronRange(
                monitor_Nhidden, monitor_Nhidden + Nout
            )

    # - Return the configuration and buffer
    return config


def blocking_read(
    read_buffer: XyloIMUReadBuffer,
    target_timestep: Optional[int] = None,
    count: Optional[int] = None,
    timeout: Optional[float] = None,
) -> Tuple[List, bool]:
    """
    Perform a blocking read on a buffer, optionally waiting for a certain count, a target timestep, or imposing a timeout

    You should not provide `count` and `target_timestep` together.

    Args:
        read_buffer (XyloReadBuffer): A buffer to read from
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
        readout_events (List[ReadoutEvent]): A list of `ReadoutEvent`s recorded from Xylo IMU
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
    vmem_out_ts = np.zeros((T_count, Nout), np.int16)
    isyn_out_ts = np.zeros((T_count, Nout_monitor), np.int16)
    spikes_ts = np.zeros((T_count, Nhidden_monitor), np.int8)
    output_ts = np.zeros((T_count, Nout), np.int8)

    # print(f"decode_accel_mode_data: T_start {T_start} T_end {T_end}; T_count {T_count}")

    # - Loop over time steps
    for ev in readout_events:
        if type(ev) is ReadoutEvent:
            timestep = ev.timestep - T_start
            vmems = ev.neuron_v_mems
            vmem_ts[timestep, 0:Nhidden_monitor] = vmems[0:Nhidden_monitor]
            vmem_out_ts[timestep, 0:Nout] = ev.output_v_mems

            isyns = ev.neuron_i_syns
            isyn_ts[timestep, 0:Nhidden_monitor] = isyns[0:Nhidden_monitor]
            isyn_out_ts[timestep, 0:Nout] = isyns[
                Nhidden_monitor : Nhidden_monitor + Nout_monitor
            ]

            spikes_ts[timestep] = ev.hidden_spikes
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
        Spikes_hid=spikes_ts,
        Spikes_out=output_ts,
    )


def decode_realtime_mode_data(
    readout_events: List[ReadoutEvent],
    Nout: int,
    T_start: int,
    T_end: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read realtime simulation mode data from a Xylo HDK

    Args:
        readout_events (List[ReadoutEvent]): A list of `ReadoutEvent`s recorded from Xylo IMU
        Nout (int): The number of output neurons to monitor
        T_start (int): Initial timestep
        T_end (int): Final timestep

    Returns:
        Tuple[np.ndarray, np.ndarray]: (`vmem_out_ts`, `output_ts`) The membrane potential and output event trains from Xylo
    """
    # - Initialise lists for recording state
    T_count = T_end - T_start + 1
    vmem_out_ts = np.zeros((T_count, Nout), np.int16)
    output_ts = np.zeros((T_count, Nout), np.int8)

    # - Loop over time steps
    for ev in readout_events:
        if type(ev) is ReadoutEvent:
            timestep = ev.timestep - T_start
            if timestep >= 0:
                vmem_out_ts[timestep, 0:Nout] = ev.output_v_mems
                output_ts[timestep] = ev.output_spikes

    # - Return Vmem and spikes
    return vmem_out_ts, output_ts


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


def get_current_timestep(
    read_buffer: XyloIMUReadBuffer,
    write_buffer: XyloIMUWriteBuffer,
    timeout: float = 3.0,
) -> int:
    """
    Retrieve the current timestep on a Xylo HDK

    Args:
        read_buffer (XyloIMUReadBuffer): A connected read buffer for the xylo HDK
        write_buffer (XyloIMUWriteBuffer): A connected write buffer for the Xylo HDK
        timeout (float): A timeout for reading

    Returns:
        int: The current timestep on the Xylo HDK
    """

    # - Clear read buffer
    read_buffer.get_events()

    # - Trigger a readout event on Xylo
    e = samna.xyloImu.event.TriggerReadout()
    write_buffer.write([e])

    # - Wait for the readout event to be sent back, and extract the timestep
    timestep = None
    continue_read = True
    start_t = time.time()
    while continue_read:
        readout_events = read_buffer.get_events()
        ev_filt = [
            e for e in readout_events if isinstance(e, samna.xyloImu.event.Readout)
        ]
        if ev_filt:
            timestep = ev_filt[0].timestep
            continue_read = False
        else:
            # - Check timeout
            continue_read &= (time.time() - start_t) < timeout

    if timestep is None:
        raise TimeoutError(f"Timeout after {timeout}s when reading current timestep.")

    # - Return the timestep
    return timestep


def config_realtime_mode(
    config: XyloConfiguration,
    dt: float,
    main_clk_rate: int,
) -> XyloConfiguration:
    """
    Set the Xylo HDK to real-time mode

    Args:
        config (XyloConfiguration): A configuration for Xylo IMU
        dt (float): The simulation time-step to use for this Module
        main_clk_rate (int): The main clock rate of Xylo in Hz

    Return:
        updated Xylo configuration
    """
    # - Select real-time operation mode
    config.operation_mode = samna.xyloImu.OperationMode.RealTime

    config.debug.always_update_omp_stat = True
    config.imu_if_input_enable = True
    config.debug.imu_if_clk_enable = True

    # - Configure Xylo IMU clock rate
    config.time_resolution_wrap = int(dt * main_clk_rate)
    IMU_IF_clk_rate = 50_000  # IMU IF clock must be 50 kHz
    config.debug.imu_if_clock_freq_div = int(main_clk_rate / IMU_IF_clk_rate - 1)

    # - Set configuration timeout
    config.input_interface.configuration_timeout = 20_000

    # - No monitoring of internal state in realtime mode
    config.debug.monitor_neuron_v_mem = None
    config.debug.monitor_neuron_i_syn = None
    config.debug.monitor_neuron_spike = None

    return config


def encode_imu_data(input: np.ndarray) -> List[samna.events.Acceleration]:
    """
    Encode imu data as `samna` events

    Args:
        input (np.ndarray): An array ``[T, 3]`` of imu data, specifying the number of timesteps, and the accelerations along x, y, z axes. The data must be quantized to int.

    Returns:
        List[samna.events.Acceleration]: A list of encoded IMU data events
    """

    imu_input = [
        samna.events.Acceleration(
            x=int(i[0]),
            y=int(i[1]),
            z=int(i[2]),
        )
        for i in input
    ]

    return imu_input


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


def set_xylo_core_clock_freq(device: XyloIMUHDK, desired_freq_MHz: float) -> float:
    """
    Set the inference core clock frequency used by Xylo

    Args:
        device (XyloIMUHDK): A Xylo device to configure
        desired_freq_MHz (float): The desired Xylo core clock frequency in MHz

    Returns:
        (float): Actual frequency obtained, in MHz
    """
    # - Determine wait period and actual obtained clock frequency
    wait_period = int(np.ceil(round(100 / desired_freq_MHz) / 2 - 1))
    actual_freq = 100 / (2 * (wait_period + 1))

    # - Configure device
    device.get_io_module().set_main_clk_rate(int(actual_freq * 1e6))

    return actual_freq


def enable_ram_access(device: XyloIMUHDK, enabled: bool) -> None:
    if enabled:
        device.get_model().open_ram_access()
    else:
        device.get_model().close_ram_access()
