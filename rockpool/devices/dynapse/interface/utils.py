"""
Dynap-SE2 samna connection utilities

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

    name change: utils/se2 -> interface/utils @ 220926

[] TODO : find_dynapse_boards is similar to xylo, can be merged
31/05/2022
"""
from typing import Any, Dict, List, Optional, Tuple

import time
import logging
import numpy as np

from rockpool.devices.dynapse.samna_alias.dynapse2 import (
    Dynapse2Destination,
    NormalGridEvent,
    Dynapse2Interface,
    Dynapse2Model,
    DeviceInfo,
)

from rockpool.timeseries import TSEvent
import os

# Try to import samna for device interfacing
try:
    import samna
    import samna.dynapse2 as se2
except:
    samna = Any
    se2 = Any
    print(
        "Device interface requires `samna` package which is not installed on the system"
    )


def __bit_path_Dynapse2Stack() -> str:
    """
    __bit_path_Dynapse2Stack returns the bitfile path of the Dynapse2Stack configuration file.

    :return: bitfile path
    :rtype: str
    """
    __dirname__ = os.path.dirname(os.path.abspath(__file__))
    __bit_file_path = os.path.join(__dirname__, "bitfiles", "Dynapse2Stack.bit")
    return __bit_file_path


def find_dynapse_boards(name: str = "DYNAP-SE2") -> List[DeviceInfo]:
    """
    find_dynapse_boards identifies the Dynap-SE2 boards plugged in to the system

    :param name: the name of the devices, defaults to "DYNAP-SE2"
    :type name: str, optional
    :raises ConnectionError: No samna device found plugged in to the system!
    :return: a list of Dynap-SE2 device info objects among all samna devices
    :rtype: List[DeviceInfo]
    """

    dynapse_list = []
    devices = samna.device.get_all_devices()

    if not devices:
        raise ConnectionError(f"No samna device found plugged in to the system!")

    # Search the dynapse boards with the right name
    for d in devices:

        if name.upper() in d.device_type_name.upper():
            dynapse_list.append(d)

    logging.info(
        f" Total of {len(dynapse_list)} {name} board(s) found with serial numbers : {[d.serial_number for d in devices]}"
    )

    return dynapse_list


def configure_fpga(
    device: DeviceInfo, bitfile: Optional[str] = None
) -> Dynapse2Interface:
    """
    configure_fpga configures the FPGA on board and builds a connection node between CPU and the device.
    It allows one to configure the device, read or write AER events to bus, and monitor the activity of device neurons

    :param device: the device object to open and configure
    :type device: DeviceInfo
    :param bitfile: the bitfile path if known, defaults to None
    :type bitfile: Optional[str], optional
    :raises IOError: Failed to configure Opal Kelly
    :return: an open and configured Dynan-SE2 interface node
    :rtype: Dynapse2Interface
    """

    device = samna.device.open_device(device)

    if bitfile is None:
        bitfile = __bit_path_Dynapse2Stack()

    if not device.configure_opal_kelly(bitfile):
        raise IOError("Failed to configure Opal Kelly")

    logging.info(
        f"{device.get_device_type_name()} is connected, configured and ready for operation!"
    )

    return device


def get_model(board: Dynapse2Interface, reset: bool = True) -> Dynapse2Model:
    """
    get_model obtain a `Dynapse2Model` from an already opened dynapse2interface node

    :param board: the Dynan-SE2 interface node. (Like a file) It should be opened beforehand.
    :type board: Dynapse2Interface
    :param reset: reset the model or not, defaults to True
    :type reset: bool, optional
    :return: a dynapse2 model object that can be used to configure the device
    :rtype: Dynapse2Model
    """

    model: Dynapse2Model = board.get_model()

    if reset:
        model.reset(se2.ResetType.PowerCycle, 0b1)
        model.clear_error_queue()

    return model


def disconnect(board: Dynapse2Interface) -> None:
    """
    disconnect breaks the connection between CPU and the device

    :param board: an opened Dynan-SE2 interface node
    :type board: Dynapse2Interface
    """
    logging.warn(f"{board.get_device_type_name()} disconnected!")
    return samna.device.close_device(board)


def dispatch_timeseries_to_device(
    board: Dynapse2Interface,
    timeseries: TSEvent,
    channel_map: Optional[Dict[int, Dynapse2Destination]],
    offset_fpga: bool = True,
    offset: float = 0.0,
    dt: float = 1e-6,
) -> TSEvent:
    """
    dispatch_timeseries_to_device creates AER packages using a timeseries object as a reference and sends them to device

    :param board: the Dynan-SE2 interface node. (Like a file) It should be opened beforehand.
    :type board: Dynapse2Interface
    :param timeseries: the timeseries to be converted into list of Dynap-SE2 AER packages
    :type timeseries: TSEvent
    :param channel_map: the mapping between timeseries channels and the destinations
    :type channel_map: Optional[Dict[int, Dynapse2Destination]]
    :param offset_fpga: offset the timeseries depending on the current FPGA clock, defaults to True
    :type offset_fpga: bool, optional
    :param offset: user defined offset in seconds, defaults to True
    :type offset: float, optional
    :param dt: the FPGA timestep resolution, defaults to 1e-6
    :type dt: float, optional
    :return: the shifted (or not shifted) input timeseries
    :rtype: TSEvent
    """

    # Get the current FPGA time at the moment and offset the events accordingly
    if offset_fpga:
        offset += fpga_time(board)

    timeseries = timeseries.start_at(timeseries.t_start + offset)

    # Convert the TSEvent object to a list of AER packages
    input_events = tsevent_to_aer(
        timeseries=timeseries, channel_map=channel_map, return_samna=True, dt=dt
    )

    # Write the events to the bus
    board.grid_bus_write_events(input_events)

    return timeseries


def capture_timeseries_from_device(
    board: Dynapse2Interface,
    duration: float,
    poll_step: float = 10e-3,
    dt: float = 1e-6,
    *args,
    **kwargs,
) -> Tuple[TSEvent, Dict[int, Dynapse2Destination]]:
    """
    capture_timeseries_from_device records the device's output and converts this to a timeseries object

    :param board: the Dynan-SE2 interface node. (Like a file) It should be opened beforehand.
    :type board: Dynapse2Interface
    :param duration: the duration of capturing
    :type duration: float
    :param poll_step: the pollling step, 10 ms means the CPU fetches events from FPGA in every 10 ms, defaults to 10e-3
    :type poll_step: float, optional
    :param dt: the FPGA timestep resolution, defaults to 1e-6
    :type dt: float, optional
    :return: ts, cmap
        ts: the timeseries object that is converted referenced on the event buffer
        cmap: the mapping between timeseries channels and the destinations
    :rtype: Tuple[TSEvent, Dict[int, Dynapse2Destination]]
    """

    buffer = capture_events_from_device(
        board=board, duration=duration, poll_step=poll_step
    )
    return aer_to_tsevent(buffer=buffer, dt=dt, *args, **kwargs)


def capture_events_from_device(
    board: Dynapse2Interface, duration: float, poll_step: float = 10e-3
) -> List[NormalGridEvent]:
    """
    capture_events_from_device records the device's output and stores in an event buffer

    :param board: the Dynan-SE2 interface node. (Like a file) It should be opened beforehand.
    :type board: Dynapse2Interface
    :param duration: the duration of capturing
    :type duration: float
    :param poll_step: the pollling step, 10 ms means the CPU fetches events from FPGA in every 10 ms, defaults to 10e-3
    :type poll_step: float, optional
    :return: the event buffer, a list of Dynap-SE2 AER packages captured
    :rtype: List[NormalGridEvent]
    """
    record = []

    # Initial time
    tic = toc = time.time()

    # Polling
    while toc - tic < duration:
        buffer = board.read_events()
        if len(buffer) > 0:
            record += [NormalGridEvent.from_samna(data) for data in buffer]

        time.sleep(poll_step)
        toc = time.time()

    return record


def tsevent_to_aer(
    timeseries: TSEvent,
    channel_map: Optional[Dict[int, Dynapse2Destination]],
    return_samna: bool = True,
    dt: float = 1e-6,
) -> List[NormalGridEvent]:
    """
    tsevent_to_aer converts a TSEvent timeseries object to a list of AER packages.
    It uses a channel map to map the channels to destinations, and by default it returns a list of samna objects.

    :param timeseries: the timeseries to be converted into list of Dynap-SE2 AER packages
    :type timeseries: TSEvent
    :param channel_map: the mapping between timeseries channels and the destinations
    :type channel_map: Optional[Dict[int, Dynapse2Destination]]
    :param return_samna: return actual samna objects or not(aliases), defaults to True
    :type return_samna: bool, optional
    :param dt: the FPGA timestep resolution, defaults to 1e-6
    :type dt: float, optional
    :raises ValueError: Channel map does not map the channels of the timeseries provided!
    :return: a list of Dynap-SE2 AER packages
    :rtype: List[NormalGridEvent]
    """
    buffer = []
    __channels = set(timeseries.channels)

    # Create the default channel map is not provided. NOT RECOMMENDED!
    if channel_map is None:
        channel_map = {
            c: Dynapse2Destination(core=[True] * 4, x_hop=0, y_hop=0, tag=c)
            for c in __channels
        }

    if not __channels <= set(channel_map.keys()):
        raise ValueError(
            "Channel map does not map the channels of the timeseries provided!"
        )

    # Create the AER list
    for t, c in timeseries:
        timestamp = int(np.around((t / dt)))
        destination = channel_map[c]
        event = NormalGridEvent(destination, timestamp)
        if return_samna:
            event = event.samna_object(se2.NormalGridEvent)
        buffer.append(event)

    return buffer


def aer_to_tsevent(
    buffer: List[NormalGridEvent], dt: float = 1e-6, *args, **kwargs
) -> Tuple[TSEvent, Dict[int, Dynapse2Destination]]:
    """
    aer_to_tsevent converts a list of Dynap-SE2 AER packages to a `TSEvent` timeseries object

    :param buffer: the event buffer, a list of Dynap-SE2 AER packages
    :type buffer: List[NormalGridEvent]
    :param dt: the FPGA timestep resolution, defaults to 1e-6
    :type dt: float, optional
    :return: ts, cmap
        ts: the timeseries object that is converted referenced on the event buffer
        cmap: the mapping between timeseries channels and the destinations
    :rtype: Tuple[TSEvent, Dict[int, Dynapse2Destination]]
    """

    # Get a reverse channel map
    cmap = extract_channel_map(buffer)
    rcmap = {v: k for k, v in cmap.items()}

    # Create the event/channel lists
    times = []
    channels = []
    for event in buffer:
        times.append(event.timestamp * dt)
        channels.append(rcmap[event.event])

    # Construct a timeseries object
    ts = TSEvent(
        times=times,
        channels=channels,
        t_start=(times[0] - dt),
        t_stop=(times[-1] + dt),
        *args,
        **kwargs,
    )

    return ts, cmap


def extract_channel_map(
    buffer: List[NormalGridEvent],
) -> Dict[int, Dynapse2Destination]:
    """
    extract_channel_map obtains a channel map from a list of dummy AER packages (samna alias)

    :param buffer: the list of AER packages
    :type buffer: List[NormalGridEvent]
    :return: the mapping between timeseries channels and the destinations
    :rtype: Dict[int, Dynapse2Destination]
    """
    destinations = []

    for data in buffer:
        if data.event not in destinations:
            destinations.append(data.event)

    channel_map = dict(zip(range(len(destinations)), destinations))

    return channel_map


def fpga_time(
    board: Dynapse2Interface,
    reset: bool = False,
    timeout: float = 10e-3,
    retry: int = 100,
    dt: float = 1e-6,
) -> float:
    """
    fpga_time bounces a dummy event from FPGA to get the exact FPGA time at that moment.

    :param board: the Dynan-SE2 interface node. (Like a file) It should be opened beforehand.
    :type board: Dynapse2Interface
    :param reset: reset the FPGA timeline or not, defaults to False
    :type reset: bool, optional
    :param timeout: the time to wait for the event to bounce back, defaults to 100e-3
    :type timeout: float, optional
    :param retry: number of retrials in the case that event is not returned back, defaults to 100
    :type retry: int, optional
    :param dt: the FPGA time step, defaults to 1e-6
    :type dt: float, optional
    :raises TimeoutError: FPGA could not respond!
    :return: the current FPGA time in seconds (roughly)
    :rtype: float
    """

    # Flush the buffers
    board.output_read()

    # Reset FPGA
    if reset:
        reset = board.reset_fpga()
        while not reset:
            reset = board.reset_fpga()
            time.sleep(timeout)

    # Generate a dummy event
    event = NormalGridEvent(
        event=Dynapse2Destination(
            core=[True, True, True, True], x_hop=-1, y_hop=-1, tag=2047
        ),
        timestamp=int(timeout / dt),
    ).samna_object(se2.NormalGridEvent)

    # Send 3 events to the device
    board.input_interface_write_events(0, [event] * 3)
    time.sleep(timeout)

    # Try to catch them and read the timestamp
    for __break in range(retry):
        evs = board.output_read()
        if len(evs) > 0:
            return evs[-1] * dt
        else:
            time.sleep(timeout)
            timeout *= 2

    raise TimeoutError(f"FPGA could not respond, increase number of trials or timeout!")
