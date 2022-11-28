"""
Dynap-SE2 samna connection utilities

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

    name change: utils/se2 -> interface/utils @ 220926

[] TODO : find_dynapse_boards is similar to xylo, can be merged
[] TODO : Timeout with multiprocessing/signalling
[] TODO : Timeout error for configure_dynapse2_fpga
31/05/2022
"""
from typing import Any, Dict, List, Optional, Tuple

import time
import logging
import numpy as np

from rockpool.devices.dynapse.lookup import SE2_STACK_FPGA_FILEPATH
from rockpool.devices.dynapse.samna_alias import (
    Dynapse2Destination,
    NormalGridEvent,
    Dynapse2Interface,
    DeviceInfo,
)


# Try to import samna for device interfacing
try:
    import samna
except:
    samna = Any
    logging.warning(
        "Device interface requires `samna` package which is not installed on the system"
    )

__all__ = [
    "find_dynapse_boards",
    "configure_dynapse2_fpga",
    "disconnect",
    "capture_events_from_device",
    "aer_to_raster",
    "extract_channel_map",
    "raster_to_aer",
    "event_generator",
]


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


def configure_dynapse2_fpga(
    device: DeviceInfo,
    bitfile: Optional[str] = None,
    timeout: float = 5.0,
) -> Dynapse2Interface:
    """
    configure_dynapse2_fpga configures the FPGA on board and builds a connection node between CPU and the device.
    It allows one to configure the device, read or write AER events to bus, and monitor the activity of device neurons

    :param device: the device object to open and configure
    :type device: DeviceInfo
    :param bitfile: the bitfile path if known, defaults to None
    :type bitfile: Optional[str], optional
    :param timeout: the maximum time limit in seconds  that the device should respons, defaults to 5.0
    :type timeout: float, optional
    :raises IOError: Failed to configure Opal Kelly
    :return: an open and configured Dynan-SE2 interface node
    :rtype: Dynapse2Interface
    """

    device = samna.device.open_device(device)

    if bitfile is None:
        bitfile = SE2_STACK_FPGA_FILEPATH

    if not device.configure_opal_kelly(bitfile):
        raise IOError("Failed to configure Opal Kelly")

    logging.info(
        f"{device.get_device_type_name()} is connected, configured and ready for operation!"
    )

    return device


def disconnect(board: Dynapse2Interface) -> None:
    """
    disconnect breaks the connection between CPU and the device

    :param board: an opened Dynan-SE2 interface node
    :type board: Dynapse2Interface
    """
    logging.warn(f"{board.get_device_type_name()} disconnected!")
    return samna.device.close_device(board)


def capture_events_from_device(
    board: Dynapse2Interface,
    duration: float,
    control_tag: int,
    poll_step: float = 10e-3,
) -> List[NormalGridEvent]:
    """
    capture_events_from_device records the device's output and stores in an event buffer

    :param board: the Dynan-SE2 interface node. (Like a file) It should be opened beforehand.
    :type board: Dynapse2Interface
    :param duration: the minimum duration of capturing
    :type duration: float
    :param control_tag: a tag used in special occacions such as current time reading. Do not capture events with this tag, defaults to 2047
    :type control_tag: int, optional
    :param poll_step: the pollling step, 10 ms means the CPU fetches events from FPGA in every 10 ms, defaults to 10e-3
    :type poll_step: float, optional
    :return: the event buffer, a list of Dynap-SE2 AER packages captured
    :rtype: List[NormalGridEvent]
    """

    record = []

    # Initial time
    tic = toc = time.time()

    # Fixed duration Polling
    while toc - tic < duration:

        buffer = board.read_events()
        if len(buffer) > 0:
            record += [
                NormalGridEvent.from_samna(data)
                for data in buffer
                if data.event.tag != control_tag
            ]
        time.sleep(poll_step)
        toc = time.time()

    return record


def aer_to_raster(
    buffer: List[NormalGridEvent],
    stop_time: float,
    start_time: float = 0,
    dt: float = 1e-3,
    dt_fpga: float = 1e-6,
) -> Tuple[np.ndarray, Dict[int, Dynapse2Destination]]:
    """
    aer_to_raster converts a list of Dynap-SE2 AER packages to a discrete raster record
    The events does not meet the start and stop time criteria are descarded

    :param buffer: the event buffer, a list of Dynap-SE2 AER packages
    :type buffer: List[NormalGridEvent]
    :param stop_time: the stop time cut-off for the events.
    :type stop_time: float
    :param start_time: the start time cut-off for the events, defaults to 0
    :type start_time: float, optional
    :param dt: the raster's timestep resolution, defaults to 1e-3
    :type dt: float, optional
    :param dt_fpga: the FPGA timestep resolution, defaults to 1e-6
    :type dt_fpga: float, optional
    :return: ts, cmap
        raster_out: the raster record referenced on the event buffer
        cmap: the mapping between raster channels and the destinations
    :rtype: Tuple[np.ndarray, Dict[int, Dynapse2Destination]]
    """

    # Get a reverse channel map
    cmap = extract_channel_map(buffer)
    rcmap = {v: k for k, v in cmap.items()}

    # Create the event/channel lists
    times = []
    channels = []
    for event in buffer:
        times.append(event.timestamp * dt_fpga)
        channels.append(rcmap[event.event])

    # sort time and channel arrays in the same order
    idx = np.argsort(times)
    times = np.array(times)[idx]
    channels = np.array(channels)[idx]

    # generate the output raster
    time_course = np.arange(start_time, stop_time, dt)
    raster_out = np.zeros((len(time_course), len(cmap)))

    # Save the data meeting the start and stop time criteria and discard the rest
    for i, t in enumerate(times):
        idx = np.searchsorted(time_course, t)
        if idx < len(raster_out):
            raster_out[idx][channels[i]] += 1
    return raster_out, cmap


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


def raster_to_aer(
    raster: np.ndarray,
    start_time: float = 0.0,
    channel_map: Optional[Dict[int, Dynapse2Destination]] = None,
    return_samna: bool = True,
    dt: float = 1e-3,
    dt_fpga: float = 1e-6,
) -> List[NormalGridEvent]:
    """
    raster_to_aer converts a discrete raster record to a list of AER packages.
    It uses a channel map to map the channels to destinations, and by default it returns a list of samna objects.

    :param raster: the discrete timeseries to be converted into list of Dynap-SE2 AER packages
    :type raster: np.ndarray
    :param start_time: the start time of the record in seconds, defaults to 0.0
    :type start_time: float
    :param channel_map: the mapping between timeseries channels and the destinations
    :type channel_map: Optional[Dict[int, Dynapse2Destination]]
    :param return_samna: return actual samna objects or not(aliases), defaults to True
    :type return_samna: bool, optional
    :param dt: the raster's timestep resolution, defaults to 1e-3
    :type dt: float, optional
    :param dt_fpga: the FPGA timestep resolution, defaults to 1e-6
    :type dt_fpga: float, optional
    :raises ValueError: Raster should be 2 dimensional!
    :raises ValueError: Channel map does not map the channels of the timeseries provided!
    :return: a list of Dynap-SE2 AER packages
    :rtype: List[NormalGridEvent]
    """

    if len(raster.shape) != 2:
        raise ValueError("Raster should be 2 dimensional!")

    buffer = []
    duration = raster.shape[0] * dt
    num_channels = raster.shape[1]
    __time_course = np.arange(start_time, start_time + duration, dt)

    # Default channel map is NOT RECOMMENDED!
    if channel_map is None:
        channel_map = {
            c: Dynapse2Destination(core=[True] * 4, x_hop=-1, y_hop=-1, tag=c)
            for c in range(num_channels)
        }

    if not num_channels <= len(set(channel_map.keys())):
        raise ValueError(
            "Channel map does not map the channels of the timeseries provided!"
        )

    # Create the AER list
    for spikes, time in zip(raster, __time_course):

        destinations = np.argwhere(spikes).flatten()
        timestamp = int(np.around((time / dt_fpga)))
        events = [
            NormalGridEvent(channel_map[dest], timestamp + i)
            for i, dest in enumerate(destinations)
        ]
        if return_samna:
            events = [
                event.samna_object(samna.dynapse2.NormalGridEvent) for event in events
            ]
        buffer.extend(events)

    return buffer


def event_generator(
    event_time: float,
    core: List[bool] = [True, True, True, True],
    x_hop: int = -1,
    y_hop: int = -1,
    tag: np.uint = 2047,
    dt_fpga: float = 1e-6,
) -> NormalGridEvent:
    """
    event_generator a Dynap-SE2 event generator utility function, can be used to generate dummy events

    :param event_time: the time that the event happened in seconds
    :type event_time: float
    :param core: the core mask used while sending the events, defaults to [True, True, True, True]
            [1,1,1,1] means all 4 cores are on the target
            [0,0,1,0] means the event will arrive at core 2 only
    :type core: List[bool], optional
    :param x_hop: number of chip hops on x axis, defaults -1
    :type x_hop: int, optional
    :param y_hop: number of chip hops on y axis, defaults to -1
    :type y_hop: int, optional
    :param tag: globally multiplexed locally unique event tag which is used to identify the connection between two neurons, defaults to 2047
    :type tag: np.uint, optional
    :param dt_fpga: the FPGA timestep resolution, defaults to 1e-6
    :type dt_fpga: float, optional
    :return: a virtual samna AER package for DynapSE2
    :rtype: NormalGridEvent
    """

    event = NormalGridEvent(
        event=Dynapse2Destination(core, x_hop, y_hop, tag),
        timestamp=int(event_time / dt_fpga),
    ).samna_object(samna.dynapse2.NormalGridEvent)
    # .to_samna()

    return event
