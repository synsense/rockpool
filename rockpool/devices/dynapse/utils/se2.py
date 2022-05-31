"""
Dynap-SE2 samna connection utilities

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
31/05/2022
"""
from typing import Any

import time
import logging

from rockpool.devices.dynapse.samna_alias.dynapse2 import (
    Dynapse2Destination,
    NormalGridEvent,
    Dynapse2Interface,
    Dynapse2Model,
)

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


def fpga_time(
    board: Dynapse2Interface,
    reset: bool = False,
    timeout: float = 10e-3,
    retry: int = 100,
    fpga_resolution: float = 1e-6,
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
    :param fpga_resolution: the FPGA time step, defaults to 1e-6
    :type fpga_resolution: float, optional
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
        timestamp=int(timeout / fpga_resolution),
    ).samna_object(se2.NormalGridEvent)

    # Send 3 events to the device
    board.input_interface_write_events(0, [event] * 3)
    time.sleep(timeout)

    # Try to catch them and read the timestamp
    for __break in range(retry):
        evs = board.output_read()
        if len(evs) > 0:
            return evs[-1] * fpga_resolution
        else:
            time.sleep(timeout)
            timeout *= 2

    raise TimeoutError(f"FPGA could not respond, increase number of trials or timeout!")


def connect(bitfile: str) -> Dynapse2Interface:
    """
    connect build a connection node between CPU and the device. It allows one to configure the device,
    read or write AER events to bus, and monitor the activity of device neurons

    :param bitfile: the FPGA configuration bitfile
    :type bitfile: str
    :raises ConnectionError: No device connected to the system!
    :raises IOError: Failed to configure Opal Kelly
    :return: an opened Dynan-SE2 interface node
    :rtype: Dynapse2Interface
    """

    devices = samna.device.get_unopened_devices()
    if not devices:
        raise ConnectionError("No device connected to the system!")

    board: Dynapse2Interface = samna.device.open_device(devices[0])

    if not board.configure_opal_kelly(bitfile):
        raise IOError("Failed to configure Opal Kelly")

    logging.info(
        f"{devices[0].device_type_name} with serial number:{devices[0].serial_number} is ready!"
    )

    return board


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
    logging.info(f"{board.get_device_type_name()} disconnected!")
    return samna.device.close_device(board)
