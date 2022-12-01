"""
Dynap-SE2 samna connection utilities

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

31/05/2022
"""
from typing import Any, List
import logging
from rockpool.devices.dynapse.samna_alias import DeviceInfo


# Try to import samna for device interfacing
try:
    import samna
except:
    samna = Any
    logging.warning(
        "Device interface requires `samna` package which is not installed on the system"
    )

__all__ = ["find_dynapse_boards"]


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
