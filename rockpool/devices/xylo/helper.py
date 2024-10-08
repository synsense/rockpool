"""
Helper function used to check board version and import matching packages.
"""

from typing import Tuple, List
from types import ModuleType

from pkg_resources import parse_version

import samna

__all__ = ["find_xylo_hdks", "check_firmware_versions"]


def find_xylo_hdks() -> Tuple[List["XyloHDK"], List[ModuleType], List[str]]:
    """
    Enumerate connected Xylo HDKs, and import the corresponding support module

    Returns:
        (Tuple[List["XyloHDK"], List[ModuleType], List[str]]): (hdks, modules, versions)
        hdks (List): A (possibly empty) list of HDK devices
        modules (List): A (possibly empty) list of python modules providing support for the corresponding Xylo HDK
        versions (List): A (possibly empty) list containing the version string for each detected HDK
    """
    # - Get a list of devices
    device_list = samna.device.get_all_devices()

    xylo_hdks = []
    xylo_support_modules = []
    xylo_versions = []

    for d in device_list:
        if d.device_type_name == "XyloA2TestBoard":
            dev = samna.device.open_device(d)

            if not check_firmware_versions(dev, "0.11.5", "1.1.3"):
                raise ValueError(
                    "The firmware of the connected Xylo HDK is unsupported, and must be upgraded."
                )

            print(
                "The connected Xylo HDK contains a XyloAudio v2 (SYNS61201). Importing `rockpool.devices.xylo.syns61201`"
            )
            import rockpool.devices.xylo.syns61201 as x2

            xylo_hdks.append(dev)
            xylo_support_modules.append(x2)
            xylo_versions.append("syns61201")

        elif (
            d.device_type_name == "XyloDevKit" or d.device_type_name == "XyloTestBoard"
        ):
            print(
                "The connected Xylo HDK contains a Xylo SNN core (SYNS61300). Importing `rockpool.devices.xylo.syns61300`"
            )

            import rockpool.devices.xylo.syns61300 as x1

            xylo_hdks.append(samna.device.open_device(d))
            xylo_support_modules.append(x1)
            xylo_versions.append("syns61300")

        elif d.device_type_name == "XyloImuTestBoard":
            dev = samna.device.open_device(d)

            if not check_firmware_versions(dev, "0.11.5", "1.1.3"):
                raise ValueError(
                    "The firmware of the connected Xylo HDK is unsupported, and must be upgraded."
                )

            print(
                "The connected Xylo HDK contains a XyloIMU. Importing `rockpool.devices.xylo.syns63300`"
            )
            import rockpool.devices.xylo.syns63300 as imu

            xylo_hdks.append(dev)
            xylo_support_modules.append(imu)
            xylo_versions.append("syns63300")

        elif d.device_type_name == "XyloAudio3TestBoard":
            dev = samna.device.open_device(d)

            if not check_firmware_versions(dev, "0.0.0", "1.4.0"):
                raise ValueError(
                    "The firmware of the connected Xylo HDK is unsupported, and must be upgraded."
                )

            print(
                "The connected Xylo HDK contains a XyloAudio 3. Importing `rockpool.devices.xylo.syns65302`"
            )
            import rockpool.devices.xylo.syns65302 as a3

            xylo_hdks.append(dev)
            xylo_support_modules.append(a3)
            xylo_versions.append("syns65302")

    return xylo_hdks, xylo_support_modules, xylo_versions


def check_firmware_versions(
    dev: "XyloHDK", min_fxtree_ver: str, min_unifirm_ver: str
) -> bool:
    """
    Verify the firmware versions on an HDK meet a minimum standard

    Arguments:
        dev (XyloHDK): A connected HDK to check
        min_fxtree_ver (str): The minimum version string for the FX3 chip firmware
        min_unifirm_ver (str): The minimum version string for the FPGA Unifirm version
    """
    # - Read device firmware versions
    vers = dev.get_firmware_versions()

    if parse_version(vers.fxtree) < parse_version(min_fxtree_ver):
        return False

    if parse_version(vers.unifirm) < parse_version(min_unifirm_ver):
        return False

    return True
