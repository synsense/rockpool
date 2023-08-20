"""
Helper function used to check board version and import matching packages.
"""

from pkg_resources import parse_version

from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)

if backend_available("samna"):
    import samna

    def find_xylo_hdks():
        """
        Enumerate connected Xylo HDKs, and import the corresponding support module

        Returns:
            List[AFEHDK]: A (possibly empty) list of HDK devices
            List[module]: A (possibly empty) list of python modules providing support for the corresponding Xylo HDK
            List[str]: A (possibly empty) list containing the version string for each detected HDK
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
                    "The connected Xylo HDK contains a Xylo Audio v2 (SYNS61201). Importing `rockpool.devices.xylo.syns61201`"
                )
                import rockpool.devices.xylo.syns61201 as x2

                xylo_hdks.append(dev)
                xylo_support_modules.append(x2)
                xylo_versions.append("syns61201")

            elif (
                d.device_type_name == "XyloDevKit"
                or d.device_type_name == "XyloTestBoard"
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
                    "The connected Xylo HDK contains a Xylo IMU. Importing `rockpool.devices.xylo.imu`"
                )
                import rockpool.devices.xylo.imu as imu

                xylo_hdks.append(dev)
                xylo_support_modules.append(imu)
                xylo_versions.append("xylo-imu")

        return xylo_hdks, xylo_support_modules, xylo_versions

else:
    find_xylo_hdks = missing_backend_shim("find_xylo_hdks", "samna")


def check_firmware_versions(dev, min_fxtree_ver, min_unifirm_ver) -> bool:
    # - Read device firmware versions
    vers = dev.get_firmware_versions()

    if parse_version(vers.fxtree) < parse_version(min_fxtree_ver):
        return False

    if parse_version(vers.unifirm) < parse_version(min_unifirm_ver):
        return False

    return True
