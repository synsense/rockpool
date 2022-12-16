"""
Helper function used to check board version and import matching packages.
"""

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
            List[AFEHDK]: A (possibly empty) list of AFE HDK nodes
            List[module]: A (possibly empty) list of python modules providing support for the corresponding Xylo HDK
        """
        # - Get a list of devices
        device_list = samna.device.get_all_devices()

        xylo_hdks = []
        xylo_support_modules = []
        xylo_versions = []

        for d in device_list:
            if d.device_type_name == "XyloA2TestBoard":
                print(
                    "The connected Xylo HDK contains a Xylo Audio v2 (SYNS61201). Importing `rockpool.devices.xylo.syns61201`"
                )
                import rockpool.devices.xylo.syns61201 as x2

                xylo_hdks.append(samna.device.open_device(d))
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

        return xylo_hdks, xylo_support_modules, xylo_versions

else:
    find_xylo_hdks = missing_backend_shim("find_xylo_hdks", "samna")
