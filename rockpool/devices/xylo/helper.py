"""
Helper function used to check board version and import matching packages.
"""

import samna


def check_version():
    """
    Helper function used to check board version and import matching libraries.

    Returns:
        List[AFEHDK]: A (possibly empty) list of AFE HDK nodes.
        module: suitable package.
    """
    # - Get a list of devices
    device_list = samna.device.get_all_devices()

    if not len(device_list):
        raise ValueError("Find No board!")

    else:
        for d in device_list:
            if d.device_type_name == "XyloA2TestBoard":
                version = 2
                print(
                    "The Xylo board version is 2. Importing rockpool.devices.xylo.syns61201 as x "
                )
                import rockpool.devices.xylo.syns61201 as x

                device = [samna.device.open_device(d)]
                return device, x
            elif (
                d.device_type_name == "XyloDevKit"
                or d.device_type_name == "XyloTestBoard"
            ):
                version = 1
                import rockpool.devices.xylo.syns61300 as x

                device = [samna.device.open_device(d)]
                return device, x
                print(
                    "The Xylo board version is 1. Importing rockpool.devices.xylo.syns61300 as x "
                )
            else:
                raise ValueError("No board version matches!")
