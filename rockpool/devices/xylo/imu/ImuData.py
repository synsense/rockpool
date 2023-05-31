from typing import Any, Optional, Tuple, Union
import samna
import time
import math
import numpy as np


from rockpool.nn.modules.module import Module
from . import xylo_imu_devkit_utils as hdkutils
from .xylo_imu_devkit_utils import XyloIMUHDK


__all__ = ["XyloIMUData"]


class XyloIMUData(Module):
    """
    Interface to the IMU sensor on a Xylo-Imu HDK

    This module uses ``samna`` to interface to the IMU hardware on a Xylo-IMU HDK. It permits recording from the IMU sensor.

    To record from the module, use the :py:meth:`~.XyloImuData.evolve` method. You need to pass this method an empty matrix, with the desired number of time-steps. The time-step ``dt`` is specified at module instantiation.
    """

    def __init__(
        self,
        device: XyloIMUHDK,
        frequenccy: float = 200.0,
        timesteps: int = 1000,
        # dt: float = 0.005,
        *args,
        **kwargs,
    ):
        """
        Instantiate a XyloIMUData Module, via a samna backend

        Args:
            device (XyloIMUHDK): A connected XyloIMUHDK device.
            shape (tuple): The data shape (axis, timesteps) to record live IMU data
            frequency (float): The frequency to read data from IMU sensor. Default: 200.0
            timesteps (int): The specify number of timesteps specified to be recorded. Default: 1000
            dt (float): The simulation time-step to use for this Module
        """

        # Check device validation
        if device is None:
            raise ValueError("`device` must be a valid, opened Xylo IMU HDK device.")

        # - Store the device
        self._device: XyloIMUHDK = device
        """ `.XyloHDK`: The Xylo HDK used by this module """

        # - Register buffers to read and write events
        self._read_buffer, self._write_buffer, mc = hdkutils.Initialise_imu_sensor(
            device
        )

        # Store the IMU sensor
        self._mc = mc

        # Store the dt
        self.dt = 1 / frequenccy

        # Config the IMU sensor to ready for data reading
        self.config_imu_sensor(mc, 5000000)

    def config_imu_sensor(self, mcdevice, time_interval=5000000):
        mcdevice.setup()
        mcdevice.set_auto_read_period(5000000)
        mcdevice.auto_read_enable(True)

    def evolve(
        self,
        input_data,
        timeout: Optional[float] = None,
        record: bool = False,
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        Use the IMU sensor to record live IMU data and return

        Args:
            input_data (np.ndarray): An array ``[T, 3]``, specifying the number of time-steps to record.

        Returns:
            (np.ndarray, dict, dict) output_events, {}, {}
        """
        Nt, Nc = input_data.shape

        if Nc != 3:
            raise ValueError(
                f"The specified data should have 3 channels! Recording data with shape [{Nt, Nc}]."
            )

        # - Discard the batch dimension
        input_data = input_data[0]

        out = []
        count = 0

        # - Determine a read timeout
        timeout = 2 * Nt * self.dt if timeout is None else timeout

        t_start = time.time()
        t_timeout = t_start + timeout

        while count < int(Nt):
            evts = self._read_buffer.get_events()
            for e in evts:
                if isinstance(e, samna.events.Acceleration):
                    count += 1
                    x = e.x * 4 / math.pow(2, 14)
                    y = e.y * 4 / math.pow(2, 14)
                    z = e.z * 4 / math.pow(2, 14)
                output = [x, y, z]
                out.append(output)

                # - Check for read timeout
                if time.time() > t_timeout:
                    raise TimeoutError(f"IMUSensor: Read timeout of {timeout} sec.")

        out = np.array(output)

        return out, {}, {}


if __name__ == "__main__":
    dk = samna.device.open_device("XyloImuTestBoard:0")
    imu_record = XyloIMUData()
    data, _, _ = imu_record(np.zeros([1000, 3]))
    print(data)
