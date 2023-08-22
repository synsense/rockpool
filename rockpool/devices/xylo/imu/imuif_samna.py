from typing import Optional, Tuple

from rockpool.devices.xylo.imu.xylo_imu_devkit_utils import XyloIMUHDK
from rockpool.devices.xylo.imu.preprocessing.identity_net import IdentityNet
from .imu_monitor import XyloIMUMonitor
from rockpool.nn.modules import Module
from rockpool.utilities.backend_management import backend_available

if backend_available("samna"):
    from samna.xyloImu.configuration import InputInterfaceConfig
else:
    InputInterfaceConfig = Any

from rockpool.typehints import FloatVector

__all__ = ["IMUIFSamna"]


N_CHANNEL = 15
"""Fixed number of channels employed"""

CLOCK_RATE = 200
"""Fixed computation step rate of 200Hz for Xylo IMU"""


class IMUIFSamna(Module):
    def __init__(
        self,
        device: Optional[XyloIMUHDK] = None,
        interface_params: dict = {},
        prerecorded_imu_input: bool = True,
    ) -> None:
        """
        Implements the interface between the Xylo IMU IF module and the `rockpool` framework.
        It's a workaround to read the IMU IF output data

        NOTE : The output is slightly different from the actual output of the Xylo IMU IF module in the case that there are consecutive spikes in the output of the Xylo IMU IF module.

        Args:
            device (Optional[XyloIMUHDK], optional):  device (XyloIMUHDK): An opened `samna` device to a Xylo dev kit. Defaults to None.
            interface_params (dict, optional): IMUIF params. Defaults to {}.
            prerecorded_imu_input (bool, optional): If ``True``, use prerocorded imu data from PC as input. If ``False``, use the live IMU sensor on the HDK. Defaults to True.
        """
        # - Check input arguments
        if device is None:
            raise ValueError("`device` must be a valid, opened `XyloIMUHDK` device.")

        super().__init__(shape=(0, N_CHANNEL), spiking_input=False, spiking_output=True)

        self.dummy_net = IdentityNet(
            device=None, n_channel=N_CHANNEL, clock_rate=CLOCK_RATE
        )

        self.monitor = XyloIMUMonitor(
            device=device,
            config=self.dummy_net.config,
            interface_params=interface_params,
            prerecorded_imu_input=prerecorded_imu_input,
        )

    def evolve(
        self,
        input_data: FloatVector,
        record: bool = False,
        read_timeout: Optional[float] = None,
    ) -> Tuple[FloatVector, dict, dict]:
        """
        Wrap the `evolve` method of the `XyloIMUMonitor` class
        Evolve a network on the Xylo HDK in Real-time mode

        Args:
            input_data (FloatVector): Pre-recorded IMU data, (Tx3)
            record (bool): ``False``, do not return a recording dictionary. Recording internal state is not supported by :py:class:`.XyloIMUMonitor`
            read_timeout (float): A duration in seconds for a read timeout. Default: 2x the real-time duration of the evolution

        Returns:
            Tuple[FloatVector, dict, dict]:
                * output_data (FloatVector): The output spike train
                * state (dict): The state dictionary from the network
                * rec (dict): The record dictionary from the network
        """
        return self.monitor.evolve(
            input_data=input_data, record=record, read_timeout=read_timeout
        )
