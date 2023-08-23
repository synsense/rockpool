from typing import Optional, Tuple

from rockpool.devices.xylo.imu.imu_monitor import XyloIMUMonitor
from rockpool.devices.xylo.imu.xylo_imu_devkit_utils import XyloIMUHDK
from rockpool.devices.xylo.imu.xylo_sim import XyloSim
from rockpool.nn.modules import Module
from rockpool.parameters import SimulationParameter
from rockpool.typehints import FloatVector
from rockpool.utilities.backend_management import backend_available

if backend_available("samna"):
    from samna.xyloImu.configuration import InputInterfaceConfig, XyloConfiguration
else:
    InputInterfaceConfig = Any
    XyloConfiguration = Any

import numpy as np

from rockpool.devices.xylo.imu.xylo_samna import (
    XyloIMUHDK,
    XyloSamna,
    config_from_specification,
)

__all__ = ["IMUIFSamna", "IdentityNet"]


N_CHANNEL = 15
"""Fixed number of channels employed"""

CLOCK_RATE = 200
"""Fixed computation step rate of 200Hz for Xylo IMU"""


class IdentityNet(Module):
    """
    A simple identity network, to be used as a dummy network which aimed to return the input as the output.
    Since there is no other option to record the output of IMUIF directly, `IdentityNet` will serve as a buffer on SNN core, making it possible to implement a `IMUIFSamna` module

    NOTE : THIS IS A HACKY SOLUTION
        * In the case the hidden state reading gets buggy, it'll fail on hardware
    """

    def __init__(
        self,
        device: Optional[XyloIMUHDK] = None,
        n_channel: int = 15,
        clock_rate: int = 200,
    ) -> None:
        """
        Object constructor

        Args:
            device (Optional[XyloIMUHDK], optional): The device to use. If None, use XyloSim. Defaults to None.
            n_channel (int, optional): Number of channels employed. Defaults to 15.
            clock_rate (int, optional): The nominal clock rate of the input unit. Defaults to 200.
        """
        super().__init__(
            shape=(n_channel, n_channel), spiking_input=True, spiking_output=True
        )

        self.clock_rate = SimulationParameter(clock_rate, shape=())
        self.dt = SimulationParameter(1 / clock_rate, shape=())

        # Hard-coded specs for the network
        self.__specs = {
            "weights_in": np.expand_dims(2 * np.eye(n_channel, dtype="int8"), axis=-1),
            "weights_out": 2 * np.eye(n_channel, dtype="int8"),
            "weights_rec": np.expand_dims(
                -1 * np.eye(n_channel, dtype="int8"), axis=-1
            ),
            "dash_mem": np.ones((n_channel,), dtype="int16"),
            "dash_mem_out": np.ones((n_channel,), dtype="int16"),
            "dash_syn": np.ones((n_channel,), dtype="int16"),
            "dash_syn_out": np.ones((n_channel,), dtype="int16"),
            "threshold": np.ones((n_channel,), dtype="int16"),
            "threshold_out": np.ones((n_channel,), dtype="int16"),
            "bias": None,
            "bias_out": None,
            "weight_shift_in": 0,
            "weight_shift_rec": 0,
            "weight_shift_out": 0,
            "aliases": None,
            "dt": self.dt,
            "output_mode": "Spike",
        }

        if device is None:
            self.model = XyloSim.from_specification(**self.__specs)

        else:
            self.model = XyloSamna(device=device, config=self.config, dt=self.dt)

    def reset_state(self) -> None:
        """Reset the underlying network."""
        self.model.reset_state()

    @property
    def config(self) -> XyloConfiguration:
        """
        Get the configuration from the specs
        """
        __config, is_valid, msg = config_from_specification(**self.__specs)
        if not is_valid:
            print(msg)
        return __config

    def evolve(
        self, input_data: FloatVector, record: bool = True
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        Evolve the underlying network for one time step.
        Read the output spikes from the hidden neurons instead of the output neurons.
        Readout neurons cannot be used reliably because:
            * Buffering the input spikes and reading the same exact output from the readout neurons is technically beyond the limits of the SNN core
            * Readout neurons do not have the capacity to spike more than once per time step, but the hidden spikes can

        Args:
            input_data (FloatVector): The input to the network.
            record (bool, optional): Dummy variable, not used. Defaults to False.

        Returns:
            Tuple[np.ndarray, dict, dict]:
                - np.ndarray: The output of the network.
                - dict: The state of the network.
                - dict: The record dictionary.
        """
        _, state, rec = self.model(input_data, record=True)
        out = rec["Spikes"]
        return out, state, rec


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
