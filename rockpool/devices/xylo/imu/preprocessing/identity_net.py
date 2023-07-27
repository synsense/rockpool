from typing import Optional, Tuple

import numpy as np

from rockpool.devices.xylo.imu.xylo_samna import (
    XyloIMUHDK,
    XyloSamna,
    config_from_specification,
)
from rockpool.devices.xylo.imu.xylo_sim import XyloSim
from rockpool.nn.modules import Module
from rockpool.typehints import FloatVector
from rockpool.parameters import SimulationParameter

__all__ = ["IdentityNet"]


class IdentityNet(Module):
    """
    A simple identity network, to be used as a dummy network which aimed to return the input as the output.
    Since there is no other option to record the output of IMUIF directly, `IdentityNet` will serve as a buffer on SNN core, making it possible to implement a `IMUIFSamna` module
    """

    def __init__(
        self,
        device: Optional[XyloIMUHDK] = None,
        n_channel: int = 15,
        clock_rate: int = 200,
        speed_up_factor: int = 2,
    ) -> None:
        """
        Object constructor

        Args:
            device (Optional[XyloIMUHDK], optional): The device to use. If None, use XyloSim. Defaults to None.
            n_channel (int, optional): Number of channels employed. Defaults to 15.
            clock_rate (int, optional): The nominal clock rate of the input unit. Defaults to 200.
            speed_up_factor (int, optional): The speed up factor, that is, the ratio between the clock rate of the input unit and the clock rate of the SNN core. Defaults to 2.
        """
        super().__init__(
            shape=(n_channel, n_channel), spiking_input=True, spiking_output=True
        )

        self.clock_rate = SimulationParameter(clock_rate, shape=())
        self.speed_up_factor = SimulationParameter(speed_up_factor, shape=())
        self.dt = SimulationParameter(1 / (clock_rate * speed_up_factor), shape=())

        # Hard-coded specs for the network, to be replaced by a yaml file
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
            print("No device provided, using XyloSim")
            self.model = XyloSim.from_specification(**self.__specs)

        else:
            print("Using XyloSamna")
            config, is_valid, msg = config_from_specification(**self.__specs)
            if not is_valid:
                print(msg)
            self.model = XyloSamna(device=device, config=config, dt=self.dt)

    def reset_state(self) -> None:
        """Reset the underlying network."""
        self.model.reset_state()

    def dilute(self, spike_train: FloatVector) -> FloatVector:
        """
        Dilute the input spike train by a factor of `speed_up_factor`
        Inject zero columns between the consecutive columns of the input spike train

        Args:
            spike_train (FloatVector): The input spike train

        Returns:
            FloatVector: The diluted spike train
        """
        spike_train = np.array(spike_train)
        __T, __C = spike_train.shape
        new_T = __T * self.speed_up_factor

        # Create an array of zeros for the new columns
        sparsed_spike_train = np.zeros((new_T, __C), dtype=spike_train.dtype)
        for i in range(0, __T):
            sparsed_spike_train[i * self.speed_up_factor, :] = spike_train[i, :]

        return sparsed_spike_train

    def undilute(self, spike_train: FloatVector) -> FloatVector:
        """
        Reverse the `dilute` operation. Remove the injected zero columns between the consecutive columns of the input spike train

        Args:
            spike_train (FloatVector): The diluted spike train

        Returns:
            FloatVector: The undiluted spike train
        """
        spike_train = np.array(spike_train)
        __T, __C = spike_train.shape
        new_T = int(__T / self.speed_up_factor)

        # Create an array of zeros for the new columns
        restored_spike_train = np.zeros((new_T, __C))
        for i in range(0, new_T):
            restored_spike_train[i, :] = spike_train[i * self.speed_up_factor, :]

        return restored_spike_train

    def evolve(
        self, input_data: FloatVector, record: bool = True
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        Evolve the underlying network for one time step. The input data is diluted by a factor of `speed_up_factor` before being fed to the network.
        Dilution is done by injecting zero columns between the consecutive columns of the input spike train
        This makes sure that the neuron and synapse are drained enough for a fresh start in the next time step

        Args:
            input_data (FloatVector): The input to the network.
            record (bool, optional): Dummy variable, not used. Defaults to False.

        Returns:
            Tuple[np.ndarray, dict, dict]:
                - np.ndarray: The output of the network.
                - dict: The state of the network.
                - dict: The record dictionary.
        """
        # Dilute the input data
        input_data = self.dilute(input_data)
        out, state, rec = self.model(input_data, record=False)

        # Process the output and restore the original shape
        __T, __C = out.shape
        out = np.vstack((out, np.zeros((1, __C))))
        out = self.undilute(out[1:])

        return out, state, rec
