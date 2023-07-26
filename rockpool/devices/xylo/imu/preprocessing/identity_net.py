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

CLOCK_RATE = 200
N_CHANNEL = 15

# Hard-coded specs, to be replaced by a YAML file
spec_identity_net = {
    "weights_in": np.expand_dims(2 * np.eye(N_CHANNEL, dtype="int8"), axis=-1),
    "weights_out": np.zeros((N_CHANNEL, N_CHANNEL), dtype="int8"),
    "weights_rec": np.expand_dims(-1 * np.eye(N_CHANNEL, dtype="int8"), axis=-1),
    "dash_mem": np.ones((N_CHANNEL,), dtype="int16"),
    "dash_mem_out": None,
    "dash_syn": np.ones((N_CHANNEL,), dtype="int16"),
    "dash_syn_out": None,
    "threshold": np.ones((N_CHANNEL,), dtype="int16"),
    "threshold_out": np.full((N_CHANNEL,), (2**15 - 1), dtype="int16"),
    "bias": None,
    "bias_out": None,
    "weight_shift_in": 0,
    "weight_shift_rec": 0,
    "weight_shift_out": 0,
    "aliases": None,
    "dt": 1 / CLOCK_RATE,
    "output_mode": "Spike",
}


class IdentityNet(Module):
    """
    A simple identity network, to be used as a dummy network which aimed to return the input as the output.
    Since there is no other option to record the output of IMUIF directly, `IdentityNet` will serve as a buffer on SNN core, making it possible to implement a `IMUIFSamna` module
    """

    def __init__(self, device: Optional[XyloIMUHDK] = None) -> None:
        super().__init__(
            shape=(N_CHANNEL, N_CHANNEL), spiking_input=True, spiking_output=True
        )

        if device is None:
            print("No device provided, using XyloSim")
            self.model = XyloSim.from_specification(**spec_identity_net)

        else:
            print("Using XyloSamna")
            config, is_valid, msg = config_from_specification(**spec_identity_net)
            if not is_valid:
                print(msg)
            self.model = XyloSamna(device=device, config=config, dt=1 / CLOCK_RATE)

        self.dt = self.model.dt

    def evolve(
        self, input_data: FloatVector, record: bool = True
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        Evolve the underlying network for one time step.
        Pop the `Spikes` key from the `rec` dictionary, and return it as the output.
        It's necessary because the recurrent spikes are the spikes that are the same as the input spikes

        Args:
            input_data (FloatVector): The input to the network.
            record (bool, optional): Dummy variable, not used. Defaults to True.

        Returns:
            Tuple[np.ndarray, dict, dict]:
                - np.ndarray: The output of the network.
                - dict: The state of the network.
                - dict: The record dictionary.
        """
        _, state, rec = self.model(input_data, record=True)
        out = rec["Spikes"]
        rec.pop("Spikes")
        return out, state, rec
