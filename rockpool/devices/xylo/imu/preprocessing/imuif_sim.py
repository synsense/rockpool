"""
Simulation of an analog IMU signal filtering front-end

Defines :py:class:`.IMUIFSim` module.

See Also:
    For example usage of the :py:class:`.IMUIFSim` Module, see :ref:`/devices/xylo-imu/imu-if.ipynb`
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from rockpool.devices.xylo.imu.preprocessing.utils import type_check
from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter
from rockpool.utilities.backend_management import backend_available

if backend_available("samna"):
    from samna.xyloImu.configuration import InputInterfaceConfig
else:
    InputInterfaceConfig = Any

__all__ = ["IMUIFSim"]


class IMUIFSim(Module):
    """
    A :py:class:`.Module` that simulates analog IMU signal preprocessing into spikes.

    This module simulates the Xylo IMU front-end stage. This is a signal-to-event core that consists of rotation removal units, low-pass filters, and a spike generator. The module takes in a 3D IMU signal and outputs a spike train.

    See Also:
        For example usage of the :py:class:`.IMUIFSim` Module, see :ref:`/devices/xylo-imu/imu-if.ipynb`
    """

    def __init__(
        self,
        shape: Optional[Union[Tuple, int]] = (3, 48),
    ):
        """ """
        super().__init__(shape=shape, spiking_input=False, spiking_output=True)

    @type_check
    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """Processes the input signal sample-by-sample and generate spikes

        Args:
            input_data (np.ndarray): batched input data recorded from IMU sensor. It should be in integer format. (BxTx3)
            record (bool, optional): If True, the intermediate results are recorded and returned. Defaults to False.

        Returns:
            Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
                output
                empty dictionary
                empty dictionary
        """

        # Shape check
        __B, __T, __C = input_data.shape
        if __C != self.size_in:
            raise ValueError(
                f"The input data has incorrect number of channels! {__C} != {self.size_in}"
            )

        return input_data, {}, {}

    @classmethod
    def from_config(cls, config: InputInterfaceConfig) -> IMUIFSim:
        ## BandPassFilter
        config.bpf_a1_values
        config.bpf_a2_values
        config.bpf_baf_values
        config.bpf_bb_values
        config.bpf_bwf_values

        config.bypass_jsvd

        config.delay_threshold
        config.enable
        config.estimator_k_setting
        config.from_json
        config.iaf_threshold_values
        config.scale_values
        config.select_iaf_output

        config.update_matrix_threshold

    @classmethod
    def from_specification(cls, *args, **kwargs) -> IMUIFSim:
        pass


if __name__ == "__main__":
    config = InputInterfaceConfig()
    print(config.bpf_a1_values)
