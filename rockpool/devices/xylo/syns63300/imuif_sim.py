"""
Simulation of an analog IMU signal filtering front-end

Defines :py:class:`.IMUIFSim` module.

See Also:
    For example usage of the :py:class:`.IMUIFSim` Module, see :ref:`/devices/xylo-imu/imu-if.ipynb`
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
from rockpool.devices.xylo.syns63300.imuif.params import DEFAULT_FILTER_BANDS
from rockpool.devices.xylo.syns63300.imuif.utils import type_check
from rockpool.devices.xylo.syns63300.imuif import (
    IAFSpikeEncoder,
    ScaleSpikeEncoder,
    RotationRemoval,
    FilterBank,
    BandPassFilter,
)
from rockpool.nn.combinators import Sequential
from rockpool.nn.modules.module import Module
from rockpool.utilities.backend_management import backend_available

from rockpool.parameters import SimulationParameter

if backend_available("samna"):
    from samna.xyloImu.configuration import InputInterfaceConfig
else:
    InputInterfaceConfig = Any

__all__ = ["IMUIFSim"]


class IMUIFSim(Module):
    """
    A :py:class:`.Module` that simulates the IMU signal preprocessing on Xylo IMU

    This module simulates the Xylo IMU front-end stage. This is a signal-to-event core that consists of rotation removal units, low-pass filters, and a spike generator. The module takes in a 3D IMU signal and outputs a spike train.

    See Also:
        For example usage of the :py:class:`.IMUIFSim` Module, see :ref:`/devices/xylo-imu/imu-if.ipynb`
    """

    def __init__(
        self,
        shape: Optional[Union[Tuple, int]] = (3, 15),
        select_iaf_output: bool = False,
        bypass_jsvd: bool = False,
        filter_list: Optional[List[BandPassFilter]] = None,
        scale_values: Union[List[int], int] = 5,
        iaf_threshold_values: Union[List[int], int] = 1024,
        num_avg_bitshift: int = 4,
        SAH_period: int = 10,
        sampling_freq: float = 200.0,
    ) -> None:
        """
        Object constructor

        Args:
            shape (Optional[Union[Tuple, int]], optional): the shape of the input-output transformation. Defaults to (3, 15).
            select_iaf_output (bool, optional): If true, the output of the module is encoded using IAF spike encoding. If false, the output of the module is encoded using scale spike encoding. Defaults to False.
            bypass_jsvd (bool, optional): If true, the module does not perform the rotation removal stage. Defaults to False.
            filter_list (Optional[List[BandPassFilter]], optional): the list of filters of the filterbank. Note that first 5 filters will apply to the first input channel, the second 5 filters apply to the second input channel, and the last 5 filters will apply to the 3rd input channel. Defaults to None, when it's None, the default values apply.
            scale_values (Union[List[int], int], optional): number of right-bit-shifts needed for down-scaling the input signal (per channel). Defaults to [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5].
            iaf_threshold_values (Union[List[int], int], optional): the thresholds of the IAF neurons (quantized). Default to [1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024].
            num_avg_bitshift (int): number of bit shifts used in the low-pass filter implementation. Default to 4.
                The effective window length of the low-pass filter will be `2**num_avg_bitshift`
            SAH_period (int): Sampling period that the signal is sampled and held, in number of timesteps. Defaults to 10.
            sampling_freq (float): Sampling frequency of the IMU interface. Default: ``200.``
        """
        super().__init__(shape=shape, spiking_input=False, spiking_output=True)

        if filter_list is None:
            filter_list = [
                BandPassFilter.from_specification(*band)
                for band in DEFAULT_FILTER_BANDS
            ]

        if len(filter_list) != self.size_out:
            raise ValueError(
                f"the number of filters {len(filter_list)} does not match the number of output channels {self.size_out}"
            )

        filter_bank = FilterBank(shape, *filter_list)

        if select_iaf_output:
            spike_encoder = IAFSpikeEncoder(
                shape=(self.size_out, self.size_out), threshold=iaf_threshold_values
            )
        else:
            spike_encoder = ScaleSpikeEncoder(
                shape=(self.size_out, self.size_out), num_scale_bits=scale_values
            )

        if bypass_jsvd:
            mod_IMUIF = Sequential(filter_bank, spike_encoder)
        else:
            rotation_removal = RotationRemoval(
                shape=(self.size_in, self.size_in),
                num_avg_bitshift=num_avg_bitshift,
                sampling_period=SAH_period,
            )
            mod_IMUIF = Sequential(rotation_removal, filter_bank, spike_encoder)

        self.model = mod_IMUIF
        """The sequential module that simulates the IMU front-end"""

        self.dt = SimulationParameter(1 / sampling_freq)
        """ (float) Time-step of the encoding simulation in seconds """

    @type_check
    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """Processes the input IMU signal sample-by-sample and generate spikes

        Args:
            input_data (np.ndarray): batched input data recorded from IMU sensor. It should be in integer format. (BxTx3)
            record (bool, optional): If True, the intermediate results are recorded and returned. Defaults to False.

        Returns:
            Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
                output
                empty dictionary
                empty dictionary
        """

        # - Auto batch and shape check
        input_data, _ = self._auto_batch(input_data)

        # - Ensure data is the correct dtype
        input_data = np.array(input_data, dtype=np.int64).astype(object)

        # - Encode data and return
        return self.model(input_data, record=record)

    @classmethod
    def from_config(cls, config: InputInterfaceConfig) -> IMUIFSim:
        """Obtain an instance of IMUIFSim from a samna configuration object

        Args:
            config (InputInterfaceConfig): a samna object that encapsulates the hardware configuration such as register values

        Raises:
            TypeError: if the input is not the samna configuration object that is expected

        Returns:
            IMUIFSim: an instance of IMUIFSim
        """
        if not isinstance(config, InputInterfaceConfig):
            raise TypeError(
                f"config must be an instance of `samna.xyloImu.configuration.InputInterfaceConfig`. We got {type(config)}"
            )
        if config.enable != True:
            warn("IMUIF is not enabled in configuration!")

        # We could not use `config.delay_threshold` here because it does not affect the simulation
        filter_list = []
        for i, (B_b, B_wf, B_af, a1, a2) in enumerate(
            zip(
                config.bpf_bb_values,
                config.bpf_bwf_values,
                config.bpf_baf_values,
                config.bpf_a1_values,
                config.bpf_a2_values,
            )
        ):
            filter_list.append(
                BandPassFilter(
                    B_b=B_b,
                    B_wf=B_wf,
                    B_af=B_af,
                    a1=a1,
                    a2=a2,
                )
            )

        return cls(
            shape=(3, 15),
            select_iaf_output=config.select_iaf_output,
            bypass_jsvd=config.bypass_jsvd,
            filter_list=filter_list,
            scale_values=config.scale_values,
            iaf_threshold_values=config.iaf_threshold_values,
            num_avg_bitshift=config.estimator_k_setting,
            SAH_period=config.update_matrix_threshold,
        )

    @classmethod
    def from_specification(cls, *args, **kwargs) -> IMUIFSim:
        raise NotImplementedError(
            "Here we do not have any high-level specification that's different than __init__ parameters."
        )

    def export_config(self) -> InputInterfaceConfig:
        """Export the current configuration of the IMUIF module

        Returns:
            InputInterfaceConfig: a samna object that encapsulates the hardware configuration such as register values
        """
        default_config = InputInterfaceConfig()
        bypass_jsvd = True
        scale_values = default_config.scale_values
        iaf_threshold_values = default_config.iaf_threshold_values
        estimator_k_setting = default_config.estimator_k_setting
        update_matrix_threshold = default_config.update_matrix_threshold

        for module in self.model:
            if isinstance(module, FilterBank):
                bpf_bb_values = module.B_b_list
                bpf_bwf_values = module.B_wf_list
                bpf_baf_values = module.B_af_list
                bpf_a1_values = [a1 for a1 in module.a1_list]
                bpf_a2_values = [a2 for a2 in module.a2_list]
            elif isinstance(module, IAFSpikeEncoder):
                select_iaf_output = True
                iaf_threshold_values = module.threshold
            elif isinstance(module, ScaleSpikeEncoder):
                select_iaf_output = False
                scale_values = module.num_scale_bits
            elif isinstance(module, RotationRemoval):
                bypass_jsvd = False
                estimator_k_setting = module.num_avg_bitshift
                update_matrix_threshold = module.sampling_period
            else:
                raise TypeError(
                    f"module {module} is not recognized as a module of IMUIF"
                )

        # We could not use `config.delay_threshold` here because it does not affect the simulation
        config = InputInterfaceConfig(
            enable=True,
            bpf_a1_values=bpf_a1_values,
            bpf_a2_values=bpf_a2_values,
            bpf_baf_values=bpf_baf_values,
            bpf_bb_values=bpf_bb_values,
            bpf_bwf_values=bpf_bwf_values,
            bypass_jsvd=bypass_jsvd,
            estimator_k_setting=estimator_k_setting,
            iaf_threshold_values=iaf_threshold_values,
            scale_values=scale_values,
            select_iaf_output=select_iaf_output,
            update_matrix_threshold=update_matrix_threshold,
        )
        return config
