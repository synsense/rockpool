"""
samna-backed module for interfacing with the Xylo-A2 AFE HW module
"""

import time
import warnings

import samna

from samna.afe2.configuration import AfeConfiguration as AFE2Configuration
from samna.afe2 import validate_configuration

from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter
from rockpool import TSEvent
from rockpool.typehints import P_float

from . import xa2_devkit_utils as hdu
from .xa2_devkit_utils import XyloA2HDK

try:
    from tqdm.autonotebook import tqdm
except ModuleNotFoundError:

    def tqdm(wrapped, *args, **kwargs):
        return wrapped


from typing import Union, Dict, Any, Tuple, Optional


__all__ = ["AFESamna"]


class AFESamna(Module):
    """
    Interface to the Audio Front-End module on a Xylo-A2 HDK

    This module uses ``samna`` to interface to the AFE hardware on a Xylo-A2 HDK. It permits recording from the AFE hardware.

    To record from the module, use the :py:meth:`~.AFESamna.evolve` method. You need to pass this method an empty matrix, with the desired number of time-steps. The time-step ``dt`` is specified at module instantiation.

    A simulation of the module is available in :py:class:`.AFESim`.

    Warnings:
        This module does not currently support manual configuration. A fixed configuration is provided which uses auto-calibration, applied when the module is instantiated. This takes approximately 50 seconds to configure, leading to slow instantiation.

    See Also:
        For information about the Audio Front-End design, and examples of using :py:class:`.AFESim` for a simulation of the AFE, see :ref:`/devices/analog-frontend-example.ipynb`.

    Examples:
        Instantiate an AFE module, connected to a Xylo-A2 HDK

        >>> from rockpool.devices.xylo import AFESamna
        >>> import rockpool.devices.xylo.syns61201.xa2_devkit_utils as xdu
        >>> afe_hdks = xdu.find_xylo_a2_boards()
        >>> afe = AFESamna(afe_hdks[0], dt = 10e-3)

        Use the module to record some audio events

        >>> import numpy as np
        >>> audio_events = afe(np.zeros([0, 100, 0]))
    """

    def __init__(
        self,
        device: XyloA2HDK,
        config: Optional[AFE2Configuration] = None,
        dt: float = 1e-3,
        amplify_level: str = "low",
        change_count: Optional[int] = None,
        hibernation_mode: bool = False,
        divisive_norm: bool = False,
        divisive_norm_params: Optional[dict] = {},
        calibration_params: Optional[dict] = {},
        read_register: bool = False,
        *args,
        **kwargs,
    ):
        """
        Instantiate an AFE module, via a samna backend

        Args:
            device (AFE2HDK): A connected AFE2 HDK device.
            config (AFE2Configuration): A samna AFE2 configuration object.
            dt (float): The desired spike time resolution in seconds.
            amplify_level(str): The level of volume gain. Defaul "low" is the one without gain.
            change_count (int): If is not None, AFE event counter will change from outputting 1 spike out of 4 into outputting 1 out of change_count.
            hibernation_mode (bool): If True, hibernation mode will be switched on, which only outputs events if it receives inputs above a threshold.
            divisive_norm (bool): If True, divisive normalization will be switched on.
            divisive_norm_params (Dict): Specify the divisive normalization parameters, should be structured as {"s": , "p": , "iaf_bias": }.
            calibration_params (Dict): Specify the calibration parameters.
            read_register (bool): If True, will print all register values of AFE after initialization.
        """
        # - Check input arguments
        if device is None:
            raise ValueError(
                "`device` must be a valid, opened Xylo AFE V2 HDK self._device."
            )

        # - Check params dict
        if (type(divisive_norm_params).__name__ != "dict") or (
            type(calibration_params).__name__ != "dict"
        ):
            raise ValueError(
                "`divisive_norm_params` and `calibration_params` must be dict."
            )

        # - Get a default configuration
        if config is not None:
            warnings.warn(
                "Setting a manual configuration for the Xylo-AFE2 is not yet supported."
            )

        if config is None:
            config = samna.afe2.configuration.AfeConfiguration()

        # - Determine how many output channels we have
        Nout = len(config.analog_top.channels)

        # - Initialise the superclass
        super().__init__(shape=(0, Nout), spiking_input=True, spiking_output=True)

        # - Store the HDK device node
        self._device = device

        # - Store the dt parameter
        self.dt: P_float = SimulationParameter(dt)

        # - Configure the HDK
        device_io = self._device.get_io_module()
        device_io.write_config(0x52, 0b11)
        # time.sleep(0.5)

        # - Create write and read buffers
        self._xylo_core_read_buf = hdu.Xylo2ReadBuffer()
        graph = samna.graph.EventFilterGraph()
        graph.sequential(
            [self._device.get_xylo_model_source_node(), self._xylo_core_read_buf]
        )

        self._afe_read_buf = hdu.AFE2ReadBuffer()
        graph = samna.graph.EventFilterGraph()
        graph.sequential([self._device.get_afe_model_source_node(), self._afe_read_buf])

        self._afe_write_buf = hdu.AFE2WriteBuffer()
        graph = samna.graph.EventFilterGraph()
        graph.sequential([self._afe_write_buf, self._device.get_afe_model_sink_node()])

        # - Check that we have a correct device version
        self._chip_version, self._chip_revision = hdu.read_afe2_module_version(
            self._afe_read_buf, self._afe_write_buf
        )
        if self._chip_version != 1 or self._chip_revision != 0:
            raise ValueError(
                f"AFE version is {(self._chip_version, self._chip_revision)}; expected (1, 0)."
            )

        # - Configure the HDK
        device_io.write_config(0x1002, 1)
        # time.sleep(0.5)
        device_io.write_config(0x0022, 1)
        # time.sleep(0.2)

        device_io.write_config(0x54, 1)
        device_io.write_config(0x0003, 1)
        device_io.write_config(0x0004, 0x03)
        device_io.write_config(0x02, 0x30)
        # time.sleep(0.5)
        # xylo_handler = device_io.get_xylo_handler()

        # - Change counter threshold
        if change_count is not None:
            if change_count < 0:
                raise ValueError(
                    f"{change_count} is negative. Must be non-negative values."
                )
            config = hdu.config_afe_channel_thresholds(config, change_count)

        # - Set up known good configuration
        print("Configuring AFE...")
        config = hdu.apply_afe2_default_config(
            afe2hdk=self._device,
            config=config,
            **calibration_params,
        )
        print("Configured AFE")

        # - Amplify input volume
        config = hdu.config_lna_amplification(config, level=amplify_level)

        # - Set up divisive normalization
        if divisive_norm:
            config = hdu.DivisiveNormalization(
                config=config,
                **divisive_norm_params,
            )

        # - Set up hibernation mode
        if hibernation_mode:
            config = hdu.config_AFE_hibernation(config)
            config.aer_2_saer.hibernation.mode = 2
            config.aer_2_saer.hibernation.reset = 1

        # - Apply configuration
        self._device.get_afe_model().apply_configuration(config)
        self._config = config

        # - Read all registers
        if read_register:
            hdu.read_all_afe2_register(self._afe_read_buf, self._afe_write_buf)

    def evolve(self, input_data, record: bool = False) -> Tuple[Any, Any, Any]:
        """
        Use the AFE HW module to record live audio and return as encoded events

        Args:
            input_data (np.ndarray): An array ``[0, T, 0]``, specifying the number of time-steps to record.

        Returns:
            (np.ndarray, dict, dict) output_events, {}, {}
        """

        # - Handle auto batching
        input_data, _ = self._auto_batch(input_data)

        # - For how long should we record?
        duration = input_data.shape[1] * self.dt

        # - Record events
        timestamps, channels = hdu.read_afe2_events_blocking(
            self._device, self._afe_write_buf, self._afe_read_buf, duration
        )

        # - Convert to an event raster
        events_ts = TSEvent(
            timestamps,
            channels,
            t_start=0.0,
            t_stop=duration,
            num_channels=self.size_out,
        ).raster(self.dt, add_events=True)

        # - Return output, state, record dict
        return events_ts, self.state(), {}

    @property
    def _version(self) -> Tuple[int, int]:
        """
        Return the version and revision numbers of the connected Xylo-AFE2 chip

        Returns:
            (int, int): version, revision
        """
        return (self._chip_version, self._chip_revision)

    def save_afe_config(self, filename):
        """
        Save an AFE configuration to disk in JSON format

        Args:
            filename (str): The filename to write to
        """
        with open(filename, "w") as f:
            f.write(self._config.to_json())
