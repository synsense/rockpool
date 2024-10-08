"""
Samna-backed bridge to Xylo dev kit for SYNS61201 Xylo core v2
"""

# - Samna imports
import samna
from samna.xyloCore2.configuration import XyloConfiguration
from samna.afe2.configuration import AfeConfiguration

from . import xa2_devkit_utils as hdkutils
from .xa2_devkit_utils import XyloA2HDK

import time
import numpy as np
from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter

# - Typing
from typing import Optional, Union, Callable, List

import warnings

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:

    def tqdm(wrapped, *args, **kwargs):
        return wrapped


# - Configure exports
__all__ = ["XyloMonitor"]


class XyloMonitor(Module):
    """
    A spiking neuron :py:class:`.Module` backed by the Xylo hardware, via `samna`.

    Use :py:func:`.config_from_specification` to build and validate a configuration for Xylo.

    """

    def __init__(
        self,
        device: XyloA2HDK,
        config: XyloConfiguration = None,
        afe_config: AfeConfiguration = None,
        dt: float = 1e-3,
        output_mode: str = "Spike",
        amplify_level: str = "low",
        change_count: Optional[int] = None,
        main_clk_rate: int = int(50e6),
        hibernation_mode: bool = False,
        divisive_norm: bool = False,
        divisive_norm_params: Optional[dict] = {},
        calibration_params: Optional[dict] = {},
        read_register: bool = False,
        power_frequency: float = 5.0,
        *args,
        **kwargs,
    ):
        """
        Instantiate a Module with Xylo dev-kit backend

        Args:
            device (XyloA2HDK): An opened `samna` device to a Xylo dev kit
            config (XyloConfiguraration): A Xylo configuration from `samna`
            afe_config (AFE2Configuration): A samna AFE2 configuration object
            dt (float): The simulation time-step to use for this Module
            output_mode (str): The readout mode for the Xylo device. This must be one of ``["Spike", "Vmem"]``. Default: "Spike", return events from the output layer.
            amplify_level(str): The level of volume gain. Defaul "low" is the one without gain.
            change_count (int): If is not None, AFE event counter will change from outputting 1 spike out of 4 into outputting 1 out of change_count.
            main_clk_rate(int): The main clock rate of Xylo.
            hibernation_mode (bool): If True, hibernation mode will be switched on, which only outputs events if it receives inputs above a threshold.
            divisive_norm (bool): If True, divisive normalization will be switched on.
            divisive_norm_params (Dict): Specify the divisive normalization parameters, should be structured as {"s": , "p": , "iaf_bias": }.
            calibration_params (Dict): Specify the calibration parameters.
            read_register (bool): If True, will print all register values of AFE and Xylo after initialization.
            power_frequency (float): The frequency of power measurement. Default: 5.0
        """

        # - Check input arguments
        if device is None:
            raise ValueError("`device` must be a valid, opened Xylo HDK device.")

        # - Check output mode specification
        if output_mode not in ["Spike", "Vmem"]:
            raise ValueError(
                f'{output_mode} is not supported. Must be one of `["Spike", "Vmem"]`.'
            )
        self._output_mode = output_mode

        # - Check params dict
        if (type(divisive_norm_params).__name__ != "dict") or (
            type(calibration_params).__name__ != "dict"
        ):
            raise ValueError(
                "`divisive_norm_params` and `calibration_params` must be dict."
            )

        # - Get a default configuration
        if config is None:
            config = samna.xyloCore2.configuration.XyloConfiguration()

        if afe_config is not None:
            warnings.warn(
                "Setting a manual configuration for the Xylo-AFE2 is not yet supported."
            )

        if afe_config is None:
            afe_config = AfeConfiguration()

        # - Store the device
        self._device: XyloA2HDK = device
        """ `.XyloHDK`: The Xylo HDK used by this module """

        # - Get the network shape
        Nin, Nhidden = np.shape(config.input.weights)
        _, Nout = np.shape(config.readout.weights)

        # - Register buffers to read and write events, monitor state
        self._read_buffer = hdkutils.new_xylo_read_buffer(device)
        self._write_buffer = hdkutils.new_xylo_write_buffer(device)
        self._state_buffer = hdkutils.new_xylo_state_monitor_buffer(device)

        self._afe_read_buffer = hdkutils.AFE2ReadBuffer()
        self._afe_write_buffer = hdkutils.AFE2WriteBuffer()
        graph = samna.graph.EventFilterGraph()
        graph.sequential(
            [self._device.get_afe_model_source_node(), self._afe_read_buffer]
        )
        graph = samna.graph.EventFilterGraph()
        graph.sequential(
            [self._afe_write_buffer, self._device.get_afe_model_sink_node()]
        )

        # - Initialise the superclass
        super().__init__(
            shape=(Nin, Nhidden, Nout), spiking_input=True, spiking_output=True
        )

        # - Store the configuration (and apply it)
        self.config: Union[
            XyloConfiguration, SimulationParameter
        ] = SimulationParameter(shape=(), init_func=lambda _: config)
        self._config = config
        if hibernation_mode:
            self._config.enable_hibernation_mode = True
        """ `.XyloConfiguration`: The HDK configuration applied to the Xylo module """

        # - Store the timestep
        self.dt: Union[float, SimulationParameter] = dt
        """ float: Simulation time-step of the module, in seconds """

        # - Store the main clock rate
        self._main_clk_rate = main_clk_rate

        # - Store the io module
        self._io = self._device.get_io_module()

        # - Initialise the xylo HDK
        hdkutils.initialise_xylo_hdk(self._write_buffer)

        # - Check that we can access the device node, and that it's a Xylo HDK
        if not hdkutils.verify_xylo_version(
            self._read_buffer, self._write_buffer, timeout=10.0
        ):
            raise ValueError(
                "Cannot verify HDK version. `device` must be an opened Xylo HDK."
            )

        self._chip_version, self._chip_revision = hdkutils.read_afe2_module_version(
            self._afe_read_buffer, self._afe_write_buffer
        )
        if self._chip_version != 1 or self._chip_revision != 0:
            raise ValueError(
                f"AFE version is {(self._chip_version, self._chip_revision)}; expected (1, 0)."
            )

        # - Change event count number of AFE
        if change_count is not None:
            if change_count < 0:
                raise ValueError(
                    f"{change_count} is negative. Must be non-negative values."
                )
            afe_config = hdkutils.config_afe_channel_thresholds(
                afe_config, change_count
            )

        # - Set up known good AFE configuration
        print("Configuring AFE...")
        afe_config = hdkutils.apply_afe2_default_config(
            afe2hdk=self._device,
            config=afe_config,
            **calibration_params,
        )
        print("Configured AFE")

        # - Amplify input volume
        afe_config = hdkutils.config_lna_amplification(afe_config, level=amplify_level)

        # - Divisive normalization
        if divisive_norm:
            afe_config = hdkutils.DivisiveNormalization(
                config=afe_config,
                **divisive_norm_params,
            )

        # - Set to hibernation mode
        if hibernation_mode:
            afe_config = hdkutils.config_AFE_hibernation(afe_config)

        # - Apply configuration
        self._device.get_afe_model().apply_configuration(afe_config)

        # - Set power measurement module
        self._power_buf, self.power_monitor = hdkutils.set_power_measure(
            self._device, power_frequency
        )

        # - Configure to auto mode
        self.auto_config(hibernation=hibernation_mode)

        # - Read AFE and Xylo configuration
        if read_register:
            hdkutils.read_all_afe2_register(
                self._afe_read_buffer, self._afe_write_buffer
            )
            hdkutils.read_all_xylo_register(self._read_buffer, self._write_buffer)

    @property
    def config(self):
        # - Return the configuration stored on Xylo HDK
        return self._device.get_xylo_model().get_configuration()

    @config.setter
    def config(self, new_config):
        # - Test for a valid configuration
        is_valid, msg = samna.xyloCore2.validate_configuration(new_config)
        if not is_valid:
            raise ValueError(f"Invalid configuration for the Xylo HDK: {msg}")

        # - Write the configuration to the device
        hdkutils.apply_configuration(
            self._device, new_config, self._read_buffer, self._write_buffer
        )

        # - Store the configuration locally
        self._config = new_config

    def auto_config(self, hibernation: bool = False):
        """
        Configure the Xylo HDK to use real-time mode

        Args:
            hibernation: specify the hibernation mode
        """
        # - First configure to manual mode and clear network states
        self.config = hdkutils.config_basic_mode(self._config)

        # - Set main clock rate
        self._io.get_xylo_handler().set_main_clk_rate(self._main_clk_rate)
        self._state_buffer.set_configuration(self._config)

        # - Set configure to auto mode
        hdkutils._auto_mode(
            io=self._io,
            read_buffer=self._read_buffer,
            write_buffer=self._write_buffer,
            write_afe_buffer=self._afe_write_buffer,
            dt=self.dt,
            main_clk_rate=self._main_clk_rate,
            hibernation_mode=hibernation,
        )
        time.sleep(1)
        self._state_buffer.reset()

    def evolve(
        self,
        input_data,
        record: bool = False,
        record_power: bool = False,
    ) -> (list, dict, dict):
        """
        Evolve a network on the Xylo HDK in Real-time mode

        Args:
            input_data (np.ndarray): An array ``[T, Nin]``, specifying the number of time-steps to record.
            record (bool): ``False``, do not return a recording dictionary. Recording internal state is not supported by :py:class:`.XyloMonitor`
            record_power (bool): If ``True``, record the power consumption during each evolve.

        Returns:
            (list, dict, dict) output_events, {}, {}
            output_events is a list that stores the output events of T time-steps.
        """
        # - Check `record` flag
        if record:
            raise ValueError(
                "Recording internal state is not supported by XyloIMUMonitor."
            )

        Nt = input_data.shape[0]
        out = []
        count = 0

        # - Clear the power recording buffer, if recording power
        if record_power:
            self._power_buf.clear_events()

        while count < int(Nt):
            if self._output_mode == "Vmem":
                output = self._state_buffer.get_output_v_mem()

            elif self._output_mode == "Spike":
                output = self._state_buffer.get_output_spike()

            if output[0]:
                self._state_buffer.reset()
                count += len(output)
                out.append([sub[-1] for sub in output])

        rec_dict = {}

        if record_power:
            # - Get all recent power events from the power measurement
            ps = self._power_buf.get_events()

            # - Separate out power meaurement events by channel
            channels = samna.xyloA2TestBoard.MeasurementChannels
            io_power = np.array([e.value for e in ps if e.channel == int(channels.Io)])
            AFE_io_power = np.array(
                [e.value for e in ps if e.channel == int(channels.IoAfe)]
            )
            logic_power = np.array(
                [e.value for e in ps if e.channel == int(channels.Logic)]
            )
            AFE_logic_power = np.array(
                [e.value for e in ps if e.channel == int(channels.LogicAfe)]
            )
            rec_dict.update(
                {
                    "io_power": io_power,
                    "logic_power": logic_power,
                    "AFE_io_power": AFE_io_power,
                    "AFE_logic_power": AFE_logic_power,
                }
            )

        return out, {}, rec_dict

    def reset_state(*args, **kwargs):
        raise NotImplementedError(
            "Reset state is not permitted for XyloAudio 2 in real-time mode"
        )
