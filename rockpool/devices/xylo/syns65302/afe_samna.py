"""
samna-backed module for interfacing with the XyloAudio 3 AudioFrontEnd HW module
"""

import samna

from samna.xyloAudio3.configuration import XyloConfiguration

from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter
from rockpool import TSEvent
from rockpool.typehints import P_float

from . import xa3_devkit_utils as hdu
from .xa3_devkit_utils import XyloAudio3HDK

import numpy as np

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:

    def tqdm(wrapped, *args, **kwargs):
        return wrapped


from typing import Union, Dict, Any, Tuple, Optional

# - Configure exports
__all__ = ["AFESamna", "load_afe_config", "save_afe_config"]

Default_Main_Clock_Rate = 50.0  # 50 MHz


class AFESamna(Module):
    """
    Interface to the Audio Front-End module on a XyloAudio 3 HDK

    This module uses ``samna`` to interface to the AFE hardware on a XyloAudio 3 HDK. It permits recording from the AFE hardware.

    To record from the module, use the :py:meth:`~.AFESamna.evolve` method. You need to pass this method an empty matrix, with the desired number of time-steps. The time-step ``dt`` is specified at module instantiation.

    A simulation of the module is available in :py:class:`.AFESim`.

    # See Also:
    #     For information about the Audio Front-End design, and examples of using :py:class:`.AFESim` for a simulation of the AFE, see :ref:`/devices/xylo-a3/AFESim3_as_transform.ipynb`.

    # Examples:
    #     Instantiate an AFE module, connected to a XyloAudio 3 HDK
    #     >>> from rockpool.devices.xylo import AFESamna
    #     >>> import rockpool.devices.xylo.syns65301.xa3_devkit_utils as xdu
    #     >>> afe_hdks = xdu.find_xylo_a3_boards()
    #     >>> afe = AFESamna(afe_hdks[0], dt = 10e-3)

    #     Use the module to record some audio events

    #     >>> import numpy as np
    #     >>> audio_events = afe(np.zeros([0, 100, 0]))
    """

    # FIXME: Currently we are using full XyloConfig, change it to AFEconfig
    def __init__(
        self,
        device: XyloAudio3HDK,
        config: Optional[XyloConfiguration] = None,
        dt: float = 1e-3,
        main_clk_rate: float = Default_Main_Clock_Rate,
        amplify_level: str = "low",
        change_count: Optional[int] = None,
        hibernation_mode: bool = False,
        divisive_norm: bool = False,
        *args,
        **kwargs,
    ):
        """
        Instantiate an AudioFrontEnd module, via a samna backend

        Args:
            device (XyloA3HDK): A connected XyloAudio 3 HDK device.
            config (XyloConfiguraration): A Xylo configuration from `samna`
            dt (float): The desired spike time resolution in seconds.
            amplify_level(str): The level of volume gain. Defaul "low" is the one without gain.
            change_count (int): If is not None, AFE event counter will change from outputting 1 spike out of 4 into outputting 1 out of change_count.
            hibernation_mode (bool): If True, hibernation mode will be switched on, which only outputs events if it receives inputs above a threshold.
            divisive_norm (bool): If True, divisive normalization will be switched on.
        """
        # - Check input arguments
        if device is None:
            raise ValueError("`device` must be a valid, opened XyloAudio 3 HDK.")

        # - Update board configuration
        board_config = samna.xyloAudio3.XyloAudio3TestBoardDefaultConfig()
        board_config.main_clock_frequency = int(main_clk_rate * 1e6)
        device.reset_board_soft(board_config)

        # - Calculate tr_wrap (clock in Hz and dt in seconds)
        tr_wrap = main_clk_rate * dt

        # - Get a default configuration
        if config is None:
            default_config = True
            config = samna.xyloAudio3.configuration.XyloConfiguration()
            config.operation_mode = samna.xyloAudio3.OperationMode.Recording

        # - Determine how many output channels we have
        _, Nout = np.shape(config.readout.weights)

        # - Initialise the superclass
        super().__init__(shape=(0, Nout), spiking_input=True, spiking_output=True)

        # - Store the HDK device node
        self._device = device

        # - Store the dt parameter
        self.dt: P_float = SimulationParameter(dt)

        # - Create write and read buffers
        self._xylo_core_read_buf = hdu.XyloAudio3ReadBuffer()
        self._read_graph = samna.graph.EventFilterGraph()
        self._read_graph.sequential(
            [self._device.get_xylo_model_source_node(), self._xylo_core_read_buf]
        )

        self._afe_read_buf = hdu.XyloAudio3ReadBuffer()
        self._afe_read_graph = samna.graph.EventFilterGraph()
        self._afe_read_graph.sequential(
            [self._device.get_afe_model_source_node(), self._afe_read_buf]
        )

        self._afe_write_buf = hdu.XyloAudio3WriteBuffer()
        self._write_graph = samna.graph.EventFilterGraph()
        self._write_graph.sequential(
            [self._afe_write_buf, self._device.get_afe_model_sink_node()]
        )

        # - Check that we have a correct device version
        self._chip_version, self._chip_revision = hdu.read_afe2_module_version(
            self._afe_read_buf, self._afe_write_buf
        )
        if self._chip_version != 1 or self._chip_revision != 0:
            raise ValueError(
                f"AFE version is {(self._chip_version, self._chip_revision)}; expected (1, 0)."
            )

        if default_config:
            config.debug.use_timestamps = False
            config.time_resolution_wrap = int(tr_wrap)
            # Choose PDM as input source.
            config.input_source = samna.xyloAudio3.InputSource.DigitalMicrophone
            # We need to set `clock_direction` to 1 (Xylo output), because there is no external clock.
            config.digital_frontend.pdm_preprocessing.clock_direction = 1
            config.digital_frontend.pdm_preprocessing.clock_edge = 0
            # Xylo clock frequency for PDM sampling can be influenced here.
            config.debug.sdm_clock_ratio = 23
            config.digital_frontend.filter_bank.dn_enable = True
            config.digital_frontend.filter_bank.use_global_iaf_threshold = True

            # TODO: Identify what parameters can be changed by passing a config
            # # - Change counter threshold
            # if change_count is not None:
            #     if change_count < 0:
            #         raise ValueError(
            #             f"{change_count} is negative. Must be non-negative values."
            #         )
            #     config = hdu.config_afe_channel_thresholds(config, change_count)

            # # - Amplify input volume
            # config = hdu.config_lna_amplification(config, level=amplify_level)

            # # - Set up hibernation mode
            # if hibernation_mode:
            #     config = hdu.config_AFE_hibernation(config)
            #     config.aer_2_saer.hibernation.mode = 2
            #     config.aer_2_saer.hibernation.reset = 1

        # - Apply configuration
        self.apply_config_blocking(config)
        # Once the configuration is set, the chip is already recording input.
        # But we want to define a clear time window in which we record, so we start the stopwatch to obtain event timesteps relative to this moment.
        # And also throw away any events we have received until now.
        # FIXME
        # stopwatch.start()
        # sink.clear_events()
        self._config = config

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
        Return the version and revision numbers of the connected XyloAudio 3 chip

        Returns:
            (int, int): version, revision
        """
        return (self._chip_version, self._chip_revision)

    def save_config(self, filename):
        """
        Save an AFE configuration to disk in JSON format

        Args:
            filename (str): The filename to write to
        """
        save_afe_config(self._config, filename)

    def __del__(self):
        """
        Delete the AFESamna object and reset the HDK.
        """
        self._read_graph.stop()
        self._afe_read_graph.stop()
        self._write_graph.stop()

        # - Reset the HDK to clean up
        self._device.reset_board_soft()

    def apply_config_blocking(self, config):
        self._device.apply_configuration(config)

        # Communication with the device is asynchronous, so we don't know when the configuration is finished.
        # Normally this is not a problem, because events are guaranteed to be sent in order.
        # But for this script we want to let Xylo run for a specific amount of time, so we need to synchronise somehow,
        # the easiest way is to read a register and wait for the response.

        # TODO - update this with rockpool code
        # source.write([samna.xyloAudio3.event.ReadRegisterValue(address=0)])
        # ready = False
        # while not ready:
        #     events = sink.get_events_blocking(timeout=1000)
        #     for ev in events:
        #         if isinstance(ev, samna.xyloAudio3.event.RegisterValue) and ev.address == 0:
        #             ready=True


def load_afe_config(filename: str) -> XyloConfiguration:
    """
    Read a Xylo configuration from disk in JSON format

    Args:
        filename (str): The filename to read from

    Returns:
        `.XyloConfiguration`: The configuration loaded from disk
    """
    # FIXME: Currently we are using full XyloConfig, change it to AFEconfig
    # - Create a new config object
    conf = XyloConfiguration()

    # - Read the configuration from file
    with open(filename) as f:
        conf.from_json(f.read())

    # - Return the configuration
    return conf


def save_afe_config(config: XyloConfiguration, filename: str) -> None:
    """
    Save a Xylo configuration to disk in JSON format

    Args:
        config (XyloConfiguration): The configuration to write
        filename (str): The filename to write to
    """
    # FIXME: Currently we are using full XyloConfig, change it to AFEconfig
    with open(filename, "w") as f:
        f.write(config.to_json())
