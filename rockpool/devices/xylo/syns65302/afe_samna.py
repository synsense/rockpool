"""
samna-backed module for interfacing with the XyloAudio 3 AudioFrontEnd HW module
"""

import numpy as np
import time
import math
from typing import Union, Any, Tuple, Optional

import samna
from samna.xyloAudio3.configuration import XyloConfiguration

from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter
from rockpool import TSEvent
from rockpool.typehints import P_float

from . import xa3_devkit_utils as hdu
from .xa3_devkit_utils import XyloAudio3HDK


# - Configure exports
__all__ = ["AFESamna", "load_afe_config", "save_afe_config"]

Default_Main_Clock_Rate = 50.0  # 50 MHz
Pdm_Clock_Rate = 1.56  # MHz


class AFESamna(Module):
    """
    Interface to the Audio Front-End module on a XyloAudio 3 HDK

    This module uses ``samna`` to interface to the Audio FrontEnd (AFE) hardware on a XyloAudio 3 HDK. It permits recording from the AFE hardware.

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

    def __init__(
        self,
        device: XyloAudio3HDK,
        config: Optional[XyloConfiguration] = None,
        dt: float = 1e-3,
        main_clk_rate: float = Default_Main_Clock_Rate,
        hibernation_mode: bool = False,
        divisive_norm: bool = True,
        sdm_clock_ratio: int = 15,
        *args,
        **kwargs,
    ):
        """
        Instantiate an AudioFrontEnd module, via a samna backend.

        This module runs on the XyloAudio 3 device, capturing data with the microphone, and outputing the generated input spikes that will be processed by the SNN core.
        Operation mode is set to ``Recording`` in this module.

        Args:
            device (XyloA3HDK): A connected XyloAudio 3 HDK device.
            config (XyloConfiguraration): A Xylo configuration from `samna`
            dt (float): The desired spike time resolution in seconds. Default: 0.001s.
            hibernation_mode (bool): If True, hibernation mode will be switched on, which only outputs events if it receives inputs above a threshold. Default: False.
            divisive_norm (bool): If True, divisive normalization will be switched on. Default: True.
        """
        # - Check input arguments
        if device is None:
            raise ValueError("`device` must be a valid, opened XyloAudio 3 HDK.")

        # - Update board configuration
        board_config = samna.xyloAudio3.XyloAudio3TestBoardDefaultConfig()
        board_config.main_clock_frequency = int(main_clk_rate * 1e6)
        device.reset_board_soft(board_config)

        self._stopwatch = device.get_stop_watch()
        """ `stopwatch`: The Xylo HDK control for timesteps """

        # - Calculate tr_wrap (clock in Hz and dt in seconds)
        tr_wrap = main_clk_rate * 1e6 * dt

        # - Get a default configuration
        if config is None:
            config = samna.xyloAudio3.configuration.XyloConfiguration()

        # - Set operation mode to Recording
        config.operation_mode = samna.xyloAudio3.OperationMode.Recording

        # - Determine how many output channels we have
        _, Nout = np.shape(config.readout.weights)

        # - Initialise the superclass
        super().__init__(shape=(0, Nout), spiking_input=True, spiking_output=True)

        # - Store the HDK device node
        self._device = device

        # - Store the dt parameter
        self.dt: P_float = SimulationParameter(dt)

        # - Register buffers to read and write events, monitor state
        self._afe_read_buffer = hdu.new_xylo_read_buffer(self._device)
        self._afe_write_buffer = hdu.new_xylo_write_buffer(self._device)

        config.debug.use_timestamps = False
        config.time_resolution_wrap = int(tr_wrap)
        # -- Choose PDM as input source.
        config.input_source = samna.xyloAudio3.InputSource.DigitalMicrophone
        # -- We need to set `clock_direction` to 1 (Xylo output), because there is no external clock.
        config.digital_frontend.pdm_preprocessing.clock_direction = 1
        config.digital_frontend.pdm_preprocessing.clock_edge = 0
        # -- Xylo clock frequency for PDM sampling
        # In theory, the calculation for SDM clock should use: int(main_clk_rate / Pdm_Clock_Rate / 2 - 1)
        config.debug.sdm_clock_ratio = (
            sdm_clock_ratio  # int(main_clk_rate / Pdm_Clock_Rate / 2 - 1)
        )
        config.digital_frontend.filter_bank.use_global_iaf_threshold = True

        # - Set hibernation mode
        config.enable_hibernation_mode = hibernation_mode

        # - Set divisive normalization
        config.digital_frontend.filter_bank.dn_enable = divisive_norm

        # - Store the configuration
        self._config: Union[
            XyloConfiguration, SimulationParameter
        ] = SimulationParameter(shape=(), init_func=lambda _: config)
        """ `.XyloConfiguration`: The HDK configuration applied to the Xylo module """

        # - Apply configuration
        hdu.apply_configuration_blocking(
            self._device, config, self._afe_read_buffer, self._afe_write_buffer
        )

    def evolve(
        self, input_data, record: bool = False, flip_and_encode: bool = False
    ) -> Tuple[Any, Any, Any]:
        """
        Use the AFE HW module to record live audio and return as encoded events

        Args:
            input_data (np.ndarray): An array ``[0, T, 0]``, specifying the number of time-steps to record.
            record (bool): Record and return all internal state of the neurons and synapses on Xylo. Default: ``False``, do not record internal state.
            flip_and_encode (bool): Determine if flip-and-encode fix should be applied to the input data. When applied, the input data will be flipped on axis=0 and concatenated to the begin of the original input data. Note that input data will have its size doubled.

        Returns:
            (np.ndarray, dict, dict) output_events, {}, {}
        """

        # - Handle auto batching
        input_data, _ = self._auto_batch(input_data)

        # - Calculate the duration of the recording
        duration = input_data.shape[1] * self.dt

        flip_and_encode_size = None
        if flip_and_encode:
            # -- Revert and repeat the input signal in the beginning to avoid boundary effects
            flip_and_encode_size = np.shape(input_data)[0]
            __input_rev = np.flip(input_data, axis=0)
            input_data = np.concatenate((__input_rev, input_data), axis=0)

        # At this point, the chip is already recording input.
        # But we want to define a clear time window in which we record, so we start the stopwatch to obtain event timesteps relative to this moment.
        # And also throw away any events we have received until now.
        self._stopwatch.start()
        self._afe_read_buffer.clear_events()

        # - Record events
        # - Wait for all the events received during the read timeout
        readout_events = []
        read_until = time.time() + duration

        # -- We still need the loop because there is no function in samna that wait for a specific ammount of time and return all events
        while (now := time.time()) < read_until:
            remaining_time = read_until - now
            readout_events += self._afe_read_buffer.get_events_blocking(
                math.ceil(remaining_time * 1000)
            )

        if len(readout_events) == 0:
            message = f"No event received in {duration}s."
            raise TimeoutError(message)

        last_timestep = readout_events[-1].timestep
        events = [
            (e.timestep, e.neuron_id)
            for e in readout_events
            if isinstance(e, samna.xyloAudio3.event.Spike)
            and e.timestep <= last_timestep
        ]

        # - Sort events by time
        if len(events) > 0:
            events = np.stack(events)
            index_array = np.argsort(events[:, 0])

            # - Convert to vectors of timesteps, neuron ids
            timesteps = events[index_array, 0]
            neuron_ids = events[index_array, 1]
        else:
            timesteps = np.zeros(0)
            neuron_ids = np.zeros(0)

        # - Convert to an event raster
        events_ts = TSEvent(
            timesteps,
            neuron_ids,
            t_start=0.0,
            t_stop=last_timestep + 1,
            num_channels=16,
        ).raster(
            1, add_events=True
        )  # the timesteps are given by the spike timesteps

        if record:
            # - Build a recorded state dictionary
            rec_dict = {
                "neuron_ids": np.array(neuron_ids),
                "timesteps": np.array(timesteps),
            }
        else:
            rec_dict = {}

        # - Apply a default configuration to stop recording mode
        self._device.get_model().apply_configuration(
            samna.xyloAudio3.configuration.XyloConfiguration()
        )
        self._stopwatch.stop()

        if flip_and_encode:
            # - Trim the part of the signal coresponding to __input_rev (which was added to avoid boundary effects)
            events_ts = events_ts[flip_and_encode_size:, :]

            # - Trim recordings
            rec_dict = {k: v[flip_and_encode_size:, :] for k, v in rec_dict.items()}

        # - Return output, state, record dict
        return events_ts, self.state(), rec_dict

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
        # - Reset the HDK to clean up
        self._device.reset_board_soft()


def load_afe_config(filename: str) -> XyloConfiguration:
    """
    Read a Xylo configuration from disk in JSON format

    Args:
        filename (str): The filename to read from

    Returns:
        `.XyloConfiguration`: The configuration loaded from disk
    """
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
    with open(filename, "w") as f:
        f.write(config.to_json())
