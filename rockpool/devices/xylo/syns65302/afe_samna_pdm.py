"""
Provides :py:class:`.AFESamnaPDM`
"""

import numpy as np
import samna
import time
import math

from rockpool import TSEvent

try:
    from tqdm.autonotebook import tqdm
except:
    tqdm = lambda x: x

from samna.xyloAudio3.configuration import (
    XyloConfiguration,
    # PdmPreprocessingConfig,
    # DigitalFrontendConfig,
)

PdmPreprocessingConfig = None
DigitalFrontendConfig = None

from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter
from . import xa3_devkit_utils as hdkutils
from .afe import params

# - Typing
from typing import Union, Tuple
from warnings import warn


XyloAudio3HDK = samna.xyloAudio3.XyloAudio3TestBoard
XyloAudio3ReadBuffer = samna.BasicSinkNode_xylo_audio3_event_output_event
XyloAudio3WriteBuffer = samna.BasicSourceNode_xylo_audio3_event_input_event

__all__ = ["AFESamnaPDM"]


class AFESamnaPDM(Module):
    """
    A spiking neuron :py:class:`.Module` backed by the Xylo hardware, via `samna`, with PDM input.
    """

    def __init__(
        self,
        device: XyloAudio3HDK,
        config: XyloConfiguration = None,
        # pdm_config: PdmPreprocessingConfig = None,
        # dfe_config: DigitalFrontendConfig = None,
        dt: float = 1024e-6,
        main_clk_rate: int = params.SYSTEM_CLOCK_RATE,
        dn_active: bool = True,
        hibernation_mode: bool = False,
        *args,
        **kwargs,
    ):
        """
        Instantiate a Module with Xylo dev-kit backend
        This module uses PdmEvents as input source. It bypass the microphone, feeding the input data (PdmEvents) to the filter bank.
        The output responses are the generated input spikes that will be fed into the SNN core.

        Args:
            device (XyloAudio3HDK): An opened `samna` device to a XyloAudio 3 dev kit
            config (XyloConfiguration): A Xylo configuration from `samna`
            dt (float): The simulation time-step to use for this Module
            main_clk_rate (int): The main clock rate of Xylo, in Hz. Default: 50.000.000 Hz.
            dn_active (bool): If True, divisive normalization will be used. Defaults to True.
            hibernation_mode (bool): If True, hibernation mode will be switched on, which only outputs events if it receives inputs above a threshold. Defaults to False.


        Raises:
            `ValueError`: If ``device`` is not set. ``device`` must be a ``XyloAudio3HDK``.
            `ValueError`: If ``operation_mode`` is ``RealTime``. Valid options are ``Manual`` or ``AcceleratedTime``.
        """

        # - Check input arguments
        if device is None:
            raise ValueError("`device` must be a valid, opened Xylo HDK device.")

        # - Configure master clock and communication bus clocks
        hdkutils.set_xylo_core_clock_freq(device, float(main_clk_rate / 1e6))

        self._stopwatch = device.get_stop_watch()
        """ `stopwatch`: The Xylo HDK control for timesteps """

        # - Calculate tr_wrap (clock in Hz and dt in seconds)
        tr_wrap = main_clk_rate * dt

        # - Get a default configuration
        if config is None:
            config = samna.xyloAudio3.configuration.XyloConfiguration()

        # - Set operation mode to Recording
        config.operation_mode = samna.xyloAudio3.OperationMode.Recording

        # - Input source must be PdmEvents
        config.input_source = samna.xyloAudio3.InputSource.PdmEvents

        # - Get the network shape
        Nin, _ = np.shape(config.input.weights)
        Nhidden, _ = np.shape(config.hidden.weights)
        _, Nout = np.shape(config.readout.weights)

        # - Initialise the superclass
        super().__init__(
            shape=(Nin, Nhidden, Nout), spiking_input=True, spiking_output=True
        )

        # - Store the device
        self._device: XyloAudio3HDK = device
        """ `.XyloAudio3HDK`: The Xylo HDK used by this module """

        # - Register buffers to read and write events
        self._read_buffer: XyloAudio3ReadBuffer = hdkutils.new_xylo_read_buffer(device)
        """ `.XyloAudio3ReadBuffer`: The read buffer for the connected HDK """
        self._write_buffer: XyloAudio3WriteBuffer = hdkutils.new_xylo_write_buffer(
            device
        )
        """ `.XyloAudio3WriteBuffer`: The write buffer for the connected HDK """

        # - Store the timestep
        self._dt: Union[float, SimulationParameter] = dt
        """ float: Simulation time-step of the module, in seconds """

        # - Sleep time post sending spikes on each time-step, in manual mode
        self._sleep_time: float = 0e-3
        """ float: Post-stimulation sleep time in seconds """

        # - Configure parameters for recording mode
        config.debug.use_timestamps = False
        config.time_resolution_wrap = int(tr_wrap)

        # - Configure parameters for PdmEvents as input source
        config.digital_frontend.filter_bank.dn_enable = dn_active
        config.digital_frontend.pdm_preprocessing.clock_direction = 0
        config.digital_frontend.pdm_preprocessing.clock_edge = 0
        config.digital_frontend.hibernation_mode_enable = hibernation_mode

        # - Store the SNN core configuration
        self._config: Union[
            XyloConfiguration, SimulationParameter
        ] = SimulationParameter(shape=(), init_func=lambda _: config)
        """ `.XyloConfiguration`: The HDK configuration applied to the Xylo module """

        # - Apply configuration on the board
        hdkutils.apply_configuration_blocking(
            self._device, self._config, self._read_buffer, self._write_buffer
        )

    @property
    def config(self):
        # - Return the configuration stored on Xylo HDK
        return self._device.get_model().get_configuration()

    @config.setter
    def config(self, new_config):
        # - Test for a valid configuration
        is_valid, msg = samna.xyloAudio3.validate_configuration(new_config)
        if not is_valid:
            raise ValueError(f"Invalid configuration for the Xylo HDK: {msg}")

        # - Write the configuration to the device
        hdkutils.apply_configuration(self._device, new_config)
        time.sleep(self._sleep_time)

        self._config = new_config

    def __del__(self):
        # - Apply a default configuration to stop recording mode
        hdkutils.apply_configuration(
            self._device, samna.xyloAudio3.configuration.XyloConfiguration()
        )

        # - Reset the HDK to clean up
        hdkutils.reset_board_blocking(
            self._device, self._read_buffer, self._write_buffer
        )

    def reset_state(self) -> "AFESamnaPDM":
        # - Reset neuron and synapse state on Xylo
        # -- To reset Samna and Firmware, we need to send a configuration with different operation mode
        # -- In AFESamnaPDM the operation mode is `Recording`. The default operation mode is `AcceleratedTime`
        hdkutils.apply_configuration(
            self._device, samna.xyloAudio3.configuration.XyloConfiguration()
        )

        # - Reapply the user defined configuration
        hdkutils.apply_configuration(self._device, self._config)
        return self

    def evolve(
        self,
        input: np.ndarray,
        record: bool = False,
        flip_and_encode: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        Use the AudioFrontEnd HW module to process PDM events and return its results as encoded events.

        Sends a series of events to the Xylo HDK and returns the input events that would be used to feed the SNN core.

        Args:
            input (np.ndarray): A vector ``(Tpdm, 1)`` with a PDM-encoded audio signal, ``1`` or ``0``. The PDM clock is always 1.5625 MHz. 32 PDM samples correspond to one audio sample passed to the band-pass filterbank (i.e. 48.828125 kHz). The network ``dt`` is independent of this sampling rate, but should be an even divisor of 48.828125 MHz (e.g. 1024 us).
            record (bool): Record and return all internal state of the neurons and synapses on Xylo. Default: ``False``, do not record internal state.
            read_timeout (Optional[float]): Set an explicit read timeout for the entire simulation time. This should be sufficient for the simulation to complete, and for data to be returned. Default: 5s.
            flip_and_encode (bool): Determine if flip-and-encode should be applied to the input data. When applied, the input data will be flipped on axis=0 and concatenated to the begin of the original input data. Note that input data will have its size doubled.

        Returns:
            (np.ndarray, dict, dict): ``output``, ``new_state``, ``record_dict``.
            ``output`` is a raster ``(T, Nout)``, containing events for each channel in each time bin. Time bins in ``output`` correspond to the time bins in ``input``.
            ``new_state`` is an empty dictionary. The Xylo HDK does not permit querying or setting state.
            ``record_dict`` is a dictionary containing recorded internal state of Xylo during evolution, if the ``record`` argument is ``True``. Otherwise this is an empty dictionary.

        Raises:
            `ValueError`: If ``operation_mode`` is ``RealTime``. Valid options are ``Manual`` or ``AcceleratedTime``.
            `TimeoutError`: If reading data times out during the evolution. An explicity timeout can be set using the `read_timeout` argument.
        """

        # - Calculate sample rates and `dt`-length window
        PDM_samples_per_dt = params.PDM_SAMPLING_RATE * self._dt

        # - Check window length, should be integer
        if not np.allclose(PDM_samples_per_dt % 1, 0.0):
            warn(
                f"Non-integer number of PDM samples per network `dt`. Network evolution will not be accurate. PDM_samples_per_dt: {PDM_samples_per_dt}; PDM_sample_rate: {params.PDM_SAMPLING_RATE/1e6} Hz."
            )
        PDM_samples_per_dt = int(PDM_samples_per_dt)

        # - Check input length
        if np.size(input) % PDM_samples_per_dt > 0:
            warn(
                f"Input PDM audio trace does not fit evenly into `dt`. Audio will be trimmed at the end of the sample. input size: {np.size(input)}; PDM_samples_per_dt: {PDM_samples_per_dt}."
            )

        # - Calculate duration of the input data
        duration = len(input) / params.PDM_SAMPLING_RATE
        margin = duration * 0.1

        # - Compute number of `dt` time-steps
        # -- This is the expected timestep count from the spikes
        num_dt = np.size(input) // PDM_samples_per_dt

        if flip_and_encode:
            # -- Flip the input signal in the beginning to avoid boundary effects
            __input_rev = np.flip(input, axis=0)

            # - Bin the flipped samples into `dt`-length windows and trim
            input_rev_raster = np.reshape(
                __input_rev[: num_dt * PDM_samples_per_dt], [-1, PDM_samples_per_dt]
            )

        # - Bin input samples into `dt`-length windows and trim
        input_raster = np.reshape(
            input[: num_dt * PDM_samples_per_dt], [-1, PDM_samples_per_dt]
        )

        # - Clear the buffer to collect events
        self._read_buffer.clear_events()

        pdm_events = []
        if flip_and_encode:
            # - Add the flipped PDM data
            for input_sample in tqdm(input_rev_raster):
                pdm_events.extend(
                    samna.xyloAudio3.event.AfeSample(data=int(i)) for i in input_sample
                )

            # - Send the flipped data to the chip
            self._write_buffer.write(pdm_events)

            # - Send a read register event to know when all the spikes that can be dropped were received
            self._write_buffer.write(
                [samna.xyloAudio3.event.ReadRegisterValue(address=0)]
            )

        # Start timestep counter, we care about timesteps for the input data only
        self._stopwatch.start()
        pdm_events = []
        # - Add all PDM data at once and extract activity
        for input_sample in tqdm(input_raster):
            pdm_events.extend(
                samna.xyloAudio3.event.AfeSample(data=int(i)) for i in input_sample
            )

        self._write_buffer.write(pdm_events)
        # - Send a read register event to know when all the spikes were received
        self._write_buffer.write([samna.xyloAudio3.event.ReadRegisterValue(address=0)])

        # - Record events
        # -- Wait for all the events received during the read timeout
        readout_events = []
        read_until = time.time() + duration + margin
        if flip_and_encode:
            read_until *= 2

        # -- We still need the loop because there is no function in samna that wait for a specific ammount of time and return all events
        ready = False
        dropped = False
        while (now := time.time()) < read_until and not ready:
            remaining_time = read_until - now
            readout_events += self._read_buffer.get_events_blocking(
                math.ceil(remaining_time)
            )
            for ev in readout_events:
                if (
                    isinstance(ev, samna.xyloAudio3.event.RegisterValue)
                    and ev.address == 0
                ):
                    if flip_and_encode and not dropped:
                        readout_events = []
                        dropped = True
                    else:
                        ready = True
                        readout_events.pop()

        if len(readout_events) == 0:
            message = f"No event received in {duration}s."
            raise TimeoutError(message)

        if not ready:
            message = f"Didnt receive all the spikes in {duration}s"

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
            t_start=timesteps[0],
            t_stop=last_timestep + 1,
            num_channels=16,
        ).raster(
            1, add_events=True
        )  # the timesteps are given by the spike timesteps

        if record:
            # - Build a recorded state dictionary
            rec_dict = {
                "Spikes_in": np.stack(events_ts),
                "neuron_ids": np.array(neuron_ids),
                "timesteps": np.array(timesteps),
            }
        else:
            rec_dict = {}

        self._stopwatch.stop()

        # - Return output, state, record dict
        return events_ts, self.state(), rec_dict
