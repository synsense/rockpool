"""
Provides :py:class:`.AFESamnaPDM`
"""

import numpy as np
import samna
import time
import copy

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

XyloAudio3HDK = samna.xyloAudio3.XyloAudio3TestBoard
XyloAudio3ReadBuffer = samna.BasicSinkNode_xylo_audio3_event_output_event
XyloAudio3WriteBuffer = samna.BasicSourceNode_xylo_audio3_event_input_event


# - Typing
from typing import Optional, Union, List, Tuple
from warnings import warn

__all__ = ["XyloSamnaPDM"]


class XyloSamnaPDM(Module):
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
        output_mode: str = "Spike",
        power_frequency: Optional[float] = 100.0,
        dn_active: bool = True,
        hibernation_mode: bool = False,
        *args,
        **kwargs,
    ):
        """
        Instantiate a Module with Xylo dev-kit backend
        This module uses PdmEvents as input source. It bypass the microphone, feeding the input data (PdmEvents) to the filter bank.
        The output responses are both the generated input spikes that will be fed into the SNN core and the result of the SNN core processing.

        Args:
            device (XyloAudio3HDK): An opened `samna` device to a XyloAudio 3 dev kit.
            config (XyloConfiguration): A Xylo configuration from `samna`.
            dt (float): The simulation time-step to use for this Module. Default: 1024e-6.
            output_mode (str): The readout mode for the Xylo device. This must be one of ``["Spike", "Isyn", "Vmem"]``. Default: "Spike", return events from the output layer.
            power_frequency (float): The frequency of power measurement in Hz. Default: 100.0Hz.
            dn_active (bool): If True, divisive normalization will be used. Defaults to True.
            hibernation_mode (bool): If True, hibernation mode will be switched on, which only outputs events if it receives inputs above a threshold. Defaults to False.


        Raises:
            `ValueError`: If ``device`` is not set. ``device`` must be a ``XyloAudio3HDK``.
            `ValueError`: If ``output_mode`` is not ``Spike``, ``Vmem`` or ``Isyn``.
            `ValueError`: If ``operation_mode`` is ``RealTime``. Valid options are ``Manual`` or ``AcceleratedTime``.

        """

        # - Check input arguments
        if device is None:
            raise ValueError("`device` must be a valid, opened Xylo HDK device.")

        # - Check output mode specification
        if output_mode not in ["Spike", "Vmem", "Isyn"]:
            raise ValueError(
                f'{output_mode} is not supported. Must be one of `["Spike", "Vmem", "Isyn"]`.'
            )

        # - Get a default configuration
        if config is None:
            config = samna.xyloAudio3.configuration.XyloConfiguration()

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
        self._sleep_time = 0e-3
        """ float: Post-stimulation sleep time in seconds """

        # - Configure parameters for PdmEvents as input source
        config.digital_frontend.filter_bank.dn_enable = dn_active
        config.digital_frontend.pdm_preprocessing.clock_direction = 0
        config.digital_frontend.pdm_preprocessing.clock_edge = 0
        config.digital_frontend.hibernation_mode_enable = hibernation_mode
        config.digital_frontend.filter_bank.use_global_iaf_threshold = 1

        if config.operation_mode == samna.xyloAudio3.OperationMode.AcceleratedTime:
            config.debug.always_update_omp_stat = True

        time.sleep(self._sleep_time)

        # - For AFESamnaPDM, operation mode can be either manual or accelerated time
        if config.operation_mode == samna.xyloAudio3.OperationMode.RealTime:
            raise ValueError(
                "`operation_mode` can't be RealTime for XyloSamnaPDM. Options are Manual or AcceleratedTime."
            )

        # - Store the power frequency
        self._power_frequency: float = power_frequency
        """ float: Frequency of power monitoring, in Hz """

        # - Keep a registry of the current recording mode, to save unnecessary reconfiguration
        self._last_record_mode: Optional[bool] = None
        """ bool: The most recent (and assumed still valid) recording mode """

        # - Set power measurement module
        (
            self._power_buf,
            self._power_monitor,
            self._stopwatch,
        ) = hdkutils.set_power_measurement(self._device, self._power_frequency)

        # - Store the SNN core configuration
        self._config: Union[
            XyloConfiguration, SimulationParameter
        ] = SimulationParameter(shape=(), init_func=lambda _: config)
        """ `.XyloConfiguration`: The HDK configuration applied to the Xylo module """

        # - Apply configuration on the board
        hdkutils.apply_configuration(self._device, self._config)

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
        if self._power_monitor:
            self._power_monitor.stop_auto_power_measurement()

        if self._stopwatch:
            self._stopwatch.stop()

        # - Reset the HDK to clean up
        hdkutils.reset_board_blocking(
            self._device, self._read_buffer, self._write_buffer
        )

    def reset_state(self) -> "XyloSamnaPDM":
        # - Reset neuron and synapse state on Xylo
        # - To reset Samna and Firmware, we need to send a configuration with different operation mode
        new_config = samna.xyloAudio3.configuration.XyloConfiguration()
        # -- Guarantee a diferent operation mode, sine XyloSamna does not use Recording mode
        new_config.operation_mode = samna.xyloAudio3.OperationMode.Recording
        hdkutils.apply_configuration(self._device, new_config)
        # - Reapply the user defined configuration
        hdkutils.apply_configuration_blocking(
            self._device, self._config, self._read_buffer, self._write_buffer
        )
        return self

    def evolve(
        self,
        input: np.ndarray,
        record: bool = False,
        record_power: bool = False,
        read_timeout: float = 5.0,
        flip_and_encode: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        Evolve a network on the XyloAudio 3 HDK in either single-step manual mode or accelerated time mode.
        For debug purposes only. Uses 'samna.xylo.OperationMode.Manual' or 'samna.xylo.OperationMode.AcceleratedTime' in samna.

        Sends a series of events to the Xylo HDK, evolves the network over the input events, and returns the output events produced during the input period.

        Args:
            input (np.ndarray): A vector ``(Tpdm, 1)`` with a PDM-encoded audio signal, ``1`` or ``0``. The PDM clock is always 1.5625 MHz. 32 PDM samples correspond to one audio sample passed to the band-pass filterbank (i.e. 48.828125 kHz). The network ``dt`` is independent of this sampling rate, but should be an even divisor of 48.828125 MHz (e.g. 1024 us).
            record (bool): Record and return all internal state of the neurons and synapses on Xylo. Default: ``False``, do not record internal state.
            record_power (bool): Iff ``True``, record the power consumption during each evolve.
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

        if record != self._last_record_mode:
            self._config.debug.debug_status_update_enable = record
            self._last_record_mode = record
            hdkutils.apply_configuration(self._device, self._config)

        if record_power:
            # clear the record buffer
            self._power_buf.get_events()

        # - Check again if operation mode is either manual or accelerated time
        if self._config.operation_mode == samna.xyloAudio3.OperationMode.RealTime:
            raise ValueError(
                "`operation_mode` can't be RealTime for XyloSamnaPDM. Options are Manual or AcceleratedTime."
            )

        # - Get some information about the network size
        Nin, Nhidden, Nout = self.shape

        # - Calculate sample rates and `dt`-length window
        PDM_sample_rate = 1562500
        PDM_samples_per_dt = PDM_sample_rate * self._dt

        # - Check window length, should be integer
        if not np.allclose(PDM_samples_per_dt % 1, 0.0):
            warn(
                f"Non-integer number of PDM samples per network `dt`. Network evolution will not be accurate. PDM_samples_per_dt: {PDM_samples_per_dt}; PDM_sample_rate: {PDM_sample_rate} Hz."
            )
        PDM_samples_per_dt = int(PDM_samples_per_dt)

        # - Compute number of `dt` time-steps
        num_dt = np.size(input) // PDM_samples_per_dt

        # - Check input length
        if np.size(input) % PDM_samples_per_dt > 0:
            warn(
                f"Input PDM audio trace does not fit evenly into `dt`. Audio will be trimmed at the end of the sample. input size: {np.size(input)}; PDM_samples_per_dt: {PDM_samples_per_dt}."
            )

        # - Bin samples into `dt`-length windows and trim
        input_raster = np.reshape(
            input[: num_dt * PDM_samples_per_dt], [-1, PDM_samples_per_dt]
        )

        flip_and_encode_size = None
        if flip_and_encode:
            # -- Revert and repeat the input signal in the beginning to avoid boundary effects
            flip_and_encode_size = np.shape(input_raster)[0]
            __input_rev = np.flip(input_raster, axis=0)
            input_raster = np.concatenate((__input_rev, input_raster), axis=0)

        # - Initialise lists for recording state
        input_spikes = []
        vmem_ts = []
        isyn_ts = []
        isyn2_ts = []
        vmem_out_ts = []
        isyn_out_ts = []
        spikes_ts = []
        output_ts = []

        # - Send PDM data and extract activity
        for input_sample in tqdm(input_raster):
            # - Send PDM events for this dt

            pdm_events = [
                samna.xyloAudio3.event.AfeSample(data=int(i)) for i in input_sample
            ]
            self._write_buffer.write(pdm_events)

            # - Read input spikes
            if record:
                input_spikes.append(
                    hdkutils.read_input_spikes(self._read_buffer, self._write_buffer)[
                        :Nin
                    ]
                )

            # - Trigger processing: if manual mode, advance time step manually
            if self._config.operation_mode == samna.xyloAudio3.OperationMode.Manual:
                hdkutils.advance_time_step(self._write_buffer)
                # - Wait until xylo has finished the simulation of this time step
                t_start = time.time()
                is_timeout = False
                while not hdkutils.is_xylo_ready(self._read_buffer, self._write_buffer):
                    if time.time() - t_start > read_timeout:
                        is_timeout = True
                        break

                if is_timeout:
                    raise TimeoutError

            else:
                # - Trigger processing: in accelerated time, time steps are advanced automatically
                # given the time step on the spike. AfeSamples do not have timestep so sending
                # trigger processing to do so.
                self._write_buffer.write([samna.xyloAudio3.event.TriggerProcessing()])

            # - Read all synapse and neuron states for this time step
            if record:
                this_state = hdkutils.read_neuron_synapse_state(
                    self._read_buffer, self._write_buffer, Nin, Nhidden, Nout
                )
                vmem_ts.append(this_state.V_mem_hid)
                isyn_ts.append(this_state.I_syn_hid)
                isyn2_ts.append(this_state.I_syn2_hid)
                vmem_out_ts.append(this_state.V_mem_out)
                isyn_out_ts.append(this_state.I_syn_out)
                spikes_ts.append(this_state.Spikes_hid)

            # - Read the output event register
            output_events = hdkutils.read_output_events(
                self._read_buffer, self._write_buffer
            )[:Nout]
            output_ts.append(output_events)

        if record:
            # - Build a recorded state dictionary
            rec_dict = {
                "Spikes_in": np.stack(input_spikes),
                "Vmem": np.array(vmem_ts),
                "Isyn": np.array(isyn_ts),
                "Isyn2": np.array(isyn2_ts),
                "Spikes": np.array(spikes_ts),
                "Vmem_out": np.array(vmem_out_ts),
                "Isyn_out": np.array(isyn_out_ts),
                # "times": np.arange(start_timestep, final_timestep + 1),
            }
        else:
            rec_dict = {}

        if record_power:
            # - Get all recent power events from the power measurement
            ps = self._power_buf.get_events()

            # - Separate out power meaurement events by channel
            channels = samna.xyloAudio3.MeasurementChannels
            io_power = np.array([e.value for e in ps if e.channel == int(channels.Io)])
            analog_power = np.array(
                [e.value for e in ps if e.channel == int(channels.AnalogLogic)]
            )
            digital_power = np.array(
                [e.value for e in ps if e.channel == int(channels.DigitalLogic)]
            )

            rec_dict.update(
                {
                    "io_power": io_power,
                    "analog_power": analog_power,
                    "digital_power": digital_power,
                }
            )

        output_ts = np.array(output_ts)

        if flip_and_encode:
            # Trim the part of the signal coresponding to __input_rev (which was added to avoid boundary effects)
            output_ts = output_ts[flip_and_encode_size:]

            # # Trim recordings
            rec_dict = {k: v[flip_and_encode_size:] for k, v in rec_dict.items()}

        # - Return the output spikes, the (empty) new state dictionary, and the recorded state dictionary
        return output_ts, {}, rec_dict
