"""
Provides :py:class:`.XyloSamnaPDM`
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
        snn_config: XyloConfiguration = None,
        pdm_config: PdmPreprocessingConfig = None,
        dfe_config: DigitalFrontendConfig = None,
        register_config: dict = None,
        dt: float = 1024e-6,
        output_mode: str = "Spike",
        record: Optional[bool] = False,
        power_frequency: Optional[float] = 5.0,
        dn_active: bool = True,
        *args,
        **kwargs,
    ):
        """
        Instantiate a Module with Xylo dev-kit backend

        Args:
            device (XyloAudio3HDK): An opened `samna` device to a Xylo dev kit
            config (XyloConfiguration): A Xylo configuration from `samna`
            dt (float): The simulation time-step to use for this Module
            output_mode (str): The readout mode for the Xylo device. This must be one of ``["Spike", "Isyn", "Vmem"]``. Default: "Spike", return events from the output layer.
            record (bool): Record and return all internal state of the neurons and synapses on Xylo. Default: ``False``, do not record internal state.
            power_frequency (float): The frequency of power measurement. Default: 5.0

        Raises:
            `Warning`: For XyloSamnaPDM ``config.input_source`` must be set to ``PdmEvents``
        """

        # - Check input arguments
        if device is None:
            raise ValueError("`device` must be a valid, opened Xylo HDK device.")

        # - Check output mode specification
        if output_mode not in ["Spike", "Vmem", "Isyn"]:
            raise ValueError(
                f'{output_mode} is not supported. Must be one of `["Spike", "Vmem", "Isyn"]`.'
            )
        self._output_mode = output_mode

        # - Get a default configuration
        if snn_config is None:
            snn_config = samna.xyloAudio3.configuration.XyloConfiguration()

        if snn_config.input_source is not samna.xyloAudio3.InputSource.PdmEvents:
            warn(
                "Changing `input_source` configuration to PdmEvents. For real monitoring or spiking signals use `XyloMonitor` or `XyloSamna`."
            )
            snn_config.input_source = samna.xyloAudio3.InputSource.PdmEvents

        # - Get the network shape
        Nin, _ = np.shape(snn_config.input.weights)
        Nhidden, _ = np.shape(snn_config.hidden.weights)

        _, Nout = np.shape(snn_config.readout.weights)

        # - Initialise the superclass
        super().__init__(
            shape=(Nin, Nhidden, Nout), spiking_input=True, spiking_output=True
        )

        # HACK record tag was moved to the constructor of the class instead of evolve
        # updating the configuration in evolve is leading to erratic behavior because of a mismatch between samna and firmware
        if record:
            # - Switch on reporting of input spike register pointer value
            snn_config.debug.debug_status_update_enable = 1
        else:
            snn_config.debug.debug_status_update_enable = 0

        # - Store record option
        self._record: Optional[bool] = record
        """ bool: Record and return all internal state of the neurons and synapses on Xylo """

        # - Store the device
        self._device: XyloAudio3HDK = device
        """ `.XyloHDK`: The Xylo HDK used by this module """

        # - Register buffers to read and write events
        self._read_buffer = hdkutils.new_xylo_read_buffer(device)
        """ `.XyloAudio3ReadBuffer`: The read buffer for the connected HDK """

        self._write_buffer = hdkutils.new_xylo_write_buffer(device)
        """ `.XyloAudio3WriteBuffer`: The write buffer for the connected HDK """

        # - Store the timestep
        self.dt: Union[float, SimulationParameter] = dt
        """ float: Simulation time-step of the module, in seconds """

        # - Sleep time post sending spikes on each time-step, in manual mode
        self._sleep_time = 0e-3
        """ float: Post-stimulation sleep time in seconds """

        # - TODO: AFAIK these parameters are necesary to run with PDM configurations
        # there is a discussion need in order to understand if we will only overwrite parameters
        # or assume that the user knows what parameters he needs.
        snn_config.debug.sdm_clock_ratio = 48
        snn_config.digital_frontend.filter_bank.dn_enable = dn_active
        snn_config.digital_frontend.hibernation_mode_enable = 0
        snn_config.digital_frontend.filter_bank.use_global_iaf_threshold = 1
        snn_config.digital_frontend.pdm_preprocessing.clock_direction = 0
        snn_config.digital_frontend.pdm_preprocessing.clock_edge = 0

        if snn_config.operation_mode == samna.xyloAudio3.OperationMode.AcceleratedTime:
            snn_config.debug.always_update_omp_stat = True

        # - Store the SNN core configuration (and apply it)
        time.sleep(self._sleep_time)

        # - For XyloSamnaPDM, operation mode can be either manual or accelerated time
        if snn_config.operation_mode == samna.xyloAudio3.OperationMode.RealTime:
            raise ValueError(
                "`operation_mode` can't be RealTime for XyloSamnaPDM. Options are Manual or AcceleratedTime."
            )

        self.config: Union[
            XyloConfiguration, SimulationParameter
        ] = SimulationParameter(shape=(), init_func=lambda _: snn_config)
        """ `.XyloConfiguration`: The HDK configuration applied to the Xylo module """

        self.snn_config = snn_config

        # - Keep a registry of the current recording mode, to save unnecessary reconfiguration
        self._last_record_mode: Optional[bool] = None
        """ bool: The most recent (and assumed still valid) recording mode """

        # - Store the timestep
        self.dt: Union[float, SimulationParameter] = dt
        """ float: Simulation time-step of the module, in seconds """

        # - Apply configuration on the board
        hdkutils.apply_configuration(self._device, self.snn_config)

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

    def reset_state(self) -> "XyloSamnaPDM":
        # - Reset neuron and synapse state on Xylo
        # -- Copy values of configuration
        operation_mode = copy.copy(self.snn_config.operation_mode)
        status_update = copy.copy(self.snn_config.debug.debug_status_update_enable)

        # - To reset Samna and Firmware, we need to send a configuration with different operation mode
        # -- Operation mode can not be RealTime in XyloSamnaPDM
        self._config.operation_mode = (
            samna.xyloAudio3.OperationMode.Manual
            if self._config.operation_mode
            == samna.xyloAudio3.OperationMode.AcceleratedTime
            else samna.xyloAudio3.OperationMode.AcceleratedTime
        )
        self.snn_config.debug.debug_status_update_enable = 0
        hdkutils.apply_configuration(self._device, self._config)

        # - Reapply the user defined configuration
        self._config.operation_mode = operation_mode
        self.snn_config.debug.debug_status_update_enable = status_update
        hdkutils.apply_configuration(self._device, self._config)
        return self

    def evolve(
        self,
        input: np.ndarray,
        record: bool = False,
        read_timeout: float = 5.0,
        *args,
        **kwargs,
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        Evolve a network on the XyloAudio 3 HDK in either single-step manual mode or accelerated time mode.
        For debug purposes only. Uses 'samna.xylo.OperationMode.Manual' or 'samna.xylo.OperationMode.AcceleratedTime' in samna.

        Sends a series of events to the Xylo HDK, evolves the network over the input events, and returns the output events produced during the input period.

        Args:
            input (np.ndarray): A vector ``(Tpdm, 1)`` with a PDM-encoded audio signal, ``1`` or ``0``. The PDM clock is always 1.5625 MHz. 32 PDM samples correspond to one audio sample passed to the band-pass filterbank (i.e. 48.828125 kHz). The network ``dt`` is independent of this sampling rate, but should be an even divisor of 48.828125 MHz (e.g. 1024 us).
            record (bool): Deprecated parameter. Please use ``record`` from the class initialization.
            read_timeout (Optional[float]): Set an explicit read timeout for the entire simulation time. This should be sufficient for the simulation to complete, and for data to be returned. Default: ``None``, set a reasonable default timeout.

        Returns:
            (np.ndarray, dict, dict): ``output``, ``new_state``, ``record_dict``.
            ``output`` is a raster ``(T, Nout)``, containing events for each channel in each time bin. Time bins in ``output`` correspond to the time bins in ``input``.
            ``new_state`` is an empty dictionary. The Xylo HDK does not permit querying or setting state.
            ``record_dict`` is a dictionary containing recorded internal state of Xylo during evolution, if the ``record`` argument is ``True``. Otherwise this is an empty dictionary.

        Raises:
            `TimeoutError`: If reading data times out during the evolution. An explicity timeout can be set using the `read_timeout` argument.
        """

        # - Get some information about the network size
        Nin, Nhidden, Nout = self.shape

        # - Check again if operation mode is either manual or accelerated time
        if self.snn_config.operation_mode == samna.xyloAudio3.OperationMode.RealTime:
            raise ValueError(
                "`operation_mode` can't be RealTime for XyloSamnaPDM. Options are Manual or AcceleratedTime."
            )

        # HACK record is not working inside evolve and was transferred to the class initialization
        if record:
            warn(
                "`record` is now a parameter in the class initialization and its behavior is not updated in the `evolve` method."
            )
        record = self._record

        # - Calculate sample rates and `dt`-length window
        PDM_sample_rate = 1562500
        PDM_samples_per_dt = PDM_sample_rate * self.dt

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

        # - Initialise lists for recording state
        input_spikes = []
        vmem_ts = []
        isyn_ts = []
        isyn2_ts = []
        vmem_out_ts = []
        isyn_out_ts = []
        spikes_ts = []
        output_ts = []

        triggered = False
        # - Send PDM data and extract activity
        for input_sample in tqdm(input_raster):
            # - Send PDM events for this dt

            pdm_events = [
                samna.xyloAudio3.event.AfeSample(data=int(i)) for i in input_sample
            ]
            self._write_buffer.write(pdm_events)
            time.sleep(0.5)

            # - Read input spikes
            if record:
                input_spikes.append(
                    hdkutils.read_input_spikes(self._read_buffer, self._write_buffer)[
                        :Nin
                    ]
                )

            # - Trigger processing: if manual mode, advance time step manually
            if self.snn_config.operation_mode == samna.xyloAudio3.OperationMode.Manual:
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

        # - Return the output spikes, the (empty) new state dictionary, and the recorded state dictionary
        return np.array(output_ts), {}, rec_dict
