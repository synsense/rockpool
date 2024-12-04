"""
Provides :py:class:`.XyloMonitor`
"""

# - Samna imports
import samna
from samna.xyloAudio3.configuration import XyloConfiguration

XyloAudio3HDK = samna.xyloAudio3.XyloAudio3TestBoard
from . import xa3_devkit_utils as hdkutils

import time
import math
import numpy as np
from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter

# - Typing
from typing import Optional, Union, Callable, List, Tuple
from warnings import warn

try:
    from tqdm.autonotebook import tqdm, trange
except ModuleNotFoundError:

    def tqdm(wrapped, *args, **kwargs):
        return wrapped

    def trange(obj, *args, **kwargs):
        return range(obj)


# - Configure exports
__all__ = ["XyloMonitor"]

Default_Main_Clock_Rate = 50.0  # 50 MHz
Pdm_Clock_Rate = 1.56


class XyloMonitor(Module):
    """
    A spiking neuron :py:class:`.Module` backed by the XyloAudio 3 hardware, via `samna`.

    :py:class:`.XyloMonitor` operates continuously in real-time, receiving and processing data from a microphone with the deployed SNN. Results are continuously output from the HDK and buffered.

    On evolution, :py:class:`.XyloMonitor` returns a chunk of buffered processed time of a specified duration.

    Use :py:func:`~.devices.xylo.syns65302.config_from_specification` to build and validate a configuration for Xylo.

    """

    # TODO: Create XyloAudioFrontendConfig and include hibernation_mode, dn_active and digital microphone in it

    def __init__(
        self,
        device: XyloAudio3HDK,
        config: Optional[XyloConfiguration] = None,
        # frontend_config: Optional[XyloAudioFrontendConfig] = None,
        output_mode: str = "Spike",
        dt: float = 1e-3,
        main_clk_rate: float = Default_Main_Clock_Rate,
        hibernation_mode: bool = False,
        power_frequency: float = 5.0,
        dn_active: bool = True,
        digital_microphone=True,
        *args,
        **kwargs,
    ):
        """
        Instantiate a Module with XyloAudio 3 dev-kit backend. XyloMonitor is used in ``RealTime`` mode. For  other operation modes please use :py:class:`.XyloSamna`.

        Args:
            device (XyloAudio3HDK): An opened `samna` device to a XyloAudio 3 dev kit
            config (XyloConfiguraration): A Xylo configuration from `samna`
            output_mode (str): The readout mode for the Xylo device. This must be one of ``["Spike", "Vmem"]``. Default: "Spike", return events from the output layer.
            dt (float): The timewindow duration, in seconds. Default: 0.001
            main_clk_rate (float): The main clock rate of Xylo, in MHz
            hibernation_mode (bool): If True, hibernation mode will be switched on, which only outputs events if it receives inputs above a threshold.
            power_frequency (float): The frequency of power measurement, in Hz. Default: 5.0
            dn_active (bool): If True, divisive normalization will be used. Defaults to True.
            digital_microphone (bool): If True, configure XyloAudio 3 to use the digital microphone, otherwise, analog microphone. Defaults to True.

        Raises:
            `ValueError`: If ``device`` isoutput_mode not set. ``device`` must be a ``XyloAudio3HDK``.
            `ValueError`: If ``output_mode`` is not ``Spike`` or ``Vmem``.
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

        # - Configure master clock and communication bus clocks
        hdkutils.set_xylo_core_clock_freq(device, main_clk_rate)

        # - Get a default configuration
        if config is None:
            config = samna.xyloAudio3.configuration.XyloConfiguration()

        # - Get a default audio frontend configuration
        # TODO: Add audio frontend configuration

        # - Get the network shape
        Nin, Nhidden = np.shape(config.input.weights)
        _, Nout = np.shape(config.readout.weights)

        # - Register buffers to read and write events, monitor state
        self._read_buffer = hdkutils.new_xylo_read_buffer(device)
        self._write_buffer = hdkutils.new_xylo_write_buffer(device)

        # - Set operation mode to be RealTime
        config.operation_mode = samna.xyloAudio3.OperationMode.RealTime
        config.digital_frontend.filter_bank.dn_enable = dn_active
        config.enable_hibernation_mode = hibernation_mode

        # - Configuration for real time in XyloAudio 3
        config.time_resolution_wrap = self._get_tr_wrap(
            ts=dt, main_clk_freq_in_mhz=main_clk_rate
        )
        config.debug.always_update_omp_stat = True
        config.digital_frontend.filter_bank.use_global_iaf_threshold = True

        if digital_microphone:
            config.input_source = samna.xyloAudio3.InputSource.DigitalMicrophone
            # - the ideal sdm clock ratio depends on the main clock rate
            # -- int(main_clk_rate / Pdm_Clock_Rate / 2 - 1)
            config.debug.sdm_clock_ratio = 24
            config.digital_frontend.pdm_preprocessing.clock_direction = 1
            config.digital_frontend.pdm_preprocessing.clock_edge = 0

        else:
            raise ValueError("Analog microphone is not available yet for XyloAudio 3.")

        # - Disable internal state monitoring
        config.debug.monitor_neuron_v_mem = []
        config.debug.monitor_neuron_spike = []
        config.debug.monitor_neuron_i_syn = []

        # - Initialise the superclass
        super().__init__(
            shape=(Nin, Nhidden, Nout), spiking_input=False, spiking_output=True
        )

        # - Store the device
        self._device: XyloAudio3HDK = device
        """ `.XyloHDK`: The Xylo HDK used by this module """

        # - Store the configuration
        self._config: Union[
            XyloConfiguration, SimulationParameter
        ] = SimulationParameter(shape=(), init_func=lambda _: config)
        """ `XyloConfiguration`: The HDK configuration applied to the Xylo module """

        # - Store the timestep
        self.dt: Union[float, SimulationParameter] = dt
        """ float: Simulation time-step of the module, in seconds """

        self._main_clk_rate = main_clk_rate

        # - Store the io module
        self._io = self._device.get_io_module()

        # - Apply the configuration
        hdkutils.apply_configuration(device, self._config)

        # # - Disable RAM access to save power
        hdkutils.enable_ram_access(self._device, False)

        self._power_monitor = None
        """Power monitor for Xylo"""

        self._power_frequency = power_frequency

        # - Set power measurement module
        (
            self._power_buf,
            self._power_monitor,
            self._stopwatch,
        ) = hdkutils.set_power_measure(self._device, power_frequency)

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

        # - Disable RAM access to save power
        hdkutils.enable_ram_access(self._device, False)

        # - Store the configuration locally
        self._config = new_config

    def _get_tr_wrap(self, ts, main_clk_freq_in_mhz):
        """
        Calculate the value of tr wrap

        Args:
            ts: time windown in seconds
            main_clk_freq_in_mhz: main clock frequency in mhz
        """
        main_clk_freq = main_clk_freq_in_mhz * 1e6  # in Hz
        tr_wrap = int(ts * main_clk_freq)
        return tr_wrap

    def __del__(self):
        """
        Delete the XyloAudio3Monitor object and reset the HDK.
        """

        # self._spike_graph.stop()
        self._stopwatch.stop()
        # - Reset the HDK to clean up
        self._device.reset_board_soft()

    def apply_configuration(self, new_config):
        # - Write the configuration to the device
        hdkutils.apply_configuration(self._device, new_config)

        # - Disable RAM access to save power
        hdkutils.enable_ram_access(self._device, False)

        # - Store the configuration locally
        self._config = new_config

    def evolve(
        self,
        input_data: np.ndarray,
        record: bool = False,
        record_power: bool = False,
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        Evolve a network on the Xylo HDK in Real-time mode.

        Args:
            input_data (np.ndarray): An array ``[T, Nin]``, specifying the number of T time-steps to record.
            record (bool): ``False``, do not return a recording dictionary. Recording internal state is not supported by :py:class:`.XyloAudio3Monitor`.
            record_power (bool): If ``True``, record the power consumption during each evolve.

        Returns:
            Tuple[np.ndarray, dict, dict] output_events, {}, rec_dict
            output_events is an array that stores the output events of T time-steps
            rec_dict is a dictionary that stores the power measurements of T time-steps
        """

        if record:
            raise ValueError(
                f"Recording internal state is not supported by :py:class:`.XyloAudio3Monitor`"
            )

        Nt = input_data.shape[0]
        read_timeout = Nt * self.dt

        # use current time to calculate how long the processing will run
        # read_until = time.time() + read_timeout
        output_events = []

        # - Disable RAM access to save power
        hdkutils.enable_ram_access(self._device, False)

        # - Clear the power buffer, if recording power
        if record_power:
            self._power_monitor.start_auto_power_measurement(self._power_frequency)
            self._power_buf.clear_events()

        # - Start processing
        # In realtime mode, sending triggerprocessing again without target timestep is not an issue (i.e., does nothing)
        # TriggerProcessing with target timestep can stop execution if target is smaller than current timestep
        self._write_buffer.write([samna.xyloAudio3.event.TriggerProcessing()])
        self._read_buffer.clear_events()

        # - Wait for all the events received during the read timeout
        readout_events, _ = hdkutils.blocking_read(
            read_buffer=self._read_buffer, timeout=read_timeout
        )

        # - Check if events were recorded
        # The reading is done for a time-window and the error is thrown if nothing was read
        if len(readout_events) == 0:
            raise TimeoutError(
                f"No events received after reading for {read_timeout} seconds."
            )

        ev_filt = [
            e for e in readout_events if isinstance(e, samna.xyloAudio3.event.Readout)
        ]

        for ev in ev_filt:
            if self._output_mode == "Vmem":
                output_events.append(ev.output_v_mems)
            elif self._output_mode == "Spike":
                output_events.append(ev.output_spikes)

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

        self._power_monitor.stop_auto_power_measurement()

        # - Return the output spikes, the (empty) new state dictionary, and the recorded power dictionary
        output_events = np.stack(output_events)
        return output_events, {}, rec_dict
