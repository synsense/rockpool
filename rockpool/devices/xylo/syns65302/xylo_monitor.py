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

    def __init__(
        self,
        device: XyloAudio3HDK,
        config: Optional[XyloConfiguration] = None,
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
        Instantiate a Module with XyloAudio 3 dev-kit backend.

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
            `ValueError`: If ``device`` is not set. ``device`` must be a ``XyloAudio3HDK``.
            `TimeoutError`: If ``output_mode`` is not ``Spike`` or ``Vmem``.
            `ValueError`: If ``operation_mode`` is not set to ``RealTime``. For  other opeartion modes please use :py:class:`.XyloSamna`.
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

        default_config = samna.xyloAudio3.XyloAudio3TestBoardDefaultConfig()
        default_config.main_clock_frequency = int(main_clk_rate * 1e6)
        device.reset_board_soft(default_config)

        # - Get a default configuration
        if config is None:
            config = samna.xyloAudio3.configuration.XyloConfiguration()
            config.operation_mode = samna.xyloAudio3.OperationMode.RealTime

        # - Get the network shape
        Nin, Nhidden = np.shape(config.input.weights)
        _, Nout = np.shape(config.readout.weights)

        # - Register buffers to read and write events, monitor state
        self._read_buffer = hdkutils.new_xylo_read_buffer(device)
        self._write_buffer = hdkutils.new_xylo_write_buffer(device)

        if config.operation_mode != samna.xyloAudio3.OperationMode.RealTime:
            raise ValueError("`operation_mode` must be RealTime for XyloMonitor.")

        config.digital_frontend.filter_bank.dn_enable = dn_active

        # - Configuration for real time in XyloAudio 3
        config.time_resolution_wrap = self._get_tr_wrap(
            ts=dt, main_clk_freq_in_mhz=main_clk_rate
        )
        config.debug.always_update_omp_stat = True
        config.digital_frontend.filter_bank.use_global_iaf_threshold = True

        if digital_microphone:
            config.input_source = samna.xyloAudio3.InputSource.DigitalMicrophone
            # - the ideal sdm clock ratio depends on the main clock rate
            config.debug.sdm_clock_ratio = int(main_clk_rate / Pdm_Clock_Rate / 2 - 1)
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

        # - Store the configuration (and apply it)
        self._config: Union[
            XyloConfiguration, SimulationParameter
        ] = SimulationParameter(shape=(), init_func=lambda _: config)
        """ `XyloConfiguration`: The HDK configuration applied to the Xylo module """

        # - Enable hibernation mode
        if hibernation_mode:
            self._config.enable_hibernation_mode = True

        # - Store the timestep
        self.dt: Union[float, SimulationParameter] = dt
        """ float: Simulation time-step of the module, in seconds """

        self._main_clk_rate = main_clk_rate

        # - Store the io module
        self._io = self._device.get_io_module()

        # - Store the configuration (and apply it)
        hdkutils.apply_configuration(device, self._config)

        self._power_monitor = None
        """Power monitor for Xylo"""

        self._evolve = False
        """ Track if evolve function was called """

        self._power_frequency = power_frequency

        # - Set power measurement module
        self._power_buf, self._power_monitor, self.stopwatch = hdkutils.set_power_measure(
            self._device, power_frequency
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
        self.stopwatch.stop()
        # - Reset the HDK to clean up
        self._device.reset_board_soft()

    def apply_configuration(self, new_config):
        # - Write the configuration to the device
        hdkutils.apply_configuration(self._device, new_config)

        # - Store the configuration locally
        self._config = new_config

    def evolve(
        self,
        record: bool = False,
        record_power: bool = False,
        read_timeout: Optional[float] = 1,
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        Evolve a network on the Xylo HDK in Real-time mode.

        Args:
            record (bool): ``False``, do not return a recording dictionary. Recording internal state is not supported by :py:class:`.XyloAudio3Monitor`
            record_power (bool): If ``True``, record the power consumption during each evolve.
            read_timeout (float): A duration in seconds for a read timeout.

        Returns:
            Tuple[np.ndarray, dict, dict] output_events, {}, rec_dict
            output_events is an array that stores the output events of T time-steps
            rec_dict is a dictionary that stores the power measurements of T time-steps
        """

        if record:
            raise ValueError(
                f"Recording internal state is not supported by :py:class:`.XyloAudio3Monitor`"
            )

        if read_timeout < self.dt:
            raise ValueError(
                "Read timeout can not be smaller then the duration of the processing step."
            )

        # use current time to calculate how long the processing will run
        read_until = time.time() + read_timeout
        output_events = []

        # - Clear the power buffer, if recording power
        if record_power:
            self._power_monitor.start_auto_power_measurement(self._power_frequency)
            self._power_buf.clear_events()

        # Start processing
        # In realtime mode, sending triggerprocessing again without target timestep is not an issue (i.e., does nothing)
        # TriggerProcessing with target timestep can stop execution if target is smaller than current timestep
        self._write_buffer.write([samna.xyloAudio3.event.TriggerProcessing()])
        self._read_buffer.clear_events()

        # - Wait for all the events received during the read timeout
        readout_events = []
        # -- We still need the loop because there is no function in samna that wait for a specific ammount of time and return all events
        while (now := time.time()) < read_until:
            remaining_time = read_until - now
            readout_events += self._read_buffer.get_events_blocking(
                math.ceil(remaining_time * 1000)
            )

        if len(readout_events) == 0:
            message = f"No event received in {read_timeout}s."
            raise TimeoutError(message)

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
        return output_events, {}, rec_dict
