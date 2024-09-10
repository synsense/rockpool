"""
Samna-backed bridge to Xylo dev kit for SYNS65302 Xylo core v3
"""

# - Samna imports
import samna
from samna.xyloAudio3.configuration import XyloConfiguration

XyloAudio3HDK = samna.xyloAudio3Boards.XyloAudio3TestBoard
from . import xylo_a3_devkit_utils as hdkutils

import time
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


class XyloMonitor(Module):
    """
    A spiking neuron :py:class:`.Module` backed by the Xylo-audio3 hardware, via `samna`.

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
        # interface_params: dict = dict(),
        power_frequency: float = 5.0,
        dn_active: bool = True,
        digital_microphone=True,
        *args,
        **kwargs,
    ):
        """
        Instantiate a Module with Xylo dev-kit backend.

        Args:
            device (XyloAudio3HDK): An opened `samna` device to a Xylo dev kit
            config (XyloConfiguraration): A Xylo configuration from `samna`
            output_mode (str): The readout mode for the Xylo device. This must be one of ``["Spike", "Vmem"]``. Default: "Spike", return events from the output layer.
            dt (float):
            main_clk_rate (float): The main clock rate of Xylo, in MHz
            hibernation_mode (bool): If True, hibernation mode will be switched on, which only outputs events if it receives inputs above a threshold.
            interface_params(dict): The dictionary of Xylo interface parameters used for the `hdkutils.config_if_module` function, the keys of which must be "num_avg_bitshif", "select_iaf_output", "sampling_period", "filter_a1_list", "filter_a2_list", "scale_values", "Bb_list", "B_wf_list", "B_af_list", "iaf_threshold_values".
            power_frequency (float): The frequency of power measurement. Default: 5.0
            dn_active (bool): If True, divisive normalization will be used. Defaults to True.
            digital_microphone (bool): If True, configure Xylo Audio3 to use the digital microphone, otherwise, analog microphone. Defaults to True.
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

        # - Get a default configuration
        if config is None:
            config = samna.xyloAudio3.configuration.XyloConfiguration()

        # - Get the network shape
        Nin, Nhidden = np.shape(config.input.weights)
        _, Nout = np.shape(config.readout.weights)

        # - Register buffers to read and write events, monitor state
        self._read_buffer = hdkutils.new_xylo_read_buffer(device)
        self._write_buffer = hdkutils.new_xylo_write_buffer(device)
        self._state_buffer = hdkutils.new_xylo_state_monitor_buffer(device)

        if config.operation_mode != samna.xyloAudio3.OperationMode.RealTime:
            raise ValueError("`operation_mode` must be RealTime for XyloMonitor.")

        config.digital_frontend.filter_bank.dn_enable = dn_active

        # - Configuration for real time in xyloA3
        config.time_resolution_wrap = self._get_tr_wrap(
            ts_in_ms=dt * 1000, main_clk_freq_in_mhz=50
        )
        config.debug.always_update_omp_stat = True
        config.digital_frontend.filter_bank.use_global_iaf_threshold = True

        if digital_microphone:
            config.input_source = samna.xyloAudio3.InputSource.Pdm
            config.debug.event_input_enable = False
            config.debug.sdm_clock_ratio = 24
            config.digital_frontend.pdm_preprocessing.clock_direction = 1
            config.digital_frontend.pdm_preprocessing.clock_edge = 0

        else:
            # config = self._enable_analog_registers(config)
            # config = self._program_analog_registers(config)
            config.input_source = samna.xyloAudio3.InputSource.Adc

        # - Build a filter graph to filter `Readout` events from Xylo
        self._spike_graph = samna.graph.EventFilterGraph()

        _, etf0, self._spike_buffer = self._spike_graph.sequential(
            [
                device.get_model_source_node(),
                "XyloAudio3OutputEventTypeFilter",
                samna.graph.JitSink(),
            ]
        )
        etf0.set_desired_type("xyloAudio3::event::Readout")

        self._spike_graph.start()

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

        # - Configure to real time mode
        self._enable_realtime_mode()

        # - Store the configuration (and apply it)
        hdkutils.apply_configuration(device, self._config)
        hdkutils.enable_real_time_mode(device)

        self._power_monitor = None
        """Power monitor for Xylo"""

        self._evolve = False
        """ Track if evolve function was called """

        # - Set power measurement module
        self._power_buf, self._power_monitor = hdkutils.set_power_measure(
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

    def _get_tr_wrap(self, ts_in_ms, main_clk_freq_in_mhz):
        """
        Calculate the value of tr wrap

        Args:
            ts_in_ms: time windown in miliseconds
            main_clk_freq_in_mhz: main clock frequency in mhz
        """
        ts_duration = ts_in_ms * 1e-3  # in second
        main_clk_freq = main_clk_freq_in_mhz * 1e6  # in Hz
        tr_wrap = int(ts_duration * main_clk_freq)
        return tr_wrap

    # def _enable_analog_registers(self, config):
    #     """
    #     Activate registers to use the analolog microphone path.

    #     Args:
    #         config: the configuration that will be applied
    #     """
    #     # bandgap
    #     config.analog_frontend.ivgen.select_default = True
    #     # ldo-analog
    #     config.analog_frontend.ldo.enable_ldo_analog = True
    #     config.analog_frontend.ldo.enable_ldo_vref_gen = True
    #     # afe-amp
    #     config.analog_frontend.ivgen.enable_afe_lna_bias = True
    #     config.analog_frontend.ivgen.enable_afe_pga_bias = True
    #     config.analog_frontend.ivgen.enable_afe_driver_bias = True
    #     config.analog_frontend.enable_lna = True
    #     config.analog_frontend.enable_pga = True
    #     config.analog_frontend.enable_drv = True
    #     config.analog_frontend.adc.convert_adc = True
    #     config.analog_frontend.adc.enable_adc_parallel = True

    #     # afe-adc
    #     config.debug.enable_adc = True
    #     # charProg.reset_SubModules_Deassert()
    #     config.analog_frontend.adc.enable_adc = True

    #     return config

    # def _program_analog_registers(self, config):
    #     """
    #     Configure registers to use the analolog microphone path.

    #     Args:
    #         config: the configuration that will be applied
    #     """
    #     # configuration values from Can's script - xylo-a3 program parameters (do not modify)
    #     # bandgap
    #     bandgap_trim_value = 95  # 0 - 127 --> ~565mV - ~635mV
    #     bandgap_trim_slope = 8  # 0 - 15 --> slope and value increases linearly
    #     # ptat
    #     ptat_trim_value = 24  # range: 0 - 31, default: 18, linear
    #     # ldo-digital
    #     ldo_digital_trim = 0  # 0-7; 0-3, 1.10V-0.95V; 4-7, 1.3V-1.15V
    #     # ldo-analog
    #     common_mode_voltage_trim = 1  # 0 - 3 --> 500mV, 550mV, 575mV, 600mV
    #     ldo_analog_trim = 0  # 0-7; 0-3, 1.10V-0.95V; 4-7, 1.3V-1.15V
    #     # afe-adc
    #     adcSpeed = 3  # 0 - 3 --> 0: 50ksps, 1: 100ksps, 3: 200ksps (2: not allowed)
    #     clockDivision = 1  # 1 - 8 --> 200ksps - 25ksps (inverse)
    #     # bias
    #     bias_mirror_linear = 3  # (default: 7) 0 - 7 --> 25nA - 200nA (linear)
    #     bias_pmos_inverse = 1  # (default: 0) 0 - 3 --> 800nA - 200nA (inverse)
    #     bias_adc = 0  # (default: 0, 4, 12): 0 - 15, 50nA - 237.5nA (linear)
    #     bias_ldo_dig = 7  # (default: 3) 0 - 7 --> 250nA - 600nA (linear)
    #     bias__ldo_dig_internal = 1  # (default: 0) 0 - 1 --> 400nA - 800nA (linear)
    #     bias_ldo_dig_ilim = 1  # (default: 0) 0 - 3, 50nA - 200nA (linear)
    #     bias_lna = 1  # (default: 1) 0 - 7 --> 50nA - 400nA (linear)
    #     bias_pga = 0  # (default: 1) 0 - 7 --> 50nA - 400nA (linear)
    #     bias_driver = 0  # (default: 3) 0 - 7 --> 100nA - 800nA (linear)

    #     # bandgap
    #     config.analog_frontend.ivgen.temperature_slope_trim_bandgap = bandgap_trim_slope
    #     config.analog_frontend.ivgen.absolute_value_trim_bandgap = bandgap_trim_value
    #     # ptat
    #     config.analog_frontend.ivgen.trim_value_ptat = ptat_trim_value
    #     # ldo-digital
    #     config.analog_frontend.ldo.vdd_digital_core_voltage = ldo_digital_trim
    #     # ldo-analog
    #     config.analog_frontend.ldo.vdd_analog_core_voltage = ldo_analog_trim
    #     config.analog_frontend.ldo.vcm_lna_voltage = common_mode_voltage_trim
    #     # afe-adc
    #     config.debug.adc_clock_ratio = clockDivision - 1
    #     config.analog_frontend.adc.adc_conversion_speed = adcSpeed
    #     # bias currents
    #     config.analog_frontend.ivgen.adc_buffer_test = False
    #     config.analog_frontend.ivgen.adc_buffer_bias = bias_adc
    #     config.analog_frontend.ivgen.ldo_digital_bias = bias_ldo_dig
    #     config.analog_frontend.ldo.ldo_digital_capacitor_stability = (
    #         bias__ldo_dig_internal
    #     )
    #     config.analog_frontend.ldo.ldo_digital_current_limit = bias_ldo_dig_ilim
    #     config.analog_frontend.ivgen.current_ptat = bias_mirror_linear
    #     config.analog_frontend.ivgen.current_mirror_input2 = bias_pmos_inverse
    #     config.analog_frontend.ivgen.afe_lna_bias = bias_lna
    #     config.analog_frontend.ivgen.afe_pga_bias = bias_pga
    #     config.analog_frontend.ivgen.afe_drv_bias = bias_driver
    #     config.debug.analog_test_mode.bypass_afe = False

    #     return config

    def _enable_realtime_mode(self):
        """
        Configure the Xylo HDK to use real-time mode.

        Args:
            interface_params (dict): specify the interface parameters
        """
        # - No monitoring of internal state in realtime mode
        self._config.debug.monitor_neuron_v_mem = {}
        self._config.debug.monitor_neuron_i_syn = {}
        self._config.debug.monitor_neuron_spike = {}

    def __del__(self):
        """
        Delete the XyloAudio3Monitor object and reset the HDK.
        """
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
            read_timeout (float): A duration in seconds for a read timeout. Default: 2x the real-time duration of the evolution

        Returns:
            Tuple[np.ndarray, dict, dict] output_events, {}, rec_dict
            output_events is an array that stores the output events of T time-steps
        """

        target_timestep = int(read_timeout * (1 / self.dt))

        # Trigger processing can only be sent once in RealTime mode.
        if not self._evolve:
            # send a trigger processing
            self._write_buffer.write(
                [samna.xyloAudio3.event.TriggerProcessing(target_timestep)]
            )
            # self._evolve = True

        timestep = 0
        output_events = []
        spikes_ts = []
        vmem_out_ts = []

        # - Clear the power recording buffer, if recording power
        if record_power:
            self._power_buf.clear_events()

        while timestep < target_timestep - 1:
            readout_events = self._read_buffer.get_events_blocking()

            ev_filt = [
                e
                for e in readout_events
                if isinstance(e, samna.xyloAudio3.event.Readout)
            ]

            if ev_filt:
                for ev in ev_filt:
                    if self._output_mode == "Vmem":
                        output_events.append(ev.output_v_mems)
                    elif self._output_mode == "Spike":
                        output_events.append(ev.output_spikes)

                if record:
                    vmem_out_ts.append(ev.output_v_mems)
                    spikes_ts.append(ev.output_spikes)

                    rec_dict = {
                        "Spikes": np.array(spikes_ts),
                        "Vmem_out": np.array(vmem_out_ts),
                    }
                else:
                    rec_dict = {}

            timestep = readout_events[-1].timestep

            if record_power:
                # - Get all recent power events from the power measurement
                ps = self._power_buf.get_events()

                # - Separate out power meaurement events by channel
                # - Channel 0: IO power
                # - Channel 1: Analog logic power
                # - Channel 2: Digital logic power

                io_power = np.array([e.value for e in ps if e.channel == 0])
                analog_power = np.array([e.value for e in ps if e.channel == 1])
                digital_power = np.array([e.value for e in ps if e.channel == 2])
                rec_dict.update(
                    {
                        "io_power": io_power,
                        "analog_power": analog_power,
                        "digital_power": digital_power,
                    }
                )

        return output_events, {}, rec_dict
