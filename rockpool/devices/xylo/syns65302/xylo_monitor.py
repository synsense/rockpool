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
        *args,
        **kwargs,
    ):
        """
        Instantiate a Module with Xylo dev-kit backend.

        Args:
            device (XyloAudio3HDK): An opened `samna` device to a Xylo dev kit
            config (XyloConfiguraration): A Xylo configuration from `samna`
            output_mode (str): The readout mode for the Xylo device. This must be one of ``["Spike", "Vmem"]``. Default: "Spike", return events from the output layer.
            main_clk_rate (float): The main clock rate of Xylo, in MHz
            hibernation_mode (bool): If True, hibernation mode will be switched on, which only outputs events if it receives inputs above a threshold.
            interface_params(dict): The dictionary of Xylo interface parameters used for the `hdkutils.config_if_module` function, the keys of which must be "num_avg_bitshif", "select_iaf_output", "sampling_period", "filter_a1_list", "filter_a2_list", "scale_values", "Bb_list", "B_wf_list", "B_af_list", "iaf_threshold_values".
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

        # - Get a default configuration
        if config is None:
            config = samna.xyloAudio3.configuration.XyloConfiguration()

        # - Get the network shape
        Nin, Nhidden = np.shape(config.input.weights)
        _, Nout = np.shape(config.readout.weights)

        # - Register buffers to read and write events, monitor state
        self._read_buffer = hdkutils.new_xylo_read_buffer(device)
        self._write_buffer = hdkutils.new_xylo_write_buffer(device)

        # - Build a filter graph to filter `Readout` events from Xylo
        self._readout_graph = samna.graph.EventFilterGraph()
        _, etf0, self._readout_buffer = self._readout_graph.sequential(
            [
                device.get_model_source_node(),
                "XyloAudio3OutputEventTypeFilter",
                samna.graph.JitSink(),
            ]
        )
        etf0.set_desired_type("xyloAudio3::event::Readout")

        self._readout_graph.start()

        # - Initialise the superclass
        super().__init__(
            shape=(Nin, Nhidden, Nout), spiking_input=False, spiking_output=True
        )

        # - Store the device
        self._device: XyloAudio3HDK = device
        """ `.XyloHDK`: The Xylo HDK used by this module """

        # - Store the configuration (and apply it)
        self.config: Union[
            XyloConfiguration, SimulationParameter
        ] = SimulationParameter(shape=(), init_func=lambda _: config)
        """ `XyloConfiguration`: The HDK configuration applied to the Xylo module """

        # - Enable hibernation mode
        if hibernation_mode:
            self.config.enable_hibernation_mode = True

        # - Store the timestep
        self.dt: Union[
            float, SimulationParameter
        ] = dt  # Fixed computation step rate of 200Hz for Xylo IMU
        """ float: Simulation time-step of the module, in seconds """

        self._main_clk_rate = main_clk_rate

        # self._main_clk_rate: float = hdkutils.set_xylo_core_clock_freq(
        #     self._device, main_clk_rate
        # )
        # - Store the io module
        self._io = self._device.get_io_module()

        # - Sleep time post sending spikes on each time-step, in manual mode
        self._sleep_time = 0e-3
        """ float: Post-stimulation sleep time in seconds """

        # - Initialise the HDK
        hdkutils.initialise_xylo_hdk(
            self._device, self._read_buffer, self._write_buffer
        )
        # - Store the configuration (and apply it)
        time.sleep(self._sleep_time)

        """ float: Xylo main clock frequency in MHz """

        # - Configure to auto mode
        # self._enable_realtime_mode(interface_params)
        self._enable_realtime_mode()

        self.power_monitor = None
        """Power monitor for Xylo"""

        # - Set power measurement module
        # self._power_buf, self.power_monitor = hdkutils.set_power_measure(
        #     self._device, power_frequency
        # )

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

    # def _enable_realtime_mode(self, interface_params: dict):
    def _enable_realtime_mode(self):
        """
        Configure the Xylo HDK to use real-time mode.

        Args:
            interface_params (dict): specify the interface parameters
        """

        # - Config the streaming mode
        config = hdkutils.config_realtime_mode(
            self._read_buffer,
            self._write_buffer,
            self._config,
            self.dt,
            int(self._main_clk_rate * 1e6),
        )

        # - Config the IMU interface and apply current configuration
        # config.input_interface = IMUIFSim(**interface_params).export_config()
        self.config = config

    def __del__(self):
        """
        Delete the XyloIMUMonitor object and reset the HDK.
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
        input_data: np.ndarray,
        record: bool = False,
        record_power: bool = False,
        read_timeout: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        Evolve a network on the Xylo HDK in Real-time mode.

        Args:
            input_data (np.ndarray): An array ``[T, 3]``, specifying the number of time-steps to record. If using external imu data input, the `input_data` is the external imu data. The first dimension is timesteps, and the last dimension is 3 channels of accelerations along x, y, z axes.
            record (bool): ``False``, do not return a recording dictionary. Recording internal state is not supported by :py:class:`.XyloIMUMonitor`
            record_power (bool): If ``True``, record the power consumption during each evolve.
            read_timeout (float): A duration in seconds for a read timeout. Default: 2x the real-time duration of the evolution

        Returns:
            Tuple[np.ndarray, dict, dict] output_events, {}, rec_dict
            output_events is an array that stores the output events of T time-steps
        """
        # - Check `record` flag
        if record:
            raise ValueError(
                "Recording internal state is not supported by XyloIMUMonitor."
            )

        # - Get data shape
        input_data, _ = self._auto_batch(input_data)
        Nb, Nt, Nc = input_data.shape

        # - Discard the batch dimension
        input_data = input_data[0]

        # - Get the current time step, determine duration
        start_timestep = (
            hdkutils.get_current_timestep(self._read_buffer, self._write_buffer) + 1
        )

        end_timestep = start_timestep + Nt - 1

        # - Determine a read timeout
        read_timeout = 3 * Nt * self.dt if read_timeout is None else read_timeout

        # - Clear the power recording buffer, if recording power
        if record_power:
            self._power_buf.clear_events()

        # - Process in real-time mode for a desired number of time steps
        self._write_buffer.write(
            [samna.xyloAudio3.event.TriggerProcessing(target_timestep=end_timestep + 1)]
        )

        # - Blocking read of events until simulation is finished
        read_events, is_timeout = hdkutils.blocking_read(
            self._readout_buffer, target_timestep=end_timestep, timeout=read_timeout
        )

        if is_timeout:
            raise TimeoutError(
                f"Reading events timeout after {read_timeout} seconds. Read {len(read_events)} events, expected {Nt}. Last event timestep: {read_events[-1].timestep if len(read_events) > 0 else 'None'}, waiting for timestep {end_timestep}."
            )

        rec_dict = {}

        if record_power:
            # - Get all recent power events from the power measurement
            ps = self._power_buf.get_events()

            # - Separate out power meaurement events by channel
            channels = samna.xyloAudio3Boards.MeasurementChannels
            io_power = np.array([e.value for e in ps if e.channel == int(channels.Io)])
            core_power = np.array(
                [e.value for e in ps if e.channel == int(channels.Core)]
            )
            rec_dict.update(
                {
                    "io_power": io_power,
                    "core_power": core_power,
                }
            )

        # - Decode data read from Xylo
        vmem_out_ts, spike_out_ts = hdkutils.decode_realtime_mode_data(
            read_events, self.size_out, start_timestep, end_timestep
        )
        out = vmem_out_ts if self._output_mode == "vmem" else spike_out_ts

        return out, {}, rec_dict
