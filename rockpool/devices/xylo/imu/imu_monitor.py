"""
Samna-backed bridge to Xylo dev kit for Xylo IMU
"""

# - Samna imports
import samna
from samna.xyloImu.configuration import XyloConfiguration

from . import xylo_imu_devkit_utils as hdkutils
from .xylo_imu_devkit_utils import XyloIMUHDK
from . import IMUIFSim

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
__all__ = ["XyloIMUMonitor"]

Default_Main_Clock_Rate = int(100e6)  # 100 MHz


class XyloIMUMonitor(Module):
    """
    A spiking neuron :py:class:`.Module` backed by the Xylo-IMU hardware, via `samna`.

    :py:class:`.XyloIMUMonitor` operates continuously in real-time, receiving and processing data from an IMU sensor with the deployed SNN. Results are continuously output from the HDK and buffered.

    On evolution, :py:class:`.XyloIMUMonitor` returns a chunk of buffered processed time of a specified duration.

    Use :py:func:`~.devices.xylo.imu.config_from_specification` to build and validate a configuration for Xylo.
    """

    def __init__(
        self,
        device: XyloIMUHDK,
        config: XyloConfiguration = None,
        output_mode: str = "Spike",
        prerecorded_imu_input: bool = False,
        main_clk_rate: Optional[int] = Default_Main_Clock_Rate,
        hibernation_mode: bool = False,
        interface_params: Optional[dict] = dict(),
        *args,
        **kwargs,
    ):
        """
        Instantiate a Module with Xylo dev-kit backend

        Args:
            device (XyloIMUHDK): An opened `samna` device to a Xylo dev kit
            config (XyloConfiguraration): A Xylo configuration from `samna`
            output_mode (str): The readout mode for the Xylo device. This must be one of ``["Spike", "Vmem"]``. Default: "Spike", return events from the output layer.
            prerecorded_imu_input (bool): If ``True``, use prerocorded imu data from PC as input. If ``False``, use the live IMU sensor on the HDK. Default: ``False``, use the IMU sensor.
            main_clk_rate(int): The main clock rate of Xylo
            hibernation_mode (bool): If True, hibernation mode will be switched on, which only outputs events if it receives inputs above a threshold.
            interface_params(dict): The dictionary of Xylo interface parameters used for the `hdkutils.config_if_module` function, the keys of which must be "num_avg_bitshif", "select_iaf_output", "sampling_period", "filter_a1_list", "filter_a2_list", "scale_values", "Bb_list", "B_wf_list", "B_af_list", "iaf_threshold_values".
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
            config = samna.xyloImu.configuration.XyloConfiguration()

        # - Get the network shape
        Nin, Nhidden = np.shape(config.input.weights)
        _, Nout = np.shape(config.readout.weights)

        # - Register buffers to read and write events, monitor state
        self._read_buffer = hdkutils.new_xylo_read_buffer(device)
        self._write_buffer = hdkutils.new_xylo_write_buffer(device)

        # - Build a filter graph to filter `Readout` events from Xylo IMU
        self._readout_graph = samna.graph.EventFilterGraph()
        _, etf0, self._readout_buffer = self._readout_graph.sequential(
            [
                device.get_model_source_node(),
                "XyloImuOutputEventTypeFilter",
                samna.graph.JitSink(),
            ]
        )
        etf0.set_desired_type("xyloImu::event::Readout")
        self._readout_graph.start()

        # - Initialise the superclass
        super().__init__(shape=(3, Nout), spiking_input=False, spiking_output=True)

        # - Store the device
        self._device: XyloIMUHDK = device
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
        self.dt: Union[float, SimulationParameter] = (
            1 / 200
        )  # Fixed computation step rate of 200Hz for Xylo IMU
        """ float: Simulation time-step of the module, in seconds """

        # - Store the main clock rate
        self._main_clk_rate = int(main_clk_rate)

        # - Store the io module
        self._io = self._device.get_io_module()

        # - Store the choice of external imu input
        self._external_imu_input = prerecorded_imu_input

        # - Select source of IMU accel data input
        if prerecorded_imu_input:
            self._device.enable_manual_input_acceleration(True)
        else:
            self._device.enable_manual_input_acceleration(False)

        # - Set main clock rate
        if self._main_clk_rate != Default_Main_Clock_Rate:
            self._io.set_main_clk_rate(self._main_clk_rate)

        # - Configure to auto mode
        self._enable_realtime_mode(interface_params)

    @property
    def config(self):
        # - Return the configuration stored on Xylo HDK
        return self._device.get_model().get_configuration()

    @config.setter
    def config(self, new_config):
        # - Test for a valid configuration
        is_valid, msg = samna.xyloImu.validate_configuration(new_config)
        if not is_valid:
            raise ValueError(f"Invalid configuration for the Xylo HDK: {msg}")

        # - Write the configuration to the device
        hdkutils.apply_configuration(self._device, new_config)

        # - Store the configuration locally
        self._config = new_config

    def _enable_realtime_mode(self, interface_params: dict):
        """
        Configure the Xylo HDK to use real-time mode

        Args:
            interface_params (dict): specify the interface parameters
        """

        # - Config the streaming mode
        config = hdkutils.config_realtime_mode(
            self._config,
            self.dt,
            self._main_clk_rate,
        )

        # - Config the IMU interface and apply current configuration
        config.input_interface = IMUIFSim(**interface_params).export_config()
        self.config = config

    def evolve(
        self,
        input_data: np.ndarray,
        record: bool = False,
        read_timeout: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        Evolve a network on the Xylo HDK in Real-time mode

        Args:
            input_data (np.ndarray): An array ``[T, 3]``, specifying the number of time-steps to record. If using external imu data input, the `input_data` is the external imu data. The first dimension is timesteps, and the last dimension is 3 channels of accelerations along x, y, z axes.
            record (bool): ``False``, do not return a recording dictionary. Recording internal state is not supported by :py:class:`.XyloIMUMonitor`
            timeout (float): A duration in seconds for a read timeout. Default: 2x the real-time duration of the evolution

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

        if Nb > 1:
            raise ValueError(
                f"Batched data are not supported by XyloIMUMonitor. Got batched input data with shape {[Nb, Nt, Nc]}."
            )

        if self._external_imu_input and Nc != 3:
            raise ValueError(
                f"When providing IMU input data, 3 channels of input are required. Received input with shape [{Nt, Nc}]."
            )

        # - Discard the batch dimension
        input_data = input_data[0]

        # - Get the current time step, determine duration
        start_timestep = (
            hdkutils.get_current_timestep(self._read_buffer, self._write_buffer) + 1
        )
        end_timestep = start_timestep + Nt - 1

        # - Determine a read timeout
        read_timeout = 3 * Nt * self.dt if read_timeout is None else read_timeout

        # - Send external IMU input to Xylo, if requested
        if self._external_imu_input:
            # - Ensure configuration of Xylo
            hdkutils.apply_configuration(self._device, self._config)

            # - Encode IMU data and send to FPGA
            imu_input = hdkutils.encode_imu_data(input_data)
            self._readout_buffer.get_events()
            self._write_buffer.write(imu_input)

        # - Process in real-time mode for a desired number of time steps
        self._write_buffer.write(
            [samna.xyloImu.event.TriggerProcessing(target_timestep=end_timestep + 1)]
        )

        # - Blocking read of events until simulation is finished
        read_events, is_timeout = hdkutils.blocking_read(
            self._readout_buffer, target_timestep=end_timestep, timeout=read_timeout
        )

        if is_timeout:
            raise TimeoutError(
                f"Reading events timeout after {read_timeout} seconds. Read {len(read_events)} events, expected {Nt}. Last event timestep: {read_events[-1].timestep if len(read_events) > 0 else 'None'}, waiting for timestep {end_timestep}."
            )

        # - Decode data read from Xylo
        vmem_out_ts, spike_out_ts = hdkutils.decode_realtime_mode_data(
            read_events, self.size_out, start_timestep, end_timestep
        )
        out = vmem_out_ts if self._output_mode == "vmem" else spike_out_ts

        return out, {}, {}
