import numpy as np

import samna
from samna.xyloImu.configuration import XyloConfiguration

# - Typing
from typing import Optional, Union, Callable, List, Tuple


from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter


from .xylo_imu_devkit_utils import XyloIMUHDK
from . import xylo_imu_devkit_utils as hdkutils
import time


class XyloIMUSamna(Module):
    """
    A spiking neuron :py:class:`.Module` backed by the Xylo hardware, via `samna`.

    Use :py:func:`.config_from_specification` to build and validate a configuration for Xylo.

    See Also:

        See the tutorials :ref:`/devices/xylo-overview.ipynb` and :ref:`/devices/torch-training-spiking-for-xylo.ipynb` for a high-level overview of building and deploying networks for Xylo.

    """

    def __init__(
        self,
        device: XyloIMUHDK,
        config: XyloConfiguration = None,
        dt: float = 1e-3,
        output_mode: str = "Spike",
        power_frequency: Optional[float] = 5.0,
        *args,
        **kwargs,
    ):
        """
        Instantiate a Module with Xylo dev-kit backend

        Args:
            device (XyloA2HDK): An opened `samna` device to a Xylo dev kit
            config (XyloConfiguration): A Xylo configuration from `samna`
            dt (float): The simulation time-step to use for this Module
            output_mode (str): The readout mode for the Xylo device. This must be one of ``["Spike", "Isyn", "Vmem"]``. Default: "Spike", return events from the output layer.
            power_frequency (float): The frequency of power measurement. Default: 5.0
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
        if config is None:
            config = samna.xyloImu.configuration.XyloConfiguration()

        # - Get the network shape
        Nin, Nhidden = np.shape(config.input.weights)
        _, Nout = np.shape(config.readout.weights)

        # - Initialise the superclass
        super().__init__(
            shape=(Nin, Nhidden, Nout), spiking_input=True, spiking_output=True
        )

        # - Store the device
        self._device: XyloIMUHDK = device
        """ `.XyloHDK`: The Xylo HDK used by this module """

        print("Store device")
        # - Register buffers to read and write events, monitor state
        self._read_buffer = hdkutils.new_xylo_read_buffer(device)
        self._write_buffer = hdkutils.new_xylo_write_buffer(device)
        self._state_buffer = hdkutils.new_xylo_state_monitor_buffer(device)

        # - Store the io module
        self._io = device.get_io_module()

        # - Store the timestep
        self.dt: Union[float, SimulationParameter] = dt
        """ float: Simulation time-step of the module, in seconds """

        # # - Check that we can access the device node, and that it's a Xylo HDK
        # if not hdkutils.verify_xylo_version(
        #     self._read_buffer, self._write_buffer, timeout=10.0
        # ):
        #     raise ValueError(
        #         "Cannot verify HDK version. `device` must be an opened Xylo HDK."
        #     )

        # - Store the configuration (and apply it)
        self.config: Union[
            XyloConfiguration, SimulationParameter
        ] = SimulationParameter(shape=(), init_func=lambda _: config)
        """ `.XyloConfiguration`: The HDK configuration applied to the Xylo module """

        # - Keep a registry of the current recording mode, to save unnecessary reconfiguration
        self._last_record_mode: Optional[bool] = None
        """ bool: The most recent (and assumed still valid) recording mode """

        # - Store the timestep
        self.dt: Union[float, SimulationParameter] = dt
        """ float: Simulation time-step of the module, in seconds """

        # # - Set power measurement module
        # self._power_buf, self.power = hdkutils.set_power_measure(
        #     self._device, power_frequency
        # )

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
        else:
            print("Config valid")

        # - Write the configuration to the device
        hdkutils.apply_configuration(self._device, new_config)
        self._state_buffer.set_configuration(new_config)
        self._config = new_config

    def _configure_accel_time_mode(
        self, Nhidden: int, Nout: int, record: bool = False
    ) -> None:
        """
        Configure the Xylo HDK to use accelerated-time mode, with optional state recording

        Args:
            Nhidden (int): Number of hidden neurons from which to record state. Default: ``0``; do not record state from any neurons. If non-zero, state from neurons with ID 0..(Nhidden-1) inclusive will be recorded during evolution.
            Nout (int): Number of output layer neurons from which to record state. Default: ``0``; do not record state from any output neurons.
            record (bool): Iff ``True``, record state during evolution. Default: ``False``, do not record state.
        """
        if record != self._last_record_mode:
            # - Keep a registry of the last recording mode
            self._last_record_mode = record

            self.config, state_buffer = hdkutils.configure_accel_time_mode(
                self._config,
                self._state_buffer,
                Nhidden,
                Nout,
                readout=self._output_mode,
                record=record,
            )

    def _config_hibernation_mode(self):
        """
        Configure the Xylo HDK to use hibernation mode
        """
        self.config = hdkutils.config_hibernation_mode(self._config, True)

    def evolve(
        self,
        input: np.ndarray,
        record: bool = False,
        record_power: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        Evolve a network on the Xylo HDK in accelerated-time mode

        Sends a series of events to the Xylo HDK, evolves the network over the input events, and returns the output events produced during the input period. Optionally record internal state of the network, selectable with the ``record`` flag.

        Args:
            input (np.ndarray): A raster ``(T, Nin)`` specifying for each bin the number of input events sent to the corresponding input channel on Xylo, at the corresponding time point. Up to 15 input events can be sent per bin.
            record (bool): Iff ``True``, record and return all internal state of the neurons and synapses on Xylo. Default: ``False``, do not record internal state.
            record_power (bool): Iff ``True``, record the power consumption during each evolve.

        Returns:
            (np.ndarray, dict, dict): ``output``, ``new_state``, ``record_dict``.
            ``output`` is a raster ``(T, Nout)``, containing events for each channel in each time bin. Time bins in ``output`` correspond to the time bins in ``input``.
            ``new_state`` is an empty dictiionary. The Xylo HDK does not permit querying or setting state.
            ``record_dict`` is a dictionary containing recorded internal state of Xylo during evolution, if the ``record`` argument is ``True``. Otherwise this is an empty dictionary.

        Raises:
            `TimeoutError`: If reading data times out during the evolution. An explicity timeout can be set using the `read_timeout` argument.
        """

        # - Get the network size
        Nin, Nhidden, Nout = self.shape[:]

        # - Configure the recording mode
        self._configure_accel_time_mode(Nhidden, Nout, record)

        start_timestep = hdkutils.get_current_timestamp(
            self._read_buffer, self._write_buffer
        )
        final_timestamp = start_timestep + len(input) - 1

        # -- Encode input events
        input_events_list = []

        # - Locate input events
        spikes = np.argwhere(input)
        counts = input[np.nonzero(input)]

        # - Generate input events
        for timestep, channel, count in zip(spikes[:, 0], spikes[:, 1], counts):
            for _ in range(count):
                event = samna.xyloImu.event.Spike()
                event.neuron_id = channel
                event.timestamp = start_timestep + timestep
                input_events_list.append(event)

        # - Clear the read and state buffers
        self._state_buffer.reset()

        # - Write the events and trigger the simulation
        self._write_buffer.write(input_events_list)

        # - Read the simulation output data
        xylo_data = hdkutils.read_accel_mode_data(
            self._state_buffer, Nin, Nhidden, Nout, len(input)
        )

        if record:
            rec_dict = {
                "Vmem": np.array(xylo_data.V_mem_hid),
                "Isyn": np.array(xylo_data.I_syn_hid),
                "Spikes": np.array(xylo_data.Spikes_hid),
                "Vmem_out": np.array(xylo_data.V_mem_out),
                "Isyn_out": np.array(xylo_data.I_syn_out),
                "times": np.arange(start_timestep, final_timestamp + 1),
            }
        else:
            rec_dict = {}

        # - This module holds no state
        new_state = {}

        # - Return spike output, new state and record dictionary
        if self._output_mode == "Spike":
            return xylo_data.Spikes_out, new_state, rec_dict
        elif self._output_mode == "Isyn":
            return xylo_data.I_syn_out, new_state, rec_dict
        elif self._output_mode == "Vmem":
            return xylo_data.V_mem_out, new_state, rec_dict
