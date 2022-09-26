"""
Dynap-SE samna backend bridge
Handles the low-level hardware configuration under the hood and provide easy-to-use access to the user

Note : Existing modules are reconstructed considering consistency with Xylo support.

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

15/09/2022
"""

from __future__ import annotations

from typing import Any

import numpy as np


# - Rockpool imports
from rockpool.nn.modules.module import Module

# - Configure exports
__all__ = ["DynapseSamna"]


class DynapseSamna(Module):
    def __init__(self):
        None

    # @property
    # def config(self):
    #     # - Return the configuration stored on Xylo HDK
    #     return self._device.get_model().get_configuration()

    # @config.setter
    # def config(self, new_config):
    #     # - Test for a valid configuration
    #     is_valid, msg = samna.xylo.validate_configuration(new_config)
    #     if not is_valid:
    #         raise ValueError(f"Invalid configuration for the Xylo HDK: {msg}")

    #     # - Write the configuration to the device
    #     hdkutils.apply_configuration(
    #         self._device, new_config, self._read_buffer, self._write_buffer
    #     )

    # def reset_state(self) -> "XyloSamna":
    #     # - Reset neuron and synapse state on Xylo
    #     hdkutils.reset_neuron_synapse_state(
    #         self._device, self._read_buffer, self._write_buffer
    #     )
    #     return self



class XyloSamna(Module):
    """
    A spiking neuron :py:class:`.Module` backed by the Xylo hardware, via `samna`.

    Use :py:func:`.config_from_specification` to build and validate a configuration for Xylo.

    See Also:

        See the tutorials :ref:`/devices/xylo-overview.ipynb` and :ref:`/devices/torch-training-spiking-for-xylo.ipynb` for a high-level overview of building and deploying networks for Xylo.

    """

    def __init__(
        self,
        device: XyloHDK,
        config: XyloConfiguration = None,
        dt: float = 1e-3,
        output_mode: str = "Spike",
        *args,
        **kwargs,
    ):
        """
        Instantiate a Module with Xylo dev-kit backend

        Args:
            device (XyloHDK): An opened `samna` device to a Xylo dev kit
            config (XyloConfiguraration): A Xylo configuration from `samna`
            dt (float): The simulation time-step to use for this Module
            output_mode (str): The readout mode for the Xylo device. This must be one of ``["Spike", "Isyn", "Vmem"]``. Default: "Spike", return events from the output layer.
        """
        # - Check input arguments
        if device is None:
            raise ValueError("`device` must be a valid, opened Xylo HDK device.")

        # - Check output mode specification
        if output_mode not in ["Spike", "Isyn", "Vmem"]:
            raise ValueError(
                f'{output_mode} is not supported. Must be one of `["Spike", "Isyn", "Vmem"]`.'
            )
        self._output_mode = output_mode

        # - Get a default configuration
        if config is None:
            config = samna.xylo.configuration.XyloConfiguration()

        # - Get the network shape
        Nin, Nhidden = np.shape(config.input.weights)
        _, Nout = np.shape(config.readout.weights)

        # - Initialise the superclass
        super().__init__(
            shape=(Nin, Nhidden, Nout), spiking_input=True, spiking_output=True
        )

        # - Register buffers to read and write events, monitor state
        self._read_buffer = hdkutils.new_xylo_read_buffer(device)
        self._write_buffer = hdkutils.new_xylo_write_buffer(device)
        self._state_buffer = hdkutils.new_xylo_state_monitor_buffer(device)

        # - Initialise the xylo HDK
        hdkutils.initialise_xylo_hdk(self._write_buffer)

        # - Check that we can access the device node, and that it's a Xylo HDK
        if not hdkutils.verify_xylo_version(
            self._read_buffer, self._write_buffer, timeout=10.0
        ):
            raise ValueError(
                "Cannot verify HDK version. `device` must be an opened Xylo HDK."
            )

        # - Store the device
        self._device: XyloHDK = device
        """ `.XyloHDK`: The Xylo HDK used by this module """

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

        # - Zero neuron state when building a new module
        self.reset_state()

    @property
    def config(self):
        # - Return the configuration stored on Xylo HDK
        return self._device.get_model().get_configuration()

    @config.setter
    def config(self, new_config):
        # - Test for a valid configuration
        is_valid, msg = samna.xylo.validate_configuration(new_config)
        if not is_valid:
            raise ValueError(f"Invalid configuration for the Xylo HDK: {msg}")

        # - Write the configuration to the device
        hdkutils.apply_configuration(
            self._device, new_config, self._read_buffer, self._write_buffer
        )

        # - Store the configuration locally
        self._config = new_config

    def reset_state(self) -> "XyloSamna":
        # - Reset neuron and synapse state on Xylo
        hdkutils.reset_neuron_synapse_state(
            self._device, self._read_buffer, self._write_buffer
        )
        return self

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

    def evolve(
        self,
        input: np.ndarray,
        record: bool = False,
        read_timeout: float = None,
        *args,
        **kwargs,
    ) -> (np.ndarray, dict, dict):
        """
        Evolve a network on the Xylo HDK in accelerated-time mode

        Sends a series of events to the Xylo HDK, evolves the network over the input events, and returns the output events produced during the input period. Optionally record internal state of the network, selectable with the ``record`` flag.

        Args:
            input (np.ndarray): A raster ``(T, Nin)`` specifying for each bin the number of input events sent to the corresponding input channel on Xylo, at the corresponding time point. Up to 15 input events can be sent per bin.
            record (bool): Iff ``True``, record and return all internal state of the neurons and synapses on Xylo. Default: ``False``, do not record internal state.
            read_timeout (Optional[float]): Set an explicit read timeout for the entire simulation time. This should be sufficient for the simulation to complete, and for data to be returned. Default: ``None``, set a reasonable default timeout.

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

        # - Get current timestamp
        start_timestep = hdkutils.get_current_timestamp(
            self._read_buffer, self._write_buffer
        )
        final_timestep = start_timestep + len(input) - 1

        # -- Encode input events
        input_events_list = []

        # - Locate input events
        spikes = np.argwhere(input)
        counts = input[np.nonzero(input)]

        # - Generate input events
        for timestep, channel, count in zip(spikes[:, 0], spikes[:, 1], counts):
            for _ in range(count):
                event = samna.xylo.event.Spike()
                event.neuron_id = channel
                event.timestamp = start_timestep + timestep
                input_events_list.append(event)

        # - Add an extra event to ensure readout for entire input extent
        event = samna.xylo.event.Spike()
        event.timestamp = final_timestep + 1
        input_events_list.append(event)

        # - Clear the input event count register to make sure the dummy event is ignored
        for addr in [0x0C, 0x0D, 0x0E, 0x0F]:
            event = samna.xylo.event.WriteRegisterValue()
            event.address = addr
            input_events_list.append(event)

        # - Clear the read and state buffers
        self._state_buffer.reset()
        self._read_buffer.get_events()

        # - Write the events and trigger the simulation
        self._write_buffer.write(input_events_list)

        # - Determine a reasonable read timeout
        if read_timeout is None:
            read_timeout = len(input) * self.dt * Nhidden / 800.0
            read_timeout = read_timeout * 10.0 if record else read_timeout

        # - Wait until the simulation is finished
        read_events, is_timeout = hdkutils.blocking_read(
            self._read_buffer,
            timeout=max(read_timeout, 1.0),
            target_timestamp=final_timestep,
        )

        if is_timeout:
            message = f"Processing didn't finish for {read_timeout}s. Read {len(read_events)} events"
            readout_events = [e for e in read_events if hasattr(e, "timestamp")]

            if len(readout_events) > 0:
                message += f", first timestamp: {readout_events[0].timestamp}, final timestamp: {readout_events[-1].timestamp}, target timestamp: {final_timestep}"
            raise TimeoutError(message)

        # - Read the simulation output data
        xylo_data = hdkutils.read_accel_mode_data(
            self._state_buffer, Nin, Nhidden, Nout
        )

        if record:
            # - Build a recorded state dictionary
            rec_dict = {
                "Vmem": np.array(xylo_data.V_mem_hid),
                "Isyn": np.array(xylo_data.I_syn_hid),
                "Isyn2": np.array(xylo_data.I_syn2_hid),
                "Spikes": np.array(xylo_data.Spikes_hid),
                "Vmem_out": np.array(xylo_data.V_mem_out),
                "Isyn_out": np.array(xylo_data.I_syn_out),
                "times": np.arange(start_timestep, final_timestep + 1),
            }
        else:
            rec_dict = {}

        # - This module accepts no state
        new_state = {}

        # - Return spike output, new state and record dictionary
        if self._output_mode == "Spike":
            return xylo_data.Spikes_out, new_state, rec_dict
        elif self._output_mode == "Isyn":
            return xylo_data.I_syn_out, new_state, rec_dict
        elif self._output_mode == "Vmem":
            return xylo_data.V_mem_out, new_state, rec_dict
