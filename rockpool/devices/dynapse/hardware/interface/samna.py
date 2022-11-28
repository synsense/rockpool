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

from typing import Any, Dict, Optional, List, Tuple
import time

import numpy as np

# - Rockpool imports
from rockpool.nn.modules.module import Module
from rockpool.devices.dynapse.samna_alias import (
    Dynapse2Destination,
    DeviceInfo,
    Dynapse2Interface,
    Dynapse2Model,
    Dynapse2Configuration,
)

from .utils import (
    aer_to_raster,
    event_generator,
    raster_to_aer,
    capture_events_from_device,
    configure_dynapse2_fpga,
)

# Try to import samna for device interfacing
try:
    import samna
except:
    samna = Any
    print(
        "Device interface requires `samna` package which is not installed on the system"
    )

# - Configure exports
__all__ = ["DynapseSamna"]

DT_FPGA = 1e-6


class DynapseSamna(Module):
    """
    DynapSim solves dynamical chip equations for the DPI neuron and synapse models.
    Receives configuration as bias currents and solves membrane and synapse dynamics using ``jax`` backend.

    :Parameters:
    :param shape: Two dimensions ``(Nin, Nout)``, which defines a input and output conections of DynapSE neurons.
    :type shape: Tuple[int]
    :param device: the Dynan-SE2 the device object to open and configure
    :type device: DeviceInfo
    :param config: a Dynan-SE2 ``samna`` configuration object
    :type config: Dynapse2Configuration
    :param dt: the simulation timestep resolution, defaults to 1e-3
    :type dt: float, optional
    :param control_tag: a tag used in special occacions such as current time reading. Do not capture events with this tag, defaults to 2047
    :type control_tag: int, optional


    """

    def __init__(
        self,
        shape: Tuple[int],
        device: DeviceInfo,
        config: Optional[Dynapse2Configuration] = None,
        dt: float = 1e-3,
        control_tag: int = 2047,
    ):

        if np.size(shape) != 2:
            raise ValueError("`shape` must be a two-element tuple `(Nin, Nout)`.")

        if device is None:
            raise ValueError("`device` must be a valid Dynap-SE2 HDK device.")

        # - Initialise the superclass
        super().__init__(shape=shape, spiking_input=True, spiking_output=True)

        # Configure the FPGA, now only Stack board is available
        self.board: Dynapse2Interface = configure_dynapse2_fpga(device)
        self.dt = dt
        self.dt_fpga = DT_FPGA
        self.control_tag = control_tag

        # Do the initial reading, it prepares the board for further simulation
        self.current_timestamp()

        # Config requires board
        if config is not None:
            self.config = config

    def evolve(
        self,
        input_data: Optional[np.ndarray] = None,
        channel_map: Optional[Dict[int, Dynapse2Destination]] = None,
        read_timeout: float = 1.0,
        offset_fpga: bool = True,
        offset: float = 100e-3,
        record: bool = False,
    ) -> Tuple[np.ndarray, Dict, Dict]:
        """
        evolve simulates the network on Dynap-SE2 HDK in real-time
        The function first converts raster plot to a sequence of AER packages and dispatches to the device.
        Then reads the output buffers

        :param input_data: A raster ``(T, Nin)`` specifying for each bin the number of input events sent to the corresponding input channel on Dynap-SE2, at the corresponding time point.
        :type input_data: np.ndarray
        :param channel_map: the mapping between input timeseries channels and the destinations
        :type channel_map: Optional[Dict[int, Dynapse2Destination]]
        :param read_timeout: the maximum time to wait until reading finishes, defaults to None
        :type read_timeout: float, optional
        :param offset_fpga: offset the timeseries depending on the current FPGA clock, defaults to True
        :type offset_fpga: bool, optional
        :param offset: user defined offset in seconds, defaults to 100e-3
        :type offset: float, optional
        :param record: record the states in each timestep of evolution or not, defaults to False
        :type record: bool, optional
        :return: spikes_ts, states, record_dict
            :spikes_ts: is an array with shape ``(T, Nrec)`` containing the output data(spike raster) produced by the module.
            :states: is a dictionary containing the updated module state following evolution.
            :record_dict: is a dictionary containing the recorded state variables during the evolution at each time step, if the ``record`` argument is ``True`` else empty dictionary {}
        :rtype: Tuple[np.ndarray, Dict, Dict]
        """

        # Read Current FPGA timestamp, offset the events accordingly
        # Flush AER Buffers
        self.flush_aer_buffers()
        if offset_fpga:
            start_time = offset + self.current_timestamp()

        # It's OK not to provide any data
        if input_data is not None:
            assert input_data.shape[1] == self.size_in
            simulation_duration = input_data.shape[0] * self.dt

            # Convert the input data to aer sequence
            event_sequence = raster_to_aer(
                input_data,
                start_time=start_time,
                channel_map=channel_map,
                return_samna=True,
                dt=self.dt,
                dt_fpga=self.dt_fpga,
            )
            # Write AER packages to the bus
            self.board.grid_bus_write_events(event_sequence)

        else:
            simulation_duration = 0.0

        output_events = capture_events_from_device(
            self.board, simulation_duration + read_timeout, control_tag=self.control_tag
        )

        # Read the results
        stop_time = start_time + simulation_duration + read_timeout
        spikes, channel_map = aer_to_raster(
            output_events,
            start_time=start_time,
            stop_time=stop_time,
            dt=self.dt,
            dt_fpga=self.dt_fpga,
        )

        # Return
        states = {
            "channel_map": channel_map,
            "start_time": start_time,
            "stop_time": stop_time,
        }
        record_dict = {"output_events": output_events}

        if record is True:
            record_dict = {}

        return spikes, states, record_dict

    def current_timestamp(
        self,
        reading_interval: float = 10e-3,
        number_of_events: int = 10,
        retry: int = 20,
    ) -> float:
        """
        current_timestamp bounces a dummy event from FPGA to get the exact FPGA time at that moment.

        :param reading_interval: minimum time to wait for the event to bounce back, defaults to 10e-3
        :type reading_interval: float, optional
        :param number_of_events: the number of dummy events to bounce to dispatch, defaults to 10
        :type number_of_events: int, optional
        :param retry: number of retrials in the case that event is not returned back. Each time double the reading interval, defaults to 20
        :type retry: int, optional
        :raises TimeoutError: "FPGA could not respond, increase number of trials or reading interval!"
        :return: the current FPGA time in seconds
        :rtype: float
        """

        # Flush the buffers
        self.flush_aer_buffers()

        # Generate dummy events
        events = [
            event_generator(ts, dt_fpga=self.dt_fpga, tag=self.control_tag)
            for ts in np.arange(
                0, reading_interval, reading_interval / number_of_events
            )
        ]

        # Send dummy event sequence to the device
        self.board.input_interface_write_events(0, events)
        time.sleep(reading_interval)

        # Try to catch them and read the last timestamp
        evs = self.board.output_read()

        for __break in range(retry):
            if len(evs) > 0:
                self.flush_aer_buffers()
                return evs[-1] * self.dt_fpga
            else:
                time.sleep(reading_interval)
                reading_interval *= 2

            evs = self.board.output_read()

        raise TimeoutError(
            f"FPGA could not respond, increase number of trials or reading interval!"
        )

    @property
    def model(self) -> Dynapse2Model:
        """Returns the HDK model object that can be used to configure the device"""
        return self.board.get_model()

    @property
    def config(self) -> Dynapse2Configuration:
        """Returns the configuration object stored on the Dynap-SE2 board"""
        return self.model.get_configuration()

    @config.setter
    def config(self, new_config: Dynapse2Configuration) -> bool:
        # - Write the configuration to the device
        return self.model.apply_configuration(new_config)

    def hard_reset(self, password: Optional[int] = None) -> None:
        """
        hard_reset reset FPGA and whole logic implementation

        :param password: hardcoded password value to prevent this method from users, defaults to None
        :type password: Optional[int], optional
        :raises Warning: Hard reset will flush whole FPGA configuration. If you need to do this, look into the code and find the password!
        :raises ValueError: Wrong password!
        """
        if password is None:
            raise Warning(
                "Hard reset will flush whole FPGA configuration. If you need to do this, look into the code and find the password!"
            )
        elif password == 22051995:
            self.board.reset_fpga()
        else:
            raise ValueError("Wrong password!")

    def flush_aer_buffers(self, chip_mask: int = 0x1):
        """
        flush_aer_buffers applies a logic reset to a set of chips chosen

        :param chip_mask: a selection of chips to flush all buffers, defaults to 0x1 (only chip 1)
        :type chip_mask: int, optional
        :raises ValueError: Could not flush buffers!
        """
        if not self.model.reset(samna.dynapse2.ResetType.LogicReset, chip_mask):
            raise ValueError("Could not flush buffers!")
