"""
Dynap-SE samna backend bridge
Handles the low-level hardware configuration under the hood and provide easy-to-use access to the user

* With the current samna support, it's impossible to reset the FPGA time-stamp counter without re-configuring the FPGA.
* Since underlying hardware is analog, state resetting is done by forcing the capacitors to discarge.
    * Uploding a configuration object with max leak currents (both the membrane and the synapses)
* 

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

15/09/2022
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple
import logging
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
    NormalGridEvent,
)

from rockpool.devices.dynapse.lookup import SE2_STACK_FPGA_FILEPATH

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
    :param dt_fpga: the FPGA timestep resolution, defaults to 1e-6
    :type dt_fpga: float, optional
    :param control_tag: a tag used in special occacions such as current time reading. Do not capture events with this tag and control_hop, defaults to 2047
    :type control_tag: int, optional
    :param control_hop: a chip position (-7 means x_hop=-7, y_hop=-7) which does not really exist, works in cooperation with control_tag. Do not capture events coming from this hop and control tag, defauts to -7.
    :type control_hop: int, optional
    """

    def __init__(
        self,
        shape: Tuple[int],
        device: DeviceInfo,
        config: Dynapse2Configuration,
        dt: float = 1e-3,
        dt_fpga: float = 1e-6,
        control_tag: int = 2047,
        control_hop: int = -7,
    ):

        if np.size(shape) != 2:
            raise ValueError("`shape` must be a two-element tuple `(Nin, Nout)`.")

        if device is None:
            raise ValueError("`device` must be a valid Dynap-SE2 HDK device.")

        # - Initialise the superclass
        super().__init__(shape=shape, spiking_input=True, spiking_output=True)

        self.dt = dt
        self.dt_fpga = dt_fpga
        self.control_tag = control_tag
        self.control_hop = control_hop

        # Configure the FPGA, now only Stack board is available
        self.board: Dynapse2Interface = self.configure_dynapse2_fpga(device)

        # Make reset and set state configurations ready
        self.app_config = config
        self.leaky_config = self.get_leaky_config()

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

    def evolve(
        self,
        input_data: np.ndarray,
        channel_map: Optional[Dict[int, Dynapse2Destination]] = None,
        read_timeout: float = 60.0,
        offset: float = 100e-3,
        poll_step: float = 1e-3,
        record: bool = True,
    ) -> Tuple[np.ndarray, Dict, Dict]:
        """
        evolve simulates the network on Dynap-SE2 HDK in real-time
        The function first converts raster plot to a sequence of AER packages and dispatches to the device.
        Then reads the output buffers

        :param input_data: A raster ``(T, Nin)`` specifying for each bin the number of input events sent to the corresponding input channel on Dynap-SE2, at the corresponding time point.
        :type input_data: np.ndarray
        :param channel_map: the mapping between input timeseries channels and the destinations
        :type channel_map: Optional[Dict[int, Dynapse2Destination]]
        :param read_timeout: the maximum time to wait until reading finishes, defaults to 60.0
        :type read_timeout: float, optional
        :param offset: user defined start time offset in seconds, defaults to 100e-3
        :type offset: float, optional
        :param poll_step: the pollling step, 1 ms means the CPU fetches events from FPGA in every 1 ms, defaults to 1e-3
        :type poll_step: float, optional
        :param record: record the states in each timestep of evolution or not, defaults to False
        :type record: bool, optional
        :return: spikes_ts, states, record_dict
            :spikes_ts: is an array with shape ``(T, Nrec)`` containing the output data(spike raster) produced by the module.
            :states: is a dictionary containing the updated module state following evolution.
            :record_dict: is a dictionary containing the recorded state variables during the evolution at each time step, if the ``record`` argument is ``True`` else empty dictionary {}
        :rtype: Tuple[np.ndarray, Dict, Dict]
        """

        # Get the simulation duration from the input provided
        assert input_data.shape[1] == self.size_in
        simulation_duration = input_data.shape[0] * self.dt

        # Load the config only if it's absolutely neceassary; keep the neurons silent for all the other time
        self.apply_config()

        # Read Current FPGA timestamp, it will make sure the configuration loading is complete and we could offset the events accordingly
        start_time = offset + self.current_timestamp()

        # Convert the input data to aer sequence
        event_sequence = self.raster_to_aer(
            input_data, start_time=start_time, channel_map=channel_map
        )

        # Write AER packages to the bus
        self.board.grid_bus_write_events(event_sequence)

        ## Start reading the buffers
        output_events, done, _ = self.poll_events(simulation_duration, poll_step)

        # Stop the activity of the neurons
        self.discharge_capacitors()

        if not done:
            logging.info("Simulation duration was not enough to read the buffers.")
            extension, done, _ = self.poll_events(read_timeout, poll_step)
            output_events.extend(extension)

        if not done:
            logging.warn("Buffers are still not empty!!! Increase the read timeout!")

        # Convert the results to meaningful raster plots
        stop_time = start_time + simulation_duration
        spikes, channel_map = self.aer_to_raster(
            output_events,
            start_time=start_time,
            stop_time=stop_time,
        )

        # Return
        states = {}
        record_dict = {}

        if record is True:
            record_dict = {
                "output_events": output_events,
                "channel_map": channel_map,
                "start_time": start_time,
                "stop_time": stop_time,
            }

        return spikes, states, record_dict

    ### --- Configuration --- ###

    @staticmethod
    def configure_dynapse2_fpga(
        device: DeviceInfo,
        bitfile: Optional[str] = None,
    ) -> Dynapse2Interface:
        """
        configure_dynapse2_fpga configures the FPGA on board and builds a connection node between CPU and the device.
        It allows one to configure the device, read or write AER events to bus, and monitor the activity of device neurons

        :param device: the device object to open and configure
        :type device: DeviceInfo
        :param bitfile: the bitfile path if known, defaults to None
        :type bitfile: Optional[str], optional
        :raises IOError: Failed to configure Opal Kelly
        :return: an open and configured Dynan-SE2 interface node
        :rtype: Dynapse2Interface
        """

        device = samna.device.open_device(device)

        if bitfile is None:
            bitfile = SE2_STACK_FPGA_FILEPATH

        if not device.configure_opal_kelly(bitfile):
            raise IOError("Failed to configure Opal Kelly")

        logging.info(
            f"{device.get_device_type_name()} is connected, configured and ready for operation!"
        )

        return device

    def disconnect(self) -> None:
        """disconnect breaks the connection between CPU and the device"""
        logging.warn(f"{self.board.get_device_type_name()} disconnected!")
        return samna.device.close_device(self.board)

    ### --- State Handling --- ###

    @staticmethod
    def get_leaky_config() -> Dynapse2Configuration:
        """get_leaky_config returns a configuration object to discharge all the capacitors on chip"""

        leak_biases = [
            "SOIF_LEAK_N",
            "DEAM_ETAU_P",
            "DEGA_ITAU_P",
            "DENM_ETAU_P",
            "DESC_ITAU_P",
        ]

        config: Dynapse2Configuration = samna.dynapse2.Dynapse2Configuration()

        for chip in config.chips:
            for core in chip.cores:
                for bias in leak_biases:
                    core.parameters[bias].coarse_value = 5
                    core.parameters[bias].fine_value = 255

        return config

    def discharge_capacitors(self) -> None:
        """discharge_capacitors applies the leaky configuration object to the hardware model in order to discharge all the capacitors"""
        self.model.clear_error_queue()
        self.config = self.leaky_config

    def apply_config(self) -> None:
        """apply_config writes the configuration object to the device"""
        self.model.clear_error_queue()
        self.config = self.app_config

    ### --- AER Handling --- ###

    def poll_events(
        self,
        duration: float,
        poll_step: float = 1e-3,
    ) -> Tuple[List[NormalGridEvent], bool, float]:
        """
        poll_events records the device's output and stores in an event buffer

        :param duration: the maximum duration that the events will be captured
        :type duration: float
        :param poll_step: the pollling step, 1 ms means the CPU fetches events from FPGA in every 1 ms, defaults to 1e-3
        :type poll_step: float, optional
        :return: output_events, done, t_done
            :output_events: the event buffer, a list of Dynap-SE2 AER packages captured
            :done: a bloolean flag indicating if all the events read or not
            :t_done: the time in seconds that the reading is done.
        :rtype: Tuple[List[NormalGridEvent], bool, float]
        """

        output_events = []
        done = False

        # Initial time
        tic = toc = time.time()

        # Clear Errors first
        self.model.clear_error_queue()

        # Fixed duration Polling
        while (not done) and (toc - tic < duration):

            buffer = self.board.read_events()
            if len(buffer) > 0:
                for data in buffer:
                    if not self.if_control_event(data):
                        output_events.append(NormalGridEvent.from_samna(data))
                    else:
                        done = True
                        t_done = buffer[-1].timestamp * self.dt_fpga
            time.sleep(poll_step)
            toc = time.time()

        t_done = toc if not done else t_done
        return output_events, done, t_done

    def current_timestamp(
        self,
        timeout: float = 20.0,
        poll_step: float = 1e-3,
    ) -> float:
        """
        current_timestamp bounces a dummy event from FPGA to get the exact FPGA time at that moment.

        :param timeout: the maximum timeout limit in seconds, defaults to 20.0
        :type timeout: float, optional
        :param poll_step: the pollling step, 1 ms means the CPU fetches events from FPGA in every 1 ms, defaults to 1e-3
        :type poll_step: float, optional
        :raises TimeoutError: FPGA could not respond in {timeout} seconds!
        :return: the current FPGA time in seconds
        :rtype: float
        """

        # Send dummy event sequence to the device
        self.board.input_interface_write_events(0, self.control_sequence())

        # Read those events
        _, done, t_done = self.poll_events(timeout, poll_step)

        if not done:
            raise TimeoutError(
                f"FPGA could not respond in {timeout} seconds!"
            )
        else:
            return t_done

    def event_generator(
        self,
        event_time: float,
        core: List[bool] = [True, True, True, True],
        x_hop: int = -7,
        y_hop: int = -7,
        tag: np.uint = 2047,
    ) -> NormalGridEvent:
        """
        event_generator a Dynap-SE2 event generator utility function, can be used to generate dummy events

        :param event_time: the time that the event happened in seconds
        :type event_time: float
        :param core: the core mask used while sending the events, defaults to [True, True, True, True]
                [1,1,1,1] means all 4 cores are on the target
                [0,0,1,0] means the event will arrive at core 2 only
        :type core: List[bool], optional
        :param x_hop: number of chip hops on x axis, defaults -7
        :type x_hop: int, optional
        :param y_hop: number of chip hops on y axis, defaults to -7
        :type y_hop: int, optional
        :param tag: globally multiplexed locally unique event tag which is used to identify the connection between two neurons, defaults to 2047
        :type tag: np.uint, optional
        :return: a virtual samna AER package for DynapSE2
        :rtype: NormalGridEvent
        """

        event = NormalGridEvent(
            event=Dynapse2Destination(core, x_hop, y_hop, tag),
            timestamp=int(event_time / self.dt_fpga),
        ).to_samna()

        return event

    ### --- Sanity Check --- ###

    def control_event(self, event_time: float = 0.0) -> NormalGridEvent:
        """
        control_event generates a dummy event to be bounced back from the FPGA
        This event helps to read the current FPGA time or understand the simulation is done

        :param event_time: the time that the event happened in seconds
        :type event_time: float
        :return: a virtual samna AER package for control
        :rtype: NormalGridEvent
        """

        return self.event_generator(
            event_time,
            core=[True, True, True, True],
            x_hop=self.control_hop,
            y_hop=self.control_hop,
            tag=self.control_tag,
        )

    def control_sequence(
        self, event_time: float = 0.0, num_events: int = 3
    ) -> List[NormalGridEvent]:
        """
        control_sequence creates a sequence of control events.
        In general, one control event is not enough because it can easily get lost.

        :param num_events: number of event to append to the list, defaults to 3
        :type num_events: int, optional
        :return: a sequence of control events
        :rtype: List[NormalGridEvent]
        """
        return [self.control_event(event_time) for _ in range(num_events)]

    def if_control_event(self, event: NormalGridEvent) -> bool:
        """
        if_control_event returns true if the event is a control event which is bounced back from FPGA

        :param event: any Dynap-SE2 AER package to check
        :type event: NormalGridEvent
        :return: true if the package is a control event bounced back from FPGA
        :rtype: bool
        """
        return (
            (event.event.tag == self.control_tag)
            and (event.event.x_hop == self.control_hop + 1)
            and (event.event.y_hop == self.control_hop)
        )

    ### --- IO Handling --- ###

    def raster_to_aer(
        self,
        raster: np.ndarray,
        start_time: float = 0.0,
        channel_map: Optional[Dict[int, Dynapse2Destination]] = None,
    ) -> List[NormalGridEvent]:
        """
        raster_to_aer converts a discrete raster record to a list of AER packages.
        It uses a channel map to map the channels to destinations, and by default it returns a list of samna objects.

        :param raster: the discrete timeseries to be converted into list of Dynap-SE2 AER packages
        :type raster: np.ndarray
        :param start_time: the start time of the record in seconds, defaults to 0.0
        :type start_time: float
        :param channel_map: the mapping between timeseries channels and the destinations
        :type channel_map: Optional[Dict[int, Dynapse2Destination]]
        :raises ValueError: Raster should be 2 dimensional!
        :raises ValueError: Channel map does not map the channels of the timeseries provided!
        :return: a list of Dynap-SE2 AER packages
        :rtype: List[NormalGridEvent]
        """

        if len(raster.shape) != 2:
            raise ValueError("Raster should be 2 dimensional!")

        buffer = []
        duration = raster.shape[0] * self.dt
        num_channels = raster.shape[1]
        __time_course = np.arange(start_time, start_time + duration, self.dt)

        # Default channel map is NOT RECOMMENDED!
        if channel_map is None:
            channel_map = self.default_channel_map(num_channels)

        if not num_channels <= len(set(channel_map.keys())):
            raise ValueError(
                "Channel map does not map the channels of the timeseries provided!"
            )

        # Create the AER list
        for spikes, time in zip(raster, __time_course):

            destinations = np.argwhere(spikes).flatten()
            timestamp = int(np.around((time / self.dt_fpga)))
            events = [
                NormalGridEvent(channel_map[dest], timestamp + i).to_samna()
                for i, dest in enumerate(destinations)
            ]
            buffer.extend(events)

        # Append control events
        buffer.extend(self.control_sequence(__time_course[-1]))

        return buffer

    def default_channel_map(self, num_channels: int) -> Dict[int, Dynapse2Destination]:
        """
        default_channel_map creates a dummy channel map which helps to bounce the AER events back from the FPGA

        :param num_channels: number of input channels
        :type num_channels: int
        :return: a channel map which assigns the neuron ids as tags, and use the control hop to bounce the neurons
        :rtype: Dict[int, Dynapse2Destination]
        """
        channel_map = {
            c: Dynapse2Destination(
                core=[True, True, True, True],
                x_hop=self.control_hop,
                y_hop=self.control_hop,
                tag=c,
            )
            for c in range(num_channels)
        }
        return channel_map

    def aer_to_raster(
        self,
        buffer: List[NormalGridEvent],
        stop_time: float,
        start_time: float = 0,
    ) -> Tuple[np.ndarray, Dict[int, Dynapse2Destination]]:
        """
        aer_to_raster converts a list of Dynap-SE2 AER packages to a discrete raster record
        The events does not meet the start and stop time criteria are descarded

        :param buffer: the event buffer, a list of Dynap-SE2 AER packages
        :type buffer: List[NormalGridEvent]
        :param stop_time: the stop time cut-off for the events.
        :type stop_time: float
        :param start_time: the start time cut-off for the events, defaults to 0
        :return: ts, cmap
            raster_out: the raster record referenced on the event buffer
            cmap: the mapping between raster channels and the destinations
        :rtype: Tuple[np.ndarray, Dict[int, Dynapse2Destination]]
        """

        # Get a reverse channel map
        cmap = self.extract_channel_map(buffer)
        rcmap = {v: k for k, v in cmap.items()}

        # Create the event/channel lists
        times = []
        channels = []
        for event in buffer:
            times.append(event.timestamp * self.dt_fpga)
            channels.append(rcmap[event.event])

        # sort time and channel arrays in the same order
        idx = np.argsort(times)
        times = np.array(times)[idx]
        channels = np.array(channels)[idx]

        # generate the output raster
        time_course = np.arange(start_time, stop_time, self.dt)
        raster_out = np.zeros((len(time_course), len(cmap)))

        # Save the data meeting the start and stop time criteria and discard the rest
        for i, t in enumerate(times):
            idx = np.searchsorted(time_course, t)
            if idx < len(raster_out):
                raster_out[idx][channels[i]] += 1
        return raster_out, cmap

    @staticmethod
    def extract_channel_map(
        buffer: List[NormalGridEvent],
    ) -> Dict[int, Dynapse2Destination]:
        """
        extract_channel_map obtains a channel map from a list of dummy AER packages (samna alias)

        :param buffer: the list of AER packages
        :type buffer: List[NormalGridEvent]
        :return: the mapping between timeseries channels and the destinations
        :rtype: Dict[int, Dynapse2Destination]
        """
        destinations = []

        for data in buffer:
            if data.event not in destinations:
                destinations.append(data.event)

        channel_map = dict(zip(range(len(destinations)), destinations))

        return channel_map
