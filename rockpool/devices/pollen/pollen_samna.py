"""
Samna-backed bridge to Pollen dev kit
"""

# - Check that Samna is installed
from importlib import util

if util.find_spec("samna") is None:
    raise ModuleNotFoundError(
        "'samna' not found. Modules that rely on Samna will not be available."
    )

# - Samna imports
import samna
from samna.pollen.configuration import (
    PollenConfiguration,
    ReservoirNeuron,
    OutputNeuron,
)

from samna.pollen import validate_configuration

# - Rockpool imports
from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter

from ..pollen import pollen_devkit_utils as putils
from ..pollen.pollen_devkit_utils import PollenDaughterBoard

# - Numpy
import numpy as np

# - Typing
from typing import Optional, Union

# - JSON
import json

# - Configure exports
__all__ = ["config_from_specification", "save_config", "load_config", "PollenSamna"]


def config_from_specification(
    weights_in: np.ndarray,
    weights_out: np.ndarray,
    weights_rec: np.ndarray = None,
    dash_mem: np.ndarray = None,
    dash_mem_out: np.ndarray = None,
    dash_syn: np.ndarray = None,
    dash_syn_2: np.ndarray = None,
    dash_syn_out: np.ndarray = None,
    threshold: np.ndarray = None,
    threshold_out: np.ndarray = None,
    weight_shift_in: int = 0,
    weight_shift_rec: int = 0,
    weight_shift_out: int = 0,
    aliases: Optional[list] = None,
) -> (PollenConfiguration, bool, str):
    """
    Convert a full network specification to a pollen config and validate it

    See Also:
        For detailed information about the networks supported on Pollen, see :ref:`/devices/pollen-overview.ipynb`

    Args:
        weights_in (np.ndarray): A quantised 8-bit input weight matrix ``(Nin, Nin_res, 2)``. The third dimension specifies connections onto the second input synapse for each neuron. ``Nin_res`` indicates the number of hidfden-layer neurons that receive input from the input channels.
        weights_rec (np.ndarray): A quantised 8-bit recurrent weight matrix ``(Nhidden, Nhidden, 2)``. The third dimension specified connections onto the second input synapse for each neuron. Default: ``0``
        weights_out (np.ndarray): A quantised 8-bit output weight matrix ``(Nhidden, Nout)``.
        dash_mem (np.ndarray): A vector or list ``(Nhidden,)`` specifing decay bitshift for neuron state for each hidden layer neuron. Default: ``1``
        dash_mem_out (np.ndarray): A vector or list ``(Nout,)`` specifing decay bitshift for neuron state for each output neuron. Default: ``1``
        dash_syn (np.ndarray): A vector or list ``(Nhidden,)`` specifing decay bitshift for synapse 1 state for each hidden layer neuron. Default: ``1``
        dash_syn_2 (np.ndarray): A vector or list ``(Nhidden,)`` specifing decay bitshift for synapse 2 state for each hidden layer neuron. Default: ``1``
        dash_syn_out (np.ndarray): A vector or list ``(Nout,)`` specifing decay bitshift for synapse state for each output layer neuron. Default: ``1``
        threshold (np.ndarray): A vector or list ``(Nhidden,)`` specifing the firing threshold for each hidden layer neuron. Default: ``0``
        threshold_out (np.ndarray): A vector or list ``(Nout,)`` specifing the firing threshold for each output layer neuron. Default: ``0``
        weight_shift_in (int): The number of bits to left-shift each input weight. Default: ``0``
        weight_shift_rec (int): The number of bits to left-shift each recurrent weight. Default: ``0``
        weight_shift_out (int): The number of bits to left-shift each output layer weight. Default: ``0``
        aliases (Optional[list]): For each neuron in the hidden population, a list containing the alias targets for that neuron

    Returns: (:py:class:`.samna.pollen.PollenConfiguration`, bool, str): config, is_valid, message
        ``config`` will be a `PollenConfiguration`.
        ``is_valid`` will be a boolean flag ``True`` iff the configuration is valid.
        ``message`` will be an empty string if the configuration is valid, or a message indicating why the configuration is invalid.
    """
    # - Check input weights
    if weights_in.ndim != 3:
        raise ValueError("Input weights must be 3 dimensional `(Nin, Nin_res, 2)`")

    # - Check output weights
    if weights_out.ndim != 2:
        raise ValueError("Output weights must be 2 dimensional `(Nhidden, Nout)`")

    # - Get network shape
    Nin, Nin_res, _ = weights_in.shape
    Nhidden, Nout = weights_out.shape

    # - Provide default `weights_rec`
    weights_rec = (
        np.zeros((Nhidden, Nhidden, 2), "int") if weights_rec is None else weights_rec
    )

    if weights_rec.ndim != 3 or weights_rec.shape[0] != weights_rec.shape[1]:
        raise ValueError("Recurrent weights must be of shape `(Nhidden, Nhidden, 2)`")

    if Nhidden != weights_rec.shape[0]:
        raise ValueError(
            "Input weights must be consistent with recurrent weights.\n"
            f"`weights_in`: {weights_in.shape}; `weights_rec`: {weights_rec.shape}"
        )

    # - Check aliases
    if aliases is not None and len(aliases) != Nhidden:
        raise ValueError(
            f"Aliases list must have `Nhidden` entries (`Nhidden` = {Nhidden})"
        )

    # - Check bitshift TCs, assign defaults
    dash_mem = np.ones(Nhidden, "int") if dash_mem is None else dash_mem
    dash_syn = np.ones(Nhidden, "int") if dash_syn is None else dash_syn
    dash_syn_2 = np.ones(Nhidden, "int") if dash_syn_2 is None else dash_syn_2

    if (
        np.size(dash_mem) != Nhidden
        or np.size(dash_syn) != Nhidden
        or np.size(dash_syn_2) != Nhidden
    ):
        raise ValueError(
            f"`dash_mem`, `dash_syn` and `dash_syn_2` need `Nhidden` entries (`Nhidden` = {Nhidden})"
            + f" found {np.size(dash_mem)}, {np.size(dash_syn)}, {np.size(dash_syn_2)}"
        )

    dash_mem_out = np.ones(Nout, "int") if dash_mem_out is None else dash_mem_out
    dash_syn_out = np.ones(Nout, "int") if dash_syn_out is None else dash_syn_out

    if np.size(dash_mem_out) != Nout or np.size(dash_syn_out) != Nout:
        raise ValueError(
            f"`dash_mem_out` and `dash_syn_out` need `Nout` entries (`Nout` = {Nout})"
        )

    # - Check thresholds, assign defaults
    threshold = np.zeros(Nhidden, "int") if threshold is None else threshold
    threshold_out = np.zeros(Nout, "int") if threshold_out is None else threshold_out

    if threshold.size != Nhidden:
        raise ValueError(
            f"`thresholds` needs `Nhidden` entries (`Nhidden` = {Nhidden})"
        )

    if threshold_out.size != Nout:
        raise ValueError(f"`thresholds_out` needs `Nout` entries (`Nout` = {Nout})")

    # - Build the configuration
    config = PollenConfiguration()
    config.debug.clock_enable = True
    config.synapse2_enable = True
    config.reservoir.aliasing = aliases is not None
    config.input.weight_bit_shift = weight_shift_in
    config.reservoir.weight_bit_shift = weight_shift_rec
    config.readout.weight_bit_shift = weight_shift_out

    config.input.weights = weights_in[:, :, 0]
    config.input.syn2_weights = weights_in[:, :, 1]
    config.reservoir.weights = weights_rec[:, :, 0]
    config.reservoir.syn2_weights = weights_rec[:, :, 1]
    config.readout.weights = weights_out

    reservoir_neurons = []
    for i in range(len(weights_rec)):
        neuron = ReservoirNeuron()
        if aliases is not None and len(aliases[i]) > 0:
            neuron.alias_target = aliases[i][0]
        neuron.i_syn_decay = dash_syn[i]
        neuron.i_syn2_decay = dash_syn_2[i]
        neuron.v_mem_decay = dash_mem[i]
        neuron.threshold = threshold[i]
        reservoir_neurons.append(neuron)

    config.reservoir.neurons = reservoir_neurons

    readout_neurons = []
    for i in range(np.shape(weights_out)[1]):
        neuron = OutputNeuron()
        neuron.i_syn_decay = dash_syn_out[i]
        neuron.v_mem_decay = dash_mem_out[i]
        neuron.threshold = threshold_out[i]
        readout_neurons.append(neuron)

    config.readout.neurons = readout_neurons

    # - Validate the configuration and return
    is_valid, message = validate_configuration(config)
    return config, is_valid, message


def save_config(config: PollenConfiguration, filename: str) -> None:
    """
    Save a Pollen configuration to disk in JSON format

    Args:
        config (PollenConfiguration): The configuration to write
        filename (str): The filename to write to
    """
    with open(filename, "w") as f:
        f.write(config.to_json())


def load_config(filename: str) -> PollenConfiguration:
    """
    Read a Pollen configuration from disk in JSON format
    Args:
        filename (str): The filename to read from

    Returns: PollenConfiguration: The configuration loaded from disk
    """
    # - Create a new config object
    conf = PollenConfiguration()

    # - Read the configuration from file
    with open(filename) as f:
        conf.from_json(f.read())

    # - Return the configuration
    return conf


class PollenSamna(Module):
    """
    A spiking neuron :py:class:`Module` backed by the Pollen hardware, via `samna`

    Use :py:func:`.config_from_specification` to build and validate a configuration for Pollen. See :ref:`/devices/pollen-overview.ipynb` for more information about the Pollen development kit, and supported networks.
    """

    def __init__(
        self,
        device: PollenDaughterBoard,
        config: PollenConfiguration = None,
        dt: float = 1e-3,
        *args,
        **kwargs,
    ):
        """
        Instantiate a Module with Pollen dev-kit backend

        Args:
            device (PollenDaughterBoard): An opened `samna` device to a Pollen dev kit
            config (PollenConfiguraration): A Pollen configuration from `samna`
            dt (float): The simulation time-step to use for this Module
        """
        # - Get the network shape
        Nin, Nhidden = np.shape(config.input.weights)
        _, Nout = np.shape(config.readout.weights)

        # - Initialise the superclass
        super().__init__(
            shape=(Nin, Nhidden, Nout), spiking_input=True, spiking_output=True
        )

        # - Initialise the pollen HDK
        putils.initialise_pollen_hdk(device)

        # - Register a buffer to read events from Pollen
        self._event_buffer = putils.new_pollen_output_buffer(device)

        # - Check that we can access the device node, and that it's a Pollen HDK daughterboard
        if not putils.verify_pollen_version(device, self._event_buffer):
            raise ValueError("`device` must be an opened Pollen HDK daughter board.")

        # - Store the device
        self._device = device

        # - Get a default configuration
        if config is None:
            config = samna.pollen.configuration.PollenConfiguration()

        # - Store the configuration (and apply it)
        self.config: Union[
            PollenConfiguration, SimulationParameter
        ] = SimulationParameter(init_func=lambda _: config)
        """ `.PollenConfiguration`: The configuration of the Pollen module """

        # - Store the timestep

        # - Zero neuron state when building a new module
        self.reset_state()

    @property
    def config(self):
        # - Return the locally stored config
        return self._config

        # - Reading the configuration is not yet supported
        # return self._device_model.get_configuration()

    @config.setter
    def config(self, new_config):
        # - Write the configuration to the device
        putils.apply_configuration(self._device, new_config)

        # - Store the configuration locally, since reading is not supported
        self._config = new_config

    def reset_state(self) -> "PollenSamna":
        # - Reset neuron and synapse state on Pollen
        Nhidden, Nout = self.shape[-2:]
        putils.reset_neuron_synapse_state(self._device, self._config, Nhidden, Nout)

    def _evolve_record(self, input: np.ndarray) -> (np.ndarray, dict, dict):
        """
        Evolve Pollen HDK over an input step by step, recording all state

        Args:
            input (np.ndarray): Raster of input events with shape ``(T, Nin)``

        Returns: (np.array, dict, dict): output_raster, {}, recorded_dict
            ``recorded_dict`` is a dictionary containing the recorded state of the Pollen neurons over time
        """
        # - Ensure Pollen HDK is in manual mode, and enable reading memory
        conf = self._config
        # conf.debug.clock_enable = True
        conf.operation_mode = samna.pollen.configuration.OperationMode.AcceleratedTime

        # - Apply the configuration
        putils.apply_configuration(self._device, conf)

        # - Get network shape
        Nhidden, Nout = self.shape[-2:]

        # - Wait until Pollen is ready
        putils.is_pollen_ready(self._device, self._event_buffer)

        # - Reset input spike registers
        putils.reset_input_spikes(self._device)

        vmem_ts = []
        isyn_ts = []
        isyn2_ts = []
        vmem_out_ts = []
        isyn_out_ts = []
        spikes_ts = []
        output_ts = []

        # - Loop over time steps
        for timestep in range(input.shape[0]):
            # - Send input events for this time-step
            putils.send_immediate_input_spikes(self._device, input[timestep])

            # - Evolve one time-step on Pollen
            putils.advance_time_step(self._device)

            # - Wait until pollen is finished the time-step
            while not putils.is_pollen_ready(self._device, self._event_buffer):
                pass

            # - Read all synapse and neuron states for this time step
            this_state = putils.read_neuron_synapse_state(
                self._device, self._event_buffer, Nhidden, Nout
            )
            vmem_ts.append(this_state.V_mem_hid)
            isyn_ts.append(this_state.I_syn_hid)
            isyn2_ts.append(this_state.I_syn2_hid)
            vmem_out_ts.append(this_state.V_mem_out)
            isyn_out_ts.append(this_state.I_syn_out)
            spikes_ts.append(this_state.Spikes_hid)

            # - Read the output event register
            output_events = putils.read_output_events(self._device, self._event_buffer)
            output_ts.append(output_events)

        # - Build a recorded state dictionary
        rec_dict = {
            "Vmem": np.array(vmem_ts),
            "Isyn": np.array(isyn_ts),
            "Isyn2": np.array(isyn2_ts),
            "Spikes": np.array(spikes_ts),
            "Vmem_out": np.array(vmem_out_ts),
            "Isyn_out": np.array(isyn_out_ts),
        }

        # - Return output and recorded state
        return np.array(output_ts)[:, : self.shape[-1]], {}, rec_dict

    def _evolve_no_record(self, input: np.ndarray):
        """
        Evolve Pollen HDK over an input step by step, recording all state

        Args:
            input (np.ndarray): Raster of input events with shape ``(T, Nin)``

        Returns: (np.array, dict, dict): output_raster, {}, recorded_dict
            ``recorded_dict`` is a dictionary containing the recorded state of the Pollen neurons over time
        """
        # - Ensure Pollen HDK is in manual mode, and enable reading memory
        conf = self._config
        # conf.debug.clock_enable = True
        conf.operation_mode = samna.pollen.configuration.OperationMode.AcceleratedTime

        # - Apply the configuration
        putils.apply_configuration(self._device, conf)

        # - Wait until Pollen is ready
        putils.is_pollen_ready(self._device, self._event_buffer)

        # - Reset input spike registers
        putils.reset_input_spikes(self._device)

        output_ts = []

        # - Loop over time steps
        for timestep in range(input.shape[0]):
            # - Send input events for this time-step
            putils.send_immediate_input_spikes(self._device, input[timestep])

            # - Evolve one time-step on Pollen
            putils.advance_time_step(self._device)

            # - Wait until pollen is finished the time-step
            while not putils.is_pollen_ready(self._device, self._event_buffer):
                pass

            # - Read the output event register
            output_events = putils.read_output_events(self._device, self._event_buffer)
            output_ts.append(output_events)

        # - Return output and recorded state
        return np.array(output_ts)[:, : self.shape[-1]], {}, {}

    def evolve(
        self,
        input: np.ndarray,
        record: bool = False,
        read_timeout: float = 1.0,
        *args,
        **kwargs,
    ) -> (np.ndarray, dict, dict):
        # - Get the network size
        Nhidden, Nout = self.shape[-2:]

        # - Configure Pollen for accel-time mode
        monitor_neurons = Nhidden if record else None
        putils.select_accel_time_mode(self._device, self._config, monitor_neurons)

        # - Get current timestamp (always starts from zero when enabling "accelerated time" mode,
        #          but need to manually toggle this flag)
        start_timestep = 0

        # - Encode input and readout events
        input_events_list = []
        for timestep, input_data in enumerate(input):
            # - Generate input events
            has_input = False
            for channel, channel_events in enumerate(input_data):
                for _ in range(channel_events):
                    has_input = True
                    event = samna.pollen.event.Spike()
                    event.neuron = channel
                    event.timestamp = start_timestep + timestep
                    input_events_list.append(event)

        # - Add an extra event to ensure readout for entire input extent
        event = samna.pollen.event.Spike()
        event.timestamp = np.shape(input)[0]
        input_events_list.append(event)

        # - Clear the input event count register to make sure the dummy event is ignored
        for addr in [0x0C, 0x0D, 0x0E, 0x0F]:
            event = samna.pollen.event.WriteRegisterValue()
            event.address = addr
            input_events_list.append(event)

        # - Wait until Pollen is ready
        putils.is_pollen_ready(self._device, self._event_buffer)

        # - Clear the read buffer
        self._event_buffer.get_buf()

        # - Write the events and trigger the simulation
        self._device.get_io_module().write(input_events_list)

        # - Read the simulation output events
        read_events = putils.blocking_read(
            self._event_buffer,
            target_timestamp=np.shape(input)[0],
            timeout=read_timeout,
        )

        # - Decode the simulation output events
        pollen_data, times = putils.decode_accel_mode_data(read_events, Nhidden, Nout)

        if record:
            # - Build a recorded state dictionary
            rec_dict = {
                "Vmem": np.array(pollen_data.V_mem_hid),
                "Isyn": np.array(pollen_data.I_syn_hid),
                "Isyn2": np.array(pollen_data.I_syn2_hid),
                "Spikes": np.array(pollen_data.Spikes_hid),
                "Vmem_out": np.array(pollen_data.V_mem_out),
                "Isyn_out": np.array(pollen_data.I_syn_out),
                "times": times,
            }
        else:
            rec_dict = {}

        # - This module accepts no state
        new_state = {}

        # - Return spike output, new state and record dictionary
        return pollen_data.Spikes_out, new_state, rec_dict
