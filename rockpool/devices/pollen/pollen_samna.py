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
from typing import Optional, Union, Callable

from warnings import warn

import time

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
    if weights_in.ndim < 2:
        raise ValueError(
            "Input weights must be at least 2 dimensional `(Nhidden, Nout, [2])`"
        )

    enable_isyn2 = True
    if weights_in.ndim == 2:
        enable_isyn2 = False
        weights_in = np.reshape(weights_in, [*weights_in.shape, 1])

    # - Check output weights
    if weights_out.ndim != 2:
        raise ValueError("Output weights must be 2 dimensional `(Nhidden, Nout)`")

    # - Get network shape
    Nin, Nin_res, _ = weights_in.shape
    Nhidden, Nout = weights_out.shape

    # - Check input and hidden weight sizes
    if Nin_res > Nhidden:
        raise ValueError("Input weight dimension `Nin_res` must be <= `Nhidden`")

    # - Provide default `weights_rec`
    weights_rec = (
        np.zeros((Nhidden, Nhidden, 1 + enable_isyn2), "int")
        if weights_rec is None
        else weights_rec
    )

    # - Check `weights_rec`
    if weights_rec.ndim == 2:
        enable_isyn2 = False
        weights_rec = np.reshape(weights_rec, [*weights_rec.shape, 1])

    if weights_rec.ndim != 3 or weights_rec.shape[0] != weights_rec.shape[1]:
        raise ValueError("Recurrent weights must be of shape `(Nhidden, Nhidden, [2])`")

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

    # - Check data types
    if (
        weights_in.dtype.kind not in "ui"
        or weights_rec.dtype.kind not in "ui"
        or weights_out.dtype.kind not in "ui"
    ):
        warn(
            "`weights...` arguments should be provided as `int` data types. I am casting these to `int`."
        )

    if (
        threshold.dtype.kind not in "ui"
        or dash_syn.dtype.kind not in "ui"
        or dash_syn_2.dtype.kind not in "ui"
        or dash_syn_out.dtype.kind not in "ui"
        or dash_mem.dtype.kind not in "ui"
        or dash_mem_out.dtype.kind not in "ui"
    ):
        warn(
            "Neuron and synapse parameter arguments should be provided as `int` data types. I am casting these to `int`."
        )

    # - Build the configuration
    config = PollenConfiguration()
    config.debug.clock_enable = True
    config.synapse2_enable = enable_isyn2
    config.reservoir.aliasing = aliases is not None
    config.input.weight_bit_shift = weight_shift_in
    config.reservoir.weight_bit_shift = weight_shift_rec
    config.readout.weight_bit_shift = weight_shift_out
    config.input.weights = weights_in[:, :, 0].astype("int")
    config.reservoir.weights = weights_rec[:, :, 0].astype("int")
    config.readout.weights = weights_out.astype("int")

    if enable_isyn2:
        config.input.syn2_weights = weights_in[:, :, 1].astype("int")
        config.reservoir.syn2_weights = weights_rec[:, :, 1].astype("int")

    reservoir_neurons = []
    for i in range(len(weights_rec)):
        neuron = ReservoirNeuron()
        if aliases is not None and len(aliases[i]) > 0:
            neuron.alias_target = aliases[i][0].astype("int")
        neuron.i_syn_decay = dash_syn[i].astype("int")
        neuron.i_syn2_decay = dash_syn_2[i].astype("int")
        neuron.v_mem_decay = dash_mem[i].astype("int")
        neuron.threshold = threshold[i].astype("int")
        reservoir_neurons.append(neuron)

    config.reservoir.neurons = reservoir_neurons

    readout_neurons = []
    for i in range(np.shape(weights_out)[1]):
        neuron = OutputNeuron()
        neuron.i_syn_decay = dash_syn_out[i].astype("int")
        neuron.v_mem_decay = dash_mem_out[i].astype("int")
        neuron.threshold = threshold_out[i].astype("int")
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

    Returns:
        `.PollenConfiguration`: The configuration loaded from disk
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
    A spiking neuron :py:class:`Module` backed by the Pollen hardware, via `samna`.

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
        # - Check input arguments
        if device is None:
            raise ValueError("`device` must be a valid, opened Pollen HDK device.")

        # - Get a default configuration
        if config is None:
            config = samna.pollen.configuration.PollenConfiguration()

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
        self._event_buffer = putils.new_pollen_read_buffer(device)
        self._state_buffer = putils.new_pollen_state_monitor_buffer(device)

        # - Check that we can access the device node, and that it's a Pollen HDK daughterboard
        if not putils.verify_pollen_version(device, self._event_buffer):
            raise ValueError("`device` must be an opened Pollen HDK daughter board.")

        # - Store the device
        self._device: PollenDaughterBoard = device
        """ (PollenDaughterBoard) The Pollen HDK used by this module """

        # - Store the configuration (and apply it)
        self.config: Union[
            PollenConfiguration, SimulationParameter
        ] = SimulationParameter(shape=(), init_func=lambda _: config)
        """ `.PollenConfiguration`: The configuration of the Pollen module """

        # - Keep a registry of the current recording mode, to save unnecessary reconfiguration
        self._last_record_mode: Optional[bool] = None
        """ (bool) The most recent (and assumed still valid) recording mode """

        # - Store the timestep
        self.dt: Union[float, SimulationParameter] = dt
        """ float: Simulation time-step of the module """

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
        putils.reset_neuron_synapse_state(self._device)
        return self

    def _configure_accel_time_mode(self, Nhidden: int, Nout: int, record: bool):
        if record != self._last_record_mode:
            # - Keep a registry of the last recording mode
            self._last_record_mode = record

            # - Configure Pollen for accel-time mode
            m_Nhidden = Nhidden if record else 0
            m_Nout = Nout if record else 0

            # - Applies the configuration via `self.config`
            self.config, state_buffer = putils.configure_accel_time_mode(
                self._config, self._state_buffer, m_Nhidden, m_Nout
            )

    def evolve(
        self,
        input: np.ndarray,
        record: bool = False,
        read_timeout: float = None,
        *args,
        **kwargs,
    ) -> (np.ndarray, dict, dict):
        # - Get the network size
        Nhidden, Nout = self.shape[-2:]

        # - Configure the recording mode
        self._configure_accel_time_mode(Nhidden, Nout, record)

        # - Get current timestamp
        start_timestep = putils.get_current_timestamp(self._device, self._event_buffer)
        final_timestep = start_timestep + len(input) - 1

        # -- Encode input events
        input_events_list = []

        # - Locate input events
        spikes = np.argwhere(input)
        counts = input[np.nonzero(input)]

        # - Generate input events
        for timestep, channel, count in zip(spikes[:, 0], spikes[:, 1], counts):
            for _ in range(count):
                event = samna.pollen.event.Spike()
                event.neuron = channel
                event.timestamp = start_timestep + timestep
                input_events_list.append(event)

        # - Add an extra event to ensure readout for entire input extent
        event = samna.pollen.event.Spike()
        event.timestamp = final_timestep + 1
        input_events_list.append(event)

        # - Clear the input event count register to make sure the dummy event is ignored
        for addr in [0x0C, 0x0D, 0x0E, 0x0F]:
            event = samna.pollen.event.WriteRegisterValue()
            event.address = addr
            input_events_list.append(event)

        # - Clear the read and state buffers
        self._state_buffer.reset()
        self._event_buffer.get_events()

        # - Write the events and trigger the simulation
        self._device.get_model().write(input_events_list)

        # - Determine a reasonable read timeout
        if read_timeout is None:
            read_timeout = len(input) * self.dt * Nhidden / 400.0
            read_timeout = read_timeout * 5.0 if record else read_timeout

        # - Wait until the simulation is finished
        read_events, is_timeout = putils.blocking_read(
            self._event_buffer, timeout=read_timeout, target_timestamp=final_timestep
        )

        if is_timeout:
            message = f"Processing didn't finish for {read_timeout}s. Read {len(read_events)} events"
            readout_events = [e for e in read_events if hasattr(e, "timestamp")]

            if len(readout_events) > 0:
                message += f", first timestamp: {readout_events[0].timestamp}, final timestamp: {readout_events[-1].timestamp}, target timestamp: {final_timestep}"
            raise TimeoutError(message)

        # - Read the simulation output data
        pollen_data = putils.read_accel_mode_data(self._state_buffer, Nhidden, Nout)

        if record:
            # - Build a recorded state dictionary
            rec_dict = {
                "Vmem": np.array(pollen_data.V_mem_hid),
                "Isyn": np.array(pollen_data.I_syn_hid),
                "Isyn2": np.array(pollen_data.I_syn2_hid),
                "Spikes": np.array(pollen_data.Spikes_hid),
                "Vmem_out": np.array(pollen_data.V_mem_out),
                "Isyn_out": np.array(pollen_data.I_syn_out),
                "times": np.arange(start_timestep, final_timestep + 1),
            }
        else:
            rec_dict = {}

        # - This module accepts no state
        new_state = {}

        # - Return spike output, new state and record dictionary
        return pollen_data.Spikes_out, new_state, rec_dict
