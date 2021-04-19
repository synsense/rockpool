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
from rockpool.nn.modules import Module
from rockpool.parameters import SimulationParameter

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
        weights_in (np.ndarray): A quantised 8-bit input weight matrix ``(Nin, Nhidden, 2)``. The third dimension specifies connections onto the second input synapse for each neuron
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
        raise ValueError("Input weights must be 3 dimensional `(Nin, Nhidden, 2)`")

    # - Check output weights
    if weights_out.ndim != 2:
        raise ValueError("Output weights must be 2 dimensional `(Nhidden, Nout)`")

    # - Get network shape
    Nin, Nhidden, _ = weights_in.shape
    _, Nout = weights_out.shape

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

    if weights_out.shape[0] != Nhidden:
        raise ValueError(
            "Output weights must be consistent with recurrent weights.\n"
            f"`weights_rec`: {weights_rec.shape}; `weights_out`: {weights_out.shape}"
        )

    # - Check aliases
    aliases = [[]] * Nhidden if aliases is None else aliases

    if len(aliases) != Nhidden:
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
    config.synapse2_enable = True
    config.reservoir.aliasing = True
    config.input_expansion.weight_bit_shift = weight_shift_in
    config.reservoir.weight_bit_shift = weight_shift_rec
    config.readout.weight_bit_shift = weight_shift_out

    config.input_expansion.weights = weights_in[:, :, 0]
    config.input_expansion.syn2_weights = weights_in[:, :, 1]
    config.reservoir.weights = weights_rec[:, :, 0]
    config.reservoir.syn2_weights = weights_rec[:, :, 1]
    config.readout.weights = weights_out

    reservoir_neurons = []
    for i in range(len(weights_rec)):
        neuron = ReservoirNeuron()
        if len(aliases[i]) > 0:
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
        device: "samna.device",
        config: PollenConfiguration = None,
        dt: float = 1e-3,
        *args,
        **kwargs,
    ):
        """
        Instantiate a Module with Pollen dev-kit backend

        Args:
            device (samna.pollen.PollenDevice): An opened `samna` device to a Pollen dev kit
            config (PollenConfiguraration): A Pollen configuration from `samna`
            dt (float): The simulation time-step to use for this Module
        """
        # - Initialise the superclass
        super().__init__(
            shape=(16, 1000, 8), dt=dt, spiking_input=True, spiking_output=True
        )

        # - Check that we can access the device node
        pass
        self._device = device

        # - Get the device model
        self._device_model = device.get_device_model()

        # - Check that it's a pollen device
        pass

        # - Register a buffer to read events from Pollen
        self._event_buffer = samna.BufferSinkNode_dynapcnn_event_output_event()
        self._device_model.get_source_node().add_destination(
            self._event_buffer.get_input_channel()
        )

        # - Store the configuration
        if config is None:
            config = self._device_model.get_configuration()

        self.config: Union[
            PollenConfiguration, SimulationParameter
        ] = SimulationParameter(config)
        """ `.PollenConfiguration`: The configuration of the Pollen module """

    @property
    def config(self):
        return self._device_model.get_configuration()

    @config.setter
    def config(self, new_config):
        # - Write the configuration to the dev kit
        self._device_model.apply_configuration(new_config)

    def evolve(
        self, input: np.ndarray, record: bool = False, *args, **kwargs
    ) -> (np.ndarray, dict, dict):
        if record:
            pass
            # - Loop over time steps
            # - Send input spikes for this time-step
            # - Encode input events for this time-step
            # events = [samna.pollen.event.Spike() for x in range(0, 10)]

            # - Evolve one time-step
            # - Record all synapse and neuron states for each time step
            # - Store output events for this time-step

        else:
            pass
            # - Pause pollen execution
            # - Encode input event raster
            # events = [samna.pollen.event.Spike() for x in range(0, 10)]

            # - Send input event raster to pollen
            # my_dynapcnn_model.write(events)

            # - Evolve in automatic mode

            # - No recording
            record_dict = {}

        # - Read events from the output buffer
        output_events = self._event_buffer.get_events()

        # - Clip events to the simulated period

        # - Return events
        return output_events, {}, record_dict
