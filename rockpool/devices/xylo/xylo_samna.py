"""
Samna-backed bridge to Xylo dev kit
"""

# - Check that Samna is installed
from importlib import util
from pathlib import Path
from os import makedirs

# - Samna imports
import samna

from samna.xylo.configuration import (
    ReservoirNeuron,
    OutputNeuron,
)
from samna.xylo.configuration import XyloConfiguration as XyloConfiguration

from samna.xylo import validate_configuration

# - Rockpool imports
from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter
from rockpool import TSContinuous, TSEvent

from . import xylo_devkit_utils as hdkutils
from .xylo_devkit_utils import XyloHDK

# - Numpy
import numpy as np

import time

# - Typing
from typing import Optional, Union, Callable, List

from warnings import warn

try:
    from tqdm.autonotebook import tqdm
except ModuleNotFoundError:

    def tqdm(wrapped, *args, **kwargs):
        return wrapped


# - Configure exports
__all__ = ["config_from_specification", "save_config", "load_config", "XyloSamna"]


def config_from_specification(
    weights_in: np.ndarray,
    weights_out: np.ndarray,
    weights_rec: Optional[np.ndarray] = None,
    dash_mem: Optional[np.ndarray] = None,
    dash_mem_out: Optional[np.ndarray] = None,
    dash_syn: Optional[np.ndarray] = None,
    dash_syn_2: Optional[np.ndarray] = None,
    dash_syn_out: Optional[np.ndarray] = None,
    threshold: Optional[np.ndarray] = None,
    threshold_out: Optional[np.ndarray] = None,
    weight_shift_in: int = 0,
    weight_shift_rec: int = 0,
    weight_shift_out: int = 0,
    aliases: Optional[List[List[int]]] = None,
    *args,
    **kwargs,
) -> (XyloConfiguration, bool, str):
    """
    Convert a full network specification to a xylo config and validate it

    See Also:
        For detailed information about the networks supported on Xylo, see :ref:`/devices/xylo-overview.ipynb`

    Args:
        weights_in (np.ndarray): A quantised 8-bit input weight matrix ``(Nin, Nin_res, 2)``. The third dimension specifies connections onto the second input synapse for each neuron. ``Nin_res`` indicates the number of hidden-layer neurons that receive input from the input channels.
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
        aliases (Optional[List[List[int]]]): For each neuron in the hidden population, a list containing the alias targets for that neuron

    Returns: (:py:class:`.samna.xylo.XyloConfiguration`, bool, str): config, is_valid, message
        ``config`` will be a `XyloConfiguration`.
        ``is_valid`` will be a boolean flag ``True`` iff the configuration is valid.
        ``message`` will be an empty string if the configuration is valid, or a message indicating why the configuration is invalid.
    """
    # - Check input weights
    if weights_in.ndim < 2:
        raise ValueError(
            f"Input weights must be at least 2 dimensional `(Nin, Nin_res, [2])`. Found {weights_in.shape}"
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
    dash_mem = np.ones(Nhidden, "int") if dash_mem is None else np.array(dash_mem)
    dash_syn = np.ones(Nhidden, "int") if dash_syn is None else np.array(dash_syn)
    dash_syn_2 = np.ones(Nhidden, "int") if dash_syn_2 is None else np.array(dash_syn_2)

    if (
        np.size(dash_mem) != Nhidden
        or np.size(dash_syn) != Nhidden
        or np.size(dash_syn_2) != Nhidden
    ):
        raise ValueError(
            f"`dash_mem`, `dash_syn` and `dash_syn_2` need `Nhidden` entries (`Nhidden` = {Nhidden})"
            + f" found {np.size(dash_mem)}, {np.size(dash_syn)}, {np.size(dash_syn_2)}"
        )

    dash_mem_out = (
        np.ones(Nout, "int") if dash_mem_out is None else np.array(dash_mem_out)
    )
    dash_syn_out = (
        np.ones(Nout, "int") if dash_syn_out is None else np.array(dash_syn_out)
    )

    if np.size(dash_mem_out) != Nout or np.size(dash_syn_out) != Nout:
        raise ValueError(
            f"`dash_mem_out` and `dash_syn_out` need `Nout` entries (`Nout` = {Nout})"
        )

    # - Check thresholds, assign defaults
    threshold = np.zeros(Nhidden, "int") if threshold is None else np.array(threshold)
    threshold_out = (
        np.zeros(Nout, "int") if threshold_out is None else np.array(threshold_out)
    )

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
            "`weights...` arguments should be provided as `int` data types. I am rounding and casting these to `int`."
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
            "Neuron and synapse parameter arguments should be provided as `int` data types. I am rounding and casting these to `int`."
        )

    # - Round and cast all parameters to integer
    weights_in = np.round(weights_in).astype("int")
    weights_out = np.round(weights_out).astype("int")
    weights_rec = np.round(weights_rec).astype("int")
    dash_mem = np.round(dash_mem).astype("int")
    dash_mem_out = np.round(dash_mem_out).astype("int")
    dash_syn = np.round(dash_syn).astype("int")
    dash_syn_2 = np.round(dash_syn_2).astype("int")
    dash_syn_out = np.round(dash_syn_out).astype("int")
    threshold = np.round(threshold).astype("int")
    threshold_out = np.round(threshold_out).astype("int")
    weight_shift_in = np.round(weight_shift_in).astype("int")
    weight_shift_rec = np.round(weight_shift_rec).astype("int")
    weight_shift_out = np.round(weight_shift_out).astype("int")
    if aliases is not None:
        aliases = [np.round(a).astype("int") for a in aliases]

    # - Build the configuration
    config = XyloConfiguration()

    # - WORKAROUD: Ensure that RAM power is enabled, and the chip clock is running
    config.debug.clock_enable = True
    config.debug.ram_power_enable = True

    config.synapse2_enable = enable_isyn2
    config.reservoir.aliasing = aliases is not None
    config.input.weight_bit_shift = weight_shift_in
    config.reservoir.weight_bit_shift = weight_shift_rec
    config.readout.weight_bit_shift = weight_shift_out
    config.input.weights = weights_in[:, :, 0]
    config.reservoir.weights = weights_rec[:, :, 0]
    config.readout.weights = weights_out

    if enable_isyn2:
        config.input.syn2_weights = weights_in[:, :, 1]
        config.reservoir.syn2_weights = weights_rec[:, :, 1]

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


def save_config(config: XyloConfiguration, filename: str) -> None:
    """
    Save a Xylo configuration to disk in JSON format

    Args:
        config (XyloConfiguration): The configuration to write
        filename (str): The filename to write to
    """
    with open(filename, "w") as f:
        f.write(config.to_json())


def load_config(filename: str) -> XyloConfiguration:
    """
    Read a Xylo configuration from disk in JSON format

    Args:
        filename (str): The filename to read from

    Returns:
        `.XyloConfiguration`: The configuration loaded from disk
    """
    # - Create a new config object
    conf = XyloConfiguration()

    # - Read the configuration from file
    with open(filename) as f:
        conf.from_json(f.read())

    # - Return the configuration
    return conf


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
        *args,
        **kwargs,
    ):
        """
        Instantiate a Module with Xylo dev-kit backend

        Args:
            device (XyloHDK): An opened `samna` device to a Xylo dev kit
            config (XyloConfiguraration): A Xylo configuration from `samna`
            dt (float): The simulation time-step to use for this Module
        """
        # - Check input arguments
        if device is None:
            raise ValueError("`device` must be a valid, opened Xylo HDK device.")

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

            # - Configure Xylo for accel-time mode
            m_Nhidden = Nhidden if record else 0
            m_Nout = Nout if record else 0

            # - Applies the configuration via `self.config`
            self.config, state_buffer = hdkutils.configure_accel_time_mode(
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
        self._device.get_model().write(input_events_list)

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
        return xylo_data.Spikes_out, new_state, rec_dict

    def _evolve_manual(
        self,
        input: np.ndarray,
        record: bool = False,
        read_timeout: float = 5.0,
        *args,
        **kwargs,
    ) -> (np.ndarray, dict, dict):
        """
        Evolve a network on the Xylo HDK in single-step manual mode. For debug purposes only. Uses 'samna.xylo.OperationMode.Manual' in samna.

        Sends a series of events to the Xylo HDK, evolves the network over the input events, and returns the output events produced during the input period.

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

        # - Get some information about the network size
        _, Nhidden, Nout = self.shape

        # - Select single-step simulation mode
        # - Applies the configuration via `self.config`
        self.config = hdkutils.configure_single_step_time_mode(self.config)

        # - Wait until xylo is ready
        t_start = time.time()
        while not hdkutils.is_xylo_ready(self._read_buffer, self._write_buffer):
            if time.time() - t_start > read_timeout:
                raise TimeoutError("Timed out waiting for Xylo to be ready.")

        # - Get current timestamp
        start_timestep = hdkutils.get_current_timestamp(
            self._read_buffer, self._write_buffer
        )
        final_timestep = start_timestep + len(input) - 1

        # - Reset input spike registers
        hdkutils.reset_input_spikes(self._write_buffer)

        # - Initialise lists for recording state
        vmem_ts = []
        isyn_ts = []
        isyn2_ts = []
        vmem_out_ts = []
        isyn_out_ts = []
        spikes_ts = []
        output_ts = []

        # - Loop over time steps
        for timestep in tqdm(range(len(input))):
            # - Send input events for this time-step
            hdkutils.send_immediate_input_spikes(self._write_buffer, input[timestep])

            # - Evolve one time-step on Xylo
            hdkutils.advance_time_step(self._write_buffer)

            # - Wait until xylo has finished the simulation of this time step
            t_start = time.time()
            is_timeout = False
            while not hdkutils.is_xylo_ready(self._read_buffer, self._write_buffer):
                if time.time() - t_start > read_timeout:
                    is_timeout = True
                    break

            if is_timeout:
                break

            # - Read all synapse and neuron states for this time step
            if record:
                this_state = hdkutils.read_neuron_synapse_state(
                    self._read_buffer, self._write_buffer, Nhidden, Nout
                )
                vmem_ts.append(this_state.V_mem_hid)
                isyn_ts.append(this_state.I_syn_hid)
                isyn2_ts.append(this_state.I_syn2_hid)
                vmem_out_ts.append(this_state.V_mem_out)
                isyn_out_ts.append(this_state.I_syn_out)
                spikes_ts.append(this_state.Spikes_hid)

            # - Read the output event register
            output_events = hdkutils.read_output_events(
                self._read_buffer, self._write_buffer
            )
            output_ts.append(output_events)

        if record:
            # - Build a recorded state dictionary
            rec_dict = {
                "Vmem": np.array(vmem_ts),
                "Isyn": np.array(isyn_ts),
                "Isyn2": np.array(isyn2_ts),
                "Spikes": np.array(spikes_ts),
                "Vmem_out": np.array(vmem_out_ts),
                "Isyn_out": np.array(isyn_out_ts),
                "times": np.arange(start_timestep, final_timestep + 1),
            }
        else:
            rec_dict = {}

        # - Return the output spikes, the (empty) new state dictionary, and the recorded state dictionary
        return np.array(output_ts), {}, rec_dict

    def _evolve_manual_allram(
        self,
        input: np.ndarray,
        record: bool = False,
        read_timeout: float = 5.0,
        *args,
        **kwargs,
    ) -> (np.ndarray, dict, dict):
        """
        Evolve a network on the Xylo HDK in single-step manual mode, while recording the entire RAM contents of Xylo. Uses 'samna.xylo.OperationMode.Manual' in samna.

        Sends a series of events to the Xylo HDK, evolves the network over the input events, and returns the output events produced during the input period.

        Args:
            input (np.ndarray): A raster ``(T, Nin)`` specifying for each bin the number of input events sent to the corresponding input channel on Xylo, at the corresponding time point. Up to 15 input events can be sent per bin.
            record (bool): Iff ``True``, record and return all internal state of the neurons and synapses on Xylo. Default: ``False``, do not record internal state.
            read_timeout (Optional[float]): Set an explicit read timeout for the entire simulation time. This should be sufficient for the simulation to complete, and for data to be returned. Default: ``None``, set a reasonable default timeout.

        Returns:
            (np.ndarray, dict, dict): ``output``, ``new_state``, ``record_dict``.
            ``output`` is a raster ``(T, Nout)``, containing events for each channel in each time bin. Time bins in ``output`` correspond to the time bins in ``input``.
            ``new_state`` is an empty dictiionary. The Xylo HDK does not permit querying or setting state.
            ``record_dict`` is a dictionary containing recorded internal all RAM state of Xylo during evolution, if the ``record`` argument is ``True``. Otherwise this is an empty dictionary.

        Raises:
            `TimeoutError`: If reading data times out during the evolution. An explicity timeout can be set using the `read_timeout` argument.
        """

        # - Get some information about the network size
        Nin, Nhidden, Nout = self.shape

        # - Select single-step simulation mode
        self.config = hdkutils.configure_single_step_time_mode(self.config)

        # - Wait until xylo is ready
        t_start = time.time()
        while not hdkutils.is_xylo_ready(self._read_buffer, self._write_buffer):
            if time.time() - t_start > read_timeout:
                raise TimeoutError("Timed out waiting for Xylo to be ready.")

        # - Get current timestamp
        start_timestep = hdkutils.get_current_timestamp(
            self._read_buffer, self._write_buffer
        )
        final_timestep = start_timestep + len(input) - 1

        # - Reset input spike registers
        hdkutils.reset_input_spikes(self._write_buffer)

        # - Initialise lists for internal all RAM state
        vmem_ts = []
        isyn_ts = []
        isyn2_ts = []
        vmem_out_ts = []
        isyn_out_ts = []
        spikes_ts = []
        output_ts = []

        input_weight_ram_ts = []
        input_weight_2ram_ts = []
        neuron_dash_syn_ram_ts = []
        reservoir_dash_syn_2ram_ts = []
        neuron_dash_mem_ram_ts = []
        neuron_threshold_ram_ts = []
        reservoir_config_ram_ts = []
        reservoir_aliasing_ram_ts = []
        reservoir_effective_fanout_count_ram_ts = []
        recurrent_fanout_ram_ts = []
        recurrent_weight_ram_ts = []
        recurrent_weight_2ram_ts = []
        output_weight_ram_ts = []

        # - Loop over time steps
        for timestep in tqdm(range(len(input))):
            # - Send input events for this time-step
            hdkutils.send_immediate_input_spikes(self._write_buffer, input[timestep])

            # - Evolve one time-step on Xylo
            hdkutils.advance_time_step(self._write_buffer)

            # - Wait until xylo has finished the simulation of this time step
            t_start = time.time()
            is_timeout = False
            while not hdkutils.is_xylo_ready(self._read_buffer, self._write_buffer):
                if time.time() - t_start > read_timeout:
                    is_timeout = True
                    break

            if is_timeout:
                break

            # - Read all RAM states for this time step
            if record:
                this_state = hdkutils.read_allram_state(
                    self._read_buffer, self._write_buffer, Nin, Nhidden, Nout
                )
                vmem_ts.append(this_state.V_mem_hid)
                isyn_ts.append(this_state.I_syn_hid)
                isyn2_ts.append(this_state.I_syn2_hid)
                vmem_out_ts.append(this_state.V_mem_out)
                isyn_out_ts.append(this_state.I_syn_out)
                spikes_ts.append(this_state.Spikes_hid)

                input_weight_ram_ts.append(this_state.IWTRAM_state)
                input_weight_2ram_ts.append(this_state.IWT2RAM_state)
                neuron_dash_syn_ram_ts.append(this_state.NDSRAM_state)
                reservoir_dash_syn_2ram_ts.append(this_state.RDS2RAM_state)
                neuron_dash_mem_ram_ts.append(this_state.NDMRAM_state)
                neuron_threshold_ram_ts.append(this_state.NTHRAM_state)
                reservoir_config_ram_ts.append(this_state.RCRAM_state)
                reservoir_aliasing_ram_ts.append(this_state.RARAM_state)
                reservoir_effective_fanout_count_ram_ts.append(
                    this_state.REFOCRAM_state
                )
                recurrent_fanout_ram_ts.append(this_state.RFORAM_state)
                recurrent_weight_ram_ts.append(this_state.RWTRAM_state)
                recurrent_weight_2ram_ts.append(this_state.RWT2RAM_state)
                output_weight_ram_ts.append(this_state.OWTRAM_state)

            # - Read the output event register
            output_events = hdkutils.read_output_events(
                self._read_buffer, self._write_buffer
            )
            output_ts.append(output_events)

        if record:
            # - Build a recorded state dictionary
            rec_dict = {
                "Vmem": np.array(vmem_ts),
                "Isyn": np.array(isyn_ts),
                "Isyn2": np.array(isyn2_ts),
                "Spikes": np.array(spikes_ts),
                "Vmem_out": np.array(vmem_out_ts),
                "Isyn_out": np.array(isyn_out_ts),
                "times": np.arange(start_timestep, final_timestep + 1),
                "Input_weight_ram": np.array(input_weight_ram_ts),
                "Input_weight_2ram": np.array(input_weight_2ram_ts),
                "Neuron_dash_syn_ram": np.array(neuron_dash_syn_ram_ts),
                "Reservoir_dash_syn_2ram": np.array(reservoir_dash_syn_2ram_ts),
                "Neuron_dash_mem_ram": np.array(neuron_dash_mem_ram_ts),
                "Neuron_threshold_ram": np.array(neuron_threshold_ram_ts),
                "Reservoir_config_ram": np.array(reservoir_config_ram_ts),
                "Reservoir_aliasing_ram": np.array(reservoir_aliasing_ram_ts),
                "Reservoir_effective_fanout_count_ram": np.array(
                    reservoir_effective_fanout_count_ram_ts
                ),
                "Recurrent_fanout_ram": np.array(recurrent_fanout_ram_ts),
                "Recurrent_weight_ram": np.array(recurrent_weight_ram_ts),
                "Recurrent_weight_2ram": np.array(recurrent_weight_2ram_ts),
                "Output_weight_ram": np.array(output_weight_ram_ts),
            }
        else:
            rec_dict = {}

        # - Return the output spikes, the (empty) new state dictionary, and the recorded state dictionary
        return np.array(output_ts), {}, rec_dict

    def _evolve_manual_ram_register(
        self,
        input: np.ndarray,
        record: bool = False,
        read_timeout: float = 5.0,
        *args,
        **kwargs,
    ) -> (np.ndarray, dict, dict):
        """
        Evolve a network on the Xylo HDK in single-step manual mode, while recording the entire RAM and register contents of Xylo. Uses 'samna.xylo.OperationMode.Manual' in samna.

        Evolve a network on the Xylo HDK with manual mode. It is through 'samna.xylo.OperationMode.Manual' in samna.

        Sends a series of events to the Xylo HDK, evolves the network over the input events, and returns the output events produced during the input period.

        Args:
            input (np.ndarray): A raster ``(T, Nin)`` specifying for each bin the number of input events sent to the corresponding input channel on Xylo, at the corresponding time point. Up to 15 input events can be sent per bin.
            record (bool): Iff ``True``, record and return all internal state of the neurons and synapses on Xylo. Default: ``False``, do not record internal state.
            read_timeout (Optional[float]): Set an explicit read timeout for the entire simulation time. This should be sufficient for the simulation to complete, and for data to be returned. Default: ``None``, set a reasonable default timeout.

        Returns:
            (np.ndarray, dict, dict): ``output``, ``new_state``, ``record_dict``.
            ``output`` is a raster ``(T, Nout)``, containing events for each channel in each time bin. Time bins in ``output`` correspond to the time bins in ``input``.
            ``new_state`` is an empty dictiionary. The Xylo HDK does not permit querying or setting state.
            ``record_dict`` is a dictionary containing recorded internal all RAM and register state of Xylo during evolution, if the ``record`` argument is ``True``. Otherwise this is an empty dictionary.

        Raises:
            `TimeoutError`: If reading data times out during the evolution. An explicity timeout can be set using the `read_timeout` argument.
        """
        # - Get some information about the network size
        Nin, Nhidden, Nout = self.shape

        # - Select single-step simulation mode
        self.config = hdkutils.configure_single_step_time_mode(self.config)

        # - Initialise lists for recording state
        vmem_ts = []
        isyn_ts = []
        isyn2_ts = []
        vmem_out_ts = []
        isyn_out_ts = []
        spikes_ts = []
        output_ts = []

        input_weight_ram_ts = []
        input_weight_2ram_ts = []
        neuron_dash_syn_ram_ts = []
        reservoir_dash_syn_2ram_ts = []
        neuron_dash_mem_ram_ts = []
        neuron_threshold_ram_ts = []
        reservoir_config_ram_ts = []
        reservoir_aliasing_ram_ts = []
        reservoir_effective_fanout_count_ram_ts = []
        recurrent_fanout_ram_ts = []
        recurrent_weight_ram_ts = []
        recurrent_weight_2ram_ts = []
        output_weight_ram_ts = []

        # - Read all ram and register before evolve_manual
        this_state = hdkutils.read_allram_state(
            self._read_buffer, self._write_buffer, Nin, Nhidden, Nout
        )
        vmem_ts.append(this_state.V_mem_hid)
        isyn_ts.append(this_state.I_syn_hid)
        isyn2_ts.append(this_state.I_syn2_hid)
        vmem_out_ts.append(this_state.V_mem_out)
        isyn_out_ts.append(this_state.I_syn_out)
        spikes_ts.append(this_state.Spikes_hid)

        input_weight_ram_ts.append(this_state.IWTRAM_state)
        input_weight_2ram_ts.append(this_state.IWT2RAM_state)
        neuron_dash_syn_ram_ts.append(this_state.NDSRAM_state)
        reservoir_dash_syn_2ram_ts.append(this_state.RDS2RAM_state)
        neuron_dash_mem_ram_ts.append(this_state.NDMRAM_state)
        neuron_threshold_ram_ts.append(this_state.NTHRAM_state)
        reservoir_config_ram_ts.append(this_state.RCRAM_state)
        reservoir_aliasing_ram_ts.append(this_state.RARAM_state)
        reservoir_effective_fanout_count_ram_ts.append(this_state.REFOCRAM_state)
        recurrent_fanout_ram_ts.append(this_state.RFORAM_state)
        recurrent_weight_ram_ts.append(this_state.RWTRAM_state)
        recurrent_weight_2ram_ts.append(this_state.RWT2RAM_state)
        output_weight_ram_ts.append(this_state.OWTRAM_state)

        # - Create a folder to save register state in each timestamp
        folder = "./registers/"
        newFolder = Path(folder)
        if not newFolder.exists():
            makedirs(newFolder)

        # - Save the register state before evolve_manual as '-1' timestamp.
        file = folder + "register_-1.txt"
        hdkutils.export_registers(self._read_buffer, self._write_buffer, file)

        # - Wait until xylo is ready
        t_start = time.time()
        while not hdkutils.is_xylo_ready(self._read_buffer, self._write_buffer):
            if time.time() - t_start > read_timeout:
                raise TimeoutError("Timed out waiting for Xylo to be ready.")

        # - Get current timestamp
        start_timestep = hdkutils.get_current_timestamp(
            self._read_buffer, self._write_buffer
        )
        final_timestep = start_timestep + len(input) - 1

        # - Reset input spike registers
        hdkutils.reset_input_spikes(self._write_buffer)

        # - Loop over time steps
        for timestep in tqdm(range(len(input))):
            # - Send input events for this time-step
            hdkutils.send_immediate_input_spikes(self._write_buffer, input[timestep])

            # - Save register just after give input but before evolve_manual for each timestamp
            file = folder + f"register_{timestep}_spkin.txt"
            hdkutils.export_registers(self._read_buffer, self._write_buffer, file)

            # - Evolve one time-step on Xylo HDK
            hdkutils.advance_time_step(self._write_buffer)

            # - Wait until Xylo HDK has finished the simulation of this time step
            t_start = time.time()
            is_timeout = False
            while not hdkutils.is_xylo_ready(self._read_buffer, self._write_buffer):
                if time.time() - t_start > read_timeout:
                    is_timeout = True
                    break

            if is_timeout:
                break

            # - Read all RAM and register state for this time step
            if record:
                this_state = hdkutils.read_allram_state(
                    self._read_buffer, self._write_buffer, Nin, Nhidden, Nout
                )
                vmem_ts.append(this_state.V_mem_hid)
                isyn_ts.append(this_state.I_syn_hid)
                isyn2_ts.append(this_state.I_syn2_hid)
                vmem_out_ts.append(this_state.V_mem_out)
                isyn_out_ts.append(this_state.I_syn_out)
                spikes_ts.append(this_state.Spikes_hid)

                input_weight_ram_ts.append(this_state.IWTRAM_state)
                input_weight_2ram_ts.append(this_state.IWT2RAM_state)
                neuron_dash_syn_ram_ts.append(this_state.NDSRAM_state)
                reservoir_dash_syn_2ram_ts.append(this_state.RDS2RAM_state)
                neuron_dash_mem_ram_ts.append(this_state.NDMRAM_state)
                neuron_threshold_ram_ts.append(this_state.NTHRAM_state)
                reservoir_config_ram_ts.append(this_state.RCRAM_state)
                reservoir_aliasing_ram_ts.append(this_state.RARAM_state)
                reservoir_effective_fanout_count_ram_ts.append(
                    this_state.REFOCRAM_state
                )
                recurrent_fanout_ram_ts.append(this_state.RFORAM_state)
                recurrent_weight_ram_ts.append(this_state.RWTRAM_state)
                recurrent_weight_2ram_ts.append(this_state.RWT2RAM_state)
                output_weight_ram_ts.append(this_state.OWTRAM_state)

                # - Save register after evolve_manual for each timestamp
                file = folder + f"register_{timestep}.txt"
                hdkutils.export_registers(self._read_buffer, self._write_buffer, file)

            # - Read the output event register
            output_events = hdkutils.read_output_events(
                self._read_buffer, self._write_buffer
            )
            output_ts.append(output_events)

        if record:
            # - Build a recorded state dictionary
            rec_dict = {
                "Vmem": np.array(vmem_ts),
                "Isyn": np.array(isyn_ts),
                "Isyn2": np.array(isyn2_ts),
                "Spikes": np.array(spikes_ts),
                "Vmem_out": np.array(vmem_out_ts),
                "Isyn_out": np.array(isyn_out_ts),
                "times": np.arange(start_timestep, final_timestep + 1),
                "Input_weight_ram": np.array(input_weight_ram_ts),
                "Input_weight_2ram": np.array(input_weight_2ram_ts),
                "Neuron_dash_syn_ram": np.array(neuron_dash_syn_ram_ts),
                "Reservoir_dash_syn_2ram": np.array(reservoir_dash_syn_2ram_ts),
                "Neuron_dash_mem_ram": np.array(neuron_dash_mem_ram_ts),
                "Neuron_threshold_ram": np.array(neuron_threshold_ram_ts),
                "Reservoir_config_ram": np.array(reservoir_config_ram_ts),
                "Reservoir_aliasing_ram": np.array(reservoir_aliasing_ram_ts),
                "Reservoir_effective_fanout_count_ram": np.array(
                    reservoir_effective_fanout_count_ram_ts
                ),
                "Recurrent_fanout_ram": np.array(recurrent_fanout_ram_ts),
                "Recurrent_weight_ram": np.array(recurrent_weight_ram_ts),
                "Recurrent_weight_2ram": np.array(recurrent_weight_2ram_ts),
                "Output_weight_ram": np.array(output_weight_ram_ts),
            }
        else:
            rec_dict = {}

        # - Return the output spikes, the (empty) new state dictionary, and the recorded state dictionary
        return np.array(output_ts), {}, rec_dict

    def _wrap_recorded_state(self, state_dict: dict, t_start: float = 0.0) -> dict:
        args = {"dt": self.dt, "t_start": t_start}

        return {
            "Vmem": TSContinuous.from_clocked(
                state_dict["Vmem"], name="$V_{mem}$", **args
            ),
            "Isyn": TSContinuous.from_clocked(
                state_dict["Isyn"], name="$I_{syn}$", **args
            ),
            "Isyn2": TSContinuous.from_clocked(
                state_dict["Isyn2"], name="$I_{syn,2}$", **args
            ),
            "Spikes": TSEvent.from_raster(state_dict["Spikes"], name="Spikes", **args),
            "Vmem_out": TSContinuous.from_clocked(
                state_dict["Vmem_out"], name="$V_{mem,out}$", **args
            ),
            "Isyn_out": TSContinuous.from_clocked(
                state_dict["Isyn_out"], name="$I_{syn,out}$", **args
            ),
        }
