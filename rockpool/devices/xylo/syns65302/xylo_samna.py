"""
Implements :py:class:`.XyloSamna`, for running inference on a Xylo™Audio 3 HDK

Also provides :py:func:`.config_from_specification`.
"""

import numpy as np
import samna
import time
import copy


try:
    from tqdm.autonotebook import tqdm
except:
    tqdm = lambda x: x

from samna.xyloAudio3.configuration import XyloConfiguration

from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter
from . import xa3_devkit_utils as hdkutils

XyloAudio3HDK = samna.xyloAudio3.XyloAudio3TestBoard

# - Typing
from typing import Optional, Union, List, Tuple
from warnings import warn

# - Configure exports
__all__ = [
    "config_from_specification",
    "save_config",
    "load_config",
    "XyloSamna",
]


def config_from_specification(
    weights_in: np.ndarray,
    weights_out: np.ndarray,
    weights_rec: Optional[np.ndarray] = None,
    dash_mem: Optional[np.ndarray] = None,
    dash_mem_out: Optional[np.ndarray] = None,
    dash_syn: Optional[np.ndarray] = None,
    dash_syn_out: Optional[np.ndarray] = None,
    threshold: Optional[np.ndarray] = None,
    threshold_out: Optional[np.ndarray] = None,
    bias_hidden: Optional[np.ndarray] = None,
    bias_out: Optional[np.ndarray] = None,
    weight_shift_in: int = 0,
    weight_shift_rec: int = 0,
    weight_shift_out: int = 0,
    aliases: Optional[List[List[int]]] = None,
    *args,
    **kwargs,
) -> XyloConfiguration:
    """
    Convert a full network specification to a XyloAudio3 config and validate it

    See Also:
        For detailed information about the networks supported on Xylo, see :ref:`/devices/xylo-overview.ipynb`

    Args:
        weights_in (np.ndarray): A quantised 8-bit input weight matrix ``(Nin, Nin_res, 1)``. The third dimension specifies connections onto the second input synapse for each neuron. ``Nin_res`` indicates the number of hidden-layer neurons that receive input from the input channels.
        weights_out (np.ndarray): A quantised 8-bit output weight matrix ``(Nhidden, Nout)``.
        weights_rec (np.ndarray): A quantised 8-bit recurrent weight matrix ``(Nhidden, Nhidden, 1)``. The third dimension specified connections onto the second input synapse for each neuron. Default: ``0``
        dash_mem (np.ndarray): A vector or list ``(Nhidden,)`` specifing decay bitshift for neuron state for each hidden layer neuron. Default: ``1``
        dash_mem_out (np.ndarray): A vector or list ``(Nout,)`` specifing decay bitshift for neuron state for each output neuron. Default: ``1``
        dash_syn (np.ndarray): A vector or list ``(Nhidden,)`` specifing decay bitshift for synapse 1 state for each hidden layer neuron. Default: ``1``
        dash_syn_out (np.ndarray): A vector or list ``(Nout,)`` specifing decay bitshift for synapse state for each output layer neuron. Default: ``1``
        threshold (np.ndarray): A vector or list ``(Nhidden,)`` specifing the firing threshold for each hidden layer neuron. Default: ``0``
        threshold_out (np.ndarray): A vector or list ``(Nout,)`` specifing the firing threshold for each output layer neuron. Default: ``0``
        bias_hidden (np.ndarray): A vector or list ``(Nhidden,)`` specifing the bias for each hidden layer neuron. Default: ``0``
        bias_out (np.ndarray): A vector or list ``(Nout,)`` specifing the bias for each output layer neuron. Default: ``0``
        weight_shift_in (int): The number of bits to left-shift each input weight. Default: ``0``
        weight_shift_rec (int): The number of bits to left-shift each recurrent weight. Default: ``0``
        weight_shift_out (int): The number of bits to left-shift each output layer weight. Default: ``0``
        aliases (Optional[List[List[int]]]): For each neuron in the hidden population, a list containing the alias targets for that neuron

    Returns: (:py:class:`.samna.xyloAudio3.XyloConfiguration`, bool, str): config, is_valid, message
        ``config`` will be a `XyloConfiguration`.
        ``is_valid`` will be a boolean flag ``True`` iff the configuration is valid.
        ``message`` will be an empty string if the configuration is valid, or a message indicating why the configuration is invalid.
    """
    # - Check input weights
    if weights_in.ndim != 3:
        raise ValueError(
            f"Input weights must be 3 dimensional `(Nin, Nin_res, Nsyn)`. Found {weights_in.shape}"
        )

    if weights_rec.ndim != 3:
        raise ValueError(
            f"Recurrent weights must be 3 dimensional `(Nin_res, Nin_res, Nsyn)`. Found {weights_rec.shape}"
        )

    # - Check output weights
    if weights_out.ndim != 2:
        raise ValueError("Output weights must be 2 dimensional `(Nhidden, Nout)`")

    # - Get network shape
    Nin, Nin_res, Nsyn = weights_in.shape
    Nhidden, _, _ = weights_rec.shape
    Nout_res, Nout = weights_out.shape

    # - Check number of input synapses
    if Nsyn > 2:
        raise ValueError(
            f"Max of 2 input synapses are supported on XyloAudio 3. Found {Nsyn}."
        )

    # - Check input and hidden weight sizes
    if Nin_res > Nhidden:
        raise ValueError("Input weight dimension `Nin_res` must be <= `Nhidden`")

    # - Provide default `weights_rec`
    weights_rec = (
        np.zeros((Nhidden, Nhidden, 1), "int") if weights_rec is None else weights_rec
    )

    # - Check `weights_rec`
    if weights_rec.ndim != 3 or weights_rec.shape[0] != weights_rec.shape[1]:
        raise ValueError(
            "Recurrent weights must be of shape `(Nhidden, Nhidden, Nsyn)`"
        )

    # - Check aliases
    if aliases is not None and len(aliases) != Nhidden:
        raise ValueError(
            f"Aliases list must have `Nhidden` entries (`Nhidden` = {Nhidden})"
        )

    # - Check bitshift TCs, assign defaults
    dash_mem = np.ones(Nhidden, "int") if dash_mem is None else np.array(dash_mem)
    dash_syn = np.ones(Nhidden, "int") if dash_syn is None else np.array(dash_syn)
    if bias_hidden is not None:
        bias_hidden = np.round(np.array(bias_hidden)).astype("int")
    if bias_out is not None:
        bias_out = np.round(np.array(bias_out)).astype("int")

    if np.size(dash_mem) != Nhidden or np.size(dash_syn) != Nhidden:
        raise ValueError(
            f"`dash_mem`, `dash_syn` need `Nhidden` entries (`Nhidden` = {Nhidden})"
            + f" found {np.size(dash_mem)}, {np.size(dash_syn)}"
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
        or dash_syn_out.dtype.kind not in "ui"
        or dash_mem.dtype.kind not in "ui"
        or dash_mem_out.dtype.kind not in "ui"
    ):
        warn(
            "Neuron and synapse parameter arguments should be provided as `int` data types. I am rounding and casting these to `int`."
        )

    # - Round and cast all parameters to integer
    weights_in = np.round(weights_in).astype("int8")
    weights_out = np.round(weights_out).astype("int8")
    weights_rec = np.round(weights_rec).astype("int8")
    dash_mem = np.round(dash_mem).astype("int8")
    dash_mem_out = np.round(dash_mem_out).astype("int8")
    dash_syn = np.round(dash_syn).astype("int8")
    dash_syn_out = np.round(dash_syn_out).astype("int8")
    threshold = np.round(threshold).astype("int")
    threshold_out = np.round(threshold_out).astype("int")
    weight_shift_in = np.round(weight_shift_in).astype("int8")
    weight_shift_rec = np.round(weight_shift_rec).astype("int8")
    weight_shift_out = np.round(weight_shift_out).astype("int8")
    if aliases is not None:
        aliases = [np.round(a).astype("int") for a in aliases]

    # - Build the configuration
    config = samna.xyloAudio3.configuration.XyloConfiguration()

    if bias_hidden is not None or bias_out is not None:
        config.bias_enable = True

    config.hidden.aliasing = aliases is not None
    config.input.weight_bit_shift = weight_shift_in
    config.hidden.weight_bit_shift = weight_shift_rec
    config.readout.weight_bit_shift = weight_shift_out
    if weights_in.shape[1] > 128:
        warn(
            "More than 128 input expansion neurons (IEN) detected. Only the first 128 will be used."
        )
        config.input.weights = weights_in[:, :128, 0]
    else:
        config.input.weights = weights_in[:, :, 0]
    config.hidden.weights = weights_rec[:, :, 0]
    if weights_out.shape[0] > 128:
        warn(
            "More than 128 output expansion neurons (OEN) detected. Only the last 128 will be used."
        )
        config.readout.weights = weights_out[-128:, :]
    else:
        config.readout.weights = weights_out

    hidden_neurons = []
    for i in range(len(weights_rec)):
        neuron = samna.xyloAudio3.configuration.HiddenNeuron()
        if aliases is not None and len(aliases[i]) > 0:
            neuron.alias_target = aliases[i][0]

        neuron.i_syn_decay = dash_syn[i]
        neuron.v_mem_decay = dash_mem[i]
        neuron.threshold = threshold[i]
        if bias_hidden is not None:
            neuron.v_mem_bias = bias_hidden[i]
        hidden_neurons.append(neuron)

    config.hidden.neurons = hidden_neurons

    readout_neurons = []
    for i in range(np.shape(weights_out)[1]):
        neuron = samna.xyloAudio3.configuration.OutputNeuron()
        neuron.i_syn_decay = dash_syn_out[i]
        neuron.v_mem_decay = dash_mem_out[i]
        neuron.threshold = threshold_out[i]
        if bias_out is not None:
            neuron.v_mem_bias = bias_out[i]
        readout_neurons.append(neuron)

    config.readout.neurons = readout_neurons

    # - Validate the configuration and return
    is_valid, message = samna.xyloAudio3.validate_configuration(config)
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
    A spiking neuron :py:class:`.Module` backed by the XyloAudio 3 hardware, via `samna`.

    Use :py:func:`.config_from_specification` to build and validate a configuration for Xylo.

    See Also:
        See the tutorial :ref:`/devices/xylo-a3/xylo-audio3-intro.ipynb` for a high-level overview of building and deploying networks for Xylo.
    """

    def __init__(
        self,
        device: XyloAudio3HDK,
        config: XyloConfiguration = None,
        dt: float = 1e-3,
        output_mode: str = "Spike",
        record: Optional[bool] = False,
        power_frequency: Optional[float] = 5.0,
        *args,
        **kwargs,
    ):
        """
        Instantiate a Module with XyloAudio 3 dev-kit backend

        Args:
            device (XyloAudio3HDK): An opened `samna` device to a XyloAudio 3 dev kit
            config (XyloConfiguration): A Xylo configuration from `samna`
            dt (float): The simulation time-step to use for this Module
            output_mode (str): The readout mode for the Xylo device. This must be one of ``["Spike", "Isyn", "Vmem"]``. Default: "Spike", return events from the output layer.
            record (bool): Record and return all internal states of the neurons and synapses on Xylo. Default: ``False``, do not record internal state.
            power_frequency (float): The frequency of power measurement, in Hz. Default: 5.0

        Raises:
            `ValueError`: If ``device`` is not set. ``device`` must be a ``XyloAudio3HDK``
            `TimeoutError`: If ``output_mode`` is not ``Spike``, ``Vmem`` or ``ISyn``
            `ValueError`: If ``operation_mode`` is set to ``RealTime``. For ``RealTime`` please use :py:class:`.XyloMonitor`
            `Warning`: For XyloSamna ``config.input_source`` must be set to ``SpikeEvents``

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
            config = samna.xyloAudio3.configuration.XyloConfiguration()

        if config.input_source != samna.xyloAudio3.InputSource.SpikeEvents:
            warn(
                "XyloSamna is intended to be used with direct input to the SNN core. Updating config.input_source to SpikeEvents."
            )
        # - Set input source to SpikeEvents
        config.input_source = samna.xyloAudio3.InputSource.SpikeEvents

        # - Get the network shape
        Nin, Nien = np.shape(config.input.weights)
        Nhidden, _ = np.shape(config.hidden.weights)
        Noen, Nout = np.shape(config.readout.weights)

        # - Initialise the superclass
        super().__init__(
            shape=(Nin, Nhidden, Nout), spiking_input=True, spiking_output=True
        )

        # HACK record tag was moved to the constructor of the class instead of evolve
        # updating the configuration in evolve is leading to erratic behavior because of a mismatch between samna and firmware
        # In Accelerated-Time mode we can use automatic state monitoring, by setting the neuron ids we want to monitor.
        # Output Vmem and spikes are always activated, so we only monitor the hidden neurons.
        if (
            record
            and config.operation_mode == samna.xyloAudio3.OperationMode.AcceleratedTime
        ):
            config.debug.monitor_neuron_v_mem = [i for i in range(Nhidden)]
            config.debug.monitor_neuron_spike = [i for i in range(Nhidden)]
            # Output Isyn is not available by default, so we add both hidden and output neurons.
            config.debug.monitor_neuron_i_syn = [i for i in range(Nhidden + Nout)]
        else:
            config.debug.monitor_neuron_v_mem = []
            config.debug.monitor_neuron_spike = []
            config.debug.monitor_neuron_i_syn = []

        # - Store record option
        self._record: Optional[bool] = record
        """ bool: Record and return all internal state of the neurons and synapses on Xylo """

        # - Store the device
        self._device: XyloAudio3HDK = device
        """ `.XyloHDK`: The Xylo HDK used by this module """

        # - Register buffers to read and write events
        self._read_buffer = hdkutils.new_xylo_read_buffer(device)
        """ `.XyloAudio3ReadBuffer`: The read buffer for the connected HDK """

        (
            self._readout_buffer,
            self._readout_graph,
        ) = hdkutils.new_xylo_state_monitor_buffer(device)

        self._write_buffer = hdkutils.new_xylo_write_buffer(device)
        """ `.XyloAudio3WriteBuffer`: The write buffer for the connected HDK """

        # - Store the timestep
        self.dt: Union[float, SimulationParameter] = dt
        """ float: Simulation time-step of the module, in seconds """

        # - Sleep time post sending spikes on each time-step, in manual mode
        self._sleep_time = 0e-3
        """ float: Post-stimulation sleep time in seconds """

        # - For XyloSamna, operation mode can be either manual or accelerated time
        if config.operation_mode == samna.xyloAudio3.OperationMode.RealTime:
            raise ValueError(
                "`operation_mode` can't be RealTime for XyloSamna. Options are Manual or AcceleratedTime."
            )

        elif config.operation_mode == samna.xyloAudio3.OperationMode.Manual:
            warn(
                "`operation_mode is set to Manual. This mode can be used for debug purpuses together with `_evolve_manual`. Otherwise, please use `AcceleratedTime`."
            )

        # - Apply configuration
        self._config: Union[
            XyloConfiguration, SimulationParameter
        ] = SimulationParameter(shape=(), init_func=lambda _: config)

        # - Keep a registry of the current recording mode, to save unnecessary reconfiguration
        self._last_record_mode: Optional[bool] = None
        """ bool: The most recent (and assumed still valid) recording mode """

        # - Store the timestep
        self.dt: Union[float, SimulationParameter] = dt
        """ float: Simulation time-step of the module, in seconds """

        # - Store the power frequency
        self._power_frequency = power_frequency
        """ float: Frequency of power monitoring, in Hz """

        # - Apply configuration on the board
        hdkutils.apply_configuration(self._device, self._config)

    def __del__(self):
        # - Stop the readout graph buffer
        if self._readout_graph:
            self._readout_graph.stop()

    @property
    def config(self):
        """`.XyloConfiguration`: The HDK configuration applied to the Xylo module"""
        # - Return the configuration stored on Xylo HDK
        return self._device.get_model().get_configuration()

    @config.setter
    def config(self, new_config):
        """`.XyloConfiguration`: The HDK configuration applied to the Xylo module"""
        # - Test for a valid configuration
        is_valid, msg = samna.xyloAudio3.validate_configuration(new_config)
        if not is_valid:
            raise ValueError(f"Invalid configuration for the Xylo HDK: {msg}")

        # - Write the configuration to the device
        hdkutils.apply_configuration(self._device, new_config)
        time.sleep(self._sleep_time)

        self._config = new_config

    def reset_state(self) -> "XyloSamna":
        """
        Reset all states on the Xylo device
        """
        # - Reset neuron and synapse state on Xylo
        # -- Copy values of configuration
        operation_mode = copy.copy(self._config.operation_mode)
        vmem_monitor = copy.copy(self._config.debug.monitor_neuron_v_mem)
        spike_monitor = copy.copy(self._config.debug.monitor_neuron_spike)
        isyn_monitor = copy.copy(self._config.debug.monitor_neuron_i_syn)

        # - To reset Samna and Firmware, we need to send a configuration with different operation mode
        # -- Operation mode can not be RealTime in XyloSamna
        self._config.operation_mode = (
            samna.xyloAudio3.OperationMode.Manual
            if self._config.operation_mode
            == samna.xyloAudio3.OperationMode.AcceleratedTime
            else samna.xyloAudio3.OperationMode.AcceleratedTime
        )
        self._config.debug.monitor_neuron_v_mem = []
        self._config.debug.monitor_neuron_spike = []
        self._config.debug.monitor_neuron_i_syn = []
        hdkutils.apply_configuration(self._device, self._config)

        # - Reapply the user defined configuration
        self._config.operation_mode = operation_mode
        self._config.debug.monitor_neuron_v_mem = vmem_monitor
        self._config.debug.monitor_neuron_spike = spike_monitor
        self._config.debug.monitor_neuron_i_syn = isyn_monitor
        hdkutils.apply_configuration(self._device, self._config)
        return self

    def evolve(
        self,
        input: np.ndarray,
        record: bool = False,
        record_power: bool = False,
        read_timeout: Optional[float] = None,
        *args,
        **kwargs,
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        Evolve a network on the XyloAudio 3 HDK in accelerated-time mode
        Sends a series of events to the Xylo HDK, evolves the network over the input events, and returns the output events produced during the input period. Optionally record internal state of the network, selectable with the ``record`` flag.

        Args:
            input (np.ndarray): A raster ``(T, Nin)`` specifying for each bin the number of input events sent to the corresponding input channel on Xylo, at the corresponding time point. Up to 15 input events can be sent per bin.
            record (bool): Deprecated parameter. Please use ``record`` from the class initialization.
            record_power (bool): Iff ``True``, record the power consumption during each evolve.
            read_timeout (Optional[float]): Set an explicit read timeout for the entire simulation time. This should be sufficient for the simulation to complete, and for data to be returned. Default: ``None``, set a reasonable default timeout.

        Returns:
            (np.ndarray, dict, dict): ``output``, ``new_state``, ``rec_dict``.
            ``output`` is a raster ``(T, Nout)``, containing events for each channel in each time bin. Time bins in ``output`` correspond to the time bins in ``input``.
            ``new_state`` is an empty dictionary. The Xylo HDK does not permit querying or setting state.
            ``rec_dict`` is a dictionary containing recorded internal state of Xylo during evolution, if the ``record`` argument is ``True``. Otherwise this is an empty dictionary.

        Raises:
            `ValueError`: If ``operation_mode`` is not set to ``AcceleratedTime``. For ``RealTime`` please use :py:class:`.XyloMonitor`.
            `ValueError`: If input is empty.
            `TimeoutError`: If reading data times out during the evolution. An explicity timeout can be set using the `read_timeout` argument.
            Warning: if using ``record`` parameter in the evolve function.
        """

        # - Check operation mode
        if (
            self._config.operation_mode
            != samna.xyloAudio3.OperationMode.AcceleratedTime
        ):
            raise ValueError(
                "`operation_mode` needs to be `AcceleratedTime` when using evolve. For debug purposes, use `_evolve_manual`."
            )

        # HACK record is not working inside evolve and was transferred to the class initialization
        if record:
            warn(
                "`record` is now a parameter in the class initialization and its behavior is not updated in the `evolve` method."
            )
        record = self._record

        if record_power:
            raise ValueError(
                f"Power measurement is not available yet by :py:class:`.XyloSamna`"
            )

        # - Get the network size
        Nin, Nhidden, Nout = self.shape[:]
        Nhidden_monitor = Nhidden if record else 0
        Nout_monitor = Nout if record or self._output_mode == "Isyn" else 0

        # -- Control timestep per evolve section
        timestep_count = len(input)
        if not timestep_count:
            raise ValueError("Couldn't read input data size.")

        # - Get current timestep by reading a `Readout` event
        self._write_buffer.write([samna.xyloAudio3.event.TriggerReadout()])
        evts = self._readout_buffer.get_n_events(1, timeout=6000)
        if len(evts) < 1:
            raise TimeoutError(
                "Couldn't read current timestep. No response after 6000 ms."
            )

        start_timestep = evts[0].timestep + 1
        final_timestep = start_timestep + len(input) - 1

        # -- Encode input events
        input_events_list = []

        # - Locate input events
        # - Generate input events
        for i, spike_counts in enumerate(input):
            timestep = start_timestep + i
            for neuron_id, count in enumerate(spike_counts):
                spikes = [
                    samna.xyloAudio3.event.Spike(neuron_id=neuron_id, timestep=timestep)
                ] * count
                input_events_list.extend(spikes)

        # - Add a `TriggerProcessing` event to ensure all time-steps are processed
        event = samna.xyloAudio3.event.TriggerProcessing(
            target_timestep=final_timestep + 1
        )
        input_events_list.append(event)

        # - Clear the read and state buffers
        self._read_buffer.get_events()
        self._readout_buffer.get_events()

        # - Write the events and trigger the simulation
        self._write_buffer.write(input_events_list)

        # - Determine a reasonable read timeout
        if read_timeout is None:
            read_timeout = 4 * len(input)
            read_timeout = read_timeout * 30.0 if record else read_timeout
            read_timeout = int(read_timeout)

        # - Wait until the simulation is finished
        readout_events = self._readout_buffer.get_n_events(
            timestep_count, timeout=read_timeout
        )

        if len(readout_events) < timestep_count:
            message = f"Processing didn't finish for {read_timeout}s. Read {len(readout_events)} events."
            if len(readout_events) > 0:
                message += f" First timestep: {readout_events[0].timestep}, final timestep: {readout_events[-1].timestep}, target timestep: {final_timestep}"
            raise TimeoutError(message)

        # - Read the simulation output data
        xylo_data = hdkutils.decode_accel_mode_data(
            readout_events,
            Nin,
            Nhidden_monitor,
            Nout_monitor,
            Nout,
            start_timestep,
            final_timestep,
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

    def _evolve_manual(
        self,
        input: np.ndarray,
        record: bool = False,
        record_power: bool = False,
        read_timeout: Optional[float] = None,
        *args,
        **kwargs,
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        Evolve a network on the XyloAudio 3 HDK in single-step manual mode. For debug purposes only. Uses 'samna.xylo.OperationMode.Manual' in samna.

        Sends a series of events to the Xylo HDK, evolves the network over the input events, and returns the output events produced during the input period.

        Args:
            input (np.ndarray): A raster ``(T, Nin)`` specifying for each bin the number of input events sent to the corresponding input channel on Xylo, at the corresponding time point. Up to 15 input events can be sent per bin.
            record (bool): Deprecated parameter. Please use ``record`` from the class initialization.
            read_timeout (Optional[float]): Set an explicit read timeout for the entire simulation time. This should be sufficient for the simulation to complete, and for data to be returned. Default: ``None``, set a reasonable default timeout.
            record_power (bool): Iff ``True``, record the power consumption during each evolve.

        Returns:
            (np.ndarray, dict, dict): ``output``, ``new_state``, ``record_dict``.
            ``output`` is a raster ``(T, Nout)``, containing events for each channel in each time bin. Time bins in ``output`` correspond to the time bins in ``input``.
            ``new_state`` is an empty dictionary. The Xylo HDK does not permit querying or setting state.
            ``record_dict`` is a dictionary containing recorded internal state of Xylo during evolution, if the ``record`` argument is ``True``. It also contains power measurement recordings if ``record_power`` is ``True``. Otherwise this is an empty dictionary.

        Raises:
            `ValueError`: If ``operation_mode`` is set to ``RealTime``. For ``RealTime`` please use :py:class:`.XyloMonitor`.
            `TimeoutError`: If reading data times out during the evolution. An explicity timeout can be set using the `read_timeout` argument.
        """

        # - Get some information about the network size
        Nin, Nhidden, Nout = self.shape

        # - Check again operation mode (it could have changed between initializing the class and calling evolve)
        if self._config.operation_mode != samna.xyloAudio3.OperationMode.Manual:
            raise ValueError(
                "`operation_mode` needs to be Manual when using evolve_manual."
            )

        # HACK record is not working inside evolve and was transferred to the class initialization
        if record:
            warn(
                "`record` is now a parameter in the class initialization and its behavior is not updated in the `evolve` method."
            )
        record = self._record

        if record_power:
            raise ValueError(
                f"Power measurement is not available yet by :py:class:`.XyloSamna`"
            )

        # - Advance one time-step
        hdkutils.advance_time_step(self._write_buffer)

        # - Clear the read buffers
        self._read_buffer.get_events()

        # - Wait until xylo is ready
        t_start = time.time()
        while not hdkutils.is_xylo_ready(self._read_buffer, self._write_buffer):
            if time.time() - t_start > read_timeout:
                raise TimeoutError("Timed out waiting for Xylo to be ready.")

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
                raise TimeoutError

            # - Read all synapse and neuron states for this time step
            if record:
                this_state = hdkutils.read_neuron_synapse_state(
                    self._read_buffer, self._write_buffer, Nin, Nhidden, Nout
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
            )[:Nout]
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
                # "times": np.arange(start_timestep, final_timestep + 1),
            }
        else:
            rec_dict = {}

        # - Return the output spikes, the (empty) new state dictionary, and the recorded state dictionary
        return np.array(output_ts), {}, rec_dict
