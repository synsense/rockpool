"""
Samna-backed bridge to Xylo dev kit for Xylo IMU
"""

# - Samna imports
import samna
from samna.xyloImu.configuration import XyloConfiguration

from . import xylo_imu_devkit_utils as hdkutils
from .xylo_imu_devkit_utils import XyloIMUHDK

import time
import numpy as np
from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter

# - Typing
from typing import Optional, Union, Callable, List
from warnings import warn

try:
    from tqdm.autonotebook import tqdm
except ModuleNotFoundError:

    def tqdm(wrapped, *args, **kwargs):
        return wrapped


# - Configure exports
__all__ = ["XyloIMUMonitor", "config_from_specification"]

Default_Main_Clock_Rate = int(5e7)


class XyloIMUMonitor(Module):
    """
    A spiking neuron :py:class:`.Module` backed by the Xylo hardware, via `samna`.

    Use :py:func:`.config_from_specification` to build and validate a configuration for Xylo.

    """

    def __init__(
        self,
        device: XyloIMUHDK,
        config: XyloConfiguration = None,
        dt: float = 1e-3,
        output_mode: str = "Spike",
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
            dt (float): The simulation time-step to use for this Module
            output_mode (str): The readout mode for the Xylo device. This must be one of ``["Spike", "Vmem"]``. Default: "Spike", return events from the output layer.
            main_clk_rate(int): The main clock rate of Xylo
            hibernation_mode (bool): If True, hibernation mode will be switched on, which only outputs events if it receives inputs above a threshold.
            interface_params(dict): The dictionary of Xylo interface parameters used for the `hdkutils.config_if_module` function, the keys of which must be one of "num_avg_bitshif", "select_iaf_output", "sampling_period", "filter_a1_list", "filter_a2_list", "scale_values", "Bb_list", "B_wf_list", "B_af_list", "iaf_threshold_values".
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
        self._state_buffer = hdkutils.new_xylo_state_monitor_buffer(device)

        # - Initialise the superclass
        super().__init__(
            shape=(Nin, Nhidden, Nout), spiking_input=True, spiking_output=True
        )

        # - Store the device
        self._device: XyloIMUHDK = device
        """ `.XyloHDK`: The Xylo HDK used by this module """

        # - Store the configuration (and apply it)
        self.config: Union[
            XyloConfiguration, SimulationParameter
        ] = SimulationParameter(shape=(), init_func=lambda _: config)
        self._config = config
        if hibernation_mode:
            self._config.enable_hibernation_mode = True
        """ `.XyloConfiguration`: The HDK configuration applied to the Xylo module """

        # - Store the timestep
        self.dt: Union[float, SimulationParameter] = dt
        """ float: Simulation time-step of the module, in seconds """

        # - Store the main clock rate
        self._main_clk_rate = int(main_clk_rate)

        # - Store the io module
        self._io = self._device.get_io_module()
        self._io.spi_slave_enable(True)

        # - Set main clock rate
        if self._main_clk_rate != Default_Main_Clock_Rate:
            self._io.set_main_clk_rate(self._main_clk_rate)

        # - Configure to auto mode
        self.auto_config(interface_params)

        # - Send first trigger to start to run full auto mode
        hdkutils.advance_time_step(self._write_buffer)

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

    def auto_config(self, interface_params: dict):
        """
        Configure the Xylo HDK to use real-time mode

        Args:
            interface_params (dict): specify the interface parameters
        """

        # - Config the streaming mode
        config = hdkutils.config_auto_mode(
            self._config, self.dt, self._main_clk_rate, self._io
        )
        # - Config the IMU interface and apply current configuration
        self.config = hdkutils.config_if_module(config, **interface_params)
        # - Set configuration and reset state buffer
        self._state_buffer.set_configuration(self._config)
        self._state_buffer.reset()

    def evolve(self, input_data, record: bool = False) -> (list, dict, dict):
        """
        Evolve a network on the Xylo HDK in Real-time mode

        Args:
            input_data (np.ndarray): An array ``[T, Nin]``, specifying the number of time-steps to record.

        Returns:
            (list, dict, dict) output_events, {}, {}
            output_events is a list that stores the output events of T time-steps.
        """

        Nt = input_data.shape[0]
        out = []
        count = 0

        while count < int(Nt):
            if self._output_mode == "Vmem":
                output = self._state_buffer.get_output_v_mem()

            elif self._output_mode == "Spike":
                output = self._state_buffer.get_output_spike()

            if output[0]:
                self._state_buffer.reset()
                count += 1
                out.append([sub[-1] for sub in output])

        return out, {}, {}


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
):
    """
    Convert a full network specification to a xylo config and validate it

    See Also:
        For detailed information about the networks supported on Xylo, see :ref:`/devices/xylo-overview.ipynb`

    Args:
        weights_in (np.ndarray): A quantised 8-bit input weight matrix ``(Nin, Nin_res, 1)``. The third dimension specifies connections onto the second input synapse for each neuron. ``Nin_res`` indicates the number of hidden-layer neurons that receive input from the input channels.
        weights_rec (np.ndarray): A quantised 8-bit recurrent weight matrix ``(Nhidden, Nhidden, 1)``. The third dimension specified connections onto the second input synapse for each neuron. Default: ``0``
        weights_out (np.ndarray): A quantised 8-bit output weight matrix ``(Nhidden, Nout)``.
        dash_mem (np.ndarray): A vector or list ``(Nhidden,)`` specifing decay bitshift for neuron state for each hidden layer neuron. Default: ``1``
        dash_mem_out (np.ndarray): A vector or list ``(Nout,)`` specifing decay bitshift for neuron state for each output neuron. Default: ``1``
        dash_syn (np.ndarray): A vector or list ``(Nhidden,)`` specifing decay bitshift for synapse 1 state for each hidden layer neuron. Default: ``1``
        dash_syn_out (np.ndarray): A vector or list ``(Nout,)`` specifing decay bitshift for synapse state for each output layer neuron. Default: ``1``
        threshold (np.ndarray): A vector or list ``(Nhidden,)`` specifing the firing threshold for each hidden layer neuron. Default: ``0``
        threshold_out (np.ndarray): A vector or list ``(Nout,)`` specifing the firing threshold for each output layer neuron. Default: ``0``
        weight_shift_in (int): The number of bits to left-shift each input weight. Default: ``0``
        weight_shift_rec (int): The number of bits to left-shift each recurrent weight. Default: ``0``
        weight_shift_out (int): The number of bits to left-shift each output layer weight. Default: ``0``
        aliases (Optional[List[List[int]]]): For each neuron in the hidden population, a list containing the alias targets for that neuron

    Returns: (:py:class:`.samna.xyloCore2.XyloConfiguration`, bool, str): config, is_valid, message
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
    config = samna.xyloImu.configuration.XyloConfiguration()

    # general
    config.imu_if_input_enable = False
    config.debug.always_update_omp_stat = True

    if bias_hidden is not None or bias_out is not None:
        config.bias_enable = True

    config.hidden.aliasing = aliases is not None
    config.input.weight_bit_shift = weight_shift_in
    config.hidden.weight_bit_shift = weight_shift_rec
    config.readout.weight_bit_shift = weight_shift_out
    if weights_in.shape[1] > 128:
        config.input.weights = weights_in[:, :128, 0]
    else:
        config.input.weights = weights_in[:, :, 0]
    config.hidden.weights = weights_rec[:, :, 0]
    if weights_out.shape[1] > 128:
        config.readout.weights = weights_out[-128:, :]
    else:
        config.readout.weights = weights_out

    hidden_neurons = []
    for i in range(len(weights_rec)):
        neuron = samna.xyloImu.configuration.HiddenNeuron()
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
        neuron = samna.xyloImu.configuration.OutputNeuron()
        neuron.i_syn_decay = dash_syn_out[i]
        neuron.v_mem_decay = dash_mem_out[i]
        neuron.threshold = threshold_out[i]
        if bias_out is not None:
            neuron.v_mem_bias = bias_out[i]
        readout_neurons.append(neuron)

    config.readout.neurons = readout_neurons

    # - Validate the configuration and return
    is_valid, message = samna.xyloImu.validate_configuration(config)
    return config, is_valid, message
