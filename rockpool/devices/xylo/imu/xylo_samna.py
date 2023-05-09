"""
Utilities for producing a samna HW configuration for Xylo IMU devices
"""


import numpy as np
import samna

from samna.xyloImu.configuration import XyloConfiguration, InputInterfaceConfig

# - Typing
from typing import Optional, Union, Callable, List
from warnings import warn


# - Configure exports
__all__ = [
    "config_from_specification",
    "if_config_from_specification",
    "save_config",
    "load_config",
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
    Convert a full network specification to a xylo config and validate it

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

    Returns: (:py:class:`.samna.xyloImu.XyloConfiguration`, bool, str): config, is_valid, message
        ``config`` will be a `XyloConfiguration`.
        ``is_valid`` will be a boolean flag ``True`` iff the configuration is valid.
        ``message`` will be an empty string if the configuration is valid, or a message indicating why the configuration is invalid.
    """
    # - Check input weights
    if weights_in.ndim < 2:
        raise ValueError(
            f"Input weights must be at least 2 dimensional `(Nin, Nin_res [, 1])`. Found {weights_in.shape}"
        )

    # - Check output weights
    if weights_out.ndim != 2:
        raise ValueError("Output weights must be 2 dimensional `(Nhidden, Nout)`")

    # - Get network shape
    Nin, Nin_res, Nsyn = weights_in.shape
    Nhidden, Nout = weights_out.shape

    # - Check number of input synapses
    if Nsyn > 1:
        raise ValueError(
            f"Only 1 input synapse is supported on Xylo-IMU. Found {Nsyn}."
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
            "Recurrent weights must be of shape `(Nhidden, Nhidden [, 1])`"
        )

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
        raise ValueError(
            "More than 128 input expantion neurons (IEN) detected. Only 128 IEN are available on Xylo."
        )
    else:
        config.input.weights = weights_in[:, :, 0]
    config.hidden.weights = weights_rec[:, :, 0]
    if weights_out.shape[1] > 128:
        raise ValueError(
            "More than 128 output expantion neurons (OEN) detected. Only 128 OEN are available on Xylo."
        )
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


def if_config_from_specification(
    num_avg_bitshift: int = 6,
    select_iaf_output: bool = False,
    sampling_period: int = 256,
    filter_a1_list: list = [
        -64700,
        -64458,
        -64330,
        -64138,
        -63884,
        -63566,
        -63185,
        -62743,
        -62238,
        -61672,
        -61045,
        -60357,
        -59611,
        -58805,
        -57941,
    ],
    filter_a2_list: list = [0x00007CBF] + [0x00007C0A] * 14,
    scale_values: list = [8] * 15,
    Bb_list: list = [6] * 15,
    B_wf_list: list = [8] * 15,
    B_af_list: list = [9] * 15,
    iaf_threshold_values: list = [0x000007D0] * 15,
    *args,
    **kwargs,
) -> InputInterfaceConfig:
    """
    Configure the imu interface module

    Args:
        config (XyloConfiguration): a configuration for Xylo IMU
        num_avg_bitshift (int): number of bitshifts used in the low-pass filter implementation
        select_iaf_output (bool): if True, select the IAF neuron spike encoder; else, select the scale spike encoder
        sampling_period (int): sampling period
        filter_a1_list (list): list of a1 tap values
        filter_a2_list (list): list of a2 tap values
        scale_values (list): list of number of right-bit-shifts needed for down-scaling the input signal
        Bb_list (list): list of bits needed for scaling b0
        B_wf_list (list): list of bits needed for fractional part of the filter output
        B_af_list (list): list of bits needed for encoding the fractional parts of taps
        iaf_threshold_values (list): list of threshold values of IAF neurons

    Return:
        updated Xylo configuration
    """

    if_config = samna.xyloImu.configuration.InputInterfaceConfig()

    # IMU interface hyperparameters
    if_config.enable = True
    if_config.estimator_k_setting = num_avg_bitshift  # num_avg_bitshift
    if_config.select_iaf_output = select_iaf_output  # True if use IAF encoding
    if_config.update_matrix_threshold = sampling_period - 1  # sampling_period
    if_config.delay_threshold = 1
    if_config.bpf_bb_values = Bb_list
    if_config.bpf_bwf_values = B_wf_list
    if_config.bpf_baf_values = B_af_list
    if_config.bpf_a1_values = [i & 0x1FFFF for i in filter_a1_list]
    if_config.bpf_a2_values = filter_a2_list
    if_config.scale_values = scale_values  # num_scale_bits
    if_config.iaf_threshold_values = iaf_threshold_values

    return if_config


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
