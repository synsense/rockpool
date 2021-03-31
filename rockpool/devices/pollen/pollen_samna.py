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
from samna.pollen.configuration import (
    PollenConfiguration,
    ReservoirNeuron,
    OutputNeuron,
)

from samna.pollen import validate_configuration

import numpy as np
from typing import Optional


__all__ = ["config_from_specification"]


def config_from_specification(
    weights_in: np.ndarray,
    weights_rec: np.ndarray,
    weights_out: np.ndarray,
    dash_mem: np.ndarray,
    dash_mem_out: np.ndarray,
    dash_syn: np.ndarray,
    dash_syn_out: np.ndarray,
    threshold: np.ndarray,
    threshold_out: np.ndarray,
    weight_shift: int = 0,
    weight_shift_out: int = 0,
    aliases: Optional[list] = None,
):
    """
    Convert a full network specification to a pollen config and validate it

    Args:
        weights_in:
        weights_rec:
        weights_out:
        dash_mem:
        dash_mem_out:
        dash_syn:
        dash_syn_out:
        threshold:
        threshold_out:
        weight_shift:
        weight_shift_out:
        aliases:

    Returns:

    """
    # - Work out shapes
    if weights_in.ndim != 3:
        raise ValueError("Input weights must be 3 dimensional `(Nin, Nhidden, 2)`")

    if weights_rec.ndim != 3 or weights_rec.shape[0] != weights_rec.shape[1]:
        raise ValueError("Recurrent weights must be of shape `(Nhidden, Nhidden, 2)`")

    if weights_out.ndim != 2:
        raise ValueError("Output weights must be 2 dimensional")

    Nin, Nhidden, _ = weights_in.shape
    _, Nout = weights_out.shape

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
    if aliases is None:
        aliases = [[] * Nhidden]

    if len(aliases) != Nhidden:
        raise ValueError(
            f"Aliases list must have `Nhidden` entries (`Nhidden` = {Nhidden})"
        )

    # - Check bitshift TCs
    if dash_mem.size != Nhidden or dash_syn.size != Nhidden:
        raise ValueError(
            f"`dash_mem` and `dash_syn` need `Nhidden` entries (`Nhidden` = {Nhidden})"
        )

    if dash_mem_out.size != Nout or dash_syn_out.size != Nout:
        raise ValueError(
            f"`dash_mem_out` and `dash_syn_out` need `Nout` entries (`Nout` = {Nout})"
        )

    # - Check thresholds
    if threshold.size != Nhidden:
        raise ValueError(
            f"`thresholds` needs `Nhidden` entries (`Nhidden` = {Nhidden})"
        )

    if threshold_out.size != Nout:
        raise ValueError(
            f"`thresholds_out` needs `Nhidden` entries (`Nhidden` = {Nhidden})"
        )

    config = PollenConfiguration()
    config.synapse2_enable = False
    config.reservoir.aliasing = False
    config.reservoir.weight_bit_shift = weight_shift
    config.readout.weight_bit_shift = weight_shift_out

    config.input_expansion.weights = weights_in[:, :, 0]
    config.reservoir.weights = weights_rec[:, :, 0]
    config.readout.weights = weights_out[:, :, 0]

    reservoir_neurons = []
    for i in range(len(weights_rec)):
        neuron = ReservoirNeuron()
        if len(aliases[i]) > 0:
            neuron.alias_target = aliases[i][0]
        neuron.i_syn_decay = dash_syn[i][0]
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

    return config, validate_configuration(config)
