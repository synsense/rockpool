"""
Cycle count model for XyloAudio 2 SYNS61201
"""

from rockpool.typehints import FloatVector
from typing import Tuple
import samna
import numpy as np

XyloA2Config = samna.xyloCore2.configuration.XyloConfiguration

__all__ = ["cycles_model", "est_clock_freq"]


def cycles_model(
    config: XyloA2Config,
    input_sp: FloatVector = 1.0,
    hidden_sp: FloatVector = 1.0,
    output_sp: FloatVector = 1.0,
) -> Tuple[float, float]:
    """
    Calculate the average number of cycles required for a given network architecture

    This function contains a model which estimates the number of master clock cycles required for the Xylo SNN SYNS61202 inference core to compute one time-step for a given chip configuration in ``config``. Use :py:func:`.devices.xylo.syns61201.config_from_specification` to obtain a chip configuration, along with :py:meth:`.Module.as_graph` and :py:func:`.devices.xylo.syns61201.mapper`, as described in the deployment tutorials for Xylo.

    By default the model provides a "worst-case" estimation, assuming that every neuron and every input channel fire on each time-step. If desired, real input rasters and real hidden and output spike rasters can be provided for analysis. Alternative spiking probabilities can also be provided as floats ``0..1``.

    Note that when estimating spiking probablility, only boolean values are relevant --- either a spike or no spike per time step per channel. Multiple events per bin cost the same as a single event.

    Args:
        config (XyloA2Config): A XyloAudio 2 configuration for which to calculate the cycle requirements
        input_sp (FloatVector): Either a floating-point number 0..1, specifying the average input firing rate, or an actual input spike raster to use in evaluation. Default: `1.0`; estimate a worst-case scenario
        hidden_sp (FloatVector): Either a floating-point number 0..1, specifying the average hidden neuron firing rate, or an actual hidden spike raster to use in evaluation. Default: `1.0`; estimate a worst-case scenario
        output_sp (FloatVector): Either a floating-point number 0..1, specifying the average output neuron firing rate, or an actual output spike raster to use in evaluation. Default: `1.0`; estimate a worst-case scenario

    Returns:
        Tuple[float, float]: (clk_cyles_reqd, additional_isyn2_ops)
        clk_cyles_reqd (float): The average number of master clock cycles required for this configuration, for the Xylo SNN core to compute one network `dt`
        additional_isyn2_ops (float): The average additional number of operations required when Isyn2 is enabled. Will be zero if Isyn2 is disabled. Note: These operations occur in parallel to Isyn1, and do not require additional master clock cycles.
    """

    # - Cycle count magic numbers
    input_loop_cycles = 3.5
    hidden_update_cycles = 7
    output_isyn_update_cycles = 8
    fixed_hidden_neuron_cycles = 19
    fixed_output_neuron_spk_cycles = 12

    # - Spiking probabilities
    if np.size(input_sp) > 1:
        input_spk_prob = np.count_nonzero(input_sp) / np.size(input_sp)
    else:
        input_spk_prob = input_sp

    if np.size(hidden_sp) > 1:
        hidden_spk_prob = np.count_nonzero(hidden_sp) / np.size(hidden_sp)
    else:
        hidden_spk_prob = hidden_sp

    if np.size(output_sp) > 1:
        output_spk_prob = np.count_nonzero(output_sp) / np.size(output_sp)
    else:
        output_spk_prob = output_sp

    # - Extract network shapes
    Nin, _ = np.array(config.input.weights).shape
    Nhid = len(config.reservoir.neurons)
    Nien = Nhid
    _, Nout = np.array(config.readout.weights).shape
    Noen = Nhid

    # - Is synapse 2 enabled?
    is_isyn2_enabled = config.synapse2_enable

    # - Average fanout for the network
    Nhid_fanout_avg = np.mean(
        np.sum(np.abs(np.sign(np.array(config.reservoir.weights))), axis=1)
    )

    # - Number of alias sources and targets
    alias_target_count = np.zeros(Nhid)
    alias_source_count = np.zeros(Nhid)
    for id, n in enumerate(config.reservoir.neurons):
        if n.alias_target is not None:
            alias_target_count[n.alias_target] += 1
            alias_source_count[id] += 1

    is_alias_target_prob = np.mean(alias_target_count)
    is_alias_source_prob = np.mean(alias_source_count)

    additional_isyn2_ops = 0

    # - Input spike processing
    single_input_neuron_cycles = input_spk_prob * Nien * input_loop_cycles
    input_spike_processing_cycles = Nin * (single_input_neuron_cycles + 1)
    additional_isyn2_ops += is_isyn2_enabled * (0.5 + 1) * input_spk_prob * Nin * Nien

    # - Hidden neuron Isyn
    single_hidden_neuron_isyn_cycles = (
        Nhid_fanout_avg * hidden_update_cycles * hidden_spk_prob + 3
    )
    hidden_isyn_processing_cycles = single_hidden_neuron_isyn_cycles * Nhid

    additional_isyn2_ops += (
        is_isyn2_enabled * (4 + 1) * Nhid_fanout_avg * hidden_spk_prob * Nhid
    )

    # - Output neuron Isyn
    single_output_neuron_isyn_cycles = hidden_spk_prob * (
        1 + Nout * output_isyn_update_cycles
    )
    output_isyn_processing_cycles = single_output_neuron_isyn_cycles * Noen

    # - Hidden neuron spiking
    var_hidden_neuron_cycles = (
        max(hidden_spk_prob, is_alias_target_prob * is_alias_source_prob) * 2
        + max(
            hidden_spk_prob * is_alias_target_prob,
            is_alias_target_prob * is_alias_source_prob,
        )
        * 3
        + hidden_spk_prob * 1
        + max(hidden_spk_prob, is_alias_target_prob) * is_alias_source_prob * 1
    )
    total_single_hidden_neuron_spk_cycles = (
        fixed_hidden_neuron_cycles + var_hidden_neuron_cycles
    )
    hidden_spk_processing_cycles = total_single_hidden_neuron_spk_cycles * Nhid

    additional_isyn2_ops += is_isyn2_enabled * 1 * Nhid

    # - Output neuron spiking
    var_output_neuron_spk_cycles = 0 * output_spk_prob
    total_output_neuron_spk_cycles = (
        fixed_output_neuron_spk_cycles + var_output_neuron_spk_cycles
    )
    output_spk_processing_cycles = total_output_neuron_spk_cycles * Nout

    # - Total processing cycles
    est_processing_cycles = (
        input_spike_processing_cycles
        + hidden_isyn_processing_cycles
        + output_isyn_processing_cycles
        + hidden_spk_processing_cycles
        + output_spk_processing_cycles
    )

    return est_processing_cycles, additional_isyn2_ops


def est_clock_freq(config: XyloA2Config, dt: float, margin: float = 0.2):
    """
    Estimate the required master clock frequency, to run a network in real-time

    This function will perform a worst-case analysis, assuming that every input channel, every hidden neuron and every output neuron fire an event on each `dt`. An additional margin is included (Default: 20%), to guarantee that the model will run in real time at the suggested master clock frequency.

    Args:
        config (XyloA2Config):  A XyloAudio 2 configuration for which to estimate the required clock frequency
        dt (float): The required network `dt`, in seconds
        margin (float): The additional overhead safety margin to add to the estimation, as a fraction. Default: `0.2` (20%)
    """
    cycles, _ = cycles_model(config)
    return cycles * (1 + margin) / dt
