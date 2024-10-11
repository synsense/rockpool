"""
Cycle count model for Xylo™IMU SYNS63300 and Xylo™Audio 3 SYNS65302
"""

from rockpool.typehints import FloatVector
from typing import Union
import samna
import numpy as np

XyloIMUConfig = samna.xyloImu.configuration.XyloConfiguration
XyloA3Config = samna.xyloAudio3.configuration.XyloConfiguration

__all__ = ["cycles_model", "est_clock_freq"]


def cycles_model(
    config: Union[XyloIMUConfig, XyloA3Config],
    input_sp: FloatVector = 1.0,
    hidden_sp: FloatVector = 1.0,
    output_sp: FloatVector = 1.0,
) -> float:
    """
    Calculate the average number of cycles required for a given network architecture

    This function contains a model which estimates the number of master clock cycles required for the Xylo SNN SYNS61202 and SYNS65302 inference cores to compute one time-step for a given chip configuration in ``config``. Use :py:func:`~.devices.xylo.syns61201.config_from_specification` to obtain a chip configuration, along with :py:meth:`.Module.as_graph` and :py:func:`~.devices.xylo.syns61201.mapper`, as described in the deployment tutorials for Xylo.

    By default the model provides a "worst-case" estimation, assuming that every neuron and every input channel fire on each time-step. If desired, real input rasters and real hidden and output spike rasters can be provided for analysis. Alternative spiking probabilities can also be provided as floats ``0..1``.

    Note that when estimating spiking probablility, only boolean values are relevant --- either a spike or no spike per time step per channel. Multiple events per bin cost the same as a single event.

    Args:
        config (Union[XyloIMUConfig, XyloA3Config]): A Xylo configuration for which to calculate the cycle requirements
        input_sp (FloatVector): Either a floating-point number 0..1, specifying the average input firing rate, or an actual input spike raster to use in evaluation. Default: `1.0`; estimate a worst-case scenario
        hidden_sp (FloatVector): Either a floating-point number 0..1, specifying the average hidden neuron firing rate, or an actual hidden spike raster to use in evaluation. Default: `1.0`; estimate a worst-case scenario
        output_sp (FloatVector): Either a floating-point number 0..1, specifying the average output neuron firing rate, or an actual output spike raster to use in evaluation. Default: `1.0`; estimate a worst-case scenario

    Returns:
        float: The average number of master clock cycles required for this configuration, for the Xylo SNN core to compute one network `dt`
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
    Nien = np.sum(
        np.sign(np.sum(np.abs(np.sign(np.array(config.input.weights))), axis=0))
    )
    Nhid = len(config.hidden.neurons)
    _, Nout = np.array(config.readout.weights).shape
    Noen = np.sum(
        np.sign(np.sum(np.abs(np.sign(np.array(config.readout.weights))), axis=1))
    )

    # - Average fanout for the network
    Nhid_fanout_avg = np.mean(
        np.sum(np.abs(np.sign(np.array(config.hidden.weights))), axis=1)
    )

    # - Number of alias sources and targets
    alias_target_count = np.zeros(Nhid)
    alias_source_count = np.zeros(Nhid)
    for id, n in enumerate(config.hidden.neurons):
        if n.alias_target is not None:
            t = n.alias_target
            alias_target_count[t] += 1
            alias_source_count[id] += 1

    is_alias_target_prob = np.mean(alias_target_count)
    is_alias_source_prob = np.mean(alias_source_count)

    # - Input spike processing
    single_input_neuron_cycles = input_spk_prob * Nien * input_loop_cycles
    input_spike_processing_cycles = Nin * (single_input_neuron_cycles + 1)

    # - Hidden neuron Isyn
    single_hidden_neuron_isyn_cycles = (
        Nhid_fanout_avg * hidden_update_cycles * hidden_spk_prob + 3
    )
    hidden_isyn_processing_cycles = single_hidden_neuron_isyn_cycles * Nhid

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

    return est_processing_cycles


def est_clock_freq(
    config: Union[XyloIMUConfig, XyloA3Config], dt: float, margin: float = 0.2
):
    """
    Estimate the required master clock frequency, to run a network in real-time

    This function will perform a worst-case analysis, assuming that every input channel, every hidden neuron and every output neuron fire an event on each `dt`. An additional margin is included (Default: 20%), to guarantee that the model will run in real time at the suggested master clock frequency.

    Args:
        config (Union[XyloIMUConfig, XyloA3Config]):  A Xylo configuration for which to estimate the required clock frequency
        dt (float): The required network `dt`, in seconds
        margin (float): The additional overhead safety margin to add to the estimation, as a fraction. Default: `0.2` (20%)
    """
    cycles = cycles_model(config)
    return cycles * (1 + margin) / dt
