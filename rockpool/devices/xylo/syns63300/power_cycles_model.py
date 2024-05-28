"""
Cycle count model for Xylo IMU SYNS63300
"""

from rockpool.typehints import FloatVector
import samna
import numpy as np

XyloIMUConfig = samna.xyloImu.configuration.XyloConfiguration


def xylo_imu_cycles(
    config: XyloIMUConfig,
    input_sp: FloatVector = 1.0,
    hidden_sp: FloatVector = 1.0,
    output_sp: FloatVector = 1.0,
) -> float:
    """
    Calculate the average number of cycles required for a given network architecture

    Args:
        config (XyloIMUConfig): A Xylo IMU configuration for which to calculate the cycle requirements
        input_sp (FloatVector): Either a floating-point number 0..1, specifying the average input firing rate, or an actual input spike raster to use in evaluation. Default: `1.0`; estimate a worst-case scenario
        hidden_sp (FloatVector): Either a floating-point number 0..1, specifying the average hidden neuron firing rate, or an actual hidden spike raster to use in evaluation. Default: `1.0`; estimate a worst-case scenario
        output_sp (FloatVector): Either a floating-point number 0..1, specifying the average output neuron firing rate, or an actual output spike raster to use in evaluation. Default: `1.0`; estimate a worst-case scenario

    Returns:
        float: The average number of master clock cycles required for this configuration, for the Xylo SNN core to compute one network `dt`
    """

    # - Spiking probabilities
    if hasattr(input_sp, "mean"):
        input_spk_prob = input_sp.mean()
    else:
        input_spk_prob = input_sp

    if hasattr(hidden_sp, "mean"):
        hidden_spk_prob = hidden_sp.mean()
    else:
        hidden_spk_prob = hidden_sp

    if hasattr(hidden_sp, "mean"):
        output_spk_prob = output_sp.mean()
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
            for t in n.alias_target:
                alias_target_count[t] += 1
                alias_source_count[id] += 1

    is_alias_target_prob = np.mean(alias_target_count)
    is_alias_source_prob = np.mean(alias_source_count)

    # - Input spike processing
    input_loop_cycles = 3.5
    single_input_neuron_cycles = input_spk_prob * Nien * input_loop_cycles
    input_spike_processing_cycles = Nin * (single_input_neuron_cycles + 1)

    # - Hidden neuron Isyn
    hidden_update_cycles = 7
    single_hidden_neuron_isyn_cycles = (
        Nhid_fanout_avg * hidden_update_cycles * hidden_spk_prob + 3
    )
    hidden_isyn_processing_cycles = single_hidden_neuron_isyn_cycles * Nhid

    # - Output neuron Isyn
    output_isyn_update_cycles = 8
    single_output_neuron_isyn_cycles = hidden_spk_prob * (
        1 + Nout * output_isyn_update_cycles
    )
    output_isyn_processing_cycles = single_output_neuron_isyn_cycles * Noen

    # - Hidden neuron spiking
    fixed_hidden_neuron_cycles = 19
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
    fixed_output_neuron_spk_cycles = 12
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


def est_clock_freq(config: XyloIMUConfig, dt: float, margin: float = 0.2):
    """
    Estimate the required master clock frequency, to run a network in real-time

    This function will perform a worst-case analysis, assuming that every input channel, every hidden neuron and every output neuron fire an event on each `dt`

    Args:
        config (XyloIMUConfig):  A Xylo IMU configuration for which to estimate the required clock frequency
        dt (float): The required network `dt`, in seconds
        margin (float): The additional overhead safety margin to add to the estimation, as a fraction. Default: `0.2` (20%)
    """
    cycles = xylo_imu_cycles(config)
    return cycles * (1 + margin) / dt
