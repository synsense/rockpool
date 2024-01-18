import samna

import numpy as np

XyloIMUConfig = samna.xyloImu.configuration.XyloConfiguration


def xylo_imu_cycles(config: XyloIMUConfig):
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
    hidden_alias_sources_avg = 1
    hidden_alias_targets_avg = 1
    hidden_alias_avg = np.max(hidden_alias_sources_avg, hidden_alias_targets_avg)

    # - Spiking probabilities
    input_spk_prob = 1
    hidden_spk_prob = 1
    output_spk_prob = 1

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
    output_isyn_processing_cycles = single_output_neuron_isyn_cycles * Nhid

    # - Hidden neuron spiking
    fixed_hidden_neuron_cycles = 19
    var_hidden_neuron_cycles = hidden_spk_prob * (
        2 + hidden_alias_sources_avg * 3 + 1 + hidden_alias_sources_avg * 1
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
    cycles = xylo_imu_cycles(config)
    return cycles * (1 + margin) / dt
