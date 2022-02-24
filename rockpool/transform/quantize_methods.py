"""
Quantisation methods for Xylo
"""

import numpy as np
import copy

__all__ = ["global_quantize", "channel_quantize"]


def global_quantize(
    weights_in: np.ndarray,
    weights_rec: np.ndarray,
    weights_out: np.ndarray,
    threshold: np.ndarray,
    threshold_out: np.ndarray,
    dash_mem: np.ndarray,
    dash_mem_out: np.ndarray,
    dash_syn: np.ndarray,
    dash_syn_2: np.ndarray,
    dash_syn_out: np.ndarray,
    fuzzy_scaling: bool = False,
    bits_per_weight: int = 8,
    bits_per_threshold: int = 16,
    *_,
    **__,
):
    """
    Quantize a Xylo model for deployment, using global parameter scaling

    The figure below illustrates the groups of weights which are considered for quantization. Under this global method, all weights in the network are considered together when scaling and quantizing weights and thresholds. Input and recurrent weights are considered together as a group; output weights are considered separately when quantizing. Dashes are rounded and cast to integer.

                 target
           -------------------
    s   -------------------  -
    o   -**- -**- -**- -**-  -
    u   -**- -**- -**- -**-  -
    r   -**- -**- -**- -**-  -
    c   -**- -**- -**- -**-  -
    e   -**- -**- -**- -**-  -
        -------------------

    Examples:
        specs = xylo.devices.mapper(net.as_graph(), weight_dtype="float", threshold_dtype="float")
        specs.update(global_quantize(**specs, fuzzy_scaling = True))
        xylo.devices.XyloSim.from_specifications(specs)

    Args:
        weights_in (np.ndarray): Input weight matrix
        weights_rec (np.ndarray): Recurrent weight matrix
        weights_out (np.ndarray): Output weight matrix
        threshold (np.ndarray): Firing threshold for hidden neurons
        threshold_out (np.ndarray): Firing threshold for output neurons
        dash_mem (np.ndarray): Dash for membrane potential of hidden neurons
        dash_mem_out (np.ndarray): Dash for membrane potential of output neurons
        dash_syn (np.ndarray): Dash for synaptic current of hidden neurons
        dash_syn_2 (np.ndarray): Dash for second synaptic current of hidden neurons
        dash_syn_out (np.ndarray): Dash for synaptic current of output neurons
        fuzzy_scaling (bool): If ``True``, scale and clip weights to 2*std dev. If ``False`` (default), scale and clip to maximum absolute weight.
        bits_per_weight (int): Number of bits per integer signed weight. Default: ``8``
        bits_per_threshold (int): Number of bits per integer signed threshold. Default: ``16``

    Returns:
        dict: `model_quan` which can be used to update a Xylo specification dictionary
    """

    w_in = copy.copy(weights_in)
    w_rec = copy.copy(weights_rec)
    w_out = copy.copy(weights_out)
    threshold = copy.copy(threshold)
    threshold_out = copy.copy(threshold_out)
    max_w_quan = 2 ** (bits_per_weight - 1) - 1
    max_th_quan = 2 ** (bits_per_threshold - 1) - 1

    if fuzzy_scaling:
        # detect outliers
        weights_rec_ = np.concatenate([np.ravel(w_in), np.ravel(w_rec)])
        weights_rec_ = weights_rec_[weights_rec_ != 0]
        weights_out_ = np.ravel(w_out)
        weights_out_ = weights_out_[weights_out_ != 0]
        max_w = np.abs(weights_rec_.mean()) + 2 * weights_rec_.std()
        max_w_out = np.abs(weights_out_.mean()) + 2 * weights_out_.std()
    else:
        max_w = 0
        max_w = np.max([max_w, np.max(np.abs(w_in))])
        max_w = np.max([max_w, np.max(np.abs(w_rec))])
        max_w_out = np.max([0, np.max(np.abs(w_out))])

    # determine scaling value
    if max_w != 0:
        scaling = max_w_quan / max_w
    else:
        scaling = 1

    if max_w_out != 0:
        scaling_out = max_w_quan / max_w_out
    else:
        scaling_out = 1

    # scale weights
    weights_in = np.round(w_in * scaling).astype(int)
    weights_rec = np.round(w_rec * scaling).astype(int)
    weights_out = np.round(w_out * scaling_out).astype(int)

    # scale thresholds
    threshold = np.round(threshold * scaling).astype(int)
    threshold_out = np.round(threshold_out * scaling_out).astype(int)

    # if the threshold exceed boundary
    if np.abs(np.max(threshold)) > max_th_quan:
        limited_scaling = max_th_quan / np.max(threshold)
        threshold = np.round(threshold * limited_scaling).astype(int)
        weights_in = np.round(w_in * limited_scaling).astype(int)
        weights_rec = np.round(w_rec * limited_scaling).astype(int)

    if np.abs(np.max(threshold_out)) > max_th_quan:
        limited_scaling = max_th_quan / np.max(threshold_out)
        threshold_out = np.round(threshold_out * limited_scaling).astype(int)
        weights_out = np.round(w_out * limited_scaling).astype(int)
        weights_rec = np.round(w_rec * limited_scaling).astype(int)

    # round and cast all dashes to integer
    dash_mem = np.round(dash_mem).astype(int)
    dash_mem_out = np.round(dash_mem_out).astype(int)
    dash_syn = np.round(dash_syn).astype(int)
    dash_syn_2 = np.round(dash_syn_2).astype(int)
    dash_syn_out = np.round(dash_syn_out).astype(int)

    model_quan = {
        "weights_in": weights_in,
        "weights_rec": weights_rec,
        "weights_out": weights_out,
        "threshold": threshold,
        "threshold_out": threshold_out,
        "dash_mem": dash_mem,
        "dash_mem_out": dash_mem_out,
        "dash_syn": dash_syn,
        "dash_syn_2": dash_syn_2,
        "dash_syn_out": dash_syn_out,
    }

    return model_quan


def channel_quantize(
    weights_in: np.ndarray,
    weights_rec: np.ndarray,
    weights_out: np.ndarray,
    threshold: np.ndarray,
    threshold_out: np.ndarray,
    dash_mem: np.ndarray,
    dash_mem_out: np.ndarray,
    dash_syn: np.ndarray,
    dash_syn_2: np.ndarray,
    dash_syn_out: np.ndarray,
    bits_per_weight: int = 8,
    bits_per_threshold: int = 16,
    *_,
    **__,
):
    """
    Quantize a Xylo model for deployment, using per-channel parameter scaling

    The figure below illustrates the groups of weights which are considered for quantization. Under this per-channel method, all input weights to a single target neuron are considered together when scaling and quantizing weights and thresholds. Input and recurrent weights are considered together as a group; output weights are considered separately when quantizing. Dashes are rounded and cast to integer.

                 target
           -------------------
    s   -------------------  -
    o   -++- -**- -##- -oo-  -
    u   -++- -**- -##- -oo-  -
    r   -++- -**- -##- -oo-  -
    c   -++- -**- -##- -oo-  -
    e   -++- -**- -##- -oo-  -
        -------------------

    Examples:
        specs = xylo.devices.mapper(net.as_graph(), weight_dtype="float", threshold_dtype="float")
        specs.update(channel_quantize(**specs, bits_per_weight = 12))
        xylo.devices.XyloSim.from_specifications(specs)

    Args:
        weights_in (np.ndarray): Input weight matrix
        weights_rec (np.ndarray): Recurrent weight matrix
        weights_out (np.ndarray): Output weight matrix
        threshold (np.ndarray): Firing threshold for hidden neurons
        threshold_out (np.ndarray): Firing threshold for output neurons
        dash_mem (np.ndarray): Dash for membrane potential of hidden neurons
        dash_mem_out (np.ndarray): Dash for membrane potential of output neurons
        dash_syn (np.ndarray): Dash for synaptic current of hidden neurons
        dash_syn_2 (np.ndarray): Dash for second synaptic current of hidden neurons
        dash_syn_out (np.ndarray): Dash for synaptic current of output neurons
        bits_per_weight (int): Number of bits per integer signed weight. Default: ``8``
        bits_per_threshold (int): Number of bits per integer signed threshold. Default: ``16``

    Returns:
        dict: `model_quan` which can be used to update a Xylo specification dictionary
    """

    w_in = copy.copy(weights_in)
    w_rec = copy.copy(weights_rec)
    w_out = copy.copy(weights_out)
    threshold = copy.copy(threshold)
    threshold_out = copy.copy(threshold_out)
    max_w_quan = 2 ** (bits_per_weight - 1) - 1
    max_th_quan = 2 ** (bits_per_threshold - 1) - 1

    # quantize input weight, recurrent weight, threshold
    # two weight matrix need to stack together to consider per-channel quantization
    w_in_quan = np.zeros(shape=w_in.shape)
    w_rec_quan = np.zeros(shape=w_rec.shape)
    threshold_quan = np.zeros(shape=threshold.shape)

    for i in range(w_in.shape[1]):
        # if two synaptic connection is used
        if len(w_in.shape) == 3:
            max_w = 0
            max_w = np.max([max_w, np.max(np.abs(w_in[:, i, :]))])
            max_w = np.max([max_w, np.max(np.abs(w_rec[:, i, :]))])
            if max_w != 0:
                scaling = max_w_quan / max_w
                w_in_quan[:, i, :] = np.round(w_in[:, i, :] * scaling)
                w_rec_quan[:, i, :] = np.round(w_rec[:, i, :] * scaling)
                threshold_quan[i] = np.round(threshold[i] * scaling)
                # if the threshold exceed boundary
                if np.abs(threshold[i]) > max_th_quan:
                    limited_scaling = max_th_quan / threshold[i]
                    threshold_quan[i] = np.round(threshold[i] * limited_scaling)
                    w_in_quan[:, i, :] = np.round(w_in[:, i, :] * limited_scaling)
                    w_rec_quan[:, i, :] = np.round(w_rec[:, i, :] * limited_scaling)
            else:
                threshold_quan[i] = np.round(threshold[i])

        # if only one synaptic connection is used
        elif len(w_in.shape) == 2:
            max_w = 0
            max_w = np.max([max_w, np.max(np.abs(w_in[:, i]))])
            max_w = np.max([max_w, np.max(np.abs(w_rec[:, i]))])
            if max_w != 0:
                scaling = max_w_quan / max_w
                w_in_quan[:, i] = np.round(w_in[:, i] * scaling)
                w_rec_quan[:, i] = np.round(w_rec[:, i] * scaling)
                threshold_quan[i] = np.round(threshold[i] * scaling)
                # if the threshold exceed boundary
                if np.abs(threshold_quan[i]) > max_th_quan:
                    limited_scaling = max_th_quan / threshold[i]
                    threshold_quan[i] = np.round(threshold[i] * limited_scaling)
                    w_in_quan[:, i] = np.round(w_in[:, i] * limited_scaling)
                    w_rec_quan[:, i] = np.round(w_rec[:, i] * limited_scaling)
            else:
                threshold_quan[i] = np.round(threshold[i])

    # make sure matrix type is int
    w_in_quan = w_in_quan.astype(int)
    w_rec_quan = w_rec_quan.astype(int)
    threshold_quan = threshold_quan.astype(int)

    # quantize output weight, threshold_out
    w_out_quan = np.zeros(shape=w_out.shape)
    threshold_out_quan = np.zeros(shape=threshold_out.shape)

    for i in range(w_out.shape[1]):
        max_w = 0
        max_w = np.max([max_w, np.max(np.abs(w_out[:, i]))])
        if max_w != 0:
            scaling = max_w_quan / max_w
            w_out_quan[:, i] = np.round(w_out[:, i] * scaling)
            threshold_out_quan[i] = np.round(threshold_out[i] * scaling)
            # if the threshold exceed boundary
            if np.abs(threshold_out_quan[i]) > max_th_quan:
                limited_scaling = max_th_quan / threshold_out[i]
                threshold_out_quan[i] = np.round(threshold_out[i] * limited_scaling)
                w_out_quan[:, i] = np.round(w_out[:, i] * limited_scaling)
        else:
            threshold_out_quan[i] = np.round(threshold_out[i])

    # make sure matrix type is int
    w_out_quan = w_out_quan.astype(int)
    threshold_out_quan = threshold_out_quan.astype(int)

    # round and cast all dashes to integer
    dash_mem = np.round(dash_mem).astype(int)
    dash_mem_out = np.round(dash_mem_out).astype(int)
    dash_syn = np.round(dash_syn).astype(int)
    dash_syn_2 = np.round(dash_syn_2).astype(int)
    dash_syn_out = np.round(dash_syn_out).astype(int)

    model_quan = {
        "weights_in": weights_in,
        "weights_rec": weights_rec,
        "weights_out": weights_out,
        "threshold": threshold,
        "threshold_out": threshold_out,
        "dash_mem": dash_mem,
        "dash_mem_out": dash_mem_out,
        "dash_syn": dash_syn,
        "dash_syn_2": dash_syn_2,
        "dash_syn_out": dash_syn_out,
    }

    return model_quan
