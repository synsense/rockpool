"""
Quantisation methods for Xylo

Defines the post-training quasntization methods :py:func:`.global_quantize` and :py:func:`.channel_quantize`.

"""

import numpy as np
import copy

__all__ = ["global_quantize", "channel_quantize"]


def validate_weights_to_2d(data):
    assert len(data.shape) == 2, "Output weights must be 2D."
    return data


def validate_weights_to_3d(data):
    assert (
        len(data.shape) >= 2 and len(data.shape) <= 3
    ), "Weight arrays must be 2D or 3D."
    return np.expand_dims(data, -1) if len(data.shape) < 3 else data


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
    bias: np.ndarray = None,
    bias_out: np.ndarray = None,
    fuzzy_scaling: bool = False,
    bits_per_weight: int = 8,
    bits_per_threshold: int = 16,
    *_,
    **__,
):
    """
    Quantize a Xylo model for deployment, using global parameter scaling

    The figure below illustrates the groups of weights which are considered for quantization. Under this global method, all weights in the network are considered together when scaling and quantizing weights and thresholds. Input and recurrent weights are considered together as a group; output weights are considered separately when quantizing. Dashes are rounded and cast to integer.

    ::

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
        >>> specs = xylo.devices.mapper(net.as_graph(), weight_dtype="float", threshold_dtype="float")

        >>> specs.update(global_quantize(**specs, fuzzy_scaling = True))

        >>> xylo.devices.XyloSim.from_specifications(specs)

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
        bias (np.ndarray): Bias for hidden neurons
        bias_out (np.ndarray): Bias for output neurons
        fuzzy_scaling (bool): If ``True``, scale and clip weights to 2*std dev. If ``False`` (default), scale and clip to maximum absolute weight.
        bits_per_weight (int): Number of bits per integer signed weight. Default: ``8``
        bits_per_threshold (int): Number of bits per integer signed threshold. Default: ``16``

    Returns:
        dict: `model_quan` which can be used to update a Xylo specification dictionary
    """

    w_in = validate_weights_to_3d(copy.copy(weights_in))
    w_rec = validate_weights_to_3d(copy.copy(weights_rec))
    w_out = validate_weights_to_2d(copy.copy(weights_out))
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
        if not bias is None:
            max_w = np.max([max_w, np.max(np.abs(bias))])
        max_w_out = np.max([0, np.max(np.abs(w_out))])
        if not bias_out is None:
            max_w_out = np.max([max_w_out, np.max(np.abs(bias_out))])

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

    # bias
    if not bias is None:
        bias = np.round(bias * scaling).astype(int)

    # bias out
    if not bias_out is None:
        bias_out = np.round(bias_out * scaling).astype(int)

    # if the threshold exceed boundary
    if np.abs(np.max(threshold)) > max_th_quan:
        limited_scaling = max_th_quan / np.max(threshold)
        threshold = np.round(threshold * limited_scaling).astype(int)
        weights_in = np.round(weights_in * limited_scaling).astype(int)
        weights_rec = np.round(weights_rec * limited_scaling).astype(int)

    if np.abs(np.max(threshold_out)) > max_th_quan:
        limited_scaling = max_th_quan / np.max(threshold_out)
        threshold_out = np.round(threshold_out * limited_scaling).astype(int)
        weights_out = np.round(weights_out * limited_scaling).astype(int)
        weights_rec = np.round(weights_rec * limited_scaling).astype(int)

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

    if not bias is None:
        model_quan["bias"] = bias
    if not bias_out is None:
        model_quan["bias_out"] = bias_out

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
    bias: np.ndarray = None,
    bias_out: np.ndarray = None,
    bits_per_weight: int = 8,
    bits_per_threshold: int = 16,
    *_,
    **__,
):
    """
    Quantize a Xylo model for deployment, using per-channel parameter scaling

    The figure below illustrates the groups of weights which are considered for quantization. Under this per-channel method, all input weights to a single target neuron are considered together when scaling and quantizing weights and thresholds. Input and recurrent weights are considered together as a group; output weights are considered separately when quantizing. Dashes are rounded and cast to integer.

    ::

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
        >>> specs = xylo.devices.mapper(net.as_graph(), weight_dtype="float", threshold_dtype="float")

        >>> specs.update(channel_quantize(**specs, bits_per_weight = 12))

        >>> xylo.devices.XyloSim.from_specifications(specs)

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
        bias (np.ndarray): Bias for hidden neurons
        bias_out (np.ndarray): Bias for output neurons
        bits_per_weight (int): Number of bits per integer signed weight. Default: ``8``
        bits_per_threshold (int): Number of bits per integer signed threshold. Default: ``16``

    Returns:
        dict: `model_quan` which can be used to update a Xylo specification dictionary
    """

    w_in = validate_weights_to_3d(copy.copy(weights_in))
    w_rec = validate_weights_to_3d(copy.copy(weights_rec))
    w_out = validate_weights_to_2d(copy.copy(weights_out))
    threshold = copy.copy(threshold)
    threshold_out = copy.copy(threshold_out)
    max_w_quan = 2 ** (bits_per_weight - 1) - 1
    max_th_quan = 2 ** (bits_per_threshold - 1) - 1

    Nin, Nien, Nsyn = w_in.shape
    Nhid, _, _ = w_rec.shape
    Noen, Nout = w_out.shape

    # quantize input weight, recurrent weight, threshold
    # two weight matrix need to stack together to consider per-channel quantization
    w_in_quan = np.zeros(shape=w_in.shape)
    w_rec_quan = np.zeros(shape=w_rec.shape)
    threshold_quan = np.zeros(shape=threshold.shape)

    for hidden_id in range(Nhid):
        # if two synaptic connection is used
        if Nsyn > 1:
            # - Find the maximum output weight for this neuron
            max_w = 0
            max_w = (
                np.max([max_w, np.max(np.abs(w_in[:, hidden_id, :]))])
                if hidden_id < Nien
                else max_w
            )
            max_w = np.max([max_w, np.max(np.abs(w_rec[:, hidden_id, :]))])

            # - Scale and quantise weights, thresholds and biases
            if max_w != 0:
                scaling = max_w_quan / max_w
                if hidden_id < Nien:
                    w_in_quan[:, hidden_id, :] = np.round(
                        w_in[:, hidden_id, :] * scaling
                    )
                w_rec_quan[:, hidden_id, :] = np.round(w_rec[:, hidden_id, :] * scaling)
                threshold_quan[hidden_id] = np.round(threshold[hidden_id] * scaling)
                if not bias is None:
                    bias[hidden_id] = np.round(bias[hidden_id] * scaling)

                # if the threshold exceeds boundary
                if np.abs(threshold_quan[hidden_id]) > max_th_quan:
                    limited_scaling = max_th_quan / threshold[hidden_id]
                    threshold_quan[hidden_id] = np.round(
                        threshold[hidden_id] * limited_scaling
                    )
                    if hidden_id <= Nien:
                        w_in_quan[:, hidden_id, :] = np.round(
                            w_in[:, hidden_id, :] * limited_scaling
                        )
                    w_rec_quan[:, hidden_id, :] = np.round(
                        w_rec[:, hidden_id, :] * limited_scaling
                    )
            else:
                threshold_quan[hidden_id] = np.round(threshold[hidden_id])

        # if only one synaptic connection is used
        elif Nsyn == 1:
            # - Find the maximum output weight for this neuron
            max_w = 0
            max_w = (
                np.max([max_w, np.max(np.abs(w_in[:, hidden_id]))])
                if hidden_id < Nien
                else max_w
            )
            max_w = np.max([max_w, np.max(np.abs(w_rec[:, hidden_id]))])

            # - Scale and quantise weights, thresholds and biases
            if max_w != 0:
                scaling = max_w_quan / max_w
                if hidden_id < Nien:
                    w_in_quan[:, hidden_id] = np.round(w_in[:, hidden_id] * scaling)

                w_rec_quan[:, hidden_id] = np.round(w_rec[:, hidden_id] * scaling)
                threshold_quan[hidden_id] = np.round(threshold[hidden_id] * scaling)

                # if the threshold exceeds boundary
                if np.abs(threshold_quan[hidden_id]) > max_th_quan:
                    limited_scaling = max_th_quan / threshold[hidden_id]
                    threshold_quan[hidden_id] = np.round(
                        threshold[hidden_id] * limited_scaling
                    )
                    if hidden_id <= Nien:
                        w_in_quan[:, hidden_id] = np.round(
                            w_in[:, hidden_id] * limited_scaling
                        )
                    w_rec_quan[:, hidden_id] = np.round(
                        w_rec[:, hidden_id] * limited_scaling
                    )
            else:
                threshold_quan[hidden_id] = np.round(threshold[hidden_id])

    # quantize output weight, threshold_out
    w_out_quan = np.zeros(shape=w_out.shape)
    threshold_out_quan = np.zeros(shape=threshold_out.shape)

    for output_id in range(Nout):
        max_w = 0
        max_w = np.max([max_w, np.max(np.abs(w_out[:, output_id]))])
        if max_w != 0:
            scaling = max_w_quan / max_w
            w_out_quan[:, output_id] = np.round(w_out[:, output_id] * scaling)
            threshold_out_quan[output_id] = np.round(threshold_out[output_id] * scaling)
            if not bias_out is None:
                bias_out[output_id] = np.round(bias_out[output_id] * scaling)
            # if the threshold exceed boundary
            if np.abs(threshold_out_quan[output_id]) > max_th_quan:
                limited_scaling = max_th_quan / threshold_out[output_id]
                threshold_out_quan[output_id] = np.round(
                    threshold_out[output_id] * limited_scaling
                )
                w_out_quan[:, output_id] = np.round(
                    w_out[:, output_id] * limited_scaling
                )
        else:
            threshold_out_quan[output_id] = np.round(threshold_out[output_id])

    # make sure all types are int
    weights_in = w_in_quan.astype(int)
    weights_rec = w_rec_quan.astype(int)
    weights_out = w_out_quan.astype(int)
    dash_mem = np.round(dash_mem).astype(int)
    dash_mem_out = np.round(dash_mem_out).astype(int)
    dash_syn = np.round(dash_syn).astype(int)
    dash_syn_2 = np.round(dash_syn_2).astype(int)
    dash_syn_out = np.round(dash_syn_out).astype(int)
    threshold = threshold_quan.astype(int)
    threshold_out = threshold_out_quan.astype(int)

    if not bias is None:
        bias = np.round(bias).astype(int)
    if not bias_out is None:
        bias_out = np.round(bias_out).astype(int)

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

    if not bias is None:
        model_quan["bias"] = bias
    if not bias_out is None:
        model_quan["bias_out"] = bias_out

    return model_quan
