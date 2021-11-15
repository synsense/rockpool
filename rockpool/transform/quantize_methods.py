import numpy as np
import copy


def global_quantize(model, fuzzy_scaling: bool = False):
    """
    scale the weight matrix globally

    Args:
        model: mapper model dict
        fuzzy_scaling: leave outliers

    Returns:
        quantized model_dict to be wrapped as XyloConfiguration()

                 target
           -------------------
    s   -------------------  -
    o   -**- -**- -**- -**-  -
    u   -**- -**- -**- -**-  -
    r   -**- -**- -**- -**-  -
    c   -**- -**- -**- -**-  -
    e   -**- -**- -**- -**-  -
        -------------------
    """

    global_model = copy.copy(model)
    w_in = global_model['weights_in']
    w_rec = global_model['weights_rec']
    w_out = global_model['weights_out']
    threshold = global_model['threshold']
    threshold_out = global_model['threshold_out']
    w_max_bit = 8
    max_w_quan = 2 ** (w_max_bit - 1) - 1

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
        scaling = 0

    if max_w_out != 0:
        scaling_out = max_w_quan / max_w_out
    else:
        scaling_out = 0

    # scale weights
    weights_in = np.round(w_in * scaling).astype(int)
    weights_in = weights_in.astype(int)
    weights_rec = np.round(w_rec * scaling).astype(int)
    weights_out = np.round(w_out * scaling_out).astype(int)

    # scale thresholds
    threshold = np.round(threshold * scaling).astype(int)
    threshold_out = np.round(threshold_out * scaling).astype(int)

    model_quan = {'weights_in': weights_in,
                  'weights_rec': weights_rec,
                  'weights_out': weights_out,
                  'threshold': threshold,
                  'threshold_out': threshold_out}

    del global_model['dt']
    del global_model['mapped_graph']
    global_model.update(model_quan)

    return global_model


def channel_quantize(model):
    """
    scale the weight matrix per target neuron (per weight matrix column in weight matrix),
    in Xylo each target neuron may receive events from up to 16*2 pre-synaptic neurons

    Args:
        model: mapper model dict

    Returns:
        quantized model_dict to be wrapped as XyloConfiguration()

                 target
           -------------------
    s   -------------------  -
    o   -++- -**- -##- -oo-  -
    u   -++- -**- -##- -oo-  -
    r   -++- -**- -##- -oo-  -
    c   -++- -**- -##- -oo-  -
    e   -++- -**- -##- -oo-  -
        -------------------
    """

    channel_model = copy.copy(model)
    w_in = channel_model['weights_in']
    w_rec = channel_model['weights_rec']
    w_out = channel_model['weights_out']
    threshold = channel_model['threshold']
    threshold_out = channel_model['threshold_out']
    w_max_bit = 8
    max_w_quan = 2 ** (w_max_bit - 1) - 1

    # quantize input weight, recurrent weight, threshold
    # two weight matrix need to be stack together to consider per-channel quantization
    w_in_quan = np.zeros(shape=w_in.shape)
    w_rec_quan = np.zeros(shape=w_rec.shape)
    threshold_quan = np.zeros(shape=threshold.shape)

    for i in range(w_in.shape[1]):
        max_w = 0
        max_w = np.max([max_w, np.max(np.abs(w_in[:, i, :]))])
        max_w = np.max([max_w, np.max(np.abs(w_rec[:, i, :]))])
        if max_w != 0:
            scaling = max_w_quan / max_w
            w_in_quan[:, i, :] = np.round(w_in[:, i, :] * scaling)
            w_rec_quan[:, i, :] = np.round(w_rec[:, i, :] * scaling)
            threshold_quan[i] = np.round(threshold * scaling)

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
            threshold_out_quan[i] = np.round(threshold_out * scaling)

    # make sure matrix type is int
    w_out_quan = w_out_quan.astype(int)
    threshold_out_quan = threshold_out_quan.astype(int)

    model_quan = {'weights_in': w_in_quan,
                  'weights_rec': w_rec_quan,
                  'weights_out': w_out_quan,
                  'threshold': threshold_quan,
                  'threshold_out': threshold_out_quan}

    del channel_model['dt']
    del channel_model['mapped_graph']
    channel_model.update(model_quan)

    return channel_model
