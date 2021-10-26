import numpy as np

def calc_bitshift_decay(tau, dt):
    """
    Approximate the exponential decay by bit shift decay

    Args:
        tau: exponential decay tau
        dt: unit time step

    Returns:
        the bit shift value in digital circuit
    """

    bitsh = np.log2(np.array(tau) / dt)
    bitsh[bitsh < 0] = 0
    return bitsh


def global_quantize(model,
                    threshold_out: list = [20000],
                    weight_shift: int = 2,
                    weight_shift_out: int = 3,
                    fuzzy_scaling: bool = False,
                    weight_max_bit: int = 8,
                    dt: float = 0.01,
                    ):

    """
    scale the weight matrix globally

    Args:
        model: PyTorch model in CUDA
        threshold_out: output neuron threshold
        weight_shift: scale input and recurrent weight to have more spikes
        weight_shift_out: scale output weight to have more spikes
        fuzzy_scaling: leave outliers
        weight_max_bit: weight resolution, 8-bits in Xylo
        dt: unit time step

    Returns:
        model_dict to be wrapped as PollenConfiguration()

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

    # TODO 'to_weight_matrix()', 'to_vmem_taus()', 'to_syn_taus()' to be replaced by general function
    w_in, w_rec, w_out, aliases = model.to_weight_matrix()
    tau_mem, tau_mem_out = model.to_vmem_taus()
    tau_syns, tau_syns_out = model.to_syn_taus()

    # get threshold relative to reset potential
    v_th_rel = model.spk1.threshold

    if fuzzy_scaling:
        # detect outliers
        weights_rec_ = np.concatenate([np.ravel(w_in), np.ravel(w_rec)])
        weights_rec_ = weights_rec_[weights_rec_ != 0]
        weights_out_ = np.ravel(w_out)
        weights_out_ = weights_out_[weights_out_ != 0]
        max_val = np.abs(weights_rec_.mean()) + 2 * weights_rec_.std()
        max_val_out = np.abs(weights_out_.mean()) + 2 * weights_out_.std()
    else:
        max_val = 0
        max_val = np.max([max_val, np.max(np.abs(w_in))])
        max_val = np.max([max_val, np.max(np.abs(w_rec))])
        # max_val = np.max([max_val, v_th_rel])

        max_val_out = np.max([0, np.max(np.abs(w_out))])
        # max_val_out = np.max([max_val_out, v_th_rel])

    max_quant_val = 2 ** (weight_max_bit - 1) - 1

    # determine scaling value
    scaling = max_quant_val / max_val
    scaling_out = max_quant_val / max_val_out

    # scale weights
    weights_in = np.round(w_in * scaling).astype(int)
    weights_rec = np.round(w_rec * scaling).astype(int)
    weights_out = np.round(w_out * scaling_out).astype(int)

    # thresholds
    threshold = np.repeat(np.round(v_th_rel * scaling).astype(int) << weight_shift, len(w_rec))
    # dummy threshold for Vmem collection
    if len(threshold_out) == 1:
        threshold_out = np.array(threshold_out * w_out.shape[1]).astype(int)
    # with optimized true threshold
    else:
        threshold_out = np.array(threshold_out).astype(int)

    # membrane and synaptic decay based on bit shift
    dash_mem = np.round(calc_bitshift_decay(tau_mem * dt, dt)).astype(int)
    dash_syn = np.round(calc_bitshift_decay(tau_syns * dt, dt)).astype(int)
    dash_mem_out = np.round(calc_bitshift_decay(tau_mem_out * dt, dt)).astype(int)
    dash_syn_out = np.round(calc_bitshift_decay(tau_syns_out * dt, dt)).astype(int)

    return {"weights_in": weights_in,
            "weights_rec": weights_rec,
            "weights_out": weights_out,
            "aliases": aliases,
            "threshold": threshold,
            "threshold_out": threshold_out,
            "weight_shift": weight_shift,
            "weight_shift_out": weight_shift_out,
            "dash_mem": dash_mem,
            "dash_mem_out": dash_mem_out,
            "dash_syn": dash_syn,
            "dash_syn_out": dash_syn_out, }


def channel_quantize(model,
                     threshold_out: list = [20000],
                     weight_shift: int = 2,
                     weight_shift_out: int = 3,
                     weight_max_bit: int = 8,
                     dt: float = 0.01,
                     res_num: int = 16,
                     ):

    """
    scale the weight matrix per target neuron (per weight matrix column in weight matrix), in Xylo each target neuron may receive events from up to 16*2 pre-synaptic neurons

    Args:
        model: PyTorch model in CUDA
        threshold_out: output neuron threshold
        weight_shift: scale input and recurrent weight to have more spikes
        weight_shift_out: scale output weight to have more spikes
        weight_max_bit: weight resolution, 8-bits in Xylo
        dt: unit time step
        res_num: number of the first layer of neurons receive spikes from input channels

    Returns:
        model_dict to be wrapped as PollenConfiguration()

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

    # TODO 'to_weight_matrix()', 'to_vmem_taus()', 'to_syn_taus()' to be replaced by general function
    w_in, w_rec, w_out, aliases = model.to_weight_matrix()
    tau_mem, tau_mem_out = model.to_vmem_taus()
    tau_syns, tau_syns_out = model.to_syn_taus()

    # get threshold relative to reset potential
    threshold = model.spk1.threshold

    weight_in_quan = np.zeros(shape=w_in.shape)
    threshold_in_quan = np.zeros(w_in.shape[1])

    # quantize input weight
    for i in range(w_in.shape[1]):
        max_weight = 0
        max_weight = np.max([max_weight, np.max(np.abs(w_in[:, i, :]))])
        max_weight_quan = 2 ** (weight_max_bit - 1) - 1
        if max_weight != 0:
            scaling = max_weight_quan / max_weight
            weight_in_quan[:, i, :] = np.round(w_in[:, i, :] * scaling).astype(int)
            threshold_in_quan[i] = np.round(threshold * scaling)

    # make sure matrix type is int
    weight_in_quan = weight_in_quan.astype(int)

    # quantize recurrent weight
    weight_rec_quan = np.zeros(shape=w_rec.shape)
    threshold_rec_quan = np.zeros(w_rec.shape[1])
    for i in range(w_rec.shape[1]):
        max_weight = 0
        max_weight = np.max([max_weight, np.max(np.abs(w_rec[:, i, :]))])
        max_weight_quan = 2 ** (weight_max_bit - 1) - 1
        if max_weight != 0:
            scaling = max_weight_quan / max_weight
            weight_rec_quan[:, i, :] = np.round(w_rec[:, i, :] * scaling).astype(int)
            threshold_rec_quan[i] = np.round(threshold * scaling) # 0 ~ res_num no weights, no threshold

    # make sure matrix type is int
    weight_rec_quan = weight_rec_quan.astype(int)

    # concatenate all reservior neuron threshold (including 0 ~ res_num)
    threshold_rec_quan[0: res_num] = threshold_in_quan[0: res_num]
    threshold_rec_quan = threshold_rec_quan.astype(int)

    # quantize output weight
    weight_out_quan = np.zeros(shape=w_out.shape)
    max_weight_quan = 2 ** (weight_max_bit - 1) - 1
    for i in range(w_out.shape[1]):
        max_weight = 0
        max_weight = np.max([max_weight, np.max(np.abs(w_out[:, i, :]))])
        if max_weight != 0:
            scaling = max_weight_quan / max_weight
            weight_out_quan[:, i, :] = np.round(w_out[:, i, :] * scaling).astype(int)

    # make sure matrix type is int
    weight_out_quan = weight_out_quan.astype(int)

    # dummy threshold for Vmem collection
    if len(threshold_out) == 1:
        threshold_out = np.array(threshold_out * w_out.shape[1]).astype(int)
    # with optimized true threshold
    else:
        threshold_out = np.array(threshold_out).astype(int)

    # membrane and synaptic decay based on bit shift
    dash_mem = np.round(calc_bitshift_decay(tau_mem * dt, dt)).astype(int)
    dash_syn = np.round(calc_bitshift_decay(tau_syns * dt, dt)).astype(int)
    dash_mem_out = np.round(calc_bitshift_decay(tau_mem_out * dt, dt)).astype(int)
    dash_syn_out = np.round(calc_bitshift_decay(tau_syns_out * dt, dt)).astype(int)

    return {"weights_in": weight_in_quan,
            "weights_rec": weight_rec_quan,
            "weights_out": weight_out_quan,
            "aliases": aliases,
            "threshold": threshold_rec_quan,
            "threshold_out": threshold_out,
            "weight_shift": weight_shift,
            "weight_shift_out": weight_shift_out,
            "dash_mem": dash_mem,
            "dash_mem_out": dash_mem_out,
            "dash_syn": dash_syn,
            "dash_syn_out": dash_syn_out, }
