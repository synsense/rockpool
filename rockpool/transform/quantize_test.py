import numpy as np
import sys
import torch
# np.set_printoptions(threshold=sys.maxsize)

model = SomeNetworkModel(**model_params)
model.cuda()

class Quantization(JaxParameterTransformerMixin, JaxModule):

def calc_bitshift_decay(tau, dt):
    bitsh = np.log2(np.array(tau) / dt)
    bitsh[bitsh < 0] = 0
    return bitsh

def global_quantize(model,
                    threshold_out: list = [20000],
                    weight_shift: int = 2,
                    weight_shift_out: int = 3,
                    fuzzy_scaling: bool = False,
                    weight_max_bit: int = 8,
                    ):
    print('global quantize')

    w_in, w_rec, w_out, aliases = model.to_weight_matrix()
    tau_mem, tau_mem_out = model.to_vmem_taus()
    tau_syns, tau_syns_out = model.to_syn_taus()

    dt = 0.01

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

    print(f"Quantize: max recurrent weight: {max_val},  scaling {scaling}")
    print(f"Quantize: max readout weight: {max_val_out},  scaling {scaling_out}")

    # scale weights
    weights_in = np.round(w_in * scaling).astype(int)
    weights_rec = np.round(w_rec * scaling).astype(int)
    weights_out = np.round(w_out * scaling_out).astype(int)

    # membrane decay
    dash_mem = np.round(calc_bitshift_decay(tau_mem * dt, dt)).astype(int)
    dash_syn = np.round(calc_bitshift_decay(tau_syns * dt, dt)).astype(int)

    dash_mem_out = np.round(calc_bitshift_decay(tau_mem_out * dt, dt)).astype(int)
    dash_syn_out = np.round(calc_bitshift_decay(tau_syns_out * dt, dt)).astype(int)

    # thresholds
    threshold = np.repeat(np.round(v_th_rel * scaling).astype(int) << weight_shift, len(w_rec))
    if len(threshold_out) == 1:
        threshold_out = np.array(threshold_out * w_out.shape[1]).astype(int)
    else:
        threshold_out = np.array(threshold_out).astype(int)

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
                     fuzzy_scaling: bool = False,
                     res_num: int = 16,
                     weight_max_bit: int = 8,
                     ):

    print('per channel quantize')

    w_in, w_rec, w_out, aliases = model.to_weight_matrix()

    # print('w_in.shape', w_in.shape)
    # print('w_rec.shape', w_rec.shape)
    # print('w_out.shape', w_out.shape)
    # print('len(aliases)', len(aliases))

    tau_mem, tau_mem_out = model.to_vmem_taus()
    tau_syns, tau_syns_out = model.to_syn_taus()

    threshold = model.spk1.threshold  # ???

    weight_in_quan = np.zeros(shape=w_in.shape)
    threshold_in_quan = np.zeros(w_in.shape[1])
    for i in range(w_in.shape[1]):
        max_weight = 0
        max_weight = np.max([max_weight, np.max(np.abs(w_in[:, i, :]))])
        max_weight_quan = 2 ** (weight_max_bit - 1) - 1
        if max_weight != 0:
            scaling = max_weight_quan / max_weight
            weight_in_quan[:, i, :] = np.round(w_in[:, i, :] * scaling).astype(int)
            threshold_in_quan[i] = np.round(threshold * scaling)

    weight_in_quan = weight_in_quan.astype(int)

    weight_rec_quan = np.zeros(shape=w_rec.shape)
    threshold_rec_quan = np.zeros(w_rec.shape[1])
    for i in range(w_rec.shape[1]):
        max_weight = 0
        max_weight = np.max([max_weight, np.max(np.abs(w_rec[:, i, :]))])
        max_weight_quan = 2 ** (weight_max_bit - 1) - 1
        if max_weight != 0:
            scaling = max_weight_quan / max_weight
            weight_rec_quan[:, i, :] = np.round(w_rec[:, i, :] * scaling).astype(int)
            threshold_rec_quan[i] = np.round(threshold * scaling)

    weight_rec_quan = weight_rec_quan.astype(int)

    threshold_rec_quan[0: res] = threshold_in_quan[0: res]
    threshold_rec_quan = threshold_rec_quan.astype(int)

    weight_out_quan = np.zeros(shape=w_out.shape)

    # - first condition is for the first Vmem collection
    if len(threshold_out) == 1:
        threshold_out_dummy = np.array(threshold_out * w_out.shape[1]).astype(int)
        flag = 0
    # - second condition is for optimized threshold scale
    else:
        threshold_out_quan = np.array(threshold_out).astype(int)
        flag = 1

    max_weight_quan = 2 ** (weight_max_bit - 1) - 1
    for i in range(w_out.shape[1]):
        max_weight = 0
        max_weight = np.max([max_weight, np.max(np.abs(w_out[:, i, :]))])
        if max_weight != 0:
            scaling = max_weight_quan / max_weight
            weight_out_quan[:, i, :] = np.round(w_out[:, i, :] * scaling).astype(int)

    weight_out_quan = weight_out_quan.astype(int)

    dt = 0.01  # 1e-3???

    dash_mem = np.round(calc_bitshift_decay(tau_mem * dt, dt)).astype(int)
    dash_syn = np.round(calc_bitshift_decay(tau_syns * dt, dt)).astype(int)

    dash_mem_out = np.round(calc_bitshift_decay(tau_mem_out * dt, dt)).astype(int)
    dash_syn_out = np.round(calc_bitshift_decay(tau_syns_out * dt, dt)).astype(int)

    return {"weights_in": weight_in_quan,
            "weights_rec": weight_rec_quan,
            "weights_out": weight_out_quan,
            "aliases": aliases,
            "threshold": threshold_rec_quan,
            "threshold_out": threshold_out_dummy if flag == 0 else threshold_out_quan,
            "weight_shift": weight_shift,
            "weight_shift_out": weight_shift_out,
            "dash_mem": dash_mem,
            "dash_mem_out": dash_mem_out,
            "dash_syn": dash_syn,
            "dash_syn_out": dash_syn_out, }
