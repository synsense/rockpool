import numpy as np
from matplotlib import pyplot as plt
from brian2 import volt

plt.ion()

from rockpool import TSEvent
from nn.layers import RecIAFSpkInBrian

# from rockpool.layers import RecIAFSpkInTorch
from nn.layers import RecIAFSpkInRefrTorch
from nn.layers import RecIAFSpkInNest
from nn.layers import RecAEIFSpkInNest
from nn.layers import VirtualDynapse

# - Negative weights, so that layer doesn't spike and gets reset

np.random.seed(1)
# - Quantized weights for virtual dynapse
weights_in_quant = np.random.randint(10, size=(2, 3)) - 7
weights_rec_quant = np.random.randint(10, size=(3, 3)) - 7
baseweight = 0.1
weights_in = weights_in_quant * baseweight
weights_rec = weights_rec_quant * baseweight
bias = 0.01 * np.random.rand(3)
tau_mem, tau_syn = np.clip(0.1 * np.random.rand(2, 3), 0.01, None)

dt = 0.001
refractory = 0.002
v_thresh = -0.061
v_rest = v_reset = -0.065

# - Layer generation
rlB = RecIAFSpkInBrian(
    weights_in=weights_in,
    weights_rec=weights_rec,
    bias=bias,
    tau_mem=tau_mem,
    tau_syn_inp=tau_syn,
    tau_syn_rec=tau_syn,
    dt=dt,
    noise_std=0,
    refractory=refractory,
    v_thresh=v_thresh,
    v_rest=v_rest * volt,
    v_reset=v_reset,
    record=True,
)
# rlT = RecIAFSpkInTorch(
#     weights_in=weights_in,
#     weights_rec=weights_rec,
#     bias=bias,
#     tau_mem=tau_mem,
#     tau_syn_inp=tau_syn,
#     tau_syn_rec=tau_syn,
#     noise_std=0,
#     v_thresh=v_thresh,
#     v_rest=v_rest,
#     v_reset=v_reset,
#     dt=dt,
#     record=True,
# )
rlTR = RecIAFSpkInRefrTorch(
    weights_in=weights_in,
    weights_rec=weights_rec,
    bias=bias,
    tau_mem=tau_mem,
    tau_syn_inp=tau_syn,
    tau_syn_rec=tau_syn,
    noise_std=0,
    dt=dt,
    refractory=refractory,
    v_thresh=v_thresh,
    v_rest=v_rest,
    v_reset=v_reset,
    record=True,
)
rlN = RecIAFSpkInNest(
    weights_in=weights_in,
    weights_rec=weights_rec,
    bias=bias,
    tau_mem=tau_mem,
    tau_syn_exc=tau_syn,
    tau_syn_inh=tau_syn,
    dt=dt,
    refractory=refractory,
    v_thresh=v_thresh,
    v_rest=v_rest,
    v_reset=v_reset,
    record=True,
)

rlAEN = RecAEIFSpkInNest(
    weights_in=weights_in,
    weights_rec=weights_rec,
    dt=dt,
    bias=bias,
    tau_mem=tau_mem,
    tau_syn_exc=tau_syn,
    tau_syn_inh=tau_syn,
    v_thresh=v_thresh,
    v_reset=v_reset,
    v_rest=v_rest,
    subthresh_adapt=0.0,
    spike_adapt=0.0,
    delta_t=0.0,
    refractory=refractory,
    record=True,
)


neurons_vd = [3, 300, 513]
inputs_vd = [4, 5]
vd = VirtualDynapse(
    has_tau_mem_2=False,
    dt=dt,
    v_thresh=v_thresh - v_rest,
    tau_mem_1=list(tau_mem) + 13 * [0.02],
    tau_syn_exc=list(tau_syn) + 13 * [0.05],
    tau_syn_inh=list(tau_syn) + 13 * [0.05],
    bias=list(bias) + 13 * [0],
    baseweight_e=baseweight,
    baseweight_i=baseweight,
    refractory=refractory,
    delta_t=0,
    spike_adapt=0,
    mismatch=False,
    record=True,
)
vd.set_connections(
    connections=weights_in_quant,
    ids_pre=inputs_vd,
    ids_post=neurons_vd,
    external=True,
    add=False,
)
vd.set_connections(
    connections=weights_rec_quant,
    ids_pre=neurons_vd,
    ids_post=neurons_vd,
    external=False,
    add=False,
)
# - Another virtual dyanpse, where parameters are set after instantiation
vd0 = VirtualDynapse(
    has_tau_mem_2=False,
    dt=dt,
    v_thresh=v_thresh - v_rest,
    refractory=refractory,
    delta_t=0,
    spike_adapt=0,
    mismatch=False,
    record=True,
)
vd0.set_connections(
    connections=weights_in_quant,
    ids_pre=inputs_vd,
    ids_post=neurons_vd,
    external=True,
    add=False,
)
vd0.set_connections(
    connections=weights_rec_quant,
    ids_pre=neurons_vd,
    ids_post=neurons_vd,
    external=False,
    add=False,
)
# Set parameters for vd and vd0
vd0.baseweight_e[:3] = vd0.baseweight_i[:3] = baseweight
vd0.bias[:3] = bias
vd0.tau_mem_1[:3] = tau_mem
vd0.tau_syn_exc[:3] = tau_syn
vd0.tau_syn_inh[:3] = tau_syn


# - Input signal
# tsInEvt = None
tsInEvt = TSEvent(times=[0.02, 0.04, 0.04, 0.06, 0.62], channels=[1, 0, 1, 1, 0])

tsB = rlB.evolve(tsInEvt, duration=0.5)
# tsT = rlT.evolve(tsInEvt, duration=0.5)
tsTR = rlTR.evolve(tsInEvt, duration=0.5)
tsN = rlN.evolve(tsInEvt, duration=0.5)
tsAEN = rlAEN.evolve(tsInEvt, duration=0.5)
tsVD = vd.evolve(
    tsInEvt, duration=0.5, ids_in=inputs_vd, ids_out=neurons_vd, remap_out_channels=True
)
tsVD0 = vd0.evolve(
    tsInEvt, duration=0.5, ids_in=inputs_vd, ids_out=neurons_vd, remap_out_channels=True
)

# - Plot spike patterns
plt.figure()
for ts, col in zip(
    (tsB, tsTR, tsN, tsAEN, tsVD, tsVD0),
    ("blue", "green", "red", "purple", "orange", "pink"),
):
    ts.plot(color=col, marker="x")

# - Plot states
plt.figure()
plt.plot(rlB._v_monitor.t, rlB._v_monitor.v.T, color="blue")
rlTR.ts_rec_states.plot(color="green")
# rlTR.ts_rec_states.plot(color="green")
rlN.recorded_states.plot(color="red")
rlAEN.recorded_states.plot(color="purple")
(vd.recorded_states + v_rest).clip(channels=neurons_vd).plot(color="orange")
(vd0.recorded_states + v_rest).clip(channels=neurons_vd).plot(color="pink")
