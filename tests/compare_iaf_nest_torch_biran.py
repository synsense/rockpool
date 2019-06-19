import numpy as np
from matplotlib import pyplot as plt
from brian2 import volt

plt.ion()

from NetworksPython import TSEvent
from NetworksPython.layers import RecIAFSpkInBrian

# from NetworksPython.layers import RecIAFSpkInTorch
from NetworksPython.layers import RecIAFSpkInRefrTorch
from NetworksPython.layers import RecIAFSpkInNest

# - Negative weights, so that layer doesn't spike and gets reset

np.random.seed(1)
weights_in = (2 * np.random.rand(2, 3) - 0.7) * 0.1
weights_rec = (2 * np.random.rand(3, 3) - 0.7) * 0.1
bias = 0.01 * np.random.rand(3)
tau_mem, tau_syn = np.clip(0.1 * np.random.rand(2, 3), 0.01, None)

dt = 0.001
refractory = 0.0
v_thresh = -0.055
v_rest = -0.065
v_reset = -0.065

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

# - Input signal
tsInEvt = None
# tsInEvt = TSEvent(times=[0.02, 0.04, 0.04, 0.06, 0.12], channels=[1, 0, 1, 1, 0])

tsB = rlB.evolve(tsInEvt, duration=0.1)
# tsT = rlT.evolve(tsInEvt, duration=0.1)
tsTR = rlTR.evolve(tsInEvt, duration=0.1)
tsN = rlN.evolve(tsInEvt, duration=0.1)

# - Plot spike patterns
plt.figure()
for ts, col in zip((tsB, tsTR, tsN), ("blue", "green", "red")):
    ts.plot(color=col)

# - Plot states
plt.figure()
plt.plot(rlB._v_monitor.t, rlB._v_monitor.v.T, color="blue")
rlTR.tscRecStates.plot(color="orange")
#rlTR.ts_rec_states.plot(color="green")
plt.plot(np.arange(rlN.record_states.shape[1]) * dt, rlN.record_states.T, color="red")
