import numpy as np
from matplotlib import pyplot as plt

plt.ion()

from NetworksPython import TSEvent
from NetworksPython.layers import RecIAFSpkInBrian
from brian2 import volt

# from NetworksPython.layers import RecIAFSpkInTorch
from NetworksPython.layers import RecIAFSpkInRefrTorch
from NetworksPython.layers import RecIAFSpkInNest

# - Negative weights, so that layer doesn't spike and gets reset
np.random.seed(1)
weights_in = np.random.rand(2, 3) - 1
weights_rec = 2 * np.random.rand(3, 3) - 1
bias = 2 * np.random.rand(3) - 1
tau_mem, tau_syn = np.clip(np.random.rand(2, 3), 0.01, None)
dt = 0.001
refractory = 0.01
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
    tau_syn=tau_syn,
    dt=dt,
    refractory=refractory,
    v_thresh=v_thresh,
    v_rest=v_rest,
    v_reset=v_reset,
    record=True,
)

# - Input signal
tsInEvt = TSEvent(times=[0.02, 0.04, 0.04, 0.06, 0.12], channels=[1, 0, 1, 1, 0])

tsB = rlB.evolve(tsInEvt, duration=0.1)
# tsT = rlT.evolve(tsInEvt, duration=0.1)
tsTR = rlTR.evolve(tsInEvt, duration=0.1)
tsN = rlN.evolve(tsInEvt, duration=0.1)

# - Plot spike patterns
for ts in (tsB, tsTR, tsN):
    ts.plot()

# - Plot states
plt.figure()
plt.plot(rlB._v_monitor.t, rlB._v_monitor.v.T, color="blue")
# rlT.tscRecStates.plot(color="orange")
rlTR.ts_rec_states.plot(color="green")
plt.plot(np.arange(rlN.record_states.shape[1]) * dt, rlN.record_states.T, color="red")
