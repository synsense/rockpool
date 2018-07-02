import numpy as np
import brian2 as b2
from brian2 import second
from matplotlib import pyplot as plt
plt.ion()

import TimeSeries as ts

from layers.recurrent.iaf_brian import RecIAFBrian as Rec
from layers.recurrent.weights import IAFSparseNet


### --- Set parameters

tDt = 0.005  # Length of time step in seconds
tDur = 50  # Duration of simulation
tPulse = 0.1 # Duration of input pulse

nDimIn = 1

nResSize = 512  # Reservoir size
tTauN = 250 * tDt  # Reservoir neuron time constant
tTauS = 250 * tDt # Reservoir synapse time constant

# Parameters concerning reservoir weights
kwResWeights = {
    "nResSize": nResSize,
    "fDensity" : 0.4,
    # "fConnectivity": 0.4,  # Connectivity
    # "bPartitioned": False,  # Partition reservoir into excitatory/inhibitory
    # "fRatioExc": 0.5,  # Ratio of excitatory neurons
    # "fScaleInh": 1,  # Scale of inhibitory vs excitatory weights
    # "fNormalize": 0.5,
}  # Normalize matrix spectral radius

# - Recurrent weights
mfW_res = IAFSparseNet(**kwResWeights)
# - Input weights
mfW_in = 2 * (np.random.rand(nDimIn, nResSize) - 0.5)

# - Reservoir
def gen_res():
    res=Rec(mfW=mfW_res, vtTauN=1*second, vtTauSynR=1*second,
            tDt=tDt * second, strName="reservoir")
    res.monI = b2.StateMonitor(res._ngLayer, ['I_syn', 'I_total'], record=True)
    res.monV = b2.StateMonitor(res._ngLayer, 'v', record=True)
    res._net.add(res.monI, res.monV)
    return res

res = gen_res()

# - Pulse input
vtTimePulse = np.arange(0, tDur, tDt)
nPulse = int(tPulse/tDt)
vfPulse = np.zeros_like(vtTimePulse)
vfPulse[:nPulse] = 1
# - Incorporate input weights in input
mfPulse = np.atleast_2d(vfPulse).T@mfW_in
tsPulse = ts.TimeSeries(vtTimePulse, mfPulse)

# - Simulate
tsResSpikes = res.evolve(tsPulse)
vIsyn = res.monI.I_syn.T
vVpot = res.monV.v.T

# - Plot
fig, (ax0, ax1, ax2) = plt.subplots(3, sharex=True, figsize=(12,8))

# Input and spikes
ax0.plot(vtTimePulse, nResSize*vfPulse)
ax0.scatter(tsResSpikes.vtTimeTrace, tsResSpikes.vnChannels)
ax0.set_title('Input and spikes')

# Synaptic current
ax1.plot(vtTimePulse[:-1], vIsyn)
ax1.set_title('Synaptic currents')

# Membrane potential
ax2.plot(vtTimePulse[:-1], vVpot)
ax2.set_title('Membrane potentials')