import numpy as np
import brian2 as b2
from brian2 import second
from matplotlib import pyplot as plt

plt.ion()

import TimeSeries as ts
import analysis as an

from layers.recurrent.iaf_brian import RecIAFBrian as Rec
from layers.recurrent.weights import IAFSparseNet


### --- Set parameters

tDt = 0.005  # Length of time step in seconds
tDur = 20  # Duration of simulation
tPulse = 0.1  # Duration of input pulse
tStart = 5 # Start time of the pulse
fAmplitude = 0.1 # Amplitude of the pulse

nDimIn = 1

nResSize = 64  # Reservoir size
tTauN = 25 * tDt  # Reservoir neuron time constant
tTauS = 250 * tDt  # Reservoir synapse time constant

# Parameters concerning reservoir weights
kwResWeights = {
    "nResSize": nResSize,
    "fDensity": 0.4,
    # "fConnectivity": 0.4,  # Connectivity
    # "bPartitioned": False,  # Partition reservoir into excitatory/inhibitory
    # "fRatioExc": 0.5,  # Ratio of excitatory neurons
    # "fScaleInh": 1,  # Scale of inhibitory vs excitatory weights
    # "fNormalize": 0.5,
}  # Normalize matrix spectral radius

# - Recurrent weights
mfW_res = IAFSparseNet(**kwResWeights)/500
# - Input weights
mfW_in = 2 * (np.random.rand(nDimIn, nResSize) - 0.5)

# - Reservoir
def gen_res():
    res = Rec(
        mfW=mfW_res,
        vtTauN=tTauN,
        vtTauSynR=tTauS,
        tDt=tDt * second,
        strName="reservoir",
    )
    res.monI = b2.StateMonitor(res._ngLayer, ["I_syn", "I_total"], record=True)
    res.monV = b2.StateMonitor(res._ngLayer, "v", record=True)
    res._net.add(res.monI, res.monV)
    return res


res = gen_res()

# - Pulse input
vtTimePulse = np.arange(0, tDur, tDt)
nPulse = int(tPulse / tDt)
nStart = int(tStart / tDt)
vfPulse = np.zeros_like(vtTimePulse)
vfPulse[nStart:nStart+nPulse] = fAmplitude
# - Incorporate input weights in input
mfPulse = np.atleast_2d(vfPulse).T @ mfW_in
tsPulse = ts.TimeSeries(vtTimePulse, mfPulse)

# - Simulate
tsResSpikes = res.evolve(tsPulse)
vIsyn = res.monI.I_syn.T
vVpot = res.monV.v.T

# - Plot
fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, sharex=True, figsize=(12, 12))

# Input and spikes
ax0.plot(vtTimePulse, nResSize * vfPulse / fAmplitude, zorder=10, c='r', lw=2)
ax0.scatter(tsResSpikes.vtTimeTrace, tsResSpikes.vnChannels)
ax0.set_title("Input and spikes")

# Filtered spikes
__, mfSpikeRaster, __ = tsResSpikes.raster(tDt=tDt, bSamples=False)
vtTime = np.arange(len(mfFiltered)) * tDt + tsResSpikes.tStart
# mfFiltered = an.filter_exp_box(mfSpikeRaster, 25 * tDt, 50 * tDt, tDt)
# ax1.plot(vtTime, mfFiltered)
mfFreqs = 1. / an.interspike_intervals(tsResSpikes, tDt)
ax1.set_title("Filtered input")

# Synaptic current
ax2.plot(vtTimePulse[:-1], vIsyn)
ax2.set_title("Synaptic currents")

# Membrane potential
ax3.plot(vtTimePulse[:-1], vVpot)
ax3.set_title("Membrane potentials")
