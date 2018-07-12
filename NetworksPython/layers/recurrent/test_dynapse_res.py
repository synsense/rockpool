import sys
import os.path
strPathToLib = os.path.abspath(sys.path[0] + '../../../..')
sys.path.insert(1, strPathToLib)

from matplotlib import pyplot as plt
plt.ion()

import numpy as np
import seaborn as sb

from brian2 import second, amp, farad
import brian2 as b2

from NetworksPython.layers.recurrent import dynapse_brian as db
from NetworksPython.layers.recurrent.weights import In_Res_Dynapse
from NetworksPython import timeseries as ts
from NetworksPython import analysis as an


tDt = 0.0001 * second

nNumInputSamples = 40
tInputDuration = 1

# - Corrected constant parameters
dParamsNeuron = {
    # 'Io' : 1.5e-12 * amp,
    # 'Cmem' : 2e-12 * farad,
    # 'Ispkthr' : 1e-5 * amp,
    # 'Ireset' : 0 * amp,
    # 'Ith' : 500e-9 * amp,
    # 'Iagain' : 10e-12 * amp,
    'Iconst' : 4.375e-9 * amp,
}

fBaseweightE = 7e-8 * amp
fBaseweightI = 1e-7 * amp

dParamsSynapseIn = {
    # 'Io_syn' : 1.5e-12 * amp,
    # 'Csyn' : 2e-12 * farad,
    'baseweight_i' : fBaseweightI,
    'baseweight_e' : fBaseweightE,
}

dParamsSynapseRec = {
    # 'Io_syn' : 1.5e-12 * amp,
    # 'Csyn' : 2e-12 * farad,
    'baseweight_i' : fBaseweightI,
    'baseweight_e' : fBaseweightE,
}

# - Reservoir generation

nResSize = 512
# fBiasInE = fBiasInI = 1e-8 * amp  # Input baseweight
# fBiasRecE = fBiasRecI = 1e-9 * amp  # Reservoir baseweight     
# fIconst = 1e-6 * amp  # Bias current

# Recurrent weights, normalized by spectral radius
np.random.seed(1)
# mfW = weights(nResSize, tupfWExc=(1,1), tupfWInh=(1,1), fNormalize=1)[0]
# Input weights
#vfWIn = 2*np.random.rand(nResSize) - 1
vfWIn, mfW, *__ = In_Res_Dynapse(nResSize, tupfWExc=(1,1), tupfWInh=(1,1), fNormalize=1, bLeaveSpaceForInput=True)

# Reservoir
res = db.RecDynapseBrian(mfW, vfWIn, tDt=tDt)

# Update parameters
# res._sgReceiver.baseweight_i = fBiasInI
# res._sgReceiver.baseweight_e = fBiasInE
# res._sgRecurrentSynapses.baseweight_i = fBiasRecI
# res._sgRecurrentSynapses.baseweight_e = fBiasRecE
res._sgReceiver.set_params(dParamsSynapseIn)
res._sgRecurrentSynapses.set_params(dParamsSynapseRec)
res._ngLayer.set_params(dParamsNeuron)

# Monitors
stmNg = b2.StateMonitor(res._ngLayer, ['Ie0', 'Ii0', 'Ie1', 'Ii1', 'Imem', 'Iin_clip'], record=True)
res._net.add(stmNg)

# - Input
vtIn = np.sort(np.random.rand(nNumInputSamples)) * tInputDuration
vnChIn = np.random.randint(nResSize, size=nNumInputSamples)
tsIn = ts.TSEvent(vtIn, vnChIn)

# - Run simulation
tsR = res.evolve(tsIn)

# - Plot
fig, axes = plt.subplots(4, figsize=(10,15), sharex=True)
# Continuous variables
for var, axID in zip(('Ie1', 'Ii1', 'Ie0', 'Ii0', 'Iin_clip', 'Imem'), (0,0,1,1,2,3)):
    axes[axID].plot(stmNg.t/second, getattr(stmNg, var).T/amp)
    axes[axID].set_title(var)

# Preserve axis y-limits
lYlims = [ax.get_ylim() for ax in axes]

# Input spikes
for t in vtIn:
    for ax, ylims in zip((axes[0],), (lYlims[0],)):
        ax.plot([t,t], ylims, 'k--', zorder = -1, alpha=0.5)
# Reservoir spikes
# for t in tsR.vtTimeTrace:
#     for ax, ylims in zip(axes[1:], lYlims[1:]):
#         ax.plot([t,t], ylims, 'b--', zorder = -1, alpha=0.5)

# instantaneous firing rates as inverted inter-spike intervals, averaged over all neurons
vfMeanInstRate = np.nanmean(1./an.interspike_intervals(tsR, tDt=tDt/second), axis=1)
plt.figure()
plt.plot(np.arange(len(vfMeanInstRate)) * tDt/second, vfMeanInstRate)

# - Mean firing rates
# total:
fMeanTotal = np.size(tsR.vnChannels) / tInputDuration
fMeanTotalPerNeuron = fMeanTotal / nResSize
# neuron-wise
vfMeanRates = np.array([
    np.sum(tsR.vnChannels == iChannel)/tInputDuration for iChannel in range(nResSize)
])
vfRateDevs = np.array([
    np.std(1./np.diff(tsR.vtTimeTrace[tsR.vnChannels == iChannel]))
    for iChannel in range(nResSize)
])
fStdMeanRates = np.std(vfMeanRates)

print('Overall firing rate: {} Hz, ({} +- {}) Hz per Neuron)'.format(
    fMeanTotal, fMeanTotalPerNeuron, fStdMeanRates
))

# Plot mean rate distro
plt.figure()
plt.bar(np.arange(nResSize), vfMeanRates, yerr=vfRateDevs)