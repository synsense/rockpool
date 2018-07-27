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

from NetworksPython.layers.feedforward.iaf_brian import FFIAFBrian
from NetworksPython.layers.recurrent import dynapse_brian as db
from NetworksPython.layers.recurrent.weights import In_Res_Dynapse

from NetworksPython import timeseries as ts
from NetworksPython import analysis as an
from NetworksPython.network import Network

from Projects.AnomalyDetection.ECG.ecgsignal import labeled_signal

def draw_uniform(nNumSamples, fMin, fMax):
    """
    draw_uniform - Convenience function fro drawing nNumSamples uniformly
                   distributed samples between fMin and fMax 
    """
    return (fMax - fMin) * np.random.rand(nNumSamples) + fMin

# - Parameters

# Time constants
tDtRes = 0.0001 * second
tDtFF = 0.0001 * second

# Input
fHeartRate = 1
nTrials = 1

# Reservoir
dParamsNeuron = {
    'Iconst' : 4.375e-9 * amp,
}
dParamsSynapse = {
    'baseweight_i' : 7e-8 * amp,
    'baseweight_e' : 1e-7 * amp,
}
nResSize = 512

# Analogue-to-spike layer
mfWIn = 2 * np.random.rand(nResSize) - 1

fBiasMin = 0 * amp
fBiasMax = 0.015 * amp
vfBias = draw_uniform(nResSize, fBiasMin, fBiasMax)

fWeightScale = 0.3

tTauMin = 0.010
tTauMax = 0.1
vtTau = draw_uniform(nResSize, tTauMin, tTauMax)


# - Network generation
# Analogue to spike
vfWAS = (2*np.random.rand(nResSize) - 1) * fWeightScale
flAS = FFIAFBrian(vfWAS, vfBias=vfBias, vtTauN=vtTau, tDt=tDtFF)

# Reservoir
vfWIn, mfW, *__ = In_Res_Dynapse(nResSize, tupfWExc=(1,1), tupfWInh=(1,1), fNormalize=1, bLeaveSpaceForInput=True)
res = db.RecDynapseBrian(mfW, vfWIn, tDt=tDtRes)
res._sgReceiver.set_params(dParamsSynapse)
res._sgRecurrentSynapses.set_params(dParamsSynapse)
res._ngLayer.set_params(dParamsNeuron)

# Monitors
stmNg = b2.StateMonitor(res._ngLayer, ['Ie0', 'Ii0', 'Ie1', 'Ii1', 'Imem', 'Iin_clip'], record=True)
res._net.add(stmNg)

# - Input signal
vfECG = labeled_signal(nTrials, fHeartRate=fHeartRate*tDtFF)[0]
tsECG = ts.TimeSeries(np.arange(len(vfECG))*tDtFF, vfECG)

# - Run simulation
print("Evolving A-to-S layer", end="")
tsFF = flAS.evolve(tsECG)
print("Evolving reservoir      ", end="")
tsR = res.evolve(tsFF)
print("\rDone                  ")

# - Plot
fig, axes = plt.subplots(4, figsize=(10,15), sharex=True)
# Continuous variables
for var, axID in zip(('Ie1', 'Ii1', 'Ie0', 'Ii0', 'Iin_clip', 'Imem'), (0,0,1,1,2,3)):
    axes[axID].plot(stmNg.t/second, getattr(stmNg, var).T/amp)
    axes[axID].set_title(var)

# Preserve axis y-limits
lYlims = [ax.get_ylim() for ax in axes]

# instantaneous firing rates as inverted inter-spike intervals, averaged over all neurons
vfMeanInstRate = np.nanmean(1./an.interspike_intervals(tsR, tDt=tDtRes/second), axis=1)
plt.figure()
plt.plot(np.arange(len(vfMeanInstRate)) * tDtRes/second, vfMeanInstRate)

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