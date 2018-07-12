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
from NetworksPython.layers.feedforward.rate import PassThrough
from NetworksPython import timeseries as ts
from NetworksPython.network import Network
from NetworksPython import analysis as an
from Projects.AnomalyDetection.ECG.ecgsignal import labeled_signal

def draw_uniform(nNumSamples, fMin, fMax):
    """
    draw_uniform - Convenience function fro drawing nNumSamples uniformly
                   distributed samples between fMin and fMax 
    """
    return (fMax - fMin) * np.random.rand(nNumSamples) + fMin

tDt = 0.001
fHeartRate = 1
nTrials = 20

nResSize = 512

mfWIn = 2 * np.random.rand(nResSize) - 1

fBiasMin = 0 * amp
fBiasMax = 0.015 * amp
vfBias = draw_uniform(nResSize, fBiasMin, fBiasMax)

fWeightScale = 0.3

tTauMin = 0.010
tTauMax = 0.1
vtTau = draw_uniform(nResSize, tTauMin, tTauMax)

# Input weights
vfWIn = (2*np.random.rand(nResSize) - 1) * fWeightScale

# Layer
fl = FFIAFBrian(vfWIn, vfBias=vfBias, vtTauN=vtTau)

# Input signal
vfECG = labeled_signal(nTrials, fHeartRate=fHeartRate*tDt)[0]
tsIn = ts.TimeSeries(np.arange(len(vfECG))*tDt, vfECG)

# Evolve layer
tsOut = fl.evolve(tsIn)

# plot results
plt.figure()
plt.plot(tsIn.vtTimeTrace, 0.2*nResSize*tsIn.mfSamples, color='grey', lw=2)
plt.scatter(tsOut.vtTimeTrace, tsOut.vnChannels)


# # Instantaneous firing rates
# vfMeanInstRate = np.nanmean(1./an.interspike_intervals(tsOut, tDt=tDt), axis=1)
# plt.figure()
# plt.plot(np.arange(len(vfMeanInstRate)) * tDt/second, vfMeanInstRate)

# - Mean firing rates
tInputDuration = tDt * len(vfECG)
# total:
fMeanTotal = np.size(tsOut.vnChannels) / tInputDuration
fMeanTotalPerNeuron = fMeanTotal / nResSize
# neuron-wise
vfMeanRates = np.array([
    np.sum(tsOut.vnChannels == iChannel)/tInputDuration for iChannel in range(nResSize)
])
vfRateDevs = np.array([
    np.std(1./np.diff(tsOut.vtTimeTrace[tsOut.vnChannels == iChannel]))
    for iChannel in range(nResSize)
])
fStdMeanRates = np.std(vfMeanRates)

print('Overall firing rate: {} Hz, ({} +- {}) Hz per Neuron)'.format(
    fMeanTotal, fMeanTotalPerNeuron, fStdMeanRates
))

# Plot mean rate distro
plt.figure()
viSortedByRate = np.argsort(vfMeanRates)
plt.plot(np.arange(nResSize), vfMeanRates[viSortedByRate], label="Rate")
plt.plot(np.arange(nResSize), vfRateDevs[viSortedByRate], 'k--', label="Std Rate")
plt.plot(np.arange(nResSize), vfBias[viSortedByRate] * 1e5, label="Bias * 1e5")
plt.plot(np.arange(nResSize), vtTau[viSortedByRate] * 1e5, label="Tau * 1e5")
plt.plot(np.arange(nResSize), vfWIn[viSortedByRate] * 5e3, label="Weight * 1e4")
plt.legend(loc="best")