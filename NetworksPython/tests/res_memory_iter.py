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
lfAmplitude = [1] # Amplitude of the pulse
lfStd = [0.001, 0.0055, 0.001]
lfMean = [-0.01, -0.005, 0.001]

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



# - Reservoir
def gen_res(nResSize, fDensity, fMean, fStd):
    
    # - Recurrent weights
    mfW_res = IAFSparseNet(nResSize = nResSize,
       fMean = fMean / fDensity / nResSize,
       fStd = fStd / np.sqrt(fDensity),
       fDensity = fDensity,
    ) / 500

    # - Reservoir generation
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

def input_pulse(fAmplitude, mfW_in):
    # - Pulse input
    vtTimePulse = np.arange(0, tDur, tDt)
    nPulse = int(tPulse / tDt)
    nStart = int(tStart / tDt)
    vfPulse = np.zeros_like(vtTimePulse)
    vfPulse[nStart:nStart+nPulse] = fAmplitude
    # - Incorporate input weights in input
    mfPulse = np.atleast_2d(vfPulse).T @ mfW_in
    tsPulse = ts.TimeSeries(vtTimePulse, mfPulse)

    return vfPulse, tsPulse

def plot(vtTimePulse, vfPulse, tsResSpikes, vIsyn, vVpot, fAmplitude, fStd, fMean):

    # - Plot
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, sharex=True, figsize=(12, 12))

    plt.suptitle("Amp: {}, Std: {}, Mean: {}".format(
        fAmplitude, fStd, fMean))

    # Input and spikes
    ax0.plot(vtTimePulse, nResSize * vfPulse / fAmplitude, zorder=10, c='r', lw=2)
    ax0.scatter(tsResSpikes.vtTimeTrace, tsResSpikes.vnChannels)
    ax0.set_title("Input and spikes")

    # Filtered spikes
    __, mfSpikeRaster, __ = tsResSpikes.raster(tDt=tDt, bSamples=False)
    # mfFiltered = an.filter_exp_box(mfSpikeRaster, 25 * tDt, 50 * tDt, tDt)
    # vtTime = np.arange(len(mfFiltered)) * tDt + tsResSpikes.tStart
    # ax1.plot(vtTime, mfFiltered)
    # ax1.set_title("Filtered input")
    
    mfFreqs = 1. / an.interspike_intervals(tsResSpikes, tDt)
    vtTime = np.arange(len(mfFreqs)) * tDt + tsResSpikes.tStart
    ax1.plot(vtTime, mfFreqs)
    ax1.set_title("Instantaneous firing rates")

    # Synaptic current
    ax2.plot(vtTimePulse[:-1], vIsyn)
    ax2.set_title("Synaptic currents")

    # Membrane potential
    ax3.plot(vtTimePulse[:-1], vVpot)
    ax3.set_title("Membrane potentials")

    return fig, (ax0, ax1, ax2, ax3)


# - Input weights
mfW_in = 2 * (np.random.rand(nDimIn, nResSize) - 0.5)
# - Generate input pulse
vfPulse1, tsPulse1 = input_pulse(1, mfW_in)

# - Iterrate over std
for fStd in lfStd:

    # - Iterrate over means
    for fMean in lfMean:
        
        # - Iterrate over amps
        for fAmplitude in lfAmplitude:

            # - Genrate reservoir
            res = gen_res(kwResWeights['nResSize'], kwResWeights['fDensity'], fMean, fStd)

            vfPulse = fAmplitude * vfPulse1
            tsPulse = fAmplitude * tsPulse1

            # - Simulate
            tsResSpikes = res.evolve(tsPulse)
            vIsyn = res.monI.I_syn.T
            vVpot = res.monV.v.T

            # - Plot
            fig, axes = plot(tsPulse.vtTimeTrace, vfPulse, tsResSpikes, vIsyn, vVpot, fAmplitude, fStd, fMean)