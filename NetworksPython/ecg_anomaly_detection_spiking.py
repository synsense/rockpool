import numpy as np
from scipy import sparse
from typing import Callable
from brian2 import second
from matplotlib import pyplot as plt

plt.ion()

import TimeSeries as ts
from ecg import signal_and_target
import network as nw
import analysis as an

# from layers.recurrent import rate as rec
from layers.feedforward.rate import PassThrough
from layers.recurrent.iaf_brian import RecIAFBrian as Rec
from layers.feedforward.exp_synapses_manual import FFExpSyn as FFsc
# from layers.recurrent.weights import RndmSparseEINet
from layers.recurrent.weights import IAFSparseNet


### --- Set parameters

tDt = 0.005  # Length of time step in seconds
fHeartRate = 1  # Heart rate in rhythms per second

nTrialsTr = 1500  # Number ECG rhythms for training
nTrialsTe = 400  # Number ECG rhythms for testing

nDimIn = 1  # Input dimensions
nDimOut = 1  # Output dimensions

nResSize = 512  # Reservoir size
tTauN = 25 * tDt  # Reservoir neuron time constant
tTauS = 75 * tDt # Reservoir synapse time constant
tTauO = 25 * tDt  # Readout time constant

fRegularize = 0.001  # Regularization parameter for training with ridge regression

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

# Probabilities for anomalies in ECG rhythms
dProbs = {
    "complete_normal": 0.8,     # Normal ECG
    "complete_noP": 0.025,      # Missing P-wave
    "complete_Pinv": 0.025,     # Inverted P-wave
    "complete_noQRS": 0.025,    # Missing QRS complex
    "complete_Tinv": 0.025,     # Inverted T-wave
    "complete_STelev": 0.025,   # Elevated ST-segment
    "complete_STdepr": 0.025,   # Depressed ST-segment
    "complete_tach": 0.025,     # Tachycardia
    "complete_brad": 0.025,     # Bradycardia
}  

# - Kwargs for signal_and_target function
kwSignal = {
    "strTargetMethod": "fix",  # Method for labelling targets
    "dProbs": dProbs,
    "fHeartRate": fHeartRate,
    "tDt": tDt,
    "bVerbose": True,
    "nTargetWidth": int(fHeartRate / tDt),
}


def cTrain(net: nw.Network, dtsSignal: dict, bFirst: bool, bFinal: bool):
    """
    cTrain - Train layer flOut with input from dtsSignal and tsTgtTr as
             target. Use fRegularize as parameter for ridge regression.
             Training may span several batches. This funciton is to be
             passed on to net and will be called after each batch.
    :param net:     Network containing the layer
    :param dtsSignal: dict with TimeSeries from network evolutions
    :param bFirst:    bool True if this is the first batch of a training
    :param bLast:    bool True if this is the final batch of a training
    """
    # - Input to flOut
    tsInput = dtsSignal[flOut.lyrIn.strName]
    # - Coose suitable time range for target
    tsTarget = tsTgtTr.resample(tsInput.vtTimeTrace)
    # - Train the layer
    flOut.train_rr(tsTarget, tsInput, fRegularize, bFirst, bFinal)


def ts_ecg_target(nRhythms: int, **kwargs) -> (ts.TimeSeries, ts.TimeSeries):
    """
    ts_ecg_target - Generate two time series, one containing an ECG signal
                   and the other the corresponding target.
    :param nRhythms:    int Number of ECG rhythms in the input
    :tDt:               float Size of a single time step
    :kwargs:            dict Kwargs that are passed on to signal_and_target
    """

    # - Input signal and target
    vfInput, vfTarget = signal_and_target(nTrials=nRhythms, **kwargs)
    
    # - Time base
    tDt = kwargs['tDt']
    vtTime = np.arange(0, vfInput.size * tDt, tDt)[: vfInput.size]
    
    # - Genrate time series
    tsInput = ts.TimeSeries(vtTime, vfInput)
    tsTarget = ts.TimeSeries(vtTime, vfTarget)

    return tsInput, tsTarget


### --- Network generation

# - Generate weight matrices
mfW_in = 2 * (np.random.rand(nDimIn, nResSize) - 0.5)
# mfW_res = RndmSparseEINet(**kwResWeights)
mfW_res = IAFSparseNet(**kwResWeights)

# - Generate layers
flIn = PassThrough(mfW=mfW_in, tDt=tDt, tDelay=0, strName='input')
# rlRes = rec.RecRateEuler(mfW=mfW_res, vtTau=tTau, tDt=tDt, strName='res')
# flOut = ff.PassThrough(mfW=np.zeros((nResSize, nDimOut)), tDt=tDt, tDelay=0, strName='out')
rlRes = Rec(mfW=mfW_res, vtTauN=tTauN, vtTauSynR=tTauS, tDt=tDt * second, strName="reservoir")
flOut = FFsc(mfW=np.zeros((nResSize, nDimOut)), tTauSyn=tTauO, tDt=tDt, strName="output")

# - Generate network
net = nw.Network(flIn, rlRes, flOut)


### --- Training

# - Training signal
# Generate training data and time trace
tsInTr, tsTgtTr = ts_ecg_target(nTrialsTr, **kwSignal)

# d = net.evolve(tsInTr)
# plt.scatter(d['reservoir'].vtTimeTrace, d['reservoir'].vnChannels)
# plt.plot(tsInTr.vtTimeTrace, tsInTr.mfSamples*30+50, color='y')


# - Run training
net.train(cTrain, tsInTr, tDurBatch=500)
net.reset_all()


# - Sanity check with training signal
dTr = net.evolve(tsInTr)
net.reset_all()
#
# Output TimeSeries
tsOutTr = dTr[flOut.strName]

# Plot input, target and output
plt.figure()
plt.plot(tsOutTr.vtTimeTrace, tsOutTr.mfSamples)
plt.plot(tsTgtTr.vtTimeTrace, tsTgtTr.mfSamples)
plt.plot(tsInTr.vtTimeTrace, 0.2*tsInTr.mfSamples, color='k', alpha=0.3, zorder=-1)
# # Plot smoothed output
# nWindowSize = 200
# vWindow = np.ones(nWindowSize) / nWindowSize
# vfSmoothed = np.convolve(tsOutTr.mfSamples.flatten(), vWindow, 'full')
# plt.plot(tsOutTr,vtTimeTrace, vfSmoothed[-(tsOutTr.vtTimeTrace.size):])


### --- Validation run for threshold determination

# - Validation signal
# Generate test data and time trace
tsInVa, tsTgtVa = ts_ecg_target(nTrialsTe, **kwSignal)

# - Validation run
dVa = net.evolve(tsInVa)
net.reset_all()

vOutVa = dVa[flOut.strName].mfSamples
# - Determine threshold for analysis of test run
fThr = an.find_threshold(
    vOutput=vOutVa,
    vTarget=tsTgtVa.mfSamples,
    nWindow=int(fHeartRate / tDt),
    nClose=int(fHeartRate / tDt),
    nAttempts=5,
    nRecursions=5,
)
print("Using threshold: {:.3f}".format(fThr))


### --- Testing

# - Test signal
# Generate test data and time trace
tsInTe, tsTgtTe = ts_ecg_target(nTrialsTe, **kwSignal)

# - Run test
dTe = net.evolve(tsInTe)
net.reset_all()

# - Output TimeSeries
tsOutTe = dTe[flOut.strName]

# Plot input, target and output
# plt.figure()
# plt.plot(tsOutTe.vtTimeTrace, tsOutTe.mfSamples)
# plt.plot(tsTgtTe.vtTimeTrace, tsTgtTe.mfSamples)
# plt.plot([tsResTe.vtTimeTrace[0], tsResTe.vtTimeTrace[1]], [fThr, fThr], 'k--', zorder=0, lw=2)
# plt.plot(tsInTe.vtTimeTrace, 0.2*tsInTe.mfSamples, color='k', alpha=0.3, zorder=-1)

# - Analysis and plotting
print("\nAnalysis of test run:")
vOutTe = dTe[flOut.strName].mfSamples
an.analyze(
    vOutTe,
    tsTgtTe.mfSamples,
    tsInTe.mfSamples,
    fThr,
    nWindow=int(fHeartRate / tDt),
    bPlot=True,
)
