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
from layers.recurrent.weights import RndmSparseEINet


### --- Set parameters

tDt = 0.005  # Length of time step in seconds
fHeartRate = 1  # Heart rate in rhythms per second

nTrialsTr = 500  # Number ECG rhythms for training
nTrialsTe = 500  # Number ECG rhythms for testing

nDimIn = 1  # Input dimensions
nDimOut = 1  # Output dimensions

nResSize = 256  # Reservoir size
tTauN = 50 * tDt  # Reservoir time constant
tTauS = 500 * tDt  # Reservoir time constant

fRegularize = 0.001  # Regularization parameter for training with ridge regression

# Parameters concerning reservoir weights
kwResWeights = {
    "nResSize": nResSize,
    "fConnectivity": 0.4,  # Connectivity
    "bPartitioned": False,  # Partition reservoir into excitatory/inhibitory
    "fRatioExc": 0.5,  # Ratio of excitatory neurons
    "fScaleInh": 1,  # Scale of inhibitory vs excitatory weights
    "fNormalize": 0.5,
}  # Normalize matrix spectral radius

# Probabilities for anomalies in ECG rhythms
dProbs = {
    "complete_normal": 0.8,  # Normal ECG
    "complete_noP": 0.025,  # Missing P-wave
    "complete_Pinv": 0.025,  # Inverted P-wave
    "complete_noQRS": 0.025,  # Missing QRS complex
    "complete_Tinv": 0.025,  # Inverted T-wave
    "complete_STelev": 0.025,  # Elevated ST-segment
    "complete_STdepr": 0.025,  # Depressed ST-segment
    "complete_tach": 0.025,  # Tachycardia
    "complete_brad": 0.025,
}  # Bradycardia

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


### --- Network generation

# - Generate weight matrices
mfW_in = 2 * np.random.rand(nDimIn, nResSize)
mfW_res = RndmSparseEINet(**kwResWeights)

# - Generate layers
flIn = PassThrough(mfW=mfW_in, tDt=tDt, tDelay=0, strName='in')
# rlRes = rec.RecRateEuler(mfW=mfW_res, vtTau=tTau, tDt=tDt, strName='res')
# flOut = ff.PassThrough(mfW=np.zeros((nResSize, nDimOut)), tDt=tDt, tDelay=0, strName='out')
rlRes = Rec(mfW=mfW_res, vtTauN=tTauN, tDt=tDt * second, strName="reservoir")
flOut = FFsc(mfW=np.zeros((nResSize, nDimOut)), tTauSyn=tTauS, tDt=tDt, strName="output")

# - Generate network
net = nw.Network(flIn, rlRes, flOut)


### --- Training

# - Training signal
# Generate traceining data and time trace
vfEcgTr, vfTgtTr = signal_and_target(nTrials=nTrialsTr, **kwSignal)
vtTimeTr = np.arange(0, vfEcgTr.size * tDt, tDt)[: vfEcgTr.size]
# Generate TimeSeries for input and target
tsInTr = ts.TimeSeries(vtTimeTr, vfEcgTr)
# # Train with sine wave instead
# vfTgtTr = np.sin(np.linspace(0, 20, vfEcgTr.size)) + 1
tsTgtTr = ts.TimeSeries(vtTimeTr, vfTgtTr)

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
vfEcgVa, vfTgtVa = signal_and_target(nTrials=nTrialsTe, **kwSignal)
vtTimeVa = np.arange(0, vfEcgVa.size * tDt, tDt)[: vfEcgVa.size]
# Generate TimeSeries with input and target
tsInVa = ts.TimeSeries(vtTimeVa, vfEcgVa)
tsTgtVa = ts.TimeSeries(vtTimeVa, vfTgtVa)

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
vfEcgTe, vfTgtTe = signal_and_target(nTrials=nTrialsTe, **kwSignal)
vtTimeTe = np.arange(0, vfEcgTe.size * tDt, tDt)[: vfEcgTe.size]
# Generate TimeSeries with input and target
tsInTe = ts.TimeSeries(vtTimeTe, vfEcgTe)
tsTgtTe = ts.TimeSeries(vtTimeTe, vfTgtTe)

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
