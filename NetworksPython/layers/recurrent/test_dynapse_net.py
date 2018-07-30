### --- Import statements

import sys
import os.path
strPathToLib = os.path.abspath(sys.path[0] + '../../../..')
sys.path.insert(1, strPathToLib)

from matplotlib import pyplot as plt
plt.ion()

import time

import numpy as np

from brian2 import second, amp, farad
import brian2 as b2

# Layers
from NetworksPython.layers.feedforward.iaf_brian import FFIAFBrian
from NetworksPython.layers.recurrent.dynapse_brian import RecDynapseBrian
from NetworksPython.layers.recurrent.weights import In_Res_Dynapse
from NetworksPython.layers.feedforward.exp_synapses_manual import FFExpSyn

from NetworksPython import timeseries as ts
from NetworksPython import analysis as an
from NetworksPython import network as nw

from Projects.AnomalyDetection.ECG.ecgsignal import signal_and_target

### --- Parameters

# - Input data
fHeartRate = 1  # Heart rate in rhythms per second

fStdNoiseSignal = 0  # Standard deviation of input noise

nTrialsTr = 500  # Number ECG rhythms for Training
nTrialsVa = 500  # Number ECG rhythms for validation
nRepsVa = 3  # Number repetitions of validation runs
nTrialsTe = 500  # Number ECG rhythms for testing

pNormal = 0.8  # Probability of normal input rhythm
pAnomal = (1 - pNormal) / 6.  # Probability of abnormal input rhythm

# Anomaly probabilities
dProbs = {
    "complete_normal": pNormal,  # Normal ECG
    "complete_noP": pAnomal,  # Missing P-wave
    "complete_Pinv": pAnomal,  # Inverted P-wave
    "complete_noQRS": pAnomal,  # Missing QRS complex
    "complete_Tinv": pAnomal,  # Inverted T-wave
    "complete_STelev": pAnomal,  # Elevated ST-segment
    "complete_STdepr": pAnomal,  # Depressed ST-segment
}

# - Forward layer
nDimIn = 1  # Input dimensions

tDtAS = 0.005  # Length of time step in seconds (analogue-to-spike layer)

fScaleAS = 0.3

fBiasMinAS = 0 * amp
fBiasMaxAS = 0.015 * amp

tTauMinAS = 0.010
tTauMaxAS = 0.1

# - Reservoir
tDtRes = 0.0001  # Length of time step in seconds (reservoir layer)

nResSize = 512  # Reservoir size
fConnectivity = None  # Percentage of non-zero recurrent weights

fIconst = 4.375e-9 * amp  # Constant input current as bias
fBaseweightE = 7e-8 * amp  # Excitatory synapse strength
fBaseweightI = 1e-7 * amp  # Inhibitory synapse strength

# - Readout
nDimOut = 9  # Output dimensions
tDtOut = 0.005  # Length of time step in seconds (readout layer)
tTauOut = 35 * tDtOut  # Time constant of readout exponential filter

# - Training
tDurBatch = 50  # Training batch duration
fRegularize = 0.001  # Regularization parameter for training with ridge regression


# - Collect some of the parameters in dicts

# Corrected neuron parameters
dParamNeuron = {
    'Iconst' : fIconst,
}

# Corrected synapse parameters
dParamSynapse = {
    'baseweight_i' : fBaseweightI,
    'baseweight_e' : fBaseweightE,
}

# Signal parameters
kwSignal = {
    "strTargetMethod": "segment-extd",  # Method for labelling targets
    "dProbs": dProbs,
    "fHeartRate": fHeartRate,
    "tDt": tDtAS,
    "bVerbose": True,
    "nMinWidth": int(0.5 * fHeartRate / tDtAS),
    "bDetailled": True,
    "fStdNoise" : fStdNoiseSignal,
}



### --- Function definitions

def draw_uniform(nNumSamples, fMin, fMax):
    """
    draw_uniform - Convenience function fro drawing nNumSamples uniformly
                   distributed samples between fMin and fMax 
    """
    return (fMax - fMin) * np.random.rand(nNumSamples) + fMin

def cTrain(net: nw.Network, dtsSignal: dict, bFirst: bool, bFinal: bool):
    """
    cTrain - Train layers flOut and flOut1D with input from dtsSignal and tsTgtTr as
             target. Use fRegularize as parameter for ridge regression.
             Training may span several batches. This funciton is to be
             passed on to net and will be called after each batch.
    :param net:     Network containing the layer
    :param dtsSignal: dict with TimeSeries from network evolutions
    :param bFirst:    bool True if this is the first batch of a training
    :param bLast:    bool True if this is the final batch of a training
    """
    # - Input to flOut
    tsInput = dtsSignal["reservoir"]
    # - Infer time range of current batch
    tStart = dtsSignal["external"].tStart
    tStop = dtsSignal["external"].tStop
    # - Sample target within time range of current batch
    tsTarget = tsTgtTr.resample_within(tStart, tStop)
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
    vfInput, mfTarget = signal_and_target(nTrials=nRhythms, **kwargs)

    # - Time base
    tDt = kwargs["tDt"]
    vtTime = np.arange(0, vfInput.size * tDtAS, tDtAS)[: vfInput.size]

    # - Genrate time series
    tsInput = ts.TimeSeries(vtTime, vfInput)
    tsTarget = ts.TimeSeries(vtTime, mfTarget)

    return tsInput, tsTarget


### --- Network generation

# - Layer for analogue to spike conversion
vfWAS = (2*np.random.rand(nDimIn, nResSize) - 1) * fScaleAS
vfBiasAS = draw_uniform(nResSize, fBiasMinAS, fBiasMaxAS)
vtTauAS = draw_uniform(nResSize, tTauMinAS, tTauMaxAS)
flAS = FFIAFBrian(vfWAS, vfBias=vfBiasAS, vtTauN=vtTauAS, tDt=tDtAS, strName="input")

# - Reservoir
vfWIn, mfW, *__ = In_Res_Dynapse(nResSize, tupfWExc=(1,1), tupfWInh=(1,1), fNormalize=1, bLeaveSpaceForInput=True)
rlRes = RecDynapseBrian(mfW, vfWIn, tDt=tDtRes, dParamNeuron=dParamNeuron, dParamSynapse=dParamSynapse, strName="reservoir")

# - Readout
flOut = FFExpSyn(mfW=np.zeros((nResSize, nDimOut)), tTauSyn=tTauOut, tDt=tDtOut, strName="output")
# - Network
net = nw.Network(flAS, rlRes, flOut)


if __name__ == "__main__":
    ### --- Training
    print("Training")

    # - Training signal
    # Generate training data and time trace
    tsInTr, tsTgtTr = ts_ecg_target(nTrialsTr, **kwSignal)

    # - Run training
    net.train(cTrain, tsInTr, tDurBatch=tDurBatch)
    net.reset_all()


    ### --- Validation run for threshold determination

    print("Finding threshold")

    # - Lists for storing thresholds
    lvfThr = []

    # - Validation runs
    for i in range(nRepsVa):

        print("\tRun {} of {}".format(i + 1, nRepsVa), end="\r")

        # Input and target generation
        tsInVa, tsTgtVa = ts_ecg_target(nTrialsVa, **dict(kwSignal, bVerbose=False))

        # Simulation
        dVa = net.evolve(tsInVa, bVerbose=False)
        net.reset_all()

        lvfThr.append(
            an.find_all_thresholds_multi(
                mOutput=dVa["output"].mfSamples,
                mTarget=tsTgtVa.mfSamples,
                nWindow=int(fHeartRate / tDtOut),
                nClose=int(fHeartRate / tDtOut),
                fMin=0.1,
                nAttempts=16,
                nRecursions=4,
                bStrict=False,
            )
        )

        del dVa

    # - Average over stored thresholds
    vfThr = np.nanmean(lvfThr, axis=0)

    print("Using following thresholds:")
    print(vfThr)


    ### --- Testing
    print("Testing")
    # - Test signal
    # Generate test data and time trace
    tsInTe, tsTgtTe = ts_ecg_target(nTrialsTe, **kwSignal)

    # - Run test
    dTe = net.evolve(tsInTe)
    net.reset_all()

    # - Analysis and plotting
    print("\nAnalysis of test run:")
    mfOutTe = dTe['output'].mfSamples
    dAnalysis = an.analyze_multi(
        mfOutTe,
        tsTgtTe.mfSamples,
        tsInTe.mfSamples,
        vfThr[:, 0],
        nWindow=int(fHeartRate / tDtOut),
        nClose=int(fHeartRate / tDtOut),
        bPlot=True,
        bVerbose=True,
    )


    # - Show statistics for individual symptom types

    lstrSymptomFullNames = [
        "Inverted P-wave",
        "Missing P-wave",
        "Missing QRS-complex",
        "Inverted T-wave",
        "Elevated ST-segment",
        "Depressed ST-segment",
    ]

    for iSymptom in dAnalysis["dSymptoms"].keys():
        if iSymptom < len(lstrSymptomFullNames):
            print(lstrSymptomFullNames[iSymptom] + ":")
            print(
                "{:.1%}\n".format(
                    dAnalysis["dSymptoms"][iSymptom]["fSensitivity"]
                )
            )

    # - Save results
    strDateTime = time.strftime('%y-%m-%d_%H-%M-%S')
    dParams = dict(
        nResSize = nResSize,
        nTrialTr = nTrialTr,
        nTrialTe = nTrialTe,
        nTrialVa = nTrialVa,
        nRepsVa = nRepsVa,
        fHeartRate = fHeartRate,
        fStdNoiseSignal = fStdNoiseSignal,
        tDtAS = tDtAS,
        fScaleAS = fScaleAS,
        fBiasMinAS = fBiasMinAS,
        fBiasMaxAS = fBiasMaxAS,
        tTauMinAS = tTauMinAS,
        tTauMaxAS = tTauMaxAS,
        tDtRes = tDtRes,
        fConnectivity = fConnectivity,
        fIconst = fIconst,
        fBaseweightE = fBaseweightE,
        fBaseweightI = fBaseweightI,
        nDimOut = nDimOut,
        tDtOut = tDtOut,
        tTauOut = tTauOut,
        tDurBatch = tDurBatch,
        fRegularize = fRegularize,
    )
    np.savez('results_' + strDateTime, dAnalysis=dAnalysis, dProbs=dProbs, dTe=dTe, dParams=dParams)