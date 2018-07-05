import numpy as np
from brian2 import second, amp  ##
from matplotlib import pyplot as plt
import seaborn as sb

plt.ion()

import sys 
sys.path.insert(1, '../..')

# - Local imports
from NetworksPython import TimeSeries
from NetworksPython.ecg import signal_and_target
import NetworksPython.analysis as an
import NetworksPython.network as nw

# - Layers
from NetworksPython.layers.feedforward.rate import PassThrough  ##
from NetworksPython.layers.recurrent.iaf_brian import RecIAFBrian  ##
from NetworksPython.layers.feedforward.exp_synapses_manual import FFExpSyn  ##
from NetworksPython.layers.recurrent.weights import IAFSparseNet  ##


### --- Set parameters

tDt = 0.005  # Length of time step in seconds
fHeartRate = 1  # Heart rate in rhythms per second

nTrialsTr = 1000  # Number ECG rhythms for Training
nTrialsVa = 500  # Number ECG rhythms for validation
nRepsVa = 4  # Number repetitions of validation runs
nTrialsTe = 1000  # Number ECG rhythms for testing

fStdNoiseSignal = 0.05  # Standard deviation of input noise

nDimIn = 1  # Input dimensions
nDimOut = 1  # Output dimensions

nResSize = 512  # Reservoir size  ##
fConnectivity = 0.4  # Percentage of non-zero recurrent weights  ##
fMeanScale = -9e-6  # Scale mean of recurrent weights  ##
fStdScale = 11e-6  # Scale standard dev. of recurrent weights  ##

tTauN = 35 * tDt  # Reservoir neuron time constant  ##
tTauS = 350 * tDt  # Reservoir synapse time constant  ##
tTauO = 35 * tDt  # Readout time constant  ##

fBiasMin = 0 * amp
fBiasMax = 0.02 * amp
# vfBias = (fBiasMax - fBiasMin) * np.random.rand(nResSize) + fBiasMin
vfBias = 0.0105*amp

tDurBatch = 500  # Training batch duration
fRegularize = 0.001  # Regularization parameter for training with ridge regression

# Parameters concerning reservoir weights
kwResWeights = {
    "nResSize": nResSize,
    "fDensity": fConnectivity,
    "fMean" : fMeanScale / fConnectivity / nResSize,
    "fStd" : fStdScale / np.sqrt(fConnectivity),
    # "fConnectivity": 0.4,  # Connectivity
    # "bPartitioned": False,  # Partition reservoir into excitatory/inhibitory
    # "fRatioExc": 0.5,  # Ratio of excitatory neurons
    # "fScaleInh": 1,  # Scale of inhibitory vs excitatory weights
    # "fNormalize": 0.5,
}  # Normalize matrix spectral radius

# Probabilities for anomalies in ECG rhythms
pNormal = 0.8  # Probability of normal input rhythm
pAnomal = (1 - pNormal) / 6.  # Probability of abnormal input rhythm

dProbs = {
    "complete_normal": pNormal,  # Normal ECG
    "complete_noP": pAnomal,  # Missing P-wave
    "complete_Pinv": pAnomal,  # Inverted P-wave
    "complete_noQRS": pAnomal,  # Missing QRS complex
    "complete_Tinv": pAnomal,  # Inverted T-wave
    "complete_STelev": pAnomal,  # Elevated ST-segment
    "complete_STdepr": pAnomal,  # Depressed ST-segment
    # "complete_tach": pAnomal,     # Tachycardia
    # "complete_brad": pAnomal,     # Bradycardia
}

# - Kwargs for signal_and_target function
kwSignal = {
    "strTargetMethod": "segment-extd",  # Method for labelling targets
    "dProbs": dProbs,
    "fHeartRate": fHeartRate,
    "tDt": tDt,
    "bVerbose": True,
    "nMinWidth": int(0.5 * fHeartRate / tDt),
    "bDetailled": True,
    "fStdNoise" : fStdNoiseSignal,
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
    # - Infer time range of current batch
    tStart = dtsSignal["external"].tStart
    tStop = dtsSignal["external"].tStop
    # - Sample target within time range of current batch
    tsTarget = tsTgtTr.resample_within(tStart, tStop)
    # - Train the layer
    flOut.train_rr(tsTarget, tsInput, fRegularize, bFirst, bFinal)


def ts_ecg_target(nRhythms: int, **kwargs) -> (TimeSeries, TimeSeries):
    """
    ts_ecg_target - Generate two time series, one containing an ECG signal
                   and the other the corresponding target.
    :param nRhythms:    int Number of ECG rhythms in the input
    :tDt:               float Size of a single time step
    :kwargs:            dict Kwargs that are passed on to signal_and_target
    """

    # - Input signal and target
    vfInput, mfTarget = signal_and_target(nTrials=nRhythms, **kwargs)
    vfTarget1D = mfTarget.any(axis=1)
    # - Time base
    tDt = kwargs["tDt"]
    vtTime = np.arange(0, vfInput.size * tDt, tDt)[: len(vfInput)]

    # - Genrate time series
    tsInput = TimeSeries(vtTime, vfInput)
    tsTarget = TimeSeries(vtTime, vfTarget1D)

    return tsInput, tsTarget, mfTarget


### --- Network generation

# - Generate weight matrices
mfW_in = 2 * (np.random.rand(nDimIn, nResSize) - 0.5)
# mfW_in = np.random.rand(nDimIn, nResSize)
# mfW_res = RndmSparseEINet(**kwResWeights)
mfW_res = IAFSparseNet(**kwResWeights) ##

# - Generate layers
flIn = PassThrough(mfW=mfW_in, tDt=tDt, tDelay=0, strName="input")  ##
rlRes = RecIAFBrian(
    mfW=mfW_res, vtTauN=tTauN, vfBias=vfBias, vtTauSynR=tTauS, tDt=tDt * second, strName="reservoir"
)  ##
flOut = FFExpSyn(
    mfW=np.zeros((nResSize, nDimOut)), tTauSyn=tTauO, tDt=tDt, strName="output"
)  ##

# - Generate network
net = nw.Network(flIn, rlRes, flOut)


if __name__ == "__main__":

    ### --- Training

    # - Training signal
    tsInTr, tsTgtTr, __ = ts_ecg_target(nTrialsTr, **kwSignal)

    # - Run training
    net.train(cTrain, tsInTr, tDurBatch=tDurBatch)
    net.reset_all()

    # # - Plot weight distribution
    # figWeights = plt.figure()
    # plt.suptitle("Readout weights")
    # sb.distplot(flOut.mfW)


    ### --- Validation run for threshold determination

    print("Finding threshold")

    # - Lists for storing thresholds
    lfThr = []
    
    # - Validation runs
    for i in range(nRepsVa):

        print("\tRun {} of {}".format(i + 1, nRepsVa), end='\r')
        # - Signal generation
        tsInVa, tsTgtVa, __ = ts_ecg_target(nTrialsTe, **dict(kwSignal, bVerbose=False))

        # - Validation run
        dVa = net.evolve(tsInVa, bVerbose=False)
        net.reset_all()

        # - Determine threshold for analysis of test run
        lfThr.append(
            an.find_threshold(
                vOutput=dVa[flOut.strName].mfSamples,
                vTarget=tsTgtVa.mfSamples.flatten(),
                nWindow=int(fHeartRate / tDt),
                nClose=int(fHeartRate / tDt),
                nAttempts=16,
                nRecursions=4,
            )
        )

        del dVa

    fThr = np.mean(lfThr)
    print("Using threshold: {:.3f}".format(fThr))


    ### --- Testing

    # - Test signal
    tsInTe, tsTgtTe, mfTgtTe = ts_ecg_target(nTrialsTe, **kwSignal)

    # - Run test
    dTe = net.evolve(tsInTe)
    net.reset_all()

    # - Output TimeSeries
    tsOutTe = dTe[flOut.strName]

    # - Analysis and plotting
    print("\nAnalysis of test run:")
    vOutTe = dTe[flOut.strName].mfSamples
    dAnalysis = an.analyze(
        vOutTe,
        tsTgtTe.mfSamples.flatten(),
        tsInTe.mfSamples,
        fThr,
        nWindow=int(fHeartRate / tDt),
        nClose=int(fHeartRate / tDt),
        bPlot=True,
        mfTarget=mfTgtTe,
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
            print(
                lstrSymptomFullNames[iSymptom]
                + ":" 
                + str(dAnalysis["dSymptoms"][iSymptom]["fSensitivity"])
            )


    """
    ### --- Test anomaly-wise

    # dicts with data and results
    ddTSAnom = dict()
    ddResultsAnom = dict()

    for strAnomaly in (k for k in dProbs.keys() if k != "complete_normal"):
        print("\nTesting " + strAnomaly)

        # Probabilites
        kwSignalAn = kwSignal.copy()
        kwSignalAn["bVerbose"] = False
        kwSignalAn["dProbs"] = {"complete_normal": pNormal, strAnomaly: 1. - pNormal}
        # Signal generation
        tsInAn, tsTgtAn, __ = ts_ecg_target(nTrialsTe, **kwSignalAn)

        # Run test
        ddTSAnom[strAnomaly] = net.evolve(tsInAn, bVerbose=False)
        net.reset_all()

        # Analysis
        vOutAn = ddTSAnom[strAnomaly][flOut.strName].mfSamples
        ddResultsAnom[strAnomaly] = an.analyze(
            vOutAn,
            tsTgtAn.mfSamples.flatten(),
            tsInAn.mfSamples,
            fThr,
            nWindow=int(fHeartRate / tDt),
            nClose=int(fHeartRate / tDt),
            bPlot=False,
        )
        """
