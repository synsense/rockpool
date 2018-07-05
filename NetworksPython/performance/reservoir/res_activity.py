import numpy as np
from brian2 import second, amp  ##
from matplotlib import pyplot as plt
import seaborn as sb

plt.ion()

# - Local imports
import TimeSeries as ts
from ecg import signal_and_target
import analysis as an
import network as nw

# - Layers
from layers.feedforward.rate import PassThrough
from layers.recurrent.iaf_brian import RecIAFBrian
from layers.recurrent.weights import IAFSparseNet

### --- Set parameters

tDt = 0.005  # Length of time step in seconds
fHeartRate = 1  # Heart rate in rhythms per second

nTrials = 50  # Number ECG rhythms for Training
nDimIn = 1  # Input dimensionsns

nResSize = 64  # Reservoir size  ##
fConnectivity = 0.4
fMeanScale = -9e-6
fStdScale = 11e-6

tTauN = 35 * tDt  # Reservoir neuron time constant  ##
tTauS = 350 * tDt  # Reservoir synapse time constant  ##

fBiasMin = 0 * amp
fBiasMax = 0.02 * amp
# vfBias = (fBiasMax - fBiasMin) * np.random.rand(nResSize) + fBiasMin
vfBias = 0.0105*amp

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
}

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
    vfTarget1D = mfTarget.any(axis=1)
    # - Time base
    tDt = kwargs["tDt"]
    vtTime = np.arange(0, vfInput.size * tDt, tDt)[: len(vfInput)]

    # - Genrate time series
    tsInput = ts.TimeSeries(vtTime, vfInput)
    tsTarget = ts.TimeSeries(vtTime, vfTarget1D)

    return tsInput, tsTarget, mfTarget


### --- Network generation

# - Generate weight matrices
mfW_in = 2 * (np.random.rand(nDimIn, nResSize) - 0.5)
# mfW_in = np.random.rand(nDimIn, nResSize)
# mfW_res = RndmSparseEINet(**kwResWeights)
mfW_res = IAFSparseNet(**kwResWeights)  ##

# - Generate layers
flIn = PassThrough(mfW=mfW_in, tDt=tDt, tDelay=0, strName="input")
rlRes = RecIAFBrian(
    mfW=mfW_res, vtTauN=tTauN, vfBias=vfBias, vtTauSynR=tTauS, tDt=tDt * second, strName="reservoir"
)
# - Generate network
net = nw.Network(flIn, rlRes)


if __name__ == "__main__":
    
    tsIn, tsTgt, __ = ts_ecg_target(nTrials, **kwSignal)
    # - Simulation
    dRun = net.evolve(tsIn)

    # - Plotting
    an.plot_activities_2d(dRun['reservoir'], 350*tDt, tDt)
    plt.plot(tsIn.vtTimeTrace, tsIn.mfSamples/10, color='grey', lw=2, zorder=-1)