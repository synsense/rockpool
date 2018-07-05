import numpy as np
import traceback
from brian2 import second
from matplotlib import pyplot as plt

plt.ion()

# Multitrheading
from multiprocessing import Pool

nNumThreads = 4
nMaxTasksPerChild = 50

import sys

sys.path.insert(0, "/home/felix/gitlab/network-architectures/NetworksPython")

# NetworksPython package
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


# - Deprecated!!
def detailled_exceptions(func):
    """
    detailled_exceptions - Decorate functions that are passed to
                           multiprocessing.Pool with this to get
                           more detailled exceptions
    :param:     function to be passed to Pool
    :return:    function that allows for detailled exceptions
    """

    def f(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            print("")
            raise e

    return f


### --- Signal generation


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
    vfTarget = mfTarget.flatten()

    # - Time base
    tDt = kwargs["tDt"]
    vtTime = np.arange(0, len(vfInput) * tDt, tDt)[: len(vfInput)]

    # - Genrate time series
    tsInput = ts.TimeSeries(vtTime, vfInput)
    tsTarget = ts.TimeSeries(vtTime, vfTarget)

    return tsInput, tsTarget


# @profile
def gen_signal_lists(nReps: int, nRhythms: int, **kwSignal) -> (list, list):
    """
    gen_signal_lists - Genrate multiple ecg inputs and targets. Return them in lists.
    :param nReps:       int Number of ecg inputs and targets
    :param nRhythms:    int Number of ecg rhythms per input
    :kwargs kwSignal:   kwargs to be forwarded to ecg generation
    :return 
        ltsIn:          list of nReps input time series
        ltsTg:          list of nReps target time series
    """

    ltsIn = []
    ltsTg = []

    for i in range(nReps):
        tsIn, tsTg = ts_ecg_target(nRhythms, **kwSignal)
        ltsIn.append(tsIn)
        ltsTg.append(tsTg)

    return ltsIn, ltsTg


### --- Network generation
# @profile
def gen_network(dParams: dict) -> nw.Network:
    """
    gen_network - Generate network with given parameters
    """
    # - Create network with given parameters
    # Weight matrices
    tDt = dParams["tDt"]
    nDimIn = dParams["nDimIn"]
    nDimOut = dParams["nDimOut"]

    mfW_in = dParams.get(
        "mfW_in", 2 * (np.random.rand(nDimIn, dParams["nResSize"]) - 0.5)
    )
    mfW_res = dParams.get("mfW_res", IAFSparseNet(**dParams["kwResWeights"]))

    # - Generate layers
    flIn = PassThrough(mfW=mfW_in, tDt=tDt, tDelay=0, strName="input")
    # rlRes = rec.RecRateEuler(mfW=mfW_res, vtTau=tTau, tDt=tDt, strName='res')
    # flOut = ff.PassThrough(mfW=np.zeros((nResSize, nDimOut)), tDt=tDt, tDelay=0, strName='out')
    rlRes = Rec(
        mfW=mfW_res,
        vtTauN=dParams["tTauN"],
        vtTauSynR=dParams["tTauS"],
        tDt=tDt * second,
        strName="reservoir",
    )
    flOut = FFsc(
        mfW=np.zeros((dParams["nResSize"], nDimOut)),
        tTauSyn=dParams["tTauO"],
        tDt=tDt,
        strName="output",
    )

    # - Generate and return network
    return nw.Network(flIn, rlRes, flOut)


### --- Training procedure
# @profile
def single_run(
    net: nw.Network,
    tDurBatch: float,
    fRegularize: float,
    fHeartRate: float,
    tDt: float,
    tsInTr: ts.TimeSeries,
    tsTgTr: ts.TimeSeries,
    tsInVa: ts.TimeSeries,
    tsTgVa: ts.TimeSeries,
    tsInTe: ts.TimeSeries,
    tsTgTe: ts.TimeSeries,
) -> dict:
    """
    single_run - Perform a single run of training a network, finding a
                 suitable threshold and then testing the network.
                 Return a dict with results from test analysis and the
                 threshold.
    :param net:         Network to be trained and tested
    :param fRegularize: float Regularization parameter for ridge regression
    :param fHeartRate:  float Mean heart rate of the input signals
    :param tDt:         length of single time step
    :param tsInTr:      TimeSeries containing training input
    :param tsTgTr:      TimeSeries containing training target
    :param tsInVa:      TimeSeries containing validation input (for threshold)
    :param tsTgVa:      TimeSeries containing training target
    :param tsInTe:      TimeSeries containing testing input
    :param tsTgTe:      TimeSeries containing training target

    :return:            dict containing threshold and data from test analysis
    """

    def cTrain(net: nw.Network, dtsSignal: dict, bFirst: bool, bFinal: bool):
        """
        cTrain - Train layer flOut with input from dtsSignal and tsTgTr as
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
        # - Coose suitable time range for target
        tsTarget = tsTgTr.resample(tsInput.vtTimeTrace)
        # - Train the layer
        net.output.train_rr(tsTarget, tsInput, fRegularize, bFirst, bFinal)

    ### --- Train
    net.reset_all()
    net.train(cTrain, tsInTr, tDurBatch=tDurBatch, bVerbose=False)
    net.reset_all()

    # - Print clues to indicate progress
    print(".", end="")

    ### --- Validation run: Find threshold  for analysis of test run

    # - Network evolution
    dVa = net.evolve(tsInVa, bVerbose=False)
    net.reset_all()

    # - Output from validation run
    vOutVa = dVa["output"].mfSamples
    # - Determine threshold
    fThr = an.find_threshold(
        vOutput=vOutVa,
        vTarget=tsTgVa.mfSamples,
        nWindow=int(fHeartRate / tDt),
        nClose=int(fHeartRate / tDt),
        nAttempts=16,
        nRecursions=4,
    )

    print(".", end="")

    ### --- Test run

    # - Network evolution
    dTe = net.evolve(tsInTe, bVerbose=False)
    net.reset_all()

    # - Analysis and plotting
    # Output TimeSeries
    tsOutTe = dTe["output"]
    # Output data
    vOutTe = dTe["output"].mfSamples
    dResults = an.analyze(
        vOutTe,
        tsTgTe.mfSamples,
        tsInTe.mfSamples,
        fThr,
        nWindow=int(fHeartRate / tDt),
        bPlot=False,
        bVerbose=False,
    )
    dResults["fThr"] = fThr

    print("|", end="")

    return dResults


# @profile
# @detailled_exceptions
def run_param_set(
    dParams: dict,
    tDurBatch: float,
    fRegularize: float,
    fHeartRate: float,
    tDt: float,
    ltsTrainIn: list,
    ltsTrainTg: list,
    ltsValidIn: list,
    ltsValidTg: list,
    ltsTestIn: list,
    ltsTestTg: list,
) -> dict:
    """
    run_param_set - Run multiple simulations for a network described by
                    a given set of parameters. Number of runs is
                    determined by length of lists containing TimeSeries.
                    Calculate mean speceficity, sensitivity and error ratio
                    over all repetitions.
    :param dParams:     dict Set of parameters for network generation
    :param tDurBatch:   float Duration of signle training batch
    :param fRegularize: float Regularization parameter for ridge regression
    :param fHeartRate:  float Mean heart rate of the input signals
    :param tDt:         length of single time step
    :param ltsTrainIn:  list of TimeSeries containing training inputs
    :param ltsTrainTg:  list of TimeSeries containing training targets
    :param ltsValidIn:  list of TimeSeries containing validation inputs (for threshold)
    :param ltsValidTg:  list of TimeSeries containing training targets
    :param ltsTestIn:   list of TimeSeries containing testing inputs
    :param ltsTestTg:   list of TimeSeries containing training targets
                        All list must contain the same number of TimeSeries.
                        This number corresponds to the number of runs that are simulated.

    :return:            dict containing mean speceficity, sensitivity and error ratio
                        over all repetitions
    """

    # - Network generation
    net = gen_network(dParams)

    # - Count errors and anomalies
    lnTrueNeg = []  # Number of correctly identified normal intervals
    lnTruePos = []  # Number correctly detected anomalies
    lnFalseDetects = []  # Number of normal intervals with at least one wrong detection
    lnFalseNeg = []  # Number of missed anomalies

    # - Iterate over signals
    for tupSignals in zip(
        ltsTrainIn, ltsTrainTg, ltsValidIn, ltsValidTg, ltsTestIn, ltsTestTg
    ):
        # - Simulation
        dResults = single_run(net, tDurBatch, fRegularize, fHeartRate, tDt, *tupSignals)
        # - Results
        lnTrueNeg.append(dResults["nTrueNeg"])
        lnTruePos.append(dResults["nTruePos"])
        lnFalseNeg.append(dResults["nFalseNeg"])
        lnFalseDetects.append(dResults["nFalseDetectIntervals"])

    # - Conversion to arrays
    vnTrueNeg, vnTruePos, vnFalseDetects, vnFalseNeg = (
        np.array(ln) for ln in (lnTrueNeg, lnTruePos, lnFalseDetects, lnFalseNeg)
    )

    # - Evaluate values
    vnNumNormal = vnTrueNeg + vnFalseDetects
    vnNumAnomalies = vnTruePos + vnFalseNeg

    vfSpecificities = vnTrueNeg / vnNumNormal
    vfSensitivities = vnTruePos / vnNumAnomalies
    vfErrorRatios = (vnFalseNeg + vnFalseDetects) / (vnNumNormal + vnNumAnomalies)

    # Weights according to numbers of samples
    vfSpecificities *= vnNumNormal / np.mean(vnNumNormal)
    vfSensitivities *= vnNumAnomalies / np.mean(vnNumAnomalies)
    vfErrorRatios *= (vnNumNormal + vnNumAnomalies) / np.mean(
        vnNumNormal + vnNumAnomalies
    )

    # - Return dict with evaluated values
    return {
        "fSensitivity": np.mean(vfSensitivities),
        "fSensStd": np.std(vfSensitivities, ddof=1),
        "fSpecificity": np.mean(vfSpecificities),
        "fSpecifStd": np.std(vfSpecificities, ddof=1),
        "fErrorRatio": np.mean(vfErrorRatios),
        "fErrorStd": np.std(vfErrorRatios, ddof=1),
    }
