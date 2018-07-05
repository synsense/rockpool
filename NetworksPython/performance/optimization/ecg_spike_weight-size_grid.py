from grid_simulation import *


### --- Set parameters

tDt = 0.005  # Length of time step in seconds
fHeartRate = 1  # Heart rate in rhythms per second

nReps = 3
nTrialsTr = 1500  # Number ECG rhythms for training
nTrialsTe = 1000  # Number ECG rhythms for testing

nDimIn = 1  # Input dimensions
nDimOut = 1  # Output dimensions

nResSize = 512  # Reservoir size

tTauN = 35 * tDt
tTauS = 350 * tDt
tTauO = 35 * tDt

tDurBatch = 250  # Training batch duration
fRegularize = 0.001  # Regularization parameter for training with ridge regression

lfNetworkRadius = [0.0075, 0.01, 0.02]  # , Normalize network spectral radius

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

# Probabilities for anomalies in ECG rhythms
pNormal = 0.8  # Probability of normal input rhythm
pAnomal = (1 - pNormal) / 8  # Probability of abnormal input rhythm

dProbs = {
    "complete_normal": pNormal,  # Normal ECG
    "complete_noP": pAnomal,  # Missing P-wave
    "complete_Pinv": pAnomal,  # Inverted P-wave
    "complete_noQRS": pAnomal,  # Missing QRS complex
    "complete_Tinv": pAnomal,  # Inverted T-wave
    "complete_STelev": pAnomal,  # Elevated ST-segment
    "complete_STdepr": pAnomal,  # Depressed ST-segment
    "complete_tach": pAnomal,  # Tachycardia
    "complete_brad": pAnomal,  # Bradycardia
}

# - Kwargs for signal_and_target function
kwSignal = {
    "strTargetMethod": "fix",  # Method for labelling targets
    "dProbs": dProbs,
    "fHeartRate": fHeartRate,
    "tDt": tDt,
    "bVerbose": False,
    "nWidth": int(fHeartRate / tDt),
}

# @profile
def grid_iteration(
    lfNetworkRadius,
    ltsTrainIn,
    ltsTrainTg,
    ltsValidIn,
    ltsValidTg,
    ltsTestIn,
    ltsTestTg,
):
    """
    grid_iteration - Iterate over grid of parameters
    """

    # Tarameters that are not varied
    dParamsOther = {
        "nResSize": nResSize,
        "nDimIn": nDimIn,
        "nDimOut": nDimOut,
        "tDt": tDt,
        "fRegularize": fRegularize,
        "kwResWeights": kwResWeights,
        "tTauN": tTauN,
        "tTauS": tTauS,
        "tTauO": tTauO,
    }

    def gen_norm_weights(fRadius):
        """
        gen_norm_weights - generate weights using IAFSparse net and normalize them
                           so that their spectral radius is fRadius
        """
        mfW = IAFSparseNet(**kwResWeights)
        fCurrentRadius = np.amax(np.abs(np.linalg.eigvals(mfW)))
        return mfW * fRadius / fCurrentRadius

    ldAllParams = [
        dict(mfW_res=gen_norm_weights(fRadius), **dParamsOther)
        for fRadius in lfNetworkRadius
    ]

    nCountFinished = 0
    nSimsTotal = len(ldAllParams)

    def prog_report(*args):
        nonlocal nCountFinished
        nCountFinished += 1
        print(
            "\rFinished {} of {} simulations".format(nCountFinished, nSimsTotal), end=""
        )

    lltsSignals = [ltsTrainIn, ltsTrainTg, ltsValidIn, ltsValidTg, ltsTestIn, ltsTestTg]
    lSimParams = [tDurBatch, fRegularize, fHeartRate, tDt] + lltsSignals

    """
    dRatios = dict()
    for dParams, fRadius in zip(ldAllParams, lfNetworkRadius):
        dRatios[fRadius] = run_param_set(dParams, *lSimParams)
        prog_report()

    """
    # - Perform simulations in different threads and save results in dict
    with Pool(nNumThreads, maxtasksperchild=nMaxTasksPerChild) as poolTrain:
        lpoolRatios = [
            poolTrain.apply_async(
                run_param_set, args=[dParams, *lSimParams], callback=prog_report
            )
            for dParams in ldAllParams
        ]

        dRatios = {
            fNetworkRadius: ratio.get()
            for fNetworkRadius, ratio in zip(lfNetworkRadius, lpoolRatios)
        }

    print("\n\n")

    return dRatios


if __name__ == "__main__":
    # - Signal generation

    print("Generating training signals")
    ltsTrainIn, ltsTrainTg = gen_signal_lists(nReps, nTrialsTr, **kwSignal)
    print("Generating validation signals")
    ltsValidIn, ltsValidTg = gen_signal_lists(nReps, nTrialsTe, **kwSignal)
    print("Generating test signals")
    ltsTestIn, ltsTestTg = gen_signal_lists(nReps, nTrialsTe, **kwSignal)

    print("Performing simulations")
    dAnalysis = grid_iteration(
        lfNetworkRadius,
        ltsTrainIn,
        ltsTrainTg,
        ltsValidIn,
        ltsValidTg,
        ltsTestIn,
        ltsTestTg,
    )
