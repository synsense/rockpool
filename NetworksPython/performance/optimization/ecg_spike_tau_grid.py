from grid_simulation import *


### --- Set parameters

tDt = 0.005  # Length of time step in seconds
fHeartRate = 1  # Heart rate in rhythms per second

nReps = 3
nTrialsTr = 150  # Number ECG rhythms for training
nTrialsTe = 50  # Number ECG rhythms for testing

nDimIn = 1  # Input dimensions
nDimOut = 1  # Output dimensions

nResSize = 512  # Reservoir size

# vtTauN = np.array((10, 20, 40, 80, 160, 320)) * tDt
# vtTauS = np.array((10, 20, 40, 80, 160, 320)) * tDt
# vtTauO = np.array((10, 20, 40, 80, 160, 320)) * tDt
vtTauN = [35 * tDt]
vtTauS = [350 * tDt]
vtTauO = [35 * tDt]

tDurBatch = 250  # Training batch duration
fRegularize = 0.001  # Regularization parameter for training with ridge regression

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
dProbs = {
    "complete_normal": 0.8,  # Normal ECG
    "complete_noP": 0.025,  # Missing P-wave
    "complete_Pinv": 0.025,  # Inverted P-wave
    "complete_noQRS": 0.025,  # Missing QRS complex
    "complete_Tinv": 0.025,  # Inverted T-wave
    "complete_STelev": 0.025,  # Elevated ST-segment
    "complete_STdepr": 0.025,  # Depressed ST-segment
    "complete_tach": 0.025,  # Tachycardia
    "complete_brad": 0.025,  # Bradycardia
}

# - Kwargs for signal_and_target function
kwSignal = {
    "strTargetMethod": "fix",  # Method for labelling targets
    "dProbs": dProbs,
    "fHeartRate": fHeartRate,
    "tDt": tDt,
    "bVerbose": False,
    "nTargetWidth": int(fHeartRate / tDt),
}


def grid_iteration(
    vtTauN,
    vtTauS,
    vtTauO,
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

    lltsSignals = [ltsTrainIn, ltsTrainTg, ltsValidIn, ltsValidTg, ltsTestIn, ltsTestTg]

    # Tarameters that are not varied
    dParamsOther = {
        "kwResWeights": kwResWeights,
        "nResSize": nResSize,
        "nDimIn": nDimIn,
        "nDimOut": nDimOut,
        "tDt": tDt,
    }

    ldAllParams = [
        dict(tTauN=tTauN, tTauS=tTauS, tTauO=tTauO, **dParamsOther)
        for tTauN in vtTauN
        for tTauS in vtTauS
        for tTauO in vtTauO
    ]

    nCountFinished = 0
    nSimsTotal = len(ldAllParams)

    def prog_report(*args):
        nonlocal nCountFinished
        nCountFinished += 1
        print(
            "\rFinished {} of {} simulations".format(nCountFinished, nSimsTotal), end=""
        )

    # - Perform simulations in different threads and save results in dict
    with Pool(nNumThreads, maxtasksperchild=nMaxTasksPerChild) as poolTrain:
        lpoolRatios = [
            poolTrain.apply_async(
                run_param_set,
                args=[dParams, tDurBatch, fRegularize, fHeartRate, tDt, *lltsSignals],
                callback=prog_report,
            )
            for dParams in ldAllParams
        ]

        dRatios = {
            (dParams["tTauN"], dParams["tTauS"], dParams["tTauO"]): ratio.get()
            for dParams, ratio in zip(ldAllParams, lpoolRatios)
        }

    print("\n\n")

    return dRatios


if __name__ == "__main__":
    # - Signal generation

    print("Generating training signals")
    ltsTrainIn, ltsTrainTg = gen_signal_lists(nReps, nTrialsTr, **kwSignal)
    print("Generating Validation signals")
    ltsValidIn, ltsValidTg = gen_signal_lists(nReps, nTrialsTe, **kwSignal)
    print("Generating test signals")
    ltsTestIn, ltsTestTg = gen_signal_lists(nReps, nTrialsTe, **kwSignal)

    print("Performing simulations")
    dAnalysis = grid_iteration(
        vtTauN,
        vtTauS,
        vtTauO,
        ltsTrainIn,
        ltsTrainTg,
        ltsValidIn,
        ltsValidTg,
        ltsTestIn,
        ltsTestTg,
    )
