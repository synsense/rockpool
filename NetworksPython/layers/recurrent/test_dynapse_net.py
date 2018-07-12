import sys
import os.path
strPathToLib = os.path.abspath(sys.path[0] + '../../../..')
sys.path.insert(1, strPathToLib)

from matplotlib import pyplot as plt
plt.ion()

import numpy as np

from brian2 import second, amp, farad
import brian2 as b2

from NetworksPython.layers.recurrent import dynapse_brian as db
from NetworksPython.layers.recurrent.weights import In_Res_Dynapse
from NetworksPython import timeseries as ts
from NetworksPython import analysis as an


### Include code from spiking.py and test_ecg_to_spike.py

### --- Set parameters

tDtRes = 0.0001  # Length of time step in seconds - might only need that small for reservoir...
tDt = 0.005

fHeartRate = 1  # Heart rate in rhythms per second

tDurBatch = 500  # Training batch duration
fRegularize = 0.001  # Regularization parameter for training with ridge regression

# Input data
nTrialsTr = 1000  # Number ECG rhythms for Training
nTrialsVa = 500  # Number ECG rhythms for validation
nRepsVa = 4  # Number repetitions of validation runs
nTrialsTe = 1000  # Number ECG rhythms for testing

fStdNoiseSignal = 0  # Standard deviation of input noise

nDimIn = 1  # Input dimensions
nDimOut = 1  # Output dimensions

# Forward layer
fScaleFF = 0.3

# Reservoir
nResSize = 512  # Reservoir size
fConnectivity = None  # Percentage of non-zero recurrent weights

fIconst = 4.375e-9 * amp
fBaseweightE = 7e-8 * amp
fBaseweightI = 1e-7 * amp




# - Corrected constant parameters
dParamNeuron = {
    'Iconst' : fIconst,
}

dParamSynapse = {
    'baseweight_i' : fBaseweightI,
    'baseweight_e' : fBaseweightE,
}

# - Reservoir generation


# Recurrent weights, normalized by spectral radius
np.random.seed(1)

# - Weight generation
# Forward layer
# Input weights
vfWFF = (2*np.random.rand(nResSize) - 1) * fScaleFF
# Reservoir
vfWIn, mfW, *__ = In_Res_Dynapse(nResSize, tupfWExc=(1,1), tupfWInh=(1,1), fNormalize=1, bLeaveSpaceForInput=True)

# - Network generation
# Forward layer
fl = FFIAFBrian(vfWIn, vfBias=vfBias, vtTauN=vtTau)
# Reservoir
res = db.RecDynapseBrian(mfW, vfWIn, tDt=tDtRes, dParamNeuron=dParamNeuron, dParamSynapse=dParamSynapse)

# Monitors
stmNg = b2.StateMonitor(res._ngLayer, ['Ie0', 'Ii0', 'Ie1', 'Ii1', 'Imem', 'Iin_clip'], record=True)
res._net.add(stmNg)

# - Input
vtIn = np.sort(np.random.rand(nNumInputSamples)) * tInputDuration
vnChIn = np.random.randint(nResSize, size=nNumInputSamples)
tsIn = ts.TSEvent(vtIn, vnChIn)

# - Run simulation
tsR = res.evolve(tsIn)