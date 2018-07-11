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
from NetworksPython import timeseries as ts 


tDt = 0.0001 * second

nNumInputSamples = 40
tInputDuration = 0.5

# - Corrected constant parameters
dParamsNeuron = {
    # 'Io' : 1.5e-12 * amp,
    # 'Cmem' : 2e-12 * farad,
    # 'Ispkthr' : 1e-5 * amp,
    # 'Ireset' : 0 * amp,
    # 'Ith' : 500e-9 * amp,
    # 'Iagain' : 10e-12 * amp,
    'Iconst' : 1e-12 * amp,
}

dParamsSynapse = {
    # 'Io_syn' : 1.5e-12 * amp,
    # 'Csyn' : 2e-12 * farad,
    'baseweight_i' : 1e-13 * amp,
    'baseweight_e' : 1e-13 * amp,
}

# - Reservoir generation

nReSize = 8
# fBiasInE = fBiasInI = 1e-8 * amp  # Input baseweight
# fBiasRecE = fBiasRecI = 1e-9 * amp  # Reservoir baseweight     
# fIconst = 1e-6 * amp  # Bias current

# Recurrent weights, normalized by spectral radius
np.random.seed(1)
mfW = 2*np.random.rand(nReSize, nReSize) - 1
mfW /= np.amax(np.abs(np.linalg.eigvals(mfW)))

# Input weights
vfWIn = 2*np.random.rand(nReSize) - 1


# Reservoir
res = db.RecDynapseBrian(mfW, vfWIn, tDt=tDt)

# Update parameters
# res._sgReceiver.baseweight_i = fBiasInI
# res._sgReceiver.baseweight_e = fBiasInE
# res._sgRecurrentSynapses.baseweight_i = fBiasRecI
# res._sgRecurrentSynapses.baseweight_e = fBiasRecE
res._sgReceiver.set_params(dParamsSynapse)
res._sgRecurrentSynapses.set_params(dParamsSynapse)
res._ngLayer.set_params(dParamsNeuron)

# Monitors
stmNg = b2.StateMonitor(res._ngLayer, ['Ie0', 'Ii0', 'Ie1', 'Ii1', 'Imem'], record=True)
res._net.add(stmNg)

# - Input
vtIn = np.sort(np.random.rand(nNumInputSamples)) * tInputDuration
vnChIn = np.random.randint(nReSize, size=nNumInputSamples)
tsIn = ts.TSEvent(vtIn, vnChIn)

# - Run simulation
tsR = res.evolve(tsIn)

# - Plot
fig, axes = plt.subplots(3, figsize=(10,15), sharex=True)
# Continuous variables
for var, axID in zip(('Ie1', 'Ii1', 'Ie0', 'Ii0', 'Imem'), (0,0,1,1,2)):
    axes[axID].plot(stmNg.t/second, getattr(stmNg, var).T/amp)
    axes[axID].set_title(var)

# Preserve axis y-limits
lYlims = [ax.get_ylim() for ax in axes]

# Input spikes
for t in vtIn:
    for ax, ylims in zip((axes[0],), (lYlims[0],)):
        ax.plot([t,t], ylims, 'k--', zorder = -1, alpha=0.5)
# Reservoir spikes
for t in tsR.vtTimeTrace:
    for ax, ylims in zip(axes[1:], lYlims[1:]):
        ax.plot([t,t], ylims, 'b--', zorder = -1, alpha=0.5)
