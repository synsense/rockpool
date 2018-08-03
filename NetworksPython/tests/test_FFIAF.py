'''
Test spiking feedforward layers from layers.feedforward.iaf_brian
'''

import numpy as np
import sys
import os.path
strNetworkPath = os.path.abspath(sys.path[0] + "/../..")
sys.path.insert(1, strNetworkPath)

from brian2 import second

from NetworksPython import timeseries as ts
from NetworksPython.layers.feedforward import iaf_brian

# - Generic parameters
mfW = 2*np.random.rand(2,3)-1
vfBias = 2*np.random.rand(3)-1
vtTauN = np.random.rand(3)

# - Test FFIAFBrian
fl0 = iaf_brian.FFIAFBrian(
    mfW = mfW,
    vfBias = vfBias,
    vtTauN = vtTauN,
    fNoiseStd =0.1,
    tRefractoryTime = 0.001 * second,
)

tsInCont = ts.TSContinuous(
    vtTimeTrace = np.arange(15) * 0.01,
    mfSamples = np.ones((15,2))
)

vStateBefore = np.copy(fl0.vState)
fl0.evolve(tsInCont, tDuration=0.1)
assert fl0.t == 0.1
assert (vStateBefore != fl0.vState).any()

fl0.reset_all()
assert fl0.t == 0
assert (vStateBefore == fl0.vState).all()

# - Test FFIAFSpkInBrian
fl1 = iaf_brian.FFIAFBrian(
    mfW = mfW,
    vfBias = vfBias,
    vtTauN = vtTauN,
    fNoiseStd =0.1,
    tRefractoryTime = 0.001
)

tsInEvt = ts.TSEvent(
    vtTimeTrace = [0.02, 0.04, 0.04, 0.06, 0.12],
    vnChannels = [1, 0, 1, 1, 0]
)

vStateBefore = np.copy(fl1.vState)
fl1.evolve(tsInEvt, tDuration=0.1)
assert fl1.t == 0.1
assert (vStateBefore != fl1.vState).any()

fl1.reset_all()
assert fl1.t == 0
assert (vStateBefore == fl1.vState).all()
