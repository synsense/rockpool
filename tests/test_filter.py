"""
Test filtering layers from layers.internal.filter_layer
"""

import numpy as np
import time
import pylab as plt

from hyperopt import hp
from hyperopt import fmin, tpe, space_eval, Trials
from NetworksPython.timeseries import SetPlottingBackend

SetPlottingBackend("plt")



def test_chargeSingleNeuron():
    """
    single neuron test
    charge neuron exactly to threshold without crossing using the bias
    """

    from NetworksPython.layers import Filter
    from NetworksPython.timeseries import TSContinuous

    fs = 1000
    filterName = "butter"
    mfW = np.ones((1,5))
    tsInp = TSContinuous(np.arange(1000) /fs, np.random.rand(1000))


    fl0 = Filter(
        mfW=mfW,
        tDt=1/fs,
        filterName=filterName,
        fs=fs,
        strName="test"
    )

    dFl0 = fl0.evolve(tsInp)

    assert(fl0.nNumTraces == 5)
    assert(dFl0.mfSamples.mean() != 0)
