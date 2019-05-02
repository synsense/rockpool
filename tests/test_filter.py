"""
Test filtering layers from layers.internal.filter_layer
"""

import numpy as np


def test_filter_layer():
    """
    single neuron test
    charge neuron exactly to threshold without crossing using the bias
    """

    from NetworksPython.layers import Filter
    from NetworksPython.timeseries import TSContinuous
    from AuditoryProcessing.preprocessing import butter_mel

    fs = 1000
    filterName = "butter_mel"
    numInputChannels = 5
    mfW = np.ones((1,numInputChannels))
    times = np.arange(1000) /fs
    signal = np.random.rand(1000)
    tsInp = TSContinuous(times, signal)


    fl0 = Filter(
        mfW=mfW,
        tDt=1/fs,
        filterName=filterName,
        fs=fs,
        strName="test"
    )

    dFl0 = fl0.evolve(tsInp)

    filterOutput = butter_mel(signal, fs, numInputChannels, fs, False, 2)
    tsFilt = TSContinuous(times, filterOutput)

    assert np.all(np.isclose(tsFilt.mfSamples, dFl0.mfSamples))
    assert(fl0.nNumTraces == 5)
    assert(dFl0.mfSamples.mean() != 0)
