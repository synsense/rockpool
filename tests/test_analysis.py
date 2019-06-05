"""
Test analysis methods
"""

import numpy as np
from NetworksPython import TSEvent
from NetworksPython.analysis import lv, fano_factor


def test_lv_FF():

    # generate Poisson spike train

    numNeurons = 100
    numSpikes = 1000

    isis = -np.log(np.random.rand(numNeurons, numSpikes))
    spikeTimes = np.array([np.cumsum(isi) for isi in isis])
    nids = np.array([[i] * numSpikes for i in range(numNeurons)])

    # cut to min time
    minTime = np.min(spikeTimes[:, -1])
    nids = np.array([nids[i][train <= minTime] for i, train in enumerate(spikeTimes)])
    spikeTimes = np.array([train[train <= minTime] for train in spikeTimes])

    spikeTimes = np.hstack(spikeTimes)
    nids = np.hstack(nids)

    order = np.argsort(spikeTimes)
    spikeTimes = spikeTimes[order]
    nids = nids[order]

    tse = TSEvent(spikeTimes, nids)

    assert np.abs(lv(tse).all() - 1) < 0.001
    assert np.abs(fano_factor(tse).all() - 1) < 0.001


def test_entropy():

    from NetworksPython import TSEvent
    import numpy as np

    # generate Poisson spike train

    numNeurons = 10
    numSpikes = 100

    isis = -np.log(np.random.rand(numNeurons, numSpikes))
    spikeTimes = np.array([np.cumsum(isi) for isi in isis])
    nids = np.array([[i] * numSpikes for i in range(numNeurons)])

    # cut to min time
    minTime = np.min(spikeTimes[:, -1])
    nids = np.array([nids[i][train <= minTime] for i, train in enumerate(spikeTimes)])
    spikeTimes = np.array([train[train <= minTime] for train in spikeTimes])

    spikeTimes = np.hstack(spikeTimes)
    nids = np.hstack(nids)

    order = np.argsort(spikeTimes)
    spikeTimes = spikeTimes[order]
    nids = nids[order]

    tse = TSEvent(spikeTimes, nids)

    assert np.abs(lv(tse).all() - 1) < 0.001
    assert np.abs(fano_factor(tse).all() - 1) < 0.001
