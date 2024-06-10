"""
Test analysis methods
"""


def test_lv_FF():
    import numpy as np
    from rockpool import TSEvent
    from rockpool.analysis import lv, fano_factor

    # generate Poisson spike train

    numNeurons = 100
    numSpikes = 1000

    inter_spk_intvls = -np.log(np.random.rand(numNeurons, numSpikes))
    spikeTimes = np.array([np.cumsum(isi) for isi in inter_spk_intvls])
    nids = np.array([[i] * numSpikes for i in range(numNeurons)])

    # cut to min time
    minTime = np.min(spikeTimes[:, -1])
    nids = np.concatenate(
        [nids[i][train <= minTime] for i, train in enumerate(spikeTimes)]
    )
    spikeTimes = np.concatenate([train[train <= minTime] for train in spikeTimes])

    order = np.argsort(spikeTimes)
    spikeTimes = spikeTimes[order]
    nids = nids[order]

    tse = TSEvent(spikeTimes, nids, t_stop=spikeTimes[-1] + 0.001)

    assert np.abs(lv(tse).all() - 1) < 0.001
    assert np.abs(fano_factor(tse).all() - 1) < 0.001


def test_entropy():
    import numpy as np
    from rockpool import TSEvent
    from rockpool.analysis import lv, fano_factor

    # generate Poisson spike train

    numNeurons = 10
    numSpikes = 100

    inter_spk_intvls = -np.log(np.random.rand(numNeurons, numSpikes))
    spikeTimes = np.array([np.cumsum(isi) for isi in inter_spk_intvls])
    nids = np.array([[i] * numSpikes for i in range(numNeurons)])

    # cut to min time
    minTime = np.min(spikeTimes[:, -1])
    nids = np.concatenate(
        [nids[i][train <= minTime] for i, train in enumerate(spikeTimes)]
    )
    spikeTimes = np.concatenate([train[train <= minTime] for train in spikeTimes])

    order = np.argsort(spikeTimes)
    spikeTimes = spikeTimes[order]
    nids = nids[order]

    tse = TSEvent(spikeTimes, nids, t_stop=spikeTimes[-1] + 0.001)

    assert np.abs(lv(tse).all() - 1) < 0.001
    assert np.abs(fano_factor(tse).all() - 1) < 0.001
