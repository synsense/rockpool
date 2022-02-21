"""
Test weigh access and indexing for CNNWeight class
"""
import numpy as np


def test_import():
    """
    Test import of the class
    """


def test_pt_events():
    """Test PassThroughEvents layer"""
    from rockpool.nn.layers import PassThroughEvents
    from rockpool import TSEvent

    # - Input signal
    time_trace = [0.1, 0.2, 0.7, 0.8, 0.9]
    channels = [1, 2, 0, 1, 1]
    ts_input = TSEvent(time_trace, channels, t_stop=1.5)

    # - Layer
    lpt = PassThroughEvents(np.array([[0, 2], [1, 1], [0, 0]]), dt=0.4)

    # - Evolution
    ts_out, state, rec = lpt.evolve(ts_input)

    assert (
        ts_out.channels == np.array([0, 1, 1, 1, 0, 1, 0, 1])
    ).all(), "Output channels incorrect"

    assert lpt._timestep == 3
    assert lpt._module._timestep == 3

    lpt.reset_all()

    assert lpt._timestep == 0
    assert lpt._module._timestep == 0


# This is a convenience function, not a test function
def poisson_generator(rate, t_start=0.0, t_stop=1000.0, refractory=0):
    """
    Returns a SpikeTrain whose spikes are a realization of a Poisson process
    with the given rate (Hz) and stopping time t_stop (milliseconds).
    Note: t_start is always 0.0, thus all realizations are as if
    they spiked at t=0.0, though this spike is not included in the SpikeList.
    Inputs:
        rate    - the rate of the discharge (in Hz)
        t_start - the beginning of the SpikeTrain (in ms)
        t_stop  - the end of the SpikeTrain (in ms)
    """
    n = (t_stop - t_start) / 1000.0 * rate
    number = int(np.ceil(n + 3 * np.sqrt(n)))
    if number < 100:
        number = min(5 + int(np.ceil(2 * n)), 100)
    if number > 0:
        isi = np.random.exponential(1.0 / rate, number) * 1000.0

        if number > 1:
            spikes = np.add.accumulate(isi)
        else:
            spikes = isi
    else:
        spikes = np.array([])
    spikes += t_start
    i = np.searchsorted(spikes, t_stop)
    extra_spikes = []
    if i == len(spikes):
        # ISI buf overrun
        t_last = spikes[-1] + np.random.exponential(1.0 / rate, 1)[0] * 1000.0
        while t_last < t_stop:
            extra_spikes.append(t_last)
            t_last += np.random.exponential(1.0 / rate, 1)[0] * 1000.0

        spikes = np.concatenate((spikes, extra_spikes))
    else:
        spikes = np.resize(spikes, (i,))

    return spikes
