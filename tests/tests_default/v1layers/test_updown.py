"""
Test FFUpDown layer in updown.py
"""


def test_updown():
    """Test FFUpDown"""
    from rockpool import TSContinuous
    from rockpool.nn.layers import FFUpDown
    import numpy as np

    # - Generic parameters
    weights = np.random.rand(2, 4)

    # - Layer generation
    fl0 = FFUpDown(weights=weights, dt=0.01, thr_down=0.02, thr_up=0.01, tau_decay=0.1)

    # - Check layer properties
    assert fl0.size_out == 4, "Problem with size_out"
    assert fl0.size_in == 2, "Problem with size_in"
    assert (fl0.thr_down == np.array([0.02, 0.02])).all(), "Problem with thr_down"
    assert (fl0.thr_up == np.array([0.01, 0.01])).all(), "Problem with thr_up"

    # - Input signal
    tsInCont = TSContinuous(
        times=np.arange(15) * 0.01,
        samples=np.vstack(
            (np.sin(np.linspace(0, 1, 15)), np.cos(np.linspace(0, 1, 15)))
        ).T,
    )

    # - Compare states and time before and after evolution
    vStateBefore = np.copy(fl0.state()["analog_value"])
    ts0, states, rec = fl0.evolve(tsInCont, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state()["analog_value"]).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state()["analog_value"]).all()

    # - Test repeat output
    fl1 = FFUpDown(
        weights=weights,
        repeat_output=3,
        dt=0.01,
        thr_down=0.02,
        thr_up=0.01,
        tau_decay=0.1,
    )
    assert fl1.size_out == fl0.size_out
    ts1, states, rec = fl1.evolve(tsInCont, duration=0.1)
    assert (ts1.times == np.repeat(ts0.times, 3)).all()
    assert (ts1.channels == np.repeat(ts0.channels, 3)).all()
