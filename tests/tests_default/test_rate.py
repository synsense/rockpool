"""
Test rate-based Euler models in rate.py
"""

import numpy as np
import pytest


def test_imports():
    pass


def test_FFRateEuler():
    """ Test FFRateEuler """
    from rockpool import TSContinuous
    from nn.layers import FFRateEuler

    # - Generic parameters
    weights = 2 * np.random.rand(2, 3) - 1
    bias = 2 * np.random.rand(3) - 1

    # - Layer generation
    fl0 = FFRateEuler(weights=weights, bias=bias, noise_std=0.1, dt=0.01)

    # - Input signal
    tsInCont = TSContinuous(times=np.arange(15) * 0.01, samples=np.ones((15, 2)))

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.state)
    fl0.evolve(tsInCont, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state).all()

    # - Test that some errors are caught
    with pytest.raises(ValueError):
        fl0.weights = weights[:-1]

    with pytest.raises(TypeError):
        fl1 = FFRateEuler(weights=None)

    with pytest.raises(AssertionError):
        fl1 = FFRateEuler(weights=1, bias=[1, 1])

    with pytest.raises(AssertionError):
        fl1 = FFRateEuler(weights=1, tau=[1, 1])

    with pytest.raises(AssertionError):
        fl1 = FFRateEuler(weights=1, gain=[1, 1])


def test_ff_rate_euler_train():
    """Test ridge regression for FFRateEuler"""
    from rockpool import TSContinuous, TSEvent
    from nn.layers import FFRateEuler, FFExpSyn, PassThrough

    # - Layers
    size_in = 6
    size = 3
    dt = 0.001
    tau_syn = 0.15

    # - FFExpSyn layer to filter spike trains
    fl_exp_prepare = FFExpSyn(np.eye(size_in), dt=dt, tau_syn=tau_syn)

    # - Spiking input signal

    tDur = 0.01
    nSpikes = 5

    vnC = np.tile(np.arange(size_in), int(np.ceil(1.0 / nSpikes * size)))[:nSpikes]
    vtT = np.linspace(0, tDur, nSpikes, endpoint=False)
    tsIn = TSEvent(vtT, vnC, num_channels=size_in, t_stop=tDur)

    # - Filter signal
    ts_filtered = fl_exp_prepare.evolve(tsIn)

    # - Another FFExpSyn layer to compare training
    fl_exp_train = FFExpSyn(np.zeros((size_in, size)), dt=dt, tau_syn=tau_syn)
    # - Rate layers to be trained
    fl_rate = FFRateEuler(np.zeros((size_in, size)), dt=dt, noise_std=0)
    fl_pt = PassThrough(np.zeros((size_in, size)), dt=dt, noise_std=0)

    # - Target and training
    tgt_samples = np.array(
        [
            np.sin(np.linspace(0, 10 * tDur, int(tDur / dt)) + fPhase)
            for fPhase in np.linspace(0, 3, size)
        ]
    ).T
    ts_tgt = TSContinuous(np.arange(int(tDur / dt)) * dt, tgt_samples)

    fl_exp_train.train_rr(ts_tgt, tsIn, regularize=0.1, is_first=True, is_last=True)
    fl_rate.train_rr(ts_tgt, ts_filtered, regularize=0.1, is_first=True, is_last=True)
    fl_pt.train_rr(ts_tgt, ts_filtered, regularize=0.1, is_first=True, is_last=True)

    expected_weights = np.array(
        [
            [0.01538593, 0.08080652, -0.00395387],
            [0.02217629, 0.00562184, -0.02138095],
            [0.02273271, 0.00175698, -0.02248414],
            [0.02270259, 0.00110985, -0.02254557],
            [0.01615033, 0.00050189, -0.01607933],
            [0.0, 0.0, 0.0],
        ]
    )
    expected_bias = np.array([0.00163805, 0.91420448, 0.12769848])

    for lyr in (fl_exp_train, fl_rate, fl_pt):
        assert (np.isclose(lyr.weights, expected_weights, rtol=1e-4, atol=1e-2)).all()
        assert (np.isclose(lyr.bias, expected_bias, rtol=1e-4, atol=1e-2)).all()

    # - Train with z-score standardization
    fl_exp_train.train_rr(
        ts_tgt, tsIn, regularize=0.1, is_first=True, is_last=True, standardize=True
    )

    # - Train with Fisher relabeling
    fl_exp_train.train_rr(
        ts_tgt,
        tsIn,
        regularize=0.1,
        is_first=True,
        is_last=True,
        fisher_relabelling=True,
    )

    # - Train with Fisher relabeling, z-score standardization but without biases
    biases_before = fl_exp_train.bias.copy()
    fl_exp_train.train_rr(
        ts_tgt,
        tsIn,
        regularize=0.1,
        is_first=True,
        is_last=True,
        fisher_relabelling=True,
        standardize=True,
        train_biases=False,
    )
    assert (biases_before == fl_exp_train.bias).all(), "Biases should not have changed."


def test_RecRateEuler():
    """ Test RecRateEuler """
    from rockpool import TSContinuous
    from nn.layers import RecRateEuler

    # - Generic parameters
    weights = 2 * np.random.rand(2, 2) - 1
    bias = 2 * np.random.rand(2) - 1

    # - Layer generation
    fl0 = RecRateEuler(weights=weights, bias=bias, noise_std=0.1, dt=0.01)

    assert fl0.dt == 0.01

    # - Input signal
    tsInCont = TSContinuous(times=np.arange(15) * 0.01, samples=np.ones((15, 2)))

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.state)
    fl0.evolve(tsInCont, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state).all()

    # - Test that some errors are caught
    with pytest.raises(ValueError):
        fl1 = RecRateEuler(weights=np.zeros((1, 2)))

    with pytest.raises(TypeError):
        RecRateEuler(weights=np.zeros((2, 2)), tau=None)

    with pytest.raises(TypeError):
        RecRateEuler(weights=np.zeros((2, 2)), noise_std=None)
