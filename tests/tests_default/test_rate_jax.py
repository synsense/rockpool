"""
Test rate-based Euler models in rate-jax.py
"""

import numpy as np
import pytest


def test_imports():
    from nn.layers import (
        RecRateEulerJax,
        RecRateEulerJax_IO,
        ForceRateEulerJax_IO,
    )


def test_RecRateEulerJax():
    """ Test RecRateEulerJax """
    from rockpool import TSContinuous
    from nn.layers import RecRateEulerJax

    # - Generic parameters
    w_recurrent = 2 * np.random.rand(2, 2) - 1
    bias = 2 * np.random.rand(2) - 1
    tau = 20e-3 * np.ones(2)
    dt = 0.01

    # - Layer generation
    fl0 = RecRateEulerJax(
        weights=w_recurrent,
        bias=bias,
        noise_std=0.0,
        tau=tau,
        dt=dt,
    )

    # - Input signal
    tsInCont = TSContinuous(times=np.arange(15) * 0.01, samples=np.ones((15, 2)))

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.state)
    ts_output = fl0.evolve(tsInCont, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state).any()

    # - Test functional evolve
    fl0.state = vStateBefore
    ts_out_n, _, _ = fl0._evolve_functional(
        fl0._pack(),
        vStateBefore,
        tsInCont(np.arange(int(0.1 // dt)) * dt),
    )

    assert (ts_output.samples[:10, :] == ts_out_n[:10, :]).all()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state).all()

    # - Test storage
    fl0.to_dict()

    # - Test save and load
    import tempfile

    with tempfile.NamedTemporaryFile() as path_to_save:
        # - Save the layer
        fl0.save_layer(path_to_save.name)

        # - Load the layer
        lyr_loaded = RecRateEulerJax.load_from_file(path_to_save.name)

        # - Compare properties
        props_to_test = [
            "weights",
            "tau",
            "bias",
            "dt",
            "_H",
        ]
        assert all(
            [
                np.array(
                    [getattr(fl0, prop_name) == getattr(lyr_loaded, prop_name)]
                ).all()
                for prop_name in props_to_test
            ]
        )

    # - Test that some errors are caught
    with pytest.raises(ValueError):
        fl1 = RecRateEulerJax(
            weights=np.zeros((3, 2)),
            tau=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(ValueError):
        fl1 = RecRateEulerJax(
            weights=np.zeros((2, 2)),
            tau=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(ValueError):
        fl1 = RecRateEulerJax(
            weights=np.zeros((2, 2)),
            tau=np.zeros(2),
            bias=np.zeros(3),
        )

    with pytest.raises(ValueError):
        fl1 = RecRateEulerJax(
            weights=np.zeros((2, 2)),
            tau=np.zeros(2),
            bias=np.zeros(2),
            activation_func="blah",
        )


def test_RecRateEulerJax_IO():
    """ Test RecRateEulerJax_IO """
    from rockpool import TSContinuous
    from nn.layers import RecRateEulerJax_IO

    # - Generic parameters
    w_in = 2 * np.random.rand(1, 2) - 1
    w_recurrent = 2 * np.random.rand(2, 2) - 1
    w_out = 2 * np.random.rand(2, 1) - 1
    bias = 2 * np.random.rand(2) - 1
    tau = 20e-3 * np.ones(2)
    dt = 0.01

    # - Layer generation
    fl0 = RecRateEulerJax_IO(
        w_in=w_in,
        w_recurrent=w_recurrent,
        w_out=w_out,
        bias=bias,
        noise_std=0.0,
        tau=tau,
        dt=dt,
    )

    # - Input signal
    tsInCont = TSContinuous(times=np.arange(15) * dt, samples=np.ones((15, 1)))

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.state)
    ts_output = fl0.evolve(tsInCont, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state).all()

    # - Test functional evolve
    fl0.state = vStateBefore
    ts_out_n, _, _ = fl0._evolve_functional(
        fl0._pack(),
        vStateBefore,
        tsInCont(np.arange(int(0.1 // dt)) * dt),
    )

    assert (ts_output.samples[:10, :] == ts_out_n[:10, :]).all()

    # - Test storage
    fl0.to_dict()

    # - Test save and load
    import tempfile

    with tempfile.NamedTemporaryFile() as path_to_save:
        # - Save the layer
        fl0.save_layer(path_to_save.name)

        # - Load the layer
        lyr_loaded = RecRateEulerJax_IO.load_from_file(path_to_save.name)

        # - Compare properties
        props_to_test = [
            "w_in",
            "w_recurrent",
            "w_out",
            "tau",
            "bias",
            "dt",
            "_H",
        ]
        assert all(
            [
                np.array(
                    [getattr(fl0, prop_name) == getattr(lyr_loaded, prop_name)]
                ).all()
                for prop_name in props_to_test
            ]
        )

    # - Test that some errors are caught
    with pytest.raises(ValueError):
        fl1 = RecRateEulerJax_IO(
            w_in=np.zeros((1, 2)),
            w_recurrent=np.zeros((3, 2)),
            w_out=np.zeros((3, 1)),
            tau=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(ValueError):
        fl1 = RecRateEulerJax_IO(
            w_in=np.zeros((1, 2)),
            w_recurrent=np.zeros((2, 2)),
            w_out=np.zeros((3, 1)),
            tau=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(ValueError):
        fl1 = RecRateEulerJax_IO(
            w_in=np.zeros((1, 2)),
            w_recurrent=np.zeros((2, 2)),
            w_out=np.zeros((2, 1)),
            tau=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(ValueError):
        fl1 = RecRateEulerJax_IO(
            w_in=np.zeros((1, 2)),
            w_recurrent=np.zeros((2, 2)),
            w_out=np.zeros((2, 1)),
            tau=np.zeros(2),
            bias=np.zeros(3),
        )

    with pytest.raises(ValueError):
        fl1 = RecRateEulerJax_IO(
            w_in=np.zeros((1, 2)),
            w_recurrent=np.zeros((2, 2)),
            w_out=np.zeros((2, 1)),
            tau=np.zeros(2),
            bias=np.zeros(2),
            activation_func="blah",
        )


def test_ForceRateEulerJax_IO():
    """ Test ForceRateEulerJax """
    from rockpool import TSContinuous
    from nn.layers import ForceRateEulerJax_IO

    # - Generic parameters
    w_in = 2 * np.random.rand(1, 2) - 1
    w_out = 2 * np.random.rand(2, 1) - 1
    bias = 2 * np.random.rand(2) - 1
    tau = 20e-3 * np.ones(2)

    # - Layer generation
    fl0 = ForceRateEulerJax_IO(
        w_in=w_in, w_out=w_out, bias=bias, noise_std=0.1, tau=tau, dt=0.01
    )

    # - Input signal
    tsInCont = TSContinuous(times=np.arange(15) * 0.01, samples=np.ones((15, 1)))

    tsForceCont = TSContinuous(times=np.arange(15) * 0.01, samples=np.ones((15, 2)))

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.state)
    ts_output = fl0.evolve(tsInCont, tsForceCont, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state).all()

    # - Test storage
    fl0.to_dict()

    # - Test save and load
    import tempfile

    with tempfile.NamedTemporaryFile() as path_to_save:
        # - Save the layer
        fl0.save_layer(path_to_save.name)

        # - Load the layer
        lyr_loaded = ForceRateEulerJax_IO.load_from_file(path_to_save.name)

        # - Compare properties
        props_to_test = [
            "w_in",
            "w_out",
            "tau",
            "bias",
            "dt",
            "_H",
        ]
        assert all(
            [
                np.array(
                    [getattr(fl0, prop_name) == getattr(lyr_loaded, prop_name)]
                ).all()
                for prop_name in props_to_test
            ]
        )

    # - Test that some errors are caught
    with pytest.raises(ValueError):
        fl1 = ForceRateEulerJax_IO(
            w_in=np.zeros((1, 2)),
            w_out=np.zeros((3, 1)),
            tau=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(ValueError):
        fl1 = ForceRateEulerJax_IO(
            w_in=np.zeros((1, 2)),
            w_out=np.zeros((2, 1)),
            tau=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(ValueError):
        fl1 = ForceRateEulerJax_IO(
            w_in=np.zeros((1, 2)),
            w_out=np.zeros((2, 1)),
            tau=np.zeros(2),
            bias=np.zeros(3),
        )

    with pytest.raises(ValueError):
        fl1 = ForceRateEulerJax_IO(
            w_in=np.zeros((1, 2)),
            w_out=np.zeros((2, 1)),
            tau=np.zeros(2),
            bias=np.zeros(2),
            activation_func="blah",
        )


def test_FFRateEulerJax():
    """ Test FFRateEulerJax """
    from rockpool import TSContinuous
    from nn.layers import FFRateEulerJax

    # - Generic parameters
    weights = 2 * np.random.rand(1, 2) - 1
    bias = 2 * np.random.rand(2) - 1
    tau = 20e-3 * np.ones(2)
    dt = 0.01

    # - Layer generation
    fl0 = FFRateEulerJax(w_in=weights, bias=bias, noise_std=0.0, tau=tau, dt=dt)

    # - Input signal
    tsInCont = TSContinuous(times=np.arange(15) * dt, samples=np.ones((15, 1)))

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.state)
    ts_output = fl0.evolve(tsInCont, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state).all()

    # - Test functional evolve
    fl0.state = vStateBefore
    ts_out_n, _, _ = fl0._evolve_functional(
        fl0._pack(),
        vStateBefore,
        tsInCont(np.arange(int(0.1 // dt)) * dt),
    )

    assert (ts_output.samples[:10, :] == ts_out_n[:10, :]).all()

    # - Test storage
    fl0.to_dict()

    # - Test save and load
    import tempfile

    with tempfile.NamedTemporaryFile() as path_to_save:
        # - Save the layer
        fl0.save_layer(path_to_save.name)

        # - Load the layer
        lyr_loaded = FFRateEulerJax.load_from_file(path_to_save.name)

        # - Compare properties
        props_to_test = [
            "w_in",
            "tau",
            "bias",
            "dt",
            "_H",
        ]
        assert all(
            [
                np.array(
                    [getattr(fl0, prop_name) == getattr(lyr_loaded, prop_name)]
                ).all()
                for prop_name in props_to_test
            ]
        )

    # - Test that some errors are caught
    with pytest.raises(ValueError):
        fl1 = FFRateEulerJax(
            w_in=np.zeros((1, 2)),
            tau=np.zeros(3),
            bias=np.zeros(3),
        )

    with pytest.raises(ValueError):
        fl1 = FFRateEulerJax(
            w_in=np.zeros((1, 2)),
            tau=np.zeros(2),
            bias=np.zeros(3),
        )

    with pytest.raises(ValueError):
        fl1 = FFRateEulerJax(
            w_in=np.zeros((1, 2)),
            tau=np.zeros(2),
            bias=np.zeros(2),
            activation_func="blah",
        )
