"""
Test training of rate-based Euler models in rate-jax.py, using shim in train_jax_sgd.py
"""

import numpy as np
import pytest


def test_imports():
    from rockpool.layers.training import add_train_output


def test_adam():
    from rockpool.layers import RecRateEulerJax
    from rockpool.layers.training import add_train_output
    from rockpool import TSContinuous

    # - Generic parameters
    w_in = 10. * np.ones((1, 2))
    w_recurrent = 10. * np.ones((2, 2))
    w_out = 10. * np.ones((2, 1))

    # - Layer generation
    fl0 = RecRateEulerJax(
        w_in=w_in,
        w_recurrent=w_recurrent,
        w_out=w_out,
        bias=0.,
        noise_std=0.,
        tau=20.,
        dt=1,
    )

    # - Add training shim
    fl0 = add_train_output(fl0)

    # - Define simple input and target
    ts_input = TSContinuous([0, 1, 2, 3], [1, 1, 1, 1])
    ts_target = TSContinuous([0, 1, 2, 3], [0.1, 0.1, 0.1, 0.1])

    # - Initialise training
    loss_fcn, grad_fcn = fl0.train_adam(ts_input, ts_target, is_first=True)

    # - Test loss and gradient functions
    l = loss_fcn()
    g = grad_fcn()

    # - Test known loss and gradient values
    l_target = 1477
    g_target = {
        'bias': np.array([138, 138]),
        'tau': np.array([-112, -112]),
        'w_in': np.array([[138, 138]]),
        'w_out': np.array([[138], [138]]),
        'w_recurrent': np.array([[50, 50], [50, 50]]),
    }

    # - Check loss
    assert np.abs(np.round(l) - l_target) < 1, \
        'MSE loss value does not match known target'

    # - Check gradients
    for k, v in g.items():
        assert np.all(np.abs(np.round(v) - g_target[k]) < 1), \
            'Gradient value for [' + k + '] does not match known target'

    # - Perform intermediate training step
    fl0.train_adam(ts_input, ts_target)

    # - Perform final training step
    fl0.train_adam(ts_input, ts_target, is_last=True)

    target_eps = 1e-3

    state_target = [0., 0.]
    tau_target = [20.000298, 20.000298]
    bias_target = [-0.0003, -0.0003]
    w_in_target = [[9.9997, 9.9997]]
    w_out_target = [[9.9997],[9.9997]]
    w_recurrent_target = [[9.9997, 9.9997], [9.9997, 9.9997]]

    assert np.all(np.abs(fl0.state - state_target) < target_eps)
    assert np.all(np.abs(fl0.tau - tau_target) < target_eps)
    assert np.all(np.abs(fl0.bias - bias_target) < target_eps)
    assert np.all(np.abs(fl0.w_in - w_in_target) < target_eps)
    assert np.all(np.abs(fl0.w_out - w_out_target) < target_eps)
    assert np.all(np.abs(fl0.w_recurrent - w_recurrent_target) < target_eps)
