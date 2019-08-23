"""
Test training of rate-based Euler models in rate-jax.py, using shim in train_jax_sgd.py
"""

import numpy as np
import pytest


def test_imports():
    from NetworksPython.layers.utilities import add_train_output


def test_adam():
    from NetworksPython.layers import RecRateEulerJax
    from NetworksPython.layers.utilities import add_train_output
    from NetworksPython import TSContinuous

    # - Generic parameters
    w_in = 2 * np.random.rand(1, 2) - 1
    w_recurrent = 2 * np.random.rand(2, 2) - 1
    w_out = 2 * np.random.rand(2, 1) - 1

    # - Layer generation
    fl0 = RecRateEulerJax(
        w_in=w_in,
        w_recurrent=w_recurrent,
        w_out=w_out,
        bias=0,
        noise_std=0.1,
        tau=20,
        dt=1,
    )

    # - Add training shim
    fl0 = add_train_output(fl0)

    # - Define simple input and target
    ts_input = TSContinuous([0, 1, 2, 3], [0, 1, 0, 0])
    ts_target = TSContinuous([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4])

    # - Initialise training
    loss_fcn, grad_fcn = fl0.train_adam(ts_input, ts_target, is_first=True)

    # - Test loss and gradient functions
    loss_fcn()
    grad_fcn()

    # - Perform intermediate training step
    fl0.train_adam(ts_input, ts_target)

    # - Perform final training step
    fl0.train_adam(ts_input, ts_target, is_last=True)
