"""
Test stack of jax layers in jax_stack.py
"""

import numpy as np
import pytest


def test_imports():
    from rockpool.networks import JaxStack


def test_stacking():
    from rockpool.networks import JaxStack
    from rockpool.layers import FFRateEulerJax, RecRateEulerJax
    from rockpool import TSContinuous

    # - Test creating and evolving an empty stack
    js_empty = JaxStack()
    js_empty.evolve()

    # - Specify common layer parameters
    tau = 50e-3
    bias = 0.0
    dt = 1e-3

    # - Specify layer sizes and generate all layers
    layer_sizes = [1, 5, 10, 5, 1]

    layers = []
    for N_in, N_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        # - Generate random parameters for this layer
        weights = np.random.randn(N_in, N_out) / N_out
        lyr = FFRateEulerJax(weights, tau=tau, bias=bias, dt=dt)
        layers.append(lyr)
        js_empty.add_layer(lyr)

    # - Test building a stack all at once
    stack_ffwd = JaxStack(layers)

    # - Test packing parameters and state
    assert len(stack_ffwd._pack()) == len(
        layers
    ), "Number of parameter sets should match the number of layers"

    assert len(stack_ffwd.state) == len(
        layers
    ), "Number of state sets should match the number of layers"

    # - Set up input signals
    time_base = np.arange(0, 5, dt)
    ts_input1 = TSContinuous(time_base, np.sin(time_base * 2 * np.pi))
    ts_input10 = TSContinuous(time_base, np.tanh(np.random.randn(len(time_base), 10)))

    # - Evolve the stack
    ts_output = stack_ffwd.evolve(ts_input1)

    # - Test building a recurrent stack
    w_rec = np.random.rand(10, 10)
    lyrRec = RecRateEulerJax(w_rec, tau=tau, bias=bias, dt=dt)
    stack_rec = JaxStack([lyrRec, lyrRec, lyrRec])

    # - Evolve the stack
    ts_output = stack_rec.evolve(ts_input10)

    # - Test packing and unpacking parameters
    params_fwd = stack_ffwd._pack()
    params_rec = stack_rec._pack()

    stack_ffwd._unpack(params_fwd)
    stack_rec._unpack(params_rec)

    assert (
        params_fwd == stack_ffwd._pack()
    ), "Packed parameters do not match for feed-forward stack"

    assert (
        params_rec == stack_rec._pack()
    ), "Packed parameters do not match for recurrent stack"

    # - Test randomise method
    stack_ffwd.randomize_state()
    stack_rec.randomize_state()


def test_stack_functional_evolve():
    from rockpool.networks import JaxStack
    from rockpool.layers import FFRateEulerJax, RecRateEulerJax
    from rockpool import TSContinuous

    # - Specify common layer parameters
    tau = 50e-3
    bias = 0.0
    dt = 1e-3

    # - Specify layer sizes and generate all layers
    layer_sizes = [1, 5, 1]

    layers = []
    for N_in, N_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        # - Generate random parameters for this layer
        weights = np.random.randn(N_in, N_out) / N_out
        lyr = FFRateEulerJax(weights, tau=tau, bias=bias, dt=dt)
        layers.append(lyr)

    # - Build a stack all at once
    stack_ffwd = JaxStack(layers)

    # - Set up input signals
    time_base = np.arange(0, 5, dt)
    ts_input1 = TSContinuous(time_base, np.sin(time_base * 2 * np.pi))

    # - Test functional evolution
    output_t, new_state, states_t = stack_ffwd._evolve_functional(
        stack_ffwd._pack(), stack_ffwd._state, ts_input1.samples,
    )


def test_training_jax_stack():
    from rockpool.networks import JaxStack
    from rockpool.layers import FFRateEulerJax, RecRateEulerJax
    from rockpool import TSContinuous

    # - Specify common layer parameters
    tau = 50e-3
    bias = 0.0
    dt = 1e-3

    # - Specify layer sizes and generate all layers
    layer_sizes = [1, 5, 1]

    layers = []
    for N_in, N_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        # - Generate random parameters for this layer
        weights = np.random.randn(N_in, N_out) / N_out
        lyr = FFRateEulerJax(weights, tau=tau, bias=bias, dt=dt)
        layers.append(lyr)

    # - Build a stack all at once
    stack_ffwd = JaxStack(layers)

    # - Set up input and target signals
    time_base = np.arange(0, 5, dt)
    ts_input1 = TSContinuous(time_base, np.sin(time_base * 2 * np.pi))
    ts_target1 = TSContinuous(time_base, np.random.rand(len(time_base),))

    # - Test training initialisation
    stack_ffwd.train_output_target(ts_input1, ts_target1, is_first=True)

    # - Test return functions
    l, g, out_fcn = stack_ffwd.train_output_target(ts_input1, ts_target1)

    out_fcn()
