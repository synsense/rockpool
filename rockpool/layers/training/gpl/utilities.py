#
# rockpool.layers.training.utilities – Module containing training utilities
#

from .jax_trainer import JaxTrainer
from ....timeseries import TimeSeries, TSContinuous

from importlib import util

if (util.find_spec("jax") is None) or (util.find_spec("jaxlib") is None):
    raise ModuleNotFoundError(
        "'jax' and 'jaxlib' backends not found. Layers that rely on Jax will not be available."
    )

from jax import numpy as np
from jax import jit, grad, vmap
from jax.tree_util import tree_flatten, tree_unflatten

import numpy as onp

from typing import Callable, Any, Dict, List

Params = Any
State = Any


def gradient_evolution(
    lyr: JaxTrainer,
    ts_input: TimeSeries,
    ts_target: TimeSeries,
    loss_fcn: Callable,
    loss_params: Dict = {},
    progress_fcn: Callable = lambda: None,
) -> List[TSContinuous]:
    # - Get functional evolution function
    evol_func = lyr._evolve_functional

    # - Make a curried loss function, incorporating static loss parameters and evolution
    @jit
    def loss_curried(
        opt_params: Params,
        input_batch_t: np.ndarray,
        target_batch_t: np.ndarray,
        state: State,
    ) -> float:
        """
        Curried loss function; absorbs loss parameters

        :param Params opt_params:           Current values of the layer parameters, modified by optimization
        :param np.ndarray input_batch_t:    Input rasterized time series for this batch [TxO]
        :param np.ndarray target_batch_t:   Target rasterized time series for this batch [TxO]
        :param State state:                 Initial state for the layer

        :return float:                      Loss value for the parameters in `opt_params`, for the current batch
        """
        # - Call the layer evolution function
        output_batch_t, new_state, states_t = evol_func(
            opt_params, state, input_batch_t
        )

        # - Call loss function and return loss
        return loss_fcn(
            opt_params, states_t, output_batch_t, target_batch_t, **loss_params
        )

    # - Get a gradient function
    grad_fcn = jit(grad(loss_curried))

    # - Prepare time base and inputs
    if isinstance(ts_input, TimeSeries):
        # - Check that `ts_target` is also a time series
        assert isinstance(
            ts_target, TimeSeries
        ), "If `ts_input` is provided as a `TimeSeries` object, then `ts_target` must also be a `TimeSeries`."

        # - Rasterize input and target time series
        time_base, inps, num_timesteps = lyr._prepare_input(ts_input, None, None)
        target = ts_target(time_base)
    else:
        # - Use pre-rasterized time series
        inps = ts_input
        target = ts_target

    # - Repack inputs into a tensor to use `vmap`
    steps = inps.shape[0]
    inps_tensor = onp.zeros((steps, inps.shape[1], steps))
    target_tensor = onp.zeros((steps, target.shape[1], steps))

    for step in range(inps.shape[0]):
        # - Get inputs and targets
        inps_tensor[-step - 1 :, :, step] = inps[: step + 1, :]
        target_tensor[-step - 1 :, :, step] = target[: step + 1, :]

    # - Get state and parameters
    params = lyr._pack()
    state0 = lyr._state

    # - Compute gradients by mapping over inputs
    grad_partial = lambda i, t: grad_fcn(params, i, t, state0)
    gradients_t = vmap(grad_partial, in_axes=2)(inps_tensor, target_tensor)

    # - Flatten gradients tree
    gradients_t, tree_def = tree_flatten(gradients_t)

    gradients = [
        TSContinuous.from_clocked(np.reshape(grad_t, (steps, -1)), dt=lyr.dt)
        for grad_t in gradients_t
    ]

    # - Repack into the original tree format and return
    return tree_unflatten(tree_def, gradients)
    # return gradients_t, tree_def
