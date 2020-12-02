import jax.numpy as np

import jax.tree_util as tu

from typing import Tuple

from copy import deepcopy


def mse(output: np.array, target: np.array):
    return np.sum((output - target) ** 2)


def make_bounds(params: dict) -> Tuple[dict, dict]:
    # - Make copies
    lower_bounds = deepcopy(params)
    upper_bounds = deepcopy(params)

    # - Reset to -inf and inf
    lower_bounds = tu.tree_map(lambda _: -np.inf, lower_bounds)
    upper_bounds = tu.tree_map(lambda _: np.inf, upper_bounds)

    return lower_bounds, upper_bounds


def bounds_cost(params: dict, lower_bounds: dict, upper_bounds: dict) -> float:
    # - Flatten all parameter dicts
    params, tree_def_params = tu.tree_flatten(params)
    lower_bounds, tree_def_minparams = tu.tree_flatten(lower_bounds)
    upper_bounds, tree_def_maxparams = tu.tree_flatten(upper_bounds)

    if len(params) != len(lower_bounds) != len(upper_bounds):
        raise KeyError(
            "`lower_bounds` and `upper_bounds` must have the same keys as `params`."
        )

    # - Define a bounds function
    def bound(p, lower, upper):
        lb_cost = np.nanmean(np.where(p < lower, np.exp(-(p - lower)), 0))
        ub_cost = np.nanmean(np.where(p > upper, np.exp((upper - p)), 0))

        return lb_cost + ub_cost

    # - Map bounds function over parameters and return
    return np.sum(np.array(list(map(bound, params, lower_bounds, upper_bounds))))


def l2_norm(params: dict) -> float:
    params, _ = tu.tree_flatten(params)
    return np.nanmean(
        np.array(list(map(lambda p: np.sqrt(np.nanmean(p ** 2)), params)))
    )


def l0_norm_approx(params: dict, sigma=1e-4):
    params, _ = tu.tree_flatten(params)
    return np.nanmean(
        np.array(
            list(
                map(
                    lambda p: np.nanmean(np.atleast_2d(p ** 4 / (p ** 4 + sigma))),
                    params,
                )
            )
        )
    )
