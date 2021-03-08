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
    """
    Impose a cost on parameters that violate bounds constraints



    Args:
        params (dict):
        lower_bounds (dict): A dictionary of lower bounds for parameters matching your model, modified from that returned by :py:func:`make_bounds`
        upper_bounds (dict): A dictionary of upper bounds for parameters matching your model, modified from that returned by :py:func:`make_bounds`

    Returns:

    """
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
        lb_cost_all = np.exp(-(p - lower))
        ub_cost_all = np.exp(-(upper - p))

        lb_cost = np.nanmean(np.where(p < lower, lb_cost_all, 0.0))
        ub_cost = np.nanmean(np.where(p > upper, ub_cost_all, 0.0))

        return lb_cost + ub_cost

    # - Map bounds function over parameters and return
    return np.sum(np.array(list(map(bound, params, lower_bounds, upper_bounds))))


def l2sqr_norm(params: dict) -> float:
    """
    Compute the mean L2-squared-norm of the set of parameters

    This function computes the mean L2^2 norm of each parameter. The gradient of L2^2 is defined everywhere, where the gradient of L2(x) is not defined at x == 0.

    The function is given by

    .. math::

        L2^2(\vec{x}) = #(\vec{x})^{-1} \cdot \sum{x^2}

    Args:
        params (dict): A Rockpool parameter dictionary

    Returns:
        float: The mean L2-sqr-norm of all parameters, computed individually for each parameter
    """
    # - Compute the L2 norm of each parameter individually
    params, _ = tu.tree_flatten(params)
    l22_norms = np.array(list(map(lambda p: np.nanmean(p ** 2), params)))

    # - Return the mean of each L2-sqr norm
    return np.nanmean(l22_norms)


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
