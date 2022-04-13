"""
Jax functions useful for training networks using Jax Modules.

See Also:
    See :ref:`/in-depth/jax-training.ipynb` for an introduction to training networks using Jax-backed modules in Rockpool, including the functions in `.jax_loss`.
"""

import jax.numpy as np

from copy import deepcopy

import jax.tree_util as tu

from typing import Tuple

from .ctc_loss import ctc_loss_jax


def mse(output: np.array, target: np.array) -> float:
    """
    Compute the mean-squared error between output and target

    This function is designed to be used as a component in a loss function. It computes the mean-squared error

    .. math::

        \\textrm{mse}(y, \\hat{y}) = { E[{(y - \\hat{y})^2}] }

    where :math:`E[\\cdot]` is the expectation of the expression within the brackets.

    Args:
        output (np.ndarray): The network output to test, with shape ``(T, N)``
        target (np.ndarray): The target output, with shape ``(T, N)``

    Returns:
        float: The mean-squared-error cost
    """
    return np.mean((output - target) ** 2)


def sse(output: np.array, target: np.array) -> float:
    """
    Compute the sum-squared error between output and target

    This function is designed to be used as a component in a loss function. It computes the mean-squared error

    .. math::

        \\textrm{sse}(y, \\hat{y}) = \\Sigma {(y - \\hat{y})^2}

    Args:
        output (np.ndarray): The network output to test, with shape ``(T, N)``
        target (np.ndarray): The target output, with shape ``(T, N)``

    Returns:
        float: The sum-squared-error cost
    """
    return np.sum((output - target) ** 2)


def make_bounds(params: dict) -> Tuple[dict, dict]:
    """
    Convenience function to build a bounds template for a problem

    This function works hand-in-hand with :py:func:`.bounds_cost`, to enforce minimum and/or maximum parameter bounds. :py:func:`.make_bounds` accepts a set of parameters (e.g. as returned from the :py:meth:`Module.parameters` method), and returns a ready-made dictionary of bounds (with no restrictions by default).

    See Also:
        See :ref:`/in-depth/jax-training.ipynb` for examples for using :py:func:`.make_bounds` and :py:func:`.bounds_cost`.

    :py:func:`.make_bounds` returns two dictionaries, representing the lower and upper bounds respectively. Initially all entries will be set to ``-np.inf`` and ``np.inf``, indicating that no bounds should be enforced. You must edit these dictionaries to set the bounds.

    Args:
        params (dict): Dictionary of parameters defining an optimisation problem. This can be provided as the parameter dictionary returned by :py:meth:`Module.parameters`.

    Returns:
        (dict, dict): ``lower_bounds``, ``upper_bounds``. Each dictionary mimics the structure of ``params``, with initial bounds set to ``-np.inf`` and ``np.inf`` (i.e. no bounds enforced).
    """
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

    This function works hand-in-hand with :py:func:`.make_bounds` to enforce greater-than and less-than constraints on parameter values. This is designed to be used as a component of a loss function, to ensure parameter values fall in a reasonable range.

    :py:func:`.bounds_cost` imposes a value of 1.0 for each parameter element that exceeds a bound infinitesimally, increasing exponentially as the bound is exceeded, or 0.0 for each parameter within the bounds. You will most likely want to scale this by a penalty factor within your cost function.

    Warnings:
        :py:func:`.bounds_cost` does **not** clip parameters to the bounds. It is possible for parameters to exceed the bounds during optimisation. If this must be prevented, you should clip the parameters explicitly.

    See Also:
        See :ref:`/in-depth/jax-training.ipynb` for examples for using :py:func:`.make_bounds` and :py:func:`.bounds_cost`.

    Args:
        params (dict): A dictionary of parameters over which to impose bounds
        lower_bounds (dict): A dictionary of lower bounds for parameters matching your model, modified from that returned by :py:func:`.make_bounds`
        upper_bounds (dict): A dictionary of upper bounds for parameters matching your model, modified from that returned by :py:func:`.make_bounds`

    Returns:
        float: The cost to include in the cost function.
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

        lb_cost = np.nansum(np.where(p < lower, lb_cost_all, 0.0))
        ub_cost = np.nansum(np.where(p > upper, ub_cost_all, 0.0))

        return lb_cost + ub_cost

    # - Map bounds function over parameters and return
    return np.sum(np.array(list(map(bound, params, lower_bounds, upper_bounds))))


def l2sqr_norm(params: dict) -> float:
    """
    Compute the mean L2-squared-norm of the set of parameters

    This function computes the mean :math:`L_2^2` norm of each parameter. The gradient of :math:`L_2^2(x)` is defined everywhere, where the gradient of :math:`L_2(x)` is not defined at :math:`x = 0`.

    The function is given by

    .. math::

        L_2^2(x) = E[x^2]

    where :math:`E[\\cdot]` is the expecation of the expression within the brackets.

    Args:
        params (dict): A Rockpool parameter dictionary

    Returns:
        float: The mean L2-sqr-norm of all parameters, computed individually for each parameter
    """
    # - Compute the L2 norm of each parameter individually
    params, _ = tu.tree_flatten(params)
    l22_norms = np.array(list(map(lambda p: np.nanmean(p**2), params)))

    # - Return the mean of each L2-sqr norm
    return np.nanmean(l22_norms)


def l0_norm_approx(params: dict, sigma: float = 1e-4) -> float:
    """
    Compute a smooth differentiable approximation to the L0-norm

    The :math:`L_0` norm estimates the **sparsity** of a vector -- i.e. the number of non-zero elements. This function computes a smooth approximation to the :math:`L_0` norm, for use as a component in cost functions. Including this cost will encourage parameter sparsity, by penalising non-zero parameters.

    The approximation is given by

    .. math::

        L_0(x) = \\frac{x^4}{x^4 + \\sigma}

    where :math:`\\sigma`` is a small regularisation value (by default ``1e-4``).

    References:
        Wei et. al 2018. "Gradient Projection with Approximate L0 Norm Minimization for Sparse Reconstruction in Compressed Sensing", Sensors 18 (3373). doi: 10.3390/s18103373

    Args:
        params (dict): A parameter dictionary over which to compute the L_0 norm
        sigma (float): A small value to use as a regularisation parameter. Default: ``1e-4``.

    Returns:
        float: The estimated L_0 norm cost
    """
    params, _ = tu.tree_flatten(params)
    return np.nanmean(
        np.array(
            list(
                map(
                    lambda p: np.nanmean(np.atleast_2d(p**4 / (p**4 + sigma))),
                    params,
                )
            )
        )
    )


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Implements the softmax function

    .. math::

        S(x, \\tau) = \\exp(l / \\tau) / { \\Sigma { \\exp(l / \\tau)} }

        l = x - \\max(x)

    Args:
        x (np.ndarray): Input vector of scores
        temperature (float): Temperature :math:`\\tau` of the softmax. As :math:`\\tau \\rightarrow 0`, the function becomes a hard :math:`\\max` operation. Default: ``1.0``.

    Returns:
        np.ndarray: The output of the softmax.
    """
    logits = x - np.max(x)
    eta = np.exp(logits / temperature)
    return eta / np.sum(eta)


def logsoftmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Efficient implementation of the log softmax function

    .. math ::

        log S(x, \\tau) = (l / \\tau) - \\log \\Sigma { \\exp (l / \\tau) }

        l = x - \\max (x)

    Args:
        x (np.ndarray): Input vector of scores
        temperature (float): Temperature :math:`\\tau` of the softmax. As :math:`\\tau \\rightarrow 0`, the function becomes a hard :math:`\\max` operation. Default: ``1.0``.

    Returns:
        np.ndarray: The output of the logsoftmax.
    """
    logits = x - np.max(x)
    return (logits / temperature) - np.log(np.sum(np.exp(logits / temperature)))
