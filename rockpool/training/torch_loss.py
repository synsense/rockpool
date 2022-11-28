"""
Torch loss functions and regularizers useful for training networks using Torch Modules.
"""

from rockpool.utilities.backend_management import backend_available

if not backend_available("torch"):
    raise ModuleNotFoundError(
        "`Torch` backend not found. Modules that rely on Torch will not be available."
    )

from rockpool.nn.modules import TorchModule
import torch

from copy import deepcopy

from typing import Tuple

import rockpool.utilities.tree_utils as tu

__all__ = [
    "summed_exp_boundary_loss",
    "ParameterBoundaryRegularizer",
    "make_bounds",
    "bounds_cost",
]


def summed_exp_boundary_loss(data, lower_bound=None, upper_bound=None):
    """
    Compute the summed exponential error of boundary violations of an input.

    .. math::

        \\textrm{sebl}(y, y_{lower}, y_{upper}) = \\sum_i \\textrm{sebl}(y_i, y_{lower}, y_{upper})

        \\textrm{sebl}(y_i, y_{lower}, y_{upper}) =
            \\begin{cases}
            \\exp(y_i - y_{upper}),  & \\text{if $y_i > y_{upper}$} \\\\
            \\exp(y_{lower} - y_i),  & \\text{if $y_i < y_{lower}$} \\\\
            0,  & \\text{otherwise} \\\\
            \\end{cases}

    This function allows for soft parameter constraints by creating a loss for boundary violations. This can be reached by adding `summed_exp_boundary_loss(data, lower_bound, upper_bound)` to your general loss, where `data` is an arbitrary tensor and both bounds are scalars. If either of the bounds is given as `None`, its boundary will not be penalized.

    In the example below we will introduce soft constraints to `tau_mem` of the first layer of the model, so that values `tau_mem > 1e-1` and `tau_mem < 1e-3` will be punished and considered in the optimization step.

    .. code-block:: python

        # Calculate the training loss
        y_hat, _, _ = model(x)
        train_loss = F.mse_loss(y, y_hat)

        # Set soft constraints to the time constants of the first layer of the Parameter
        boundary_loss = summed_exp_boundary_loss(model[0].tau_mem, 1e-3, 1e-1)
        complete_loss = train_loss + boundary_loss

        # Do backpropagation over both losses and optimize the model parameters accordingly
        complete_loss.backward()
        optimizer.step()

    If we would only like to introduce a lower bound penalty to a parameter we can easily do that by leaving away the definition for `upper_bound`. The same works analogously for only penalizing upper bounds.

    .. code-block:: python

        boundary_loss = summed_exp_boundary_loss(model[0].thr_up, lower_bound=1e-4)
        complete_loss = train_loss + boundary_loss

        # Do backpropagation over both losses and optimize the model parameters accordingly
        complete_loss.backward()
        optimizer.step()

    Args:
        data (torch.Tensor): The data which boundary violations will be penalized, with shape (N,).
        lower_bound (float): Lower bound for the data.
        upper_bound (float): Upper bound for the data.

    Returns:
        float: Summed exponential error of boundary violations.

    """
    # - If upper_bound is given, calculate the loss, otherwise skip it
    if upper_bound:
        upper_loss = torch.exp(data - upper_bound)

        # - Only count the loss when a violation occured, in which case exp(y_i - y_upper) > 1
        upper_loss = torch.sum(upper_loss[upper_loss > 1])
    else:
        upper_loss = 0.0

    # - If lower_bound is given, calculate the loss, otherwise skip it
    if lower_bound:
        lower_loss = torch.exp(lower_bound - data)

        # - Only count the loss when a violation occured, in which case exp(y_lower - y_i) > 1
        lower_loss = torch.sum(lower_loss[lower_loss > 1])
    else:
        lower_loss = 0.0

    return lower_loss + upper_loss


class ParameterBoundaryRegularizer(TorchModule):
    """
    Class wrapper for the summed exponential error of boundary violations of an input. See :py:func:`.summed_exp_boundary_loss` for more information.
    Allows to define the boundaries of a value just once in an object.
    """

    def __init__(self, lower_bound=None, upper_bound=None):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self, input):
        return summed_exp_boundary_loss(input, self.lower_bound, self.upper_bound)


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
    lower_bounds = tu.tree_map(lower_bounds, lambda _: -float("inf"))
    upper_bounds = tu.tree_map(upper_bounds, lambda _: float("inf"))

    return lower_bounds, upper_bounds


def bounds_cost(params: dict, lower_bounds: dict, upper_bounds: dict) -> torch.Tensor:
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
        lb_cost_all = torch.exp(-(p - lower))
        ub_cost_all = torch.exp(-(upper - p))

        lb_cost = torch.sum(lb_cost_all[p < lower])
        ub_cost = torch.sum(ub_cost_all[p > upper])

        return lb_cost + ub_cost

    # - Map bounds function over parameters and return
    return torch.sum(torch.stack(list(map(bound, params, lower_bounds, upper_bounds))))
