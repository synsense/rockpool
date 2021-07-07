"""
Torch loss functions and regularizers useful for training networks using Torch Modules.
"""

from importlib import util

if util.find_spec("torch") is None:
    raise ModuleNotFoundError(
        "'Torch' backend not found. Modules that rely on Torch will not be available."
    )

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ["ParameterBoundaryRegularizer"]

def summed_exp_boundary_loss(data, lower_bound=None, upper_bound=None):
    '''
    Compute the summed exponential error of boundary violations of an input.

    .. math::

        \\textrm{sebl}(y, y_{lower}, y_{upper}) = \\sum \\textrm{sebl}(y_i, y_{lower}, y_{upper})
        \\textrm{sebl}(y_i, y_{lower}, y_{upper}) =
            \\begin{cases}
            \\exp(y_i - y_{upper}),  & \\text{if $y_i > y_{upper}$} \\\\
            \\exp(y_{lower} - y_i),  & \\text{if $y_i < y_{lower}$} \\\\
            0,  & \\text{else} \\\\
            \\end{cases}

    Args:
        data (torch.Tensor): The data which boundary violations will be penalized, with shape (N,).
        lower_bound (float): Lower bound for the data.
        upper_bound (float): Upper bound for the data.

    Returns:
        float: Summed exponential error of boundary violations.

    '''
    # - If upper_bound is given, calculate the loss, otherwise skip it
    if upper_bound:
        upper_loss = torch.exp(data - upper_bound)
        # - Only count the loss when a violation occured, in which case exp(y_i - y_upper) > 1
        upper_loss = torch.sum(upper_loss[upper_loss > 1])
    else:
        upper_loss = 0
    # - If lower_bound is given, calculate the loss, otherwise skip it
    if lower_bound:
        lower_loss = torch.exp(lower_bound - data)
        # - Only count the loss when a violation occured, in which case exp(y_lower - y_i) > 1
        lower_loss = torch.sum(lower_loss[lower_loss > 1])
    else:
        lower_loss = 0

    return lower_loss + upper_loss


class ParameterBoundaryRegularizer(nn.Module):
    '''
    Class wrapper for the summed exponential error of boundary violations of an input. See :py:func:'.summed_exp_boundary_loss' for more information.
    Allows to define the boundaries of a value just once in an object.
    '''
    def __init__(self, lower_bound=None, upper_bound=None):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self, input):
        return summed_exp_boundary_loss(input, self.lower_bound, self.upper_bound)
