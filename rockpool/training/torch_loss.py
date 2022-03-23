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

__all__ = ["summed_exp_boundary_loss", "ParameterBoundaryRegularizer"]


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
