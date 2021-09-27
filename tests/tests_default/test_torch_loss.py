"""
Unit tests for jax training utilities, loss functions
"""
# - Ensure that NaNs in compiled functions are errors
from jax.config import config

config.update("jax_debug_nans", True)


def test_imports():
    from rockpool.training.torch_loss import (
        ParameterBoundaryRegularizer,
        summed_exp_boundary_loss,
    )


def test_summed_exp_boundary_loss():
    import torch
    from rockpool.training.torch_loss import summed_exp_boundary_loss

    N = 10
    scale = 2
    upper_bound = 1
    lower_bound = -upper_bound

    # - Initialize a dummy input for each bound violation
    input_upper = scale * torch.ones(N, requires_grad=True, dtype=torch.float)
    input_lower = -scale * torch.ones(N, requires_grad=True, dtype=torch.float)
    list_input_no_violation = []
    list_input_no_violation.append(torch.ones(N, dtype=torch.float) * upper_bound)
    list_input_no_violation.append(torch.ones(N, dtype=torch.float) * lower_bound)
    list_input_no_violation.append(
        torch.randint(lower_bound, upper_bound + 1, (N,), dtype=torch.float)
    )

    # - Calculate the bound violation loss
    loss_violation_upper = summed_exp_boundary_loss(
        input_upper, float(lower_bound), float(upper_bound)
    )
    loss_violation_lower = summed_exp_boundary_loss(
        input_lower, float(lower_bound), float(upper_bound)
    )

    # - Check whether the result is equal to the expected value
    bound_violation = torch.ones(N) * (scale - upper_bound)
    bound_loss = torch.sum(torch.exp(bound_violation))
    assert loss_violation_upper == bound_loss
    assert loss_violation_lower == bound_loss

    # - Check that the loss function is differentiable in both violation cases
    loss_violation_upper.backward()
    loss_violation_lower.backward()

    # - Check that input within the boundaries doesn't generate any loss
    for inp in list_input_no_violation:
        loss_no_violation = summed_exp_boundary_loss(
            inp, float(lower_bound), float(upper_bound)
        )
        assert loss_no_violation == torch.zeros(1)


def test_ParameterBoundaryRegularizer():
    import torch
    from rockpool.training.torch_loss import ParameterBoundaryRegularizer
    import numpy as np

    torch.manual_seed(42)

    N = 10
    scale = 2
    upper_bound = 1
    lower_bound = -upper_bound

    # - Initialize a dummy input for each bound violation
    input_upper = scale * torch.ones(N, requires_grad=True)
    input_lower = -scale * torch.ones(N, requires_grad=True)
    list_input_no_violation = []
    list_input_no_violation.append(torch.ones(N) * upper_bound)
    list_input_no_violation.append(torch.ones(N) * lower_bound)
    list_input_no_violation.append(torch.randint(lower_bound, upper_bound + 1, (N,)))

    # -Initialize the regularizer
    reg = ParameterBoundaryRegularizer(lower_bound=lower_bound, upper_bound=upper_bound)

    # - Calculate the bound violation loss
    loss_violation_upper, _, _ = reg(input_upper)
    loss_violation_lower, _, _ = reg(input_lower)

    # - Check whether the result is equal to the expected value
    bound_violation = torch.ones(N) * (scale - upper_bound)
    bound_loss = torch.sum(torch.exp(bound_violation))
    assert loss_violation_upper == bound_loss
    assert loss_violation_lower == bound_loss

    # - Check that the loss function is differentiable in both violation cases
    loss_violation_upper.backward()
    loss_violation_lower.backward()

    # - Check that input within the boundaries doesn't generate any loss
    for inp in list_input_no_violation:
        loss_no_violation, _, _ = reg(inp)
        assert loss_no_violation == torch.zeros(1)
