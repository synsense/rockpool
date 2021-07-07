"""
Utilities for debugging Jax training loops
"""

from rockpool.nn.modules import JaxModule

import jax.numpy as np

from typing import Union, Iterable, Collection, Callable


def flatten(
    generic_collection: Union[Iterable, Collection], sep: str = "_"
) -> Collection:
    """
    Flattens a generic collection of collections into an ordered dictionary.

    ``generic_collection`` is a nested tree of inhomogeneous collections, such as `list`, `set`, `dict`, etc. This function iterates through this generic collection, and flattens all the leaf nodes into a single collection. The keys in the returned collection will be named after the orginal keys in ``generic_collection``, if any, and after the nesting level.

    :param Union[Iterable, Collection] generic_collection:  A nested tree of iterable types or collections, that will be flattened
    :param str sep:                                         The separator character to use when building keys in the flattened collection. Default: "_"

    :return Collection flattened_collection: A collection of all the items in ``generic_collection``, flattened into a single coellction.
    """
    import collections

    obj = collections.OrderedDict()

    def recurse(this, parent_key=""):
        if isinstance(this, dict):
            for k, v in this.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        elif np.size(this) == 1:
            obj[parent_key + sep + "0"] = this
        elif isinstance(this, collections.abc.Iterable):
            for ind, item in enumerate(this):
                recurse(item, parent_key + sep + str(ind) if parent_key else str(ind))
        else:
            obj[parent_key] = this

    recurse(generic_collection)
    return obj


def debug_evolution(
    jmod: JaxModule,
    state: dict,
    parameters: dict,
    input: np.ndarray,
) -> None:
    """
    Debug and report the presence of NaNs in network state / output

    Use this function within a training loop to search for the presence of NaNs.

    Args:
        jmod (JaxModule): The module to debug
        state (dict): The state dictionary prior to evolution
        parameters (dict): The parameter dictionary to use in debugging
        input (np.ndarray): The input data for this optimisation iteration

    Raises:
        ValueError: If any NaNs are found during evolution
    """
    str_error = ""

    # - Check current network state
    for k, v in flatten(state).items():
        if np.any(np.isnan(v)):
            str_error += "Pre-evolve network state {} contains NaNs\n".format(k)

    # - Check network parameters
    for k, v in flatten(parameters).items():
        if np.any(np.isnan(v)):
            str_error += "Network parameter {} contains NaNs\n".format(k)

    # - Check outputs
    output_ts, new_state, _ = jmod(input)
    if np.any(np.isnan(output_ts)):
        str_error += "Network output contained NaNs\n"

    # - Check network states
    for k, v in flatten(new_state).items():
        if np.any(np.isnan(v)):
            str_error += "Post-evolve network state {} contains NaNs\n".format(k)

    # - Raise the error
    if str_error:
        raise ValueError(str_error)


def debug_optimisation(
    jmod: JaxModule,
    parameters: dict,
    input: np.ndarray,
    target: np.ndarray,
    vgf_loss: Callable,
) -> None:
    """
    Debug an optimisation step, reporting the presence of NaNs in loss and gradients

    Use this function during a training loop to search for the presence of NaNs. The location and timing of the first NaN will be reported.

    Args:
        jmod (JaxModule): The module to evaluate
        parameters (dict): The module parameters to use for the evolution
        input (np.darray): The input data for the evolution with shape ``(T, Nin)``
        target (np.ndarray): The target output for the evolution with shape ``(T, Nout)``
        vgf_loss (Callable): A value and grad function computed by :py:func:`jax.value_and_grad`. Must have the signature ``vgf_loss(params, mod, input, target) -> (loss, gradients)``

    Raises:
        ValueError: If any NaNs are discovered during the loss / gradient evaluation
    """
    # - Initialise error information
    str_error = ""

    # - Evaluate loss function
    loss, gradients = vgf_loss(parameters, jmod, input, target)

    # - Check loss function
    if np.isnan(loss):
        str_error += "Loss function returned NaN\n"

    # - Check gradients
    debug_gradient = False
    for k, v in flatten(gradients).items():
        if np.any(np.isnan(v)):
            str_error += "Gradient item {} contains NaNs\n".format(k)
            debug_gradient = True

    # - Check gradients in detail
    if debug_gradient:
        # - Loop over time steps and compute gradients
        found_nan = False
        for step in range(input.shape[0]):
            gradients_limited = vgf_loss(
                parameters,
                jmod,
                input[:step, :],
                target[:step, :],
            )

            for k, v in flatten(gradients_limited).items():
                if np.any(np.isnan(v)):
                    str_error += "Gradient NaNs begin in step {}\n".format(step)
                    found_nan = True
                    break

            if found_nan:
                break

    # - Raise the error
    if str_error:
        raise ValueError(str_error)
