"""
Dynap-SE autoencoder based quantization package custom gradient implementation

* Non User Facing *
"""

from __future__ import annotations
from typing import Tuple

# JAX
import jax
from jax import numpy as jnp


__all__ = ["step_pwl_ae"]


@jax.custom_jvp
def step_pwl_ae(probs: jnp.DeviceArray) -> jnp.DeviceArray:
    """
    step_pwl_ae is heaviside step function with piece-wise linear derivative to use as thresholded probability value surrogate

    :param probs: a probability array
    :type probs: jnp.DeviceArray
    :return: the thresholded probability values
    :rtype: float
    """
    thresholded = jnp.clip(jnp.floor(probs + 0.5), 0.0)
    return thresholded


@step_pwl_ae.defjvp
def step_pwl_jvp(
    primals: Tuple[jnp.DeviceArray], tangents: Tuple[jnp.DeviceArray]
) -> Tuple[jnp.DeviceArray]:
    """
    step_pwl_jvp custom jvp function defining the custom gradient rule of the step pwl function

    :param primals: the primary variables passed as the input to the `step_pwl_ae` function
    :type primals: Tuple[jnp.DeviceArray]
    :param tangents: the first order gradient values of the primal variables
    :type tangents: Tuple[jnp.DeviceArray]
    :return: modified forward pass output and the gradient values
    :rtype: Tuple[jnp.DeviceArray]
    """
    (probs,) = primals
    (probs_dot,) = tangents
    probs_dot = probs_dot * jnp.clip(probs, 0.0)
    return step_pwl_ae(*primals), probs_dot
