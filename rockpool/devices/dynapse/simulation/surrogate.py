"""
Low level DynapSE-2 simulator surrogate gradient implementation

* Non User Facing *

[] TODO : max spikes per dt
"""
from __future__ import annotations
from typing import Tuple

from jax import custom_jvp
from jax import numpy as jnp

__all__ = ["step_pwl"]


@custom_jvp
def step_pwl(
    imem: jnp.DeviceArray,
    Ispkthr: jnp.DeviceArray,
    Ireset: jnp.DeviceArray,
    max_spikes_per_dt: int = jnp.inf,
) -> float:
    """
    step_pwl implements heaviside step function with piece-wise linear derivative to use as spike-generation surrogate

    :param imem: Input current to be compared for firing
    :type imem: jnp.DeviceArray
    :param Ispkthr: Spiking threshold current in Amperes
    :type Ispkthr: jnp.DeviceArray
    :param Ireset: Reset current after spike generation in Amperes
    :type Ireset: jnp.DeviceArray
    :return: number of spikes produced
    :rtype: float
    """
    spikes = jnp.ceil(jnp.log(imem / Ispkthr))
    n_spikes = jnp.clip(spikes, 0.0, max_spikes_per_dt)
    return n_spikes


@step_pwl.defjvp
def step_pwl_jvp(
    primals: Tuple[jnp.DeviceArray], tangents: Tuple[jnp.DeviceArray]
) -> Tuple[jnp.DeviceArray]:
    """
    step_pwl_jvp custom jvp function defining the custom gradient rule of the step pwl function

    :param primals: the primary variables passed as the input to the `step_pwl` function
    :type primals: Tuple[jnp.DeviceArray]
    :param tangents: the first order gradient values of the primal variables
    :type tangents: Tuple[jnp.DeviceArray]
    :return: modified forward pass output and the gradient values
    :rtype: Tuple[jnp.DeviceArray]
    """
    imem, Ispkthr, Ireset, max_spikes_per_dt = primals
    imem_dot, Ispkthr_dot, Ireset_dot, max_spikes_per_dt_dot = tangents
    primal_out = step_pwl(*primals)
    tangent_out = jnp.clip(jnp.ceil(imem - Ireset), 0, 1) * imem_dot
    return primal_out, tangent_out
