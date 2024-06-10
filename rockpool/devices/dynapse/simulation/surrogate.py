"""
Low level DynapSE-2 simulator surrogate gradient implementation

* Non User Facing *

[] TODO : max spikes per dt
"""

from __future__ import annotations
from typing import Tuple
import jax
from jax import custom_jvp
from jax import numpy as jnp

__all__ = ["step_pwl"]


@custom_jvp
def step_pwl(
    imem: jax.Array,
    Ispkthr: jax.Array,
    Ireset: jax.Array,
    max_spikes_per_dt: int = jnp.inf,
) -> float:
    """
    step_pwl implements heaviside step function with piece-wise linear derivative to use as spike-generation surrogate

    :param imem: Input current to be compared for firing
    :type imem: jax.Array
    :param Ispkthr: Spiking threshold current in Amperes
    :type Ispkthr: jax.Array
    :param Ireset: Reset current after spike generation in Amperes
    :type Ireset: jax.Array
    :return: number of spikes produced
    :rtype: float
    """
    spikes = jnp.ceil(jnp.log(imem / Ispkthr))
    n_spikes = jnp.clip(spikes, 0.0, max_spikes_per_dt)
    return n_spikes


@step_pwl.defjvp
def step_pwl_jvp(
    primals: Tuple[jax.Array], tangents: Tuple[jax.Array]
) -> Tuple[jax.Array]:
    """
    step_pwl_jvp custom jvp function defining the custom gradient rule of the step pwl function

    :param primals: the primary variables passed as the input to the `step_pwl` function
    :type primals: Tuple[jax.Array]
    :param tangents: the first order gradient values of the primal variables
    :type tangents: Tuple[jax.Array]
    :return: modified forward pass output and the gradient values
    :rtype: Tuple[jax.Array]
    """
    imem, Ispkthr, Ireset, max_spikes_per_dt = primals
    imem_dot, Ispkthr_dot, Ireset_dot, max_spikes_per_dt_dot = tangents
    primal_out = step_pwl(*primals)
    tangent_out = jnp.clip(jnp.ceil(imem - Ireset), 0, 1) * imem_dot
    return primal_out, tangent_out
