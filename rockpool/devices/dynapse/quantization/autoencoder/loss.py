"""
Dynap-SE autoencoder based quantization loss computation package

* Non User Facing *
"""

from __future__ import annotations
from typing import Any, Dict
from jax import numpy as jnp

# Rockpool
from rockpool.training import jax_loss as l

from .digital import DigitalAutoEncoder
from .weight_handler import WeightHandler

import jax


__all__ = ["loss_reconstruction", "penalty_negative", "penalty_reconstruction"]


def loss_reconstruction(
    encoder: DigitalAutoEncoder,
    parameters: Dict[str, Any],
    input: jax.Array,
    f_penalty: float = 1e3,
) -> float:
    """
    loss_reconstruction calculates the mean square error loss between output and the target,
    given a new parameter set. Also, adds the bound violation penalties to the loss calculated.

    :param encoder: the autoencoder object being optimized to quantize the weights
    :type encoder: DigitalAutoEncoder
    :param parameters: new parameter set for the autoencoder
    :type parameters: Dict[str, Any]
    :param f_penalty: a factor of multiplication for bound violation penalty, defaults to 1e3
    :type f_penalty: float, optional
    :return: the mean square error loss between the output and the target + bound violation penatly
    :rtype: float
    """

    # - Assign the provided parameters to the network
    net = encoder.set_attributes(parameters)
    output, code, bit_mask = net(input)

    # - Code should always be positive (reresent real current value) - #
    penalty = f_penalty * penalty_negative(code)

    # - converting the bit_mask bit2int and int2bit should produce the same decoder
    penalty += f_penalty * penalty_reconstruction(len(code), bit_mask)

    # - Calculate the loss imposing the bounds
    _loss = l.mse(output, input) + penalty
    return _loss


def penalty_negative(param: jax.Array) -> float:
    """
    penalty_negative applies a below zero limit violation penalty to any parameter

    :param param: the parameter to apply the zero limit
    :type param: jax.Array
    :return: an exponentially increasing bound loss punishing the parameter values below zero
    :rtype: float
    """
    # - Bound penalty - #
    negatives = jnp.clip(param, None, 0)
    _loss = jnp.exp(-negatives)

    ## - subtract the code length from the sum to make the penalty 0 if all the code values are 0
    penalty = jnp.nansum(_loss) - float(param.size)
    return penalty


def penalty_reconstruction(n_bits: int, bit_mask: jax.Array) -> float:
    """
    penalty_reconstruction applies a penalty if the bit_mask encoding&decoding is non-unique.
    It also assures that the rounded decoding weights are the same as the bit_mask desired, and the
    bit_mask consists of binary values.

    :param n_bits: number of bits reserved for representing the integer values
    :type n_bits: int
    :param bit_mask: the bit_mask to check if encoding&decoding is unique
    :type bit_mask: jax.Array
    :return: mean square error loss between the bit_mask found and the bitmap reconstructed after encoding decoding
    :rtype: float
    """
    int_mask = WeightHandler.bit2int_mask(n_bits, bit_mask, jnp)
    bit_mask_reconstructed = WeightHandler.int2bit_mask(n_bits, int_mask, jnp)
    penalty = l.mse(bit_mask, bit_mask_reconstructed)

    return penalty
