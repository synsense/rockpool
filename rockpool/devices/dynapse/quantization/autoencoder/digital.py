"""
Dynap-SE autoencoder based quantization digital autoencoder implementation

* Non User Facing *
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple

import numpy as np

# JAX
import jax
from jax import numpy as jnp
from jax.nn import sigmoid

# Rockpool
from rockpool.nn.modules.jax.jax_module import JaxModule
from rockpool.parameters import Parameter
from rockpool.nn.modules.native.linear import kaiming

from .gradient import step_pwl_ae


__all__ = ["DigitalAutoEncoder"]


class DigitalAutoEncoder(JaxModule):
    """
    DigitalAutoEncoder implements a specific autoencoder architecture that aims to find the
    optimal weight parameters and the bit_mask configuraiton given a weight matrix for `DynapSE` networks.

    NOTE: If intermediate code representation is known, then add a mean square error term to the
    loss function used in training. It will push the system generate the same code.

    :Parameters:

    :param shape: the input, output size of the AutoEncoder, (N,N). Usually, the flatten matrix size.
    :type shape: Tuple[int]
    :param n_code: the length of the code. It refers to the number of bias weight parameters used., defaults to 4
    :type n_code: int, optional
    :param w_en: encoder weight matrix that transforms a weight matrix to the code, defaults to None
    :type w_en: Optional[jax.Array], optional
    :param w_dec: decoder wegiht matrix that reconstructs a weight matrix from the code, defaults to None
    :type w_dec: Optional[jax.Array], optional
    :param weight_init: weight initialization function which gets a size and creates a weight, defaults to kaiming
    :type weight_init: Callable[[Tuple[int]], np.ndarray], optional
    """

    def __init__(
        self,
        shape: Tuple[int],
        n_code: int = 4,
        w_en: Optional[jax.Array] = None,
        w_dec: Optional[jax.Array] = None,
        weight_init: Callable[[Tuple[int]], np.ndarray] = kaiming,
        *args,
        **kwargs,
    ) -> None:
        """
        __init__ initialize the `DigitalAutoEncoder` module. Parameters are explained in the class docstring.
        """

        super(DigitalAutoEncoder, self).__init__(
            shape=shape,
            *args,
            **kwargs,
        )
        self.n_code = n_code

        # Weight Initialization
        _init = lambda s: jnp.array(weight_init(s))
        self.w_en = Parameter(w_en, init_func=_init, shape=(self.size_in, n_code))
        self.w_dec = Parameter(w_dec, init_func=_init, shape=(n_code, self.size_out))

    def evolve(
        self, matrix: jax.Array, record: bool = False
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        evolve implements raw rockpool JAX evolution function for a AutoEncoder module.
        The AutoEncoder architecture is stateless, threfore, there is no state to return.
        The AutoEncoder architecture is timeless, therefore, there is no time record to hold.
        It uses the rockpool jax backend for the sake of compatibility.

        :param matrix: The weight matrix to encode via a weight currents and bit_mask
        :type matrix: jax.Array
        :param record: dummy record flag, required for rockpool jax modules, defaults to False
        :type record: bool, optional
        :return: reconstructed, code, bit_mask
            :reconstructed: the reconstructed weight matrix
            :code: compressed matrix
            :bit_mask: binary decoder
        :rtype: Tuple[jax.Array, jax.Array, jax.Array]
        """
        assert matrix.size == self.size_out

        # Compress the matrix and reconstruct
        code = self.encode(matrix)
        reconstructed = self.decode(code)

        return reconstructed, code, self.bit_mask

    def encode(self, matrix: jax.Array) -> jax.Array:
        """
        encode generates the compressed version of a matrix using the encoder

        :param matrix: any matrix to compress
        :type matrix: jax.Array
        :return: the code generated, or the compressed version
        :rtype: jax.Array
        """
        assert matrix.size == self.size_in
        return matrix @ self.w_en

    def decode(self, code: jax.Array) -> jax.Array:
        """
        decode reconstructs the matrix from the code

        :param code: the compressed version of the matrix
        :type code: jax.Array
        :return: the reconstructed matrix
        :rtype: jax.Array
        """
        assert code.size == self.n_code
        return code @ self.bit_mask

    @property
    def bit_mask(self) -> jax.Array:
        """
        bit_mask applies the sigmoid to the decoder weights to scale them in 0,1 with ditribution center at .5
        Then it applies a heaviside step function with piece-wise linear derivative to obtain a valid bit_mask consisting only of bits
        """
        prob = sigmoid(self.w_dec)
        spikes = step_pwl_ae(prob)
        return spikes
