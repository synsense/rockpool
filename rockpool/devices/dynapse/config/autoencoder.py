"""
AutoEncoder package includes the implementation of a special autoencoder structure used in 
weight matrix -> Iw & bitmask conversion

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
27/01/2022

[] TODO : Refactor step_pwl
"""

from typing import Any, Callable, Dict, Optional, Tuple

# JAX
from jax import nn, custom_gradient

from jax import numpy as jnp
import numpy as np

# Rockpool
from rockpool.parameters import Parameter
from rockpool.nn.modules.jax.jax_module import JaxModule
from rockpool.nn.modules.native.linear import kaiming, xavier


@custom_gradient
def step_pwl(
    x: jnp.DeviceArray,
) -> Tuple[jnp.DeviceArray, Callable[[jnp.DeviceArray], jnp.DeviceArray]]:
    """
    step_pwl is heaviside step function with piece-wise linear derivative to use as spike-generation surrogate

    :param jnp.DeviceArray x: Input value

    :return (jnp.DeviceArray, Callable[[jnp.DeviceArray], jnp.DeviceArray]): output value and gradient function
    """
    s = jnp.clip(jnp.floor(x + 0.5), 0.0)
    return s, lambda g: (g * (x > 0),)


class AutoEncoder(JaxModule):
    """
    AutoEncoder implements a specific autoencoder architecture that aims to find the
    optimal weight parameters and the bitmask configuraiton given a weight matrix for `DynapSE` networks.

    NOTE: If intermediate code representation is known, then add a mean square error term to the
    loss function used in training. It will push the system generate the same code.

    :Parameters:

    :param shape: the input, output size of the AutoEncoder, (N,N). Usually, the flatten matrix size.
    :type shape: Tuple[int]
    :param n_code: the length of the code. It refers to the number of bias weight parameters used., defaults to 4
    :type n_code: int, optional
    :param w_en: encoder weight matrix that transforms a weight matrix to the code, defaults to None
    :type w_en: Optional[jnp.DeviceArray], optional
    :param w_dec: decoder wegiht matrix that reconstructs a weight matrix from the code, defaults to None
    :type w_dec: Optional[jnp.DeviceArray], optional
    :param weight_init: weight initialization function which gets a size and creates a weight, defaults to kaiming
    :type weight_init: Callable[[Tuple[int]], np.ndarray], optional
    """

    def __init__(
        self,
        shape: Tuple[int],
        n_code: int = 4,
        w_en: Optional[jnp.DeviceArray] = None,
        w_dec: Optional[jnp.DeviceArray] = None,
        weight_init: Callable[[Tuple[int]], np.ndarray] = kaiming,
        *args,
        **kwargs,
    ) -> None:
        """
        __init__ initialize the `AutoEncoder` module. Parameters are explained in the class docstring.
        """

        super(AutoEncoder, self).__init__(
            shape=shape,
            *args,
            **kwargs,
        )
        self.n_code = n_code

        # Weight Initialization
        _init = lambda s: jnp.array(weight_init(s))
        self.w_en = Parameter(
            w_en,
            init_func=_init,
            shape=(self.size_in, n_code),
        )
        self.w_dec = Parameter(
            w_dec,
            init_func=_init,
            shape=(n_code, self.size_out),
        )

    def evolve(
        self, matrix: jnp.DeviceArray, record: bool = False
    ) -> Tuple[jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray]:
        """
        evolve implements raw rockpool JAX evolution function for a AutoEncoder module.
        The AutoEncoder architecture is stateless, threfore, there is no state to return.
        The AutoEncoder architecture is timeless, therefore, there is no time record to hold.
        It uses the rockpool jax backend for the sake of compatibility.

        :param matrix: The weight matrix to encode via a weight currents and bitmask
        :type matrix: jnp.DeviceArray
        :param record: dummy record flag, required for rockpool jax modules, defaults to False
        :type record: bool, optional
        :return: reconstructed, code, bitmask
            :reconstructed: the reconstructed weight matrix
            :code: compressed matrix
            :bitmask: binary decoder
        :rtype: Tuple[jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray]
        """
        assert matrix.size == self.size_out

        # Compress the matrix and reconstruct
        code = self.encode(matrix)
        reconstructed = self.decode(code)

        return reconstructed, code, self.bitmask

    def encode(self, matrix: jnp.DeviceArray) -> jnp.DeviceArray:
        """
        encode generates the compressed version of a matrix using the encoder

        :param matrix: any matrix to compress
        :type matrix: jnp.DeviceArray
        :return: the code generated, or the compressed version
        :rtype: jnp.DeviceArray
        """
        assert matrix.size == self.size_in
        return matrix @ self.w_en

    def decode(self, code: jnp.DeviceArray) -> jnp.DeviceArray:
        """
        decode reconstructs the matrix from the code

        :param code: the compressed version of the matrix
        :type code: jnp.DeviceArray
        :return: the reconstructed matrix
        :rtype: jnp.DeviceArray
        """
        assert code.size == self.n_code
        return code @ self.bitmask

    @property
    def bitmask(self) -> jnp.DeviceArray:
        """
        ABSTRACT METHOD
        bitmask should transform the decoding weights to a bitmask to be used in matrix reconstruction from the code

        :raises NotImplementedError: This is an abstract method, needs overwriting
        :return: bitmask obtained biased to w_dec
        :rtype: jnp.DeviceArray
        """
        raise NotImplementedError("Use one of child classes instead!")


class AnalogAutoEncoder(AutoEncoder):
    """
    AnalogAutoEncoder is the simplest possible AutoEncoder implementation.
    It uses the weight matrix itself as a bitmask.

    NOTE : A penalty can be implemented in the loss function used in training to
    push the system generate a proper bitmask

    def penalty_reconstruction(bitmask: jnp.DeviceArray) -> float:
        _encoded = self.encode_bitmask(bitmask).round().astype(int)
        _decoded = self.decode_bitmask(_encoded).astype(float)
        penalty = l.mse(bitmask, _decoded)

        return penalty
    """

    def __init__(self, *args, **kwargs):
        """
        __init__ intialize the parent class
        """
        super(AnalogAutoEncoder, self).__init__(*args, **kwargs)

    @property
    def bitmask(self) -> jnp.DeviceArray:
        """
        bitmask returns directly the decoder weight matrix
        """
        return self.w_dec


class DigitalAutoEncoder(AutoEncoder):
    """
    DigitalAutoEncoder uses the quantized decoder weights in the calculations
    to make sure that the bitmask produced will be a `valid`, binary bitmask
    """

    def __init__(self, *args, **kwargs):
        """
        __init__ intialize the parent class
        """
        super(DigitalAutoEncoder, self).__init__(*args, **kwargs)

    @property
    def bitmask(self) -> jnp.DeviceArray:
        """
        bitmask applies the sigmoid to the decoder weights to scale them in 0,1 with ditribution center at .5
        Then it applies a heaviside step function with piece-wise linear derivative to obtain a valid bitmask consisting only of bits
        """
        prob = nn.sigmoid(self.w_dec)
        spikes = step_pwl(prob)
        return spikes
