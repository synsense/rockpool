"""
AutoEncoder package includes the implementation of a special autoencoder structure used in 
weight matrix -> Iw & bitmask conversion

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
27/01/2022

[] TODO : make sure that bitmask is jnp.DeviceArray
"""

from __future__ import annotations
import logging

from typing import Any, Callable, Dict, Optional, Tuple, Union

from copy import deepcopy
from dataclasses import dataclass

# JAX
from jax import nn, jit, value_and_grad, custom_gradient
from jax.lax import scan
from jax.experimental import optimizers

from jax import numpy as jnp
import numpy as np

# Rockpool
from rockpool.parameters import Parameter
from rockpool.training import jax_loss as l
from rockpool.nn.modules.jax.jax_module import JaxModule
from rockpool.nn.modules.native.linear import kaiming, xavier
from rockpool.devices.dynapse.config.layout import DynapSELayout
from rockpool.devices.dynapse.config.circuits import SimulationParameters
from rockpool.devices.dynapse.lookup import param_name

_SAMNA_SE1_AVAILABLE = True
_SAMNA_SE2_AVAILABLE = True

param_name_table = param_name.table
try:
    from samna.dynapse1 import Dynapse1Parameter
except ModuleNotFoundError as e:
    Dynapse1Parameter = Any

    print(
        e, "\nDynapSE1SimCore object cannot be factored from a samna config object!",
    )
    _SAMNA_SE1_AVAILABLE = False

try:
    from samna.dynapse2 import Dynapse2Parameter
except ModuleNotFoundError as e:
    Dynapse2Parameter = Any
    print(
        e, "\nDynapSE2SimCore object cannot be factored from a samna config object!",
    )
    _SAMNA_SE2_AVAILABLE = False


WeightRecord = Tuple[
    jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray,  # loss  # w_en  # w_dec
]


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
    loss function used in training. It will push the system generate the same code. However, if it's not known,
    the way the autoencoder behave changes. Look at the evolve function below! If you are not sure, then let the code search stay at the default value.

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
    :param code_search: Do the code search or not. Code search requires using thresholded decoding, defaults to True
    :type code_search: bool, optional
    """

    def __init__(
        self,
        shape: Tuple[int],
        n_code: int = 4,
        w_en: Optional[jnp.DeviceArray] = None,
        w_dec: Optional[jnp.DeviceArray] = None,
        weight_init: Callable[[Tuple[int]], np.ndarray] = kaiming,
        code_search: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """
        __init__ initialize the `AutoEncoder` module. Parameters are explained in the class docstring.
        """

        super(AutoEncoder, self).__init__(
            shape=shape, *args, **kwargs,
        )

        self.n_code = n_code

        self.code_search = float(code_search)

        # Weight Initialization
        _init = lambda s: jnp.array(weight_init(s))
        self.w_en = Parameter(w_en, init_func=_init, shape=(self.size_in, n_code),)
        self.w_dec = Parameter(w_dec, init_func=_init, shape=(n_code, self.size_out),)

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
        return matrix @ self.w_en

    @property
    def bitmask(self) -> jnp.DeviceArray:
        # Thresholding
        prob = nn.sigmoid(self.w_dec)
        spikes = step_pwl(prob)

        # Threshold or not dependin on the code_search situation! NOT USING if-else for better jit pipeline!
        _dec = spikes * self.code_search + self.w_dec * (1.0 - self.code_search)
        return _dec

    def decode(self, code: jnp.DeviceArray) -> jnp.DeviceArray:
        """
        decode decide how to use the decoder weights and how to regard the decoder weights.
        Return the decoder indicated binary weight mask if code search is being done.
        If code search is off, then do not restrict the weights here and add the boundary violation term to the loss function
        """
        return code @ self.bitmask

