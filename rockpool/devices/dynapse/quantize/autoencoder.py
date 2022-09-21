"""
Dynap-SE weight quantization package provides easy to use support

Note : Existing modules are reconstructed considering consistency with Xylo support.

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

15/09/2022

[] TODO : Think about scale and rescale
"""
from __future__ import annotations
import logging

from typing import Any, Callable, Dict, Optional, Tuple, Union

from dataclasses import dataclass
import numpy as np
from copy import deepcopy

# JAX
from jax import numpy as jnp
from jax import jit, value_and_grad
from jax.lax import scan, cond
from rockpool.nn.modules.jax.jax_module import JaxModule
from jax.example_libraries import optimizers


# Rockpool
from rockpool.training import jax_loss as l
from rockpool.devices.dynapse.quantize.weight_handler import WeightHandler


__all__ = ["autoencoder_quantization", "get_optimizer"]

from typing import Any, Callable, Dict, Optional, Tuple

# JAX
from jax import nn, custom_gradient

# Rockpool
from rockpool.parameters import Parameter
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


class DigitalAutoEncoder(JaxModule):
    """
    AutoEncoder implements a specific autoencoder architecture that aims to find the
    optimal weight parameters and the bit_mask configuraiton given a weight matrix for `DynapSE` networks.

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

        super(DigitalAutoEncoder, self).__init__(
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

        :param matrix: The weight matrix to encode via a weight currents and bit_mask
        :type matrix: jnp.DeviceArray
        :param record: dummy record flag, required for rockpool jax modules, defaults to False
        :type record: bool, optional
        :return: reconstructed, code, bit_mask
            :reconstructed: the reconstructed weight matrix
            :code: compressed matrix
            :bit_mask: binary decoder
        :rtype: Tuple[jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray]
        """
        assert matrix.size == self.size_out

        # Compress the matrix and reconstruct
        code = self.encode(matrix)
        reconstructed = self.decode(code)

        return reconstructed, code, self.bit_mask

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
        return code @ self.bit_mask

    @property
    def bit_mask(self) -> jnp.DeviceArray:
        """
        bit_mask applies the sigmoid to the decoder weights to scale them in 0,1 with ditribution center at .5
        Then it applies a heaviside step function with piece-wise linear derivative to obtain a valid bit_mask consisting only of bits
        """
        prob = nn.sigmoid(self.w_dec)
        spikes = step_pwl(prob)
        return spikes


def autoencoder_quantization(
    weights_in: np.ndarray,
    weights_rec: np.ndarray,
    Iw_base: float,
    n_epoch: int = 100000,
    bits_per_weight: Optional[int] = 4,
    optimizer: str = "adam",
    step_size: Union[float, Callable[[int], float]] = lambda i: (
        1e-4 / (1.0 + 1e-4 * i)
    ),
    opt_params: Optional[Dict[str, Any]] = {},
) -> Dict[Any, Any]:

    if bits_per_weight > 4:
        raise ValueError("Up-to 4-bits representation supported")

    __handler = WeightHandler(weights_in, weights_rec)
    __encoder = DigitalAutoEncoder(__handler.w_flat.size, bits_per_weight)

    # Optimize #

    ## - Get the optimiser functions
    init_fun, update_fun, get_params = get_optimizer(optimizer, step_size, **opt_params)

    ## - Initialize the optimizer with the initial parameters
    params0 = deepcopy(__encoder.parameters())
    opt_state = init_fun(params0)

    ## - Get the jit compiled update and value-and-gradient functions
    loss_vgf = jit(
        value_and_grad(
            lambda params: QuantizationLoss.loss_reconstruction(
                __encoder, params, jnp.array(__handler.w_flat)
            )
        )
    )
    update_fun = jit(update_fun)

    def step(
        opt_state: optimizers.OptimizerState, epoch: int
    ) -> Tuple[Dict[str, jnp.DeviceArray], optimizers.OptimizerState, jnp.DeviceArray]:
        """
        step stacks together the single iteration step operations during training

        :param opt_state: the optimizer's current state
        :type opt_state: optimizers.OptimizerState
        :param epoch: the current epoch
        :type epoch: int
        :return: params, opt_state, loss_val
            :params: the network parameters
            :opt_state: the current time step optimizer state
            :loss_val: the current loss value
        :rtype: Tuple[Dict[str, jnp.DeviceArray], optimizers.OptimizerState, jnp.DeviceArray]
        """

        params = get_params(opt_state)
        loss_val, grads = loss_vgf(params)
        opt_state = update_fun(epoch, grads, opt_state)

        # Return
        return opt_state, loss_val

    # --- Iterate over epochs --- #
    epoch = jnp.array(range(n_epoch)).reshape(-1, 1)
    opt_state, loss_t = scan(step, opt_state, epoch)

    # Read the results
    optimized_encoder = __encoder.set_attributes(get_params(opt_state))
    __, code, bit_mask = optimized_encoder(__handler.w_flat)

    # --- Get the params --- #

    ## Weight bias currents
    code = np.array(code) * Iw_base
    get_weight_param = lambda idx: code[idx] if len(code) > idx else None

    ## Quantized weights
    q_weights = WeightHandler.bit2int_mask(bits_per_weight, bit_mask)
    qw_in, qw_rec = __handler.reshape_flat_weights(q_weights)

    spec = {
        "weights_in": qw_in,
        "sign_in": __handler.sign_in,
        "weights_rec": qw_rec,
        "sign_rec": __handler.sign_rec,
        "Iw_0": get_weight_param(0),
        "Iw_1": get_weight_param(1),
        "Iw_2": get_weight_param(2),
        "Iw_3": get_weight_param(3),
    }

    return np.array(loss_t), spec


def get_optimizer(
    name: str, *args, **kwargs
) -> Tuple[optimizers.InitFn, optimizers.UpdateFn, optimizers.ParamsFn]:
    """
    _get_optimizer calls the name-requested optimizer and returns the jax optimizer functions

    :param name: the name of the optimizer
    :type name: str
    :raises ValueError: Requested optimizer is not available!
    :return: the optimizer functions
    :rtype: Tuple[optimizers.InitFn, optimizers.UpdateFn, optimizers.ParamsFn]
    """

    if name == "sgd":
        return optimizers.sgd(*args, **kwargs)
    elif name == "momentum":
        return optimizers.momentum(*args, **kwargs)
    elif name == "nesterov":
        return optimizers.nesterov(*args, **kwargs)
    elif name == "adagrad":
        return optimizers.adagrad(*args, **kwargs)
    elif name == "rmsprop":
        return optimizers.rmsprop(*args, **kwargs)
    elif name == "rmsprop_momentum":
        return optimizers.rmsprop_momentum(*args, **kwargs)
    elif name == "adam":
        return optimizers.adam(*args, **kwargs)
    elif name == "adamax":
        return optimizers.adamax(*args, **kwargs)
    elif name == "sm3":
        return optimizers.sm3(*args, **kwargs)
    else:
        raise ValueError(
            f"The optimizer : {name} is not available!"
            f"Try one of the optimizer defined in `jax.example_libraries.optimizers'` : sgd, momentum, nesterov, adagrad, rmsprop, rmsprop_momentum, adam, adamax, sm3"
        )


@dataclass
class QuantizationLoss:
    @staticmethod
    def loss_reconstruction(
        encoder: JaxModule,
        parameters: Dict[str, Any],
        input: jnp.DeviceArray,
        f_penalty: float = 1e3,
    ) -> float:
        """
        loss calculates the mean square error loss between output and the target,
        given a new parameter set. Also, adds the bound violation penalties to the loss calculated.

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
        penalty = f_penalty * QuantizationLoss.penalty_negative(code)

        # - converting the bit_mask bit2int and int2bit should produce the same decoder
        penalty += f_penalty * QuantizationLoss.penalty_reconstruction(
            len(code), bit_mask
        )

        # - Calculate the loss imposing the bounds
        _loss = l.mse(output, input) + penalty
        return _loss

    @staticmethod
    def penalty_negative(param: jnp.DeviceArray) -> float:
        """
        penalty_negative applies a below zero limit violation penalty to any parameter

        :param param: the parameter to apply the zero limit
        :type param: jnp.DeviceArray
        :return: an exponentially increasing bound loss punishing the parameter values below zero
        :rtype: float
        """
        # - Bound penalty - #
        negatives = jnp.clip(param, None, 0)
        _loss = jnp.exp(-negatives)

        ## - subtract the code length from the sum to make the penalty 0 if all the code values are 0
        penalty = jnp.nansum(_loss) - float(param.size)
        return penalty

    @staticmethod
    def penalty_reconstruction(n_bits: int, bit_mask: jnp.DeviceArray) -> float:
        """
        penalty_reconstruction applies a penalty if the bit_mask encoding&decoding is non-unique.
        It also assures that the rounded decoding weights are the same as the bit_mask desired, and the
        bit_mask consists of binary values.

        :param n_bits: number of bits reserved for representing the integer values
        :type n_bits: int
        :param bit_mask: the bit_mask to check if encoding&decoding is unique
        :type bit_mask: jnp.DeviceArray
        :return: mean square error loss between the bit_mask found and the bitmap reconstructed after encoding decoding
        :rtype: float
        """
        int_mask = WeightHandler.bit2int_mask(n_bits, bit_mask, jnp)
        bit_mask_reconstructed = WeightHandler.int2bit_mask(n_bits, int_mask, jnp)
        penalty = l.mse(bit_mask, bit_mask_reconstructed)

        return penalty