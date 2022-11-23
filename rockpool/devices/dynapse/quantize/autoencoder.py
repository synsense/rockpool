"""
Dynap-SE autoencoder based quantization package provides easy to use support

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

Previous : config/autoencoder.py @ 220127

15/09/2022

"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from dataclasses import dataclass
from copy import deepcopy

# JAX
import jax
from jax import jit, value_and_grad
from jax import numpy as jnp
from jax.nn import sigmoid
from jax.lax import scan
from jax.example_libraries import optimizers

# Rockpool
from rockpool.nn.modules.jax.jax_module import JaxModule
from rockpool.training import jax_loss as l
from rockpool.devices.dynapse.quantize.weight_handler import WeightHandler
from rockpool.parameters import Parameter
from rockpool.nn.modules.native.linear import kaiming


__all__ = ["autoencoder_quantization"]


### --- UTILITY FUNCTIONS --- ###


def autoencoder_quantization(
    ## cluster
    n_cluster: int,
    core_map: List[int],
    ## Input
    weights_in: np.ndarray,
    weights_rec: np.ndarray,
    Iscale: float,
    n_bits: Optional[int] = 4,
    ## Optimization
    fixed_epoch: bool = False,
    num_epoch: int = int(1e7),
    num_epoch_checkpoint: int = int(1e3),
    eps: float = 1e-6,
    record_loss: bool = True,
    optimizer: str = "adam",
    step_size: Union[float, Callable[[int], float]] = 1e-4,
    opt_params: Optional[Dict[str, Any]] = {},
    *args,
    **kwargs,
) -> Dict[str, Union[np.ndarray, float]]:

    spec = {
        "weights_in": [],
        "sign_in": [],
        "weights_rec": [],
        "sign_rec": [],
        "Iw_0": [],
        "Iw_1": [],
        "Iw_2": [],
        "Iw_3": [],
        "quantization_loss": [],
    }
    n_neuron = weights_rec.shape[1]

    for n in range(n_cluster):

        w_in = np.zeros_like(weights_in)
        w_rec = np.zeros_like(weights_rec)

        w_in[:, core_map == n] = weights_in[:, core_map == n]
        w_rec[:, core_map == n] = weights_rec[:, core_map == n]

        __temp = single_autoencoder(
            w_in,
            w_rec,
            Iscale[n],
            n_bits,
            fixed_epoch,
            num_epoch,
            num_epoch_checkpoint,
            eps,
            record_loss,
            optimizer,
            step_size,
            opt_params,
        )

        for key in spec:
            spec[key].append(__temp[key])

    return spec


def single_autoencoder(
    ## Input
    weights_in: np.ndarray,
    weights_rec: np.ndarray,
    Iscale: float,
    n_bits: Optional[int] = 4,
    ## Optimization
    fixed_epoch: bool = False,
    num_epoch: int = int(1e7),
    num_epoch_checkpoint: int = int(1e3),
    eps: float = 1e-6,
    record_loss: bool = True,
    optimizer: str = "adam",
    step_size: Union[float, Callable[[int], float]] = 1e-4,
    opt_params: Optional[Dict[str, Any]] = {},
    *args,
    **kwargs,
) -> Dict[str, Union[np.ndarray, float]]:
    """
    autoencoder_quantization is a utility function to use the autoencoder quantization approach
    in deployment pipelines. One can experiment with the parameters to control the autoencoder training.

    :param weights_in: input layer weights used in Dynap-SE2 simulation
    :type weights_in: Optional[np.ndarray]
    :param weights_rec: recurrent layer (in-device neurons) weights used in Dynap-SE2 simulation
    :type weights_rec: Optional[np.ndarray]
    :param Iscale: base weight scaling current in Amperes used in simulation
    :type Iscale: float
    :param n_bits: number of target weight bits, defaults to 4
    :type n_bits: Optional[int], optional
    :param fixed_epoch: used fixed number of epochs or control the convergence by loss decrease, defaults to False
    :type fixed_epoch: bool, optional
    :param num_epoch: the fixed number of epochs as global limit, defaults to 10,000,000
    :type num_epoch: int, optional
    :param num_epoch_checkpoint: at this point (number of epochs), pipeline checks the loss decrease and decide to continue or not, defaults to 1,000.
    :type num_epoch_checkpoint: int, optional
    :param eps: the epsilon tolerance value. If the loss does not decrease more than this for five consecutive checkpoints, then training stops. defaults to 1e-6
    :type eps: float, optional
    :param record_loss: record the loss evolution or not, defaults to True
    :type record_loss: bool, optional
    :param optimizer: one of the optimizer defined in `jax.example_libraries.optimizers` : , defaults to "adam"
    :type optimizer: str, optional
    :param step_size: positive scalar, or a callable representing a step size schedule that maps the iteration index to a positive scalar. , defaults to 1e-4
    :type step_size: Union[float, Callable[[int], float]], optional
    :param opt_params: optimizer parameters dictionary, defaults to {}
    :type opt_params: Optional[Dict[str, Any]]
    :return: A dictionary of quantized weights and parameters, the quantization loss
    :rtype: Dict[str, Union[np.ndarray, float]]
    """

    ### --- Initial Object Construction --- ###

    if not isinstance(Iscale, float):
        raise ValueError("Iscale should be float!")

    # weights might not be defined, still return
    if weights_in is None and weights_rec is None:
        spec = {
            "weights_in": None,
            "sign_in": None,
            "weights_rec": None,
            "sign_rec": None,
            "Iw_0": 0.0,
            "Iw_1": 0.0,
            "Iw_2": 0.0,
            "Iw_3": 0.0,
            "quantization_loss": None,
        }
        return spec

    __handler = WeightHandler(weights_in, weights_rec)
    __encoder = DigitalAutoEncoder(__handler.w_flat.size, n_bits)

    ### --- Optimization Configuration --- ###

    ## - Get the optimiser functions
    init_fun, update_fun, get_params = __get_optimizer(
        optimizer, step_size, **opt_params
    )

    ## - Initialize the optimizer with the initial parameters
    opt_state = init_fun(deepcopy(__encoder.parameters()))

    ## - Preprocess
    __scale = lambda w: (1.0 / (jnp.max(w) - jnp.min(w) + 1))
    __scale_factor = __scale(__handler.w_flat)
    w_flat = __scale_factor * jnp.array(__handler.w_flat)

    ## - Get the jit compiled update and value-and-gradient functions
    loss_vgf = jit(
        value_and_grad(
            lambda params: QuantizationLoss.loss_reconstruction(
                __encoder, params, w_flat
            )
        )
    )
    update_fun = jit(update_fun)
    run_for = jit(
        lambda epoch, state: __run_for(epoch, state, get_params, loss_vgf, update_fun)
    )

    ### --- Optimize --- ###

    ## - Check the loss decrease and decide to stop training before it reaches to num_epochs
    if not fixed_epoch:

        count = 0
        rec_loss = []
        mean_loss = jnp.inf
        epoch = jnp.array(range(num_epoch_checkpoint)).reshape(-1, 1)

        for _ in range(0, num_epoch, num_epoch_checkpoint):
            opt_state, loss_t = run_for(epoch, opt_state)

            if record_loss:
                rec_loss += list(np.array(loss_t))

            ### Check the mean loss at each num_epoch_checkpoint
            if mean_loss - jnp.mean(loss_t) < eps:
                count += 1
                if count > 5:
                    break
            else:
                count = 0
                mean_loss = jnp.mean(loss_t)

    ## - Just repeat the process for the number of epochs
    else:
        epoch = jnp.array(range(num_epoch)).reshape(-1, 1)
        opt_state, rec_loss = run_for(epoch, opt_state)

    ### ---  Read the results --- ###

    optimized_encoder = __encoder.set_attributes(get_params(opt_state))
    __, code, bit_mask = optimized_encoder(w_flat)

    ## - Quantized weights
    q_weights = WeightHandler.bit2int_mask(n_bits, bit_mask)
    qw_in, qw_rec = __handler.reshape_flat_weights(q_weights)

    ### --- Return --- ###
    Iw = np.array(code) * Iscale / np.array(__scale_factor)

    spec = {
        "weights_in": qw_in,
        "sign_in": __handler.sign_in,
        "weights_rec": qw_rec,
        "sign_rec": __handler.sign_rec,
        "Iw_0": Iw[0],
        "Iw_1": Iw[1],
        "Iw_2": Iw[2],
        "Iw_3": Iw[3],
        "quantization_loss": float(rec_loss[-1]),
    }

    return spec


def __run_for(
    epoch: jnp.array,
    opt_state: optimizers.OptimizerState,
    get_params: optimizers.ParamsFn,
    loss_vgf: Callable[[Any], Tuple[float]],
    update_fun: optimizers.UpdateFn,
) -> Tuple[optimizers.OptimizerState, jnp.DeviceArray]:
    """
    __run_for is a utility function executing jax training workflow

    :param epoch: the dummy sequence array [0,1,2,3..] standing for epoch ids to be walked through
    :type epoch: jnp.array
    :param opt_state: the optimizer's initial state
    :type opt_state: optimizers.OptimizerState
    :param get_params: the optimizer's parameter getter
    :type get_params: optimizers.ParamsFn
    :param loss_vgf: the loss function returning the loss value and the gradient value
    :type loss_vgf: Callable[[Any], Tuple[float]]
    :param update_fun: the optimizers update functions
    :type update_fun: optimizers.UpdateFn
    :return: opt_state, loss_val
        :opt_state: the last optimized state recorded at the end of the last epoch
        :loss_val: the recorded loss values over epochs
    :rtype: Tuple[optimizers.OptimizerState, jnp.DeviceArray]
    """

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
    opt_state, loss_t = scan(step, opt_state, epoch)
    return opt_state, loss_t


def __get_optimizer(
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


### --- Custom gradient implementation --- ##


@jax.custom_jvp
def step_pwl(probs: jnp.DeviceArray) -> jnp.DeviceArray:
    """
    step_pwl is heaviside step function with piece-wise linear derivative to use as thresholded probability value surrogate

    :param probs: a probability array
    :type probs: jnp.DeviceArray
    :return: the thresholded probability values
    :rtype: float
    """
    thresholded = jnp.clip(jnp.floor(probs + 0.5), 0.0)
    return thresholded


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
    (probs,) = primals
    (probs_dot,) = tangents
    probs_dot = probs_dot * jnp.clip(probs, 0.0)
    return step_pwl(*primals), probs_dot


### --- MODULES --- ###


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
        prob = sigmoid(self.w_dec)
        spikes = step_pwl(prob)
        return spikes


@dataclass
class QuantizationLoss:
    """
    QuantizationLoss gathers together the loss function utilities
    """

    @staticmethod
    def loss_reconstruction(
        encoder: DigitalAutoEncoder,
        parameters: Dict[str, Any],
        input: jnp.DeviceArray,
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
