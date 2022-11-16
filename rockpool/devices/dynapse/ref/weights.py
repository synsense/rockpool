"""
Weight Configuration package encapsulates
the implementation for Dynap-SE Weight Matrix -> CAM & Iw base conversion and the other way arond

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
21/01/2022

[] TODO : from_samna() # standalone
[] TODO : merging and disjoining weight matrices across cores # post-synaptic side is here
"""

from __future__ import annotations
import logging

from typing import Any, Callable, Dict, Optional, Tuple, Union

from copy import deepcopy
from dataclasses import dataclass

# JAX
from jax import jit, value_and_grad
from jax.lax import scan, cond
from jax.example_libraries import optimizers
from jax import numpy as jnp

# Rockpool
from rockpool.training import jax_loss as l
from rockpool.devices.dynapse.ref.autoencoder import (
    AutoEncoder,
    DigitalAutoEncoder,
    AnalogAutoEncoder,
)
from rockpool.devices.dynapse.definitions import ArrayLike, WeightRecord


@dataclass
class WeightParameters:
    """
    WeightParameters encapsulates weight currents of the configurable synapses between neurons.
    It provides a general way of handling SE2 weight current and the conversion from device
    configuration object

    :Attributes:

    :attr _optimizers: name list of all available optimizers
    :type _optimizers: List[str]

    :Parameters:

    :param weights: The weight matrix to obtain, co-depended to Iw[0], Iw[1], Iw[2], Iw[3] and intmask.
    :type weights: jnp.DeviceArray
    :param Iw: the base weight currents array, in Amperes.
    :type Iw: jnp.DeviceArray
    :param intmask: A binary value representing uint mask to select and dot product the base Iw currents (pre, post)
    :type intmask: jnp.DeviceArray

        1 = 0001 -> selected bias parameters: Iw[0]
        8 = 1000 -> selected bias parameters: Iw[3]
        5 = 0101 -> selected bias parameterss Iw[0] + Iw[2]

            array([[ 0,  1, 12,  0],
                   [11, 10,  4,  1],
                   [ 7,  0, 15, 15],
                   [13, 15, 15,  7],
                   [ 5,  3,  2, 12],
                   [ 5,  8,  5,  9]])

    :param code_length: the number of bits to reserve for a code, defaults to 4
    :type code_length: Optional[int], optional

    :Instance Variables:

    :ivar w_flat: scaled, flattened, and preprocessed weight matrix with only non-zero values
    :type w_flat: jnp.DeviceArray
    :ivar transforms: a dictionary of the transforms applied to the weight matrix
    :type transforms: Dict[str, Any]
    :ivar code_search: Looking for a code representation if True, else already known
    :type code_search: bool
    :ivar mask_search: Looking for a bitmask representation if True, else already known
    :type mask_search: bool
    :ivar ae: The autoencoder object to find a bitmask and/or Iw code representation
    :type ae: AutoEncoder
    :ivar code_implied: The Iw code implied by the Iw bits. non-zero if at least one of Iw[0], Iw[1], Iw[2], Iw[3] is provided
    :type code_implied: jnp.DeviceArray
    :ivar bitmask_implied: The bitmask implied by the intmask. non-zero if intmask is not None.
    :type bitmask_implied: jnp.DeviceArray

    NOTE : The current implementation finds the weight parameter and CAM configuration
    for one core. The core specific allocation has to be done by looking into the other parameters distribution.
    For example : one can apply k-means clustering to the tau parameters. Then according to the neuron allocation,
    one could extract the weight matrices per cores.
    """

    weights: Optional[jnp.DeviceArray] = None
    Iw: Optional[jnp.DeviceArray] = None
    intmask: Optional[jnp.DeviceArray] = None
    code_length: Optional[int] = 4

    _optimizers = [
        "sgd",
        "momentum",
        "nesterov",
        "adagrad",
        "rmsprop",
        "rmsprop_momentum",
        "adam",
        "adamax",
        "sm3",
    ]

    def __post_init__(self) -> None:
        """
        __post_init__ runs after __init__ and initializes the WeightParameters object with default values in the case that they are not specified.

        :raises ValueError: If `weight` is None, then intmask and weight bits are required to calculate the weight matrix
        """

        # Check if intmask & code_length is compatible
        self.intmask = self.intmask

        if self.weights is None:
            if self.Iw is None or self.intmask is None:
                raise ValueError(
                    "If `weight` is None, then intmask and weight bits are required to calculate the weight matrix!"
                )
            self.weights = self.weight_matrix()
            self.w_flat, self.transforms = self.preprocess(self.weights)

        elif self.weights is not None:
            self.weights = jnp.array(self.weights)
            if self.Iw is not None and self.intmask is not None:
                raise ValueError(
                    "Conflict of Interest: Define either the weight matrix or the intmask&Iw pair."
                    "Not all at the same time!"
                )

            ## - Search for the right code and and the bitmask
            self.code_search = True if self.Iw is None else False
            self.mask_search = True if self.intmask is None else False
            self.w_flat, self.transforms = self.preprocess(self.weights)

            if self.intmask is None:
                self.intmask = jnp.zeros(self.shape, int)

            ## - Initialize an autoencoder
            logging.info("Run .fit() to find weight parameters and bitmask!")
            self.ae = (
                DigitalAutoEncoder(self.w_flat.size, self.code_length)
                if self.code_search and self.mask_search
                else AnalogAutoEncoder(self.w_flat.size, self.code_length)
            )

        ## Implied bitmask and code, 0 if nothing is implied

        self.code_implied = (
            self.Iw * self.scale
            if self.Iw is not None
            else jnp.full(self.code_length, 0.0)
        )

        self.bitmask_implied = self.quantize_intmask(
            self.code_length, self.intmask[self.idx_nonzero]
        )

    def weight_matrix(self) -> jnp.DeviceArray:
        """
        weight_matrix generates a weight matrix for `DynapSE` modules using the base weight currents, and the intmask.
        In device, we have the opportunity to define 4 different base weight current. Then using a bit mask, we can compose a
        weight current defining the strength of the connection between two neurons. The parameters and usage explained below.

        :return: the weight matrix composed using the base weight parameters and the binary bit-mask.
        :rtype: jnp.DeviceArray
        """
        # To broadcast on the post-synaptic neurons : pre, post -> [(bits), post, pre].T
        bits_trans = self.quantize_intmask(self.code_length, self.intmask).T
        Iw = self.Iw if self.Iw is not None else jnp.zeros(self.code_length)
        w_rec = jnp.sum(bits_trans * Iw, axis=-1).T
        return w_rec

    def update_encoder(
        self, w_en: jnp.DeviceArray, w_dec: jnp.DeviceArray
    ) -> AutoEncoder:
        """
        update_encoder updates the parameters of the encoder object explicitly, and returns
        a copy of the original encoder with changed parameters. It does NOT CHANGE the original object!

        :param w_en: encoder weights
        :type w_en: jnp.DeviceArray
        :param w_dec: decoder weights
        :type w_dec: jnp.DeviceArray
        :return: Updated COPY of the original encoder
        :rtype: AutoEncoder
        """
        assert w_en.shape == self.ae.w_en.shape
        assert w_dec.shape == self.ae.w_dec.shape

        params = {
            "w_en": w_en,
            "w_dec": w_dec,
        }
        encoder = self.ae.set_attributes(params)
        return encoder

    ## OPTIMIZATION ##

    def loss(self, parameters: Dict[str, Any], f_penalty: float = 1e3) -> float:
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
        net = self.ae.set_attributes(parameters)
        output, code, bitmask = net(self.w_flat)

        # - Code should always be positive (reresent real current value) - #
        penalty = f_penalty * self.penalty_negative(code)

        # - Multiplexing and de-multiplexing the bitmask should produce the same bitmask
        penalty += self.penalty_reconstruction(len(code), bitmask)

        # - If the intermediate code representation is known, then push the system to generate the same - #
        penalty += cond(
            self.code_search,
            lambda _: 0.0,
            lambda code: f_penalty * l.mse(self.code_implied, code),
            code,
        )

        # - If the bitmask is given, then push the system, produce the same - #
        penalty += cond(
            self.mask_search,
            lambda _: 0.0,
            lambda bitmask: f_penalty * l.mse(self.bitmask_implied, bitmask),
            bitmask,
        )

        # - Calculate the loss imposing the bounds
        _loss = l.mse(output, self.w_flat) + penalty
        return _loss

    def fit(
        self,
        n_epoch: int = 10000,
        optimizer: str = "adam",
        step_size: Union[float, Callable[[int], float]] = lambda i: (
            1e-4 / (1.0 + 1e-4 * i)
        ),
        light: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[AutoEncoder, optimizers.OptimizerState, Dict[str, jnp.DeviceArray]]:
        """
        fit fit the autotencoder to the given weight matrix using a gradient based optimization method

        :param n_epoch: the number of epoches to iterate, defaults to 10000
        :type n_epoch: int, optional
        :param optimizer: one of the optimizer defined in `jax.example_libraries.optimizers` : , defaults to "adam"
        :type optimizer: str, optional
        :param step_size: positive scalar, or a callable representing a step size schedule that maps the iteration index to a positive scalar. , defaults to 1e-3
        :type step_size: Union[float, Callable[[int], float]], optional
        :param light: If true, use much less memory to compute, if not finds the ultimate least loss parameter set, defaults to False
        :type light: bool, optional
        :return: encoder, opt_state, loss_t
            :encoder: the best(low-loss) encoder encountered throughout iterations
            :opt_state: the last time step optimizer state
            :loss_t: loss value record over the iterations
        :rtype: Tuple[AutoEncoder, optimizers.OptimizerState, Dict[str, jnp.DeviceArray]]
        """

        ## - Get the optimiser functions
        init_fun, update_fun, get_params = self._get_optimizer(
            optimizer, step_size, *args, **kwargs
        )

        ## - Initialize the optimizer with the initial parameters
        params0 = deepcopy(self.ae.parameters())
        opt_state = init_fun(params0)

        ## - Get the jit compiled update and value-and-gradient functions
        loss_vgf = jit(value_and_grad(self.loss))
        update_fun = jit(update_fun)

        def step(
            opt_state: optimizers.OptimizerState, epoch: int
        ) -> Tuple[
            Dict[str, jnp.DeviceArray], optimizers.OptimizerState, jnp.DeviceArray
        ]:
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
            return params, opt_state, loss_val

        def iteration_full(
            opt_state: optimizers.OptimizerState, epoch: int
        ) -> Tuple[optimizers.OptimizerState, WeightRecord]:
            """
            iteration_full encapsulates `step()` function call and returns the state the loss value and the network parameters

            :param opt_state: the optimizer's current state
            :type opt_state: optimizers.OptimizerState
            :param epoch: the current epoch
            :type epoch: int
            :return: opt_state, rec
                :opt_state: the current time step optimizer state
                :rec: the step record including loss value, encoder weights, and decoder weights
            :rtype: Tuple[optimizers.OptimizerState, WeightRecord]
            """
            params, opt_state, loss_val = step(opt_state, epoch)

            # Return
            rec = (loss_val, params["w_en"], params["w_dec"])
            return opt_state, rec

        def iteration_light(
            opt_state: optimizers.OptimizerState, epoch: int
        ) -> Tuple[optimizers.OptimizerState, jnp.DeviceArray]:
            """
            iteration_light encapsulates `step()` function call and returns only the state and the loss

            :param opt_state: the optimizer's current state
            :type opt_state: optimizers.OptimizerState
            :param epoch: the current epoch
            :type epoch: int
            :return: opt_state, loss_val
                :opt_state: the current time step optimizer state
                :loss_val: the current loss value
            :rtype: Tuple[optimizers.OptimizerState, jnp.DeviceArray]
            """

            _, opt_state, loss_val = step(opt_state, epoch)

            # Return
            return opt_state, loss_val

        # --- Iterate over epochs --- #
        epoch = jnp.array(range(n_epoch)).reshape(-1, 1)

        if light:  # Use less memory
            opt_state, loss_t = scan(iteration_light, opt_state, epoch)
            params = get_params(opt_state)
            w_en, w_dec = params["w_en"], params["w_dec"]

        else:  # Find the best encoder
            opt_state, (loss_t, w_en_t, w_dec_t) = scan(
                iteration_full, opt_state, epoch
            )
            idx = jnp.argmin(loss_t)
            w_en, w_dec = w_en_t[idx], w_dec_t[idx]

        ## - Updated COPY of the original encoder
        encoder = self.update_encoder(w_en, w_dec)
        return encoder, opt_state, loss_t

    def fit_update(self, *args, **kwargs) -> None:
        """
        fit_update calls the fit method and update the related object instance variables
        """

        # Update ae
        self.ae, state, loss_t = self.fit(*args, **kwargs)

        ## Update intmask
        intmask_flat = self.multiplex_bitmask(self.ae.n_code, self.ae.bitmask)
        intmask_round = jnp.round(
            self.intmask.at[self.idx_nonzero].set(intmask_flat)
        ).astype(int)
        self.intmask = jnp.clip(intmask_round, 0, None)

        ## Update Iws
        code = self.ae.encode(self.w_flat)
        self.Iw = code / self.scale
        return loss_t

    ## OVERWRITES ##

    def __setattr__(self, __name: str, __value: Any) -> None:
        """
        __setattr__ hook when setting an object parameter for checking if codependend parameters are consistent

        :param __name: the name of the variable
        :type __name: str
        :param __value: the value to set
        :type __value: Any
        :raises ValueError: intmask includes elements exceeding the coding capacity!
        """
        if __name == "intmask" and __value is not None:
            __value = jnp.array(__value)
            if hasattr(self, "code_length") and self.code_length is not None:
                if (__value > (2 ** self.code_length - 1)).any():
                    raise ValueError(
                        "intmask includes elements exceeding the coding capacity!"
                    )

        super().__setattr__(__name, __value)

    ## - UTILITIES - ##

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
    def penalty_reconstruction(n_bits: int, bitmask: jnp.DeviceArray) -> float:
        """
        penalty_reconstruction applies a penalty if the bitmask encoding&decoding is non-unique.
        It also assures that the rounded decoding weights are the same as the bitmask desired, and the
        bitmask consists of binary values.

        :param n_bits: number of bits reserved for representing the integer values
        :type n_bits: int
        :param bitmask: the bitmask to check if encoding&decoding is unique
        :type bitmask: jnp.DeviceArray
        :return: mean square error loss between the bitmask found and the bitmap reconstructed after encoding decoding
        :rtype: float
        """
        intmask = (
            WeightParameters.multiplex_bitmask(n_bits, bitmask).round().astype(int)
        )
        bitmask_reconstructed = WeightParameters.quantize_intmask(
            n_bits, intmask
        ).astype(float)
        penalty = l.mse(bitmask, bitmask_reconstructed)

        return penalty

    @staticmethod
    def preprocess(weights: jnp.DeviceArray) -> Tuple[jnp.DeviceArray, Dict[str, Any]]:
        """
        preprocess preprocess a weight matrix to obtain a flat, non-zero, scaled
        version which would be a better candidate for gradient-based auto-encoding

        :param weights: any matrix
        :type weights: jnp.DeviceArray
        :raises ValueError: Weight matrix provided does not have a proper shape! It should be 2-dimensional with (pre,post)!
        :return: w_flat, transforms
            w_flat: scaled, flattened, and preprocessed weight matrix
            transforms: the transforms applied to the weight matrix
        :rtype: Tuple[jnp.DeviceArray, Dict[str, Any]]
        """
        if len(weights.shape) != 2:
            raise ValueError(
                "Weight matrix provided does not have a proper shape! It should be 2-dimensional with (pre,post)!"
            )

        diff = jnp.max(weights) - jnp.min(weights) if len(weights) > 0 else 0
        scale = 1.0 / diff if diff > 0 else 1.0
        idx_nonzero = weights.astype(bool)

        transforms = {
            "shape": weights.shape,
            "scale": scale,
            "idx_nonzero": idx_nonzero,
            "n_post": weights.shape[1],
        }

        w_flat = weights[idx_nonzero].flatten() * scale

        return w_flat, transforms

    @staticmethod
    def _get_optimizer(
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

    @staticmethod
    def quantize_intmask(n_bits: int, intmask: jnp.DeviceArray) -> jnp.DeviceArray:
        """
        quantize_intmask converts a integer valued bitmask to 4 dimension (4-bits) bitmask representing the indexes of the selection

            (n_bits=4)

            1 = 0001 -> selected bit: 0
            8 = 1000 -> selected bit: 3
            5 = 0101 -> selected bit 0 and 2

        :param n_bits: number of bits reserved for representing the integer values
        :type n_bits: int
        :param intmask: Integer values representing binary numbers to select (shape,)
        :type intmask: jnp.DeviceArray
        :return: an array of indices of selected bits, only binary values, (n_bits,shape)
        :rtype: jnp.DeviceArray
        """

        pattern = jnp.array([1 << n for n in range(n_bits)])  # [1,2,4,8, ..]
        intmask_ext = jnp.full((n_bits, *intmask.shape), intmask)  # (n_bits,shape)

        # Indexes of the IDs to be selected in bits list
        bitmask = jnp.bitwise_and(intmask_ext.T, pattern).T.astype(
            bool
        )  # (n_bits,shape)
        return bitmask

    @staticmethod
    def multiplex_bitmask(n_bits: int, bitmask: jnp.DeviceArray) -> jnp.DeviceArray:
        """
        multiplex_bitmask apply 4-bit selection to binary values representing select bits and generates a compressed bitmask

            (n_bits=4)

            [0,0,0,1] -> 1
            [1,0,0,0] -> 8
            [0,1,0,1] -> 5

        :param n_bits: number of bits reserved for representing the integer values
        :type n_bits: int
        :param bitmask: an array of indices of selected bits, only binary values, (n_bits,shape)
        :type bitmask: jnp.DeviceArray
        :return: integer values representing binary numbers (shape,)
        :rtype: jnp.DeviceArray
        """
        pattern = jnp.array([1 << n for n in range(n_bits)])  # [1,2,4,8, ..]
        intmask = jnp.sum(bitmask.T * pattern, axis=-1).T
        return intmask

    ## PROPERTIES ##

    @property
    def mse(self):
        """
        mse calculates the Mean Square Error loss between the reconstructed weights and the original weight matrix
        """
        loss = l.mse(self.weight_matrix(), self.weights)
        return loss

    @property
    def shape(self) -> Tuple[int]:
        """
        shape returns weight matrix shape
        """
        return self.transforms["shape"]

    @property
    def scale(self) -> float:
        """
        scale returns weight matrix scale : size / sum
        """
        return self.transforms["scale"]

    @property
    def idx_nonzero(self) -> jnp.DeviceArray:
        """
        idx_nonzero returns the indexes of the elements with the non-zero values
        """
        return self.transforms["idx_nonzero"]

    @property
    def n_post(self) -> int:
        """
        n_post returns the number of post-synaptic neurons implicitly indicated in weights
        """
        return self.transforms["n_post"]
