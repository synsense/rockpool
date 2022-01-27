"""
Weight Configuration package encapsulates
the implementation for Dynap-SE Weight Matrix -> CAM & Iw base conversion and the other way arond

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
21/01/2022
[] TODO : dual search, depended search!
[] TODO : Get Iw as vector, do not restrict yourself on 4
[] TODO : if bitmask defined, then construct w_dec ?? harder
[] TODO : make sure that bitmask is jnp.DeviceArray
[] TODO : from_samna_parameters() # requires bitmask from router
[] TODO : from_samna() # standalone
[] TODO : merging and disjoining weight matrices across cores # post-synaptic side is here
[] TODO : what happens if Iw=0, reduce the code length being aware that the order is important
[] TODO : Iw=0, reduce bits, parametrize 4
[] TODO : Try with L1 regularization
"""
from __future__ import annotations
import logging

from typing import Any, Callable, Dict, Optional, Tuple, Union

from copy import deepcopy
from dataclasses import dataclass

# JAX
from jax import jit, value_and_grad
from jax.lax import scan
from jax.experimental import optimizers

from jax import numpy as jnp
import numpy as np

# Rockpool
from rockpool.training import jax_loss as l
from rockpool.devices.dynapse.config.layout import DynapSELayout
from rockpool.devices.dynapse.config.circuits import SimulationParameters
from rockpool.devices.dynapse.config.autoencoder import (
    AutoEncoder,
    DigitalAutoEncoder,
    AnalogAutoEncoder,
)

_SAMNA_SE1_AVAILABLE = True
_SAMNA_SE2_AVAILABLE = True

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


@dataclass
class WeightParameters:
    """
    WeightParameters encapsulates weight currents of the configurable synapses between neurons. 
    It provides a general way of handling SE2 weight current and the conversion from device 
    configuration object

    :param weights: The weight matrix to obtain, co-depended to Iw_0, Iw_1, Iw_2, Iw_3 and mux.
    :type weights: jnp.DeviceArray
    :param Iw_0: the first base weight current corresponding to the 0th bit of the bit-mask, in Amperes. In DynapSE1, it's GABA_B base weigth.
    :type Iw_0: float
    :param Iw_1: the second base weight current corresponding to the 1st bit of the bit-mask, in Amperes. In DynapSE1, it's GABA_A base weigth.
    :type Iw_1: float
    :param Iw_2: the third base weight current corresponding to the 2nd bit of the bit-mask, in Amperes. In DynapSE1, it's NMDA base weigth.
    :type Iw_2: float
    :param Iw_3: the fourth base weight current corresponding to the 3rd bit of the bit-mask, in Amperes. In DynapSE1, it's AMPA base weigth.
    :type Iw_3: float
    :param mux: A binary value representing uint mask to select and dot product the base Iw currents (shape,)
    :type mux: np.ndarray
        1 = 0001 -> selected bias parameters: Iw_0
        8 = 1000 -> selected bias parameters: Iw_3
        5 = 0101 -> selected bias parameterss Iw_0 + Iw_2
            array([[[ 0,  1, 12,  0],
                    [11, 10,  4,  1],
                    [ 7,  0, 15, 15],
                    [13, 15, 15,  7],
                    [ 5,  3,  2, 12],
                    [ 5,  8,  5,  9]],

                   [[12, 13,  9,  0],
                    [12, 12, 11,  2],
                    [10, 15,  9, 14],
                    [ 6,  8, 10,  8],
                    [15,  1,  1,  9],
                    [ 5,  2,  7, 13]],

                   [[ 2, 12, 14, 10],
                    [ 0,  3,  9,  0],
                    [ 6,  1, 11,  5],
                    [ 0,  2,  7,  1],
                    [ 7, 15,  2,  6],
                    [15, 11,  7,  7]],

                   [[11, 13, 13, 12],
                    [ 2,  9,  2,  3],
                    [ 9,  2, 12,  1],
                    [11, 11,  1,  4],
                    [15,  7,  5,  7],
                    [ 0, 13,  2,  3]],

                   [[ 0,  6, 10,  3],
                    [14, 10,  4, 10],
                    [ 8, 10,  6,  6],
                    [ 2,  3, 10,  6],
                    [ 1,  8, 10, 15],
                    [ 4,  3,  2, 12]],

                   [[13,  5,  3,  6],
                    [12, 14,  5,  3],
                    [ 4,  4,  3, 14],
                    [ 3, 13, 11, 10],
                    [ 1, 13,  5, 13],
                    [15,  2,  4,  2]]])
    
    NOTE : The current implementation finds the weight parameter and CAM configuration
    for one core. The core specific allocation has to be done by looking into the other parameters distribution.
    For example : one can apply k-means clustering to the tau parameters. Then according to the neuron allocation,
    one could extract the weight matrices per cores.
    """

    weights: Optional[jnp.DeviceArray] = None
    Iw_0: Optional[float] = None
    Iw_1: Optional[float] = None
    Iw_2: Optional[float] = None
    Iw_3: Optional[float] = None
    mux: Optional[np.ndarray] = None
    layout: Optional[DynapSELayout] = None

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

        :raises ValueError: If `weight` is None, then mux and weight bits are required to calculate the weight matrix
        """

        if self.layout is None:
            self.layout = DynapSELayout()

        if self.weights is None:
            if (
                (self.Iw_0 is None)
                or (self.Iw_1 is None)
                or (self.Iw_2 is None)
                or (self.Iw_3 is None)
                or (self.mux is None)
            ):
                raise ValueError(
                    "If `weight` is None, then mux and weight bits are required to calculate the weight matrix!"
                )
            self.weights = self.weight_matrix()

        self.w_flat, self.transforms = self.preprocess(self.weights)

        if self.weights is not None:
            if self.mux is None:
                self.mux = jnp.zeros(self.shape, int)
            else:
                pass  # [] TODO : initialize decoder weights

            self.code_search = False if self.Iw.nonzero()[0].size > 2 else True
            logging.info("Run .fit() to find weight parameters and bitmask!")
            # [] TODO : Parametrize 4

            # self.ae = AutoEncoder(self.w_flat.size, 4, code_search=code_search)
            self.ae = (
                DigitalAutoEncoder(self.w_flat.size, 4)
                if self.code_search
                else AnalogAutoEncoder(self.w_flat.size, 4)
            )

        # If most of the Iw's are defined, then there is a code implied!
        self.code_implied = self.Iw * self.scale
        self.idx_optimize = self.Iw.astype(bool)

    @classmethod
    def from_samna_parameters(
        cls,
        samna_parameters: Dict[str, Union[Dynapse1Parameter, Dynapse2Parameter]],
        mux: np.ndarray,
        layout: DynapSELayout,
        *args,
        **kwargs,
    ) -> WeightParameters:
        """
        from_samna_parameters is a factory method to construct a `WeightParameters` object using a samna config object

        :param samna_parameters: a parameter dictionary inside samna config object for setting the parameter group within one core
        :type samna_parameters: Dict[str, Union[Dynapse1Parameter, Dynapse2Parameter]]
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSELayout
        :return: a `WeightParameters` object, whose parameters obtained from the hardware configuration
        :rtype: WeightParameters
        """

        simparam = SimulationParameters(samna_parameters)

        mod = cls(
            Iw_0=simparam.nominal("Iw_0"),  # GABA_B - se1
            Iw_1=simparam.nominal("Iw_1"),  # GABA_A - se1
            Iw_2=simparam.nominal("Iw_2"),  # NMDA - se1
            Iw_3=simparam.nominal("Iw_3"),  # AMPA - se1
            mux=mux,
            layout=layout,
            *args,
            **kwargs,
        )
        return mod

    def _update_encoder(
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

    def weight_matrix(self) -> jnp.DeviceArray:
        """
        weight_matrix generates a weight matrix for `DynapSE` modules using the base weight currents, and the mux.
        In device, we have the opportunity to define 4 different base weight current. Then using a bit mask, we can compose a
        weight current defining the strength of the connection between two neurons. The parameters and usage explained below.

        :return: the weight matrix composed using the base weight parameters and the binary bit-mask.
        :rtype: jnp.DeviceArray
        """
        # To broadcast on the post-synaptic neurons : pre, post, gate -> [(bits), post, pre, gate].T
        bits_trans = self.quantize_mux(self.mux.transpose(1, 0, 2)).T
        # Restore the shape : (gate, pre, post) -> pre, post, gate
        w_rec = jnp.sum(bits_trans * self.Iw, axis=-1).transpose(1, 2, 0)
        return w_rec

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

        def penalty_negative(param: jnp.DeviceArray) -> float:
            """
            penalty_negative applies a below zero limit violation penalty to any attribute

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

        def penalty_reconstruction(bitmask: jnp.DeviceArray) -> float:
            """
            penalty_reconstruction applies a penalty if the bitmask encoding&decoding is non-unique. 
            It also assures that the rounded decoding weights are the same as the bitmask desired, and the 
            bitmask consists of binary values.

            :param bitmask: the bitmask to check if encoding&decoding is unique
            :type bitmask: jnp.DeviceArray
            :return: mean square error loss between the bitmask found and the bitmap reconstructed after encoding decoding
            :rtype: float
            """
            mux = self.multiplex_bitmask(bitmask).round().astype(int)
            bitmask_reconstructed = self.quantize_mux(mux).astype(float)
            penalty = l.mse(bitmask, bitmask_reconstructed)

            return penalty

        def penalty_code_difference(code: jnp.DeviceArray) -> float:
            """
            penalty_code_difference applies a penalty if the code is different then the code indicated by
            the weight parameters

            :param code: the code to compare against the Iw-indicated code (in the case that Iws defined)
            :type code: jnp.DeviceArray
            :return: the mean square error between the Iw-indicated and auto-encoded code
            :rtype: float
            """
            target_code = code.at[self.idx_optimize].set(
                self.code_implied[self.idx_optimize]
            )
            penalty = l.mse(target_code, code)
            return penalty

        # - Assign the provided parameters to the network
        net = self.ae.set_attributes(parameters)
        output, code, bitmask = net(self.w_flat)

        # - Code Implied - #
        penalty = f_penalty * penalty_negative(code)
        penalty += f_penalty * penalty_code_difference(code)

        # - Bitmap Implied - #
        penalty += (1.0 - self.code_search) * penalty_reconstruction(bitmask)

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
        record: bool = True,
        *args,
        **kwargs,
    ) -> Tuple[AutoEncoder, optimizers.OptimizerState, Dict[str, jnp.DeviceArray]]:
        """
        fit fit the autotencoder to the given weight matrix using a gradient based optimization method

        :param n_epoch: the number of epoches to iterate, defaults to 10000
        :type n_epoch: int, optional
        :param optimizer: one of the optimizer defined in `jax.experimental.optimizers` : , defaults to "adam"
        :type optimizer: str, optional
        :param step_size: positive scalar, or a callable representing a step size schedule that maps the iteration index to a positive scalar. , defaults to 1e-3
        :type step_size: Union[float, Callable[[int], float]], optional
        :param record: record the parameter changes through iteration steps or not, defaults to True
        :type record: bool, optional
        :return: encoder, opt_state, record_dict
            :encoder: the best(low-loss) encoder encountered throughout iterations
            :opt_state: the last time step optimizer state
            :record_dict: the record dictionary including loss value, encoder weights, and decoder weights
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

        def iteration(
            opt_state: optimizers.OptimizerState, epoch: int
        ) -> Tuple[optimizers.OptimizerState, WeightRecord]:
            """
            iteration stacks together the single iteration step operations during training

            :param opt_state: the optimizer's current state
            :type opt_state: optimizers.OptimizerState
            :param epoch: the current epoch
            :type epoch: int
            :return: opt_state, rec
                :opt_state: the current time step optimizer state
                :rec: the step record including loss value, encoder weights, and decoder weights
            :rtype: Tuple[optimizers.OptimizerState, WeightRecord]
            """

            params = get_params(opt_state)
            loss_val, grads = loss_vgf(params)
            opt_state = update_fun(epoch, grads, opt_state)

            # Return
            rec = (loss_val, params["w_en"], params["w_dec"])
            return opt_state, rec

        # --- Iterate over epochs --- #
        epoch = jnp.array(range(n_epoch)).reshape(-1, 1)
        opt_state, (loss_t, w_en_t, w_dec_t) = scan(iteration, opt_state, epoch)

        ### --- RETURN --- ###
        record_dict = {}

        if record:
            record_dict = {"loss": loss_t, "w_en": w_en_t, "w_dec": w_dec_t}

        ## - Updated COPY of the original encoder
        idx = jnp.argmin(loss_t)
        encoder = self._update_encoder(w_en_t[idx], w_dec_t[idx])

        return encoder, opt_state, record_dict

    def fit_update(self, *args, **kwargs) -> None:
        """
        fit_update calls the fit method and update the related object instance variables
        # [] TODO : Instead of hard-coded update, add more functions
        """

        # Update ae
        self.ae, state, record_dict = self.fit(*args, **kwargs)

        ## Update mux
        mux_flat = self.multiplex_bitmask(self.ae.bitmask)
        self.mux = jnp.round(self.mux.at[self.idx_nonzero].set(mux_flat)).astype(int)

        ## Update Iws
        code = self.ae.encode(self.w_flat)
        self.Iw_0 = code[0] / self.scale
        self.Iw_1 = code[1] / self.scale
        self.Iw_2 = code[2] / self.scale
        self.Iw_3 = code[3] / self.scale
        return record_dict["loss"]

    def __getattribute__(self, __name: str) -> Any:
        """
        If Iw_bits are None, warn the user that they should run .fit()
        """
        _attr = super().__getattribute__(__name)
        if (
            __name == "Iw_0"
            or __name == "Iw_1"
            or __name == "Iw_2"
            or __name == "Iw_3"
            or __name == "mux"
        ):
            if _attr is None:
                pass  # [] TODO : Hook warnings in the case they return None
        return _attr

    @staticmethod
    def preprocess(weights: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        preprocess preprocess a weight matrix to obtain a flat, non-zero, scaled
        version which would be a better candidate for gradient-based auto-encoding

        :param weights: any matrix
        :type weights: np.ndarray
        :raises ValueError: Weight matrix provided does not have a proper shape! It should be 3-dimensional with (pre,post,gate)!
        :return: w_flat, transforms
            w_flat: scaled, flattened, and preprocessed weight matrix
            transforms: the transforms applied to the weight matrix
        :rtype: Tuple[np.ndarray, Dict[str, Any]]
        """
        if len(weights.shape) != 3:
            raise ValueError(
                "Weight matrix provided does not have a proper shape! It should be 3-dimensional with (pre,post,gate)!"
            )

        diff = np.max(weights) - np.min(weights)
        scale = 1.0 / diff if diff > 0 else 1.0
        idx_nonzero = weights.astype(bool)

        transforms = {
            "shape": weights.shape,
            "scale": scale,
            "idx_nonzero": idx_nonzero,
            "n_post": weights.shape[-2],
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
                f"Try one of the optimizer defined in `jax.experimental.optimizers'` : sgd, momentum, nesterov, adagrad, rmsprop, rmsprop_momentum, adam, adamax, sm3"
            )

    @staticmethod
    def quantize_mux(mux: Union[jnp.DeviceArray, np.ndarray]) -> jnp.DeviceArray:
        """
        quantize_mux converts a integer valued bitmask to 4 dimension (4-bits) bitmask representing the indexes of the selection
            
            1 = 0001 -> selected bit: 0
            8 = 1000 -> selected bit: 3
            5 = 0101 -> selected bit 0 and 2

        :param mux: Integer values representing binary numbers to select (shape,)
        :type mux: np.ndarray
        :return: an array of indices of selected bits, only binary values, (4,shape)
        :rtype: jnp.DeviceArray
        """
        bits = range(4)  # [0,1,2,3]
        bit_pattern = lambda n: (1 << n)  # 2^n

        # Indexes of the IDs to be selected in bits list
        bitmask = jnp.array([mux & bit_pattern(bit) for bit in bits], dtype=bool)
        return bitmask

    @staticmethod
    def multiplex_bitmask(bitmask: jnp.DeviceArray) -> jnp.DeviceArray:
        """
        multiplex_bitmask apply 4-bit selection to binary values representing select bits and generates a compressed bitmask 
            
            [0,0,0,1] -> 1
            [1,0,0,0] -> 8
            [0,1,0,1] -> 5

        :param bitmask: an array of indices of selected bits, only binary values, (4,shape)
        :type bitmask: jnp.DeviceArray
        :return: integer values representing binary numbers (shape,)
        :rtype: jnp.DeviceArray
        """
        mux = jnp.sum(bitmask.T * jnp.array([1, 2, 4, 8]), axis=-1).T
        return mux

    @property
    def mse(self):
        """
        mse calculates the Mean Square Error loss between the reconstructed weights and the original weight matrix
        """
        loss = l.mse(self.weight_matrix(), self.weights)
        return loss

    @property
    def Iw(self) -> jnp.DeviceArray:
        """
        Iw returns weight matrix required
        """
        i0 = self.Iw_0 if self.Iw_0 is not None else 0.0
        i1 = self.Iw_1 if self.Iw_1 is not None else 0.0
        i2 = self.Iw_2 if self.Iw_2 is not None else 0.0
        i3 = self.Iw_3 if self.Iw_3 is not None else 0.0

        return jnp.array([i0, i1, i2, i3])

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
    def idx_nonzero(self) -> np.ndarray:
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

