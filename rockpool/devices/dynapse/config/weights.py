"""
Weight Configuration package encapsulates
the implementation for Dynap-SE Weight Matrix -> CAM & Iw base conversion and the other way arond

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
21/01/2022
[] TODO : Refactor compatible with circuit parameters
[] TODO : Select training w_en or not
[] TODO : if Iws defined, then construct w_en
[] TODO : if bitmask defined, then construct w_dec ?? harder
"""
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
from rockpool.nn.modules.native.linear import kaiming

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
    :type weight_init: Callable[[int], np.ndarray], optional
    """

    def __init__(
        self,
        shape: Tuple[int],
        n_code: int = 4,
        w_en: Optional[jnp.DeviceArray] = None,
        w_dec: Optional[jnp.DeviceArray] = None,
        weight_init: Callable[[int], np.ndarray] = kaiming,
        *args,
        **kwargs,
    ) -> None:
        """
        __init__ initialize the `AutoEncoder` module. Parameters are explained in the class docstring.
        """

        super(AutoEncoder, self).__init__(
            shape=shape, *args, **kwargs,
        )

        # Encoder weights should be non-negative
        en_init = lambda s: jnp.array(abs(weight_init(s)))
        self.w_en = Parameter(w_en, init_func=en_init, shape=(self.size_in, n_code),)

        # Decoder wegiths should include both positive and negative values
        dec_init = lambda s: jnp.array(weight_init(s))
        self.w_dec = Parameter(
            w_dec, init_func=dec_init, shape=(n_code, self.size_out),
        )

        # Bounds
        self.lower_bounds, _ = l.make_bounds(self.parameters())
        self.lower_bounds["w_en"] = 0

    def evolve(self, matrix: jnp.DeviceArray, record: bool = False):
        """
        evolve implements raw rockpool JAX evolution function for a AutoEncoder module.
        The AutoEncoder architecture is stateless, threfore, there is no state to return.
        The AutoEncoder architecture is timeless, therefore, there is no time record to hold.
        It uses the rockpool jax backend for the sake of compatibility.

        :param matrix: The weight matrix to encode via a weight currents and bitmask
        :type matrix: jnp.DeviceArray
        :param record: dummy record flag, required for rockpool jax modules, defaults to False
        :type record: bool, optional
        :return: the reconstructed weight matrix (and two dummy empty dictionaries)
        :rtype: Tuple[jnp.DeviceArray, Dict[None], Dict[None]]
        """
        assert matrix.size == self.size_out

        # Compress the matrix
        code = matrix @ self.w_en

        # Reconstruct the matrix given
        reconstructed = code @ self.bitmask

        return reconstructed, {}, {}

    def code(self, matrix: jnp.DeviceArray) -> jnp.DeviceArray:
        """
        code generates the compressed version of a matrix using the encoder

        :param matrix: any matrix to compress
        :type matrix: jnp.DeviceArray
        :return: the code generated, or the compressed version
        :rtype: jnp.DeviceArray
        """
        return matrix @ self.w_en

    @property
    def bitmask(self) -> jnp.DeviceArray:
        """
        bitmask regard the decoder weights as binary and return the decoder indicated binary weight mask
        """
        prob = nn.sigmoid(self.w_dec)
        spikes = step_pwl(prob)
        return spikes


@dataclass
class WeightConfig:
    """
    WeightConfig encapsulates weight currents of the configurable synapses between neurons. 
    It provides a general way of handling SE2 weight current and the conversion from device 
    configuration object

    :param weights: The weight matrix to obtain, co-depended to Iw_0, Iw_1, Iw_2, Iw_3 and bitmask.
    :type weights: jnp.DeviceArray
    :param Iw_0: the first base weight current corresponding to the 0th bit of the bit-mask, in Amperes. In DynapSE1, it's GABA_B base weigth.
    :type Iw_0: float
    :param Iw_1: the second base weight current corresponding to the 1st bit of the bit-mask, in Amperes. In DynapSE1, it's GABA_A base weigth.
    :type Iw_1: float
    :param Iw_2: the third base weight current corresponding to the 2nd bit of the bit-mask, in Amperes. In DynapSE1, it's NMDA base weigth.
    :type Iw_2: float
    :param Iw_3: the fourth base weight current corresponding to the 3rd bit of the bit-mask, in Amperes. In DynapSE1, it's AMPA base weigth.
    :type Iw_3: float
    :param encoded_bitmask: A bit mask to select and dot product the base Iw currents (shape,)
    :type encoded_bitmask: np.ndarray
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
    Iw_0: Optional[jnp.DeviceArray] = None
    Iw_1: Optional[jnp.DeviceArray] = None
    Iw_2: Optional[jnp.DeviceArray] = None
    Iw_3: Optional[jnp.DeviceArray] = None
    encoded_bitmask: Optional[np.ndarray] = None

    _optimizers = [
        [
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
    ]

    def __post_init__(self) -> None:
        """
        __post_init__ runs after __init__ and initializes the WeightConfig object with default values in the case that they are not specified.

        :raises ValueError: If `weight` is None, then bitmask and weight bits are required to calculate the weight matrix
        """

        if self.weights is None:
            if (
                (self.Iw_0 is None)
                or (self.Iw_1 is None)
                or (self.Iw_2 is None)
                or (self.Iw_3 is None)
                or (self.encoded_bitmask is None)
            ):
                raise ValueError(
                    "If `weight` is None, then bitmask and weight bits are required to calculate the weight matrix!"
                )
            self.weights = self.weight_matrix()

        self.w_flat, self.transforms = self.preprocess(self.weights)

        if self.weights is not None:
            if self.encoded_bitmask is None:
                self.encoded_bitmask = jnp.zeros(self.shape, int)
            else:
                pass  # [] TODO : initialize decoder weights

            if (
                (self.Iw_0 is not None)
                or (self.Iw_1 is not None)
                or (self.Iw_2 is not None)
                or (self.Iw_3 is not None)
            ):
                pass  # [] TODO : initialize or partly initialize the encoder weights

            logging.info("Run .fit() to find weight parameters and bitmask!")
            self.ae = AutoEncoder(self.w_flat.size, 4)

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
        weight_matrix generates a weight matrix for `DynapSE` modules using the base weight currents, and the bitmask.
        In device, we have the opportunity to define 4 different base weight current. Then using a bit mask, we can compose a
        weight current defining the strength of the connection between two neurons. The parameters and usage explained below.

        :return: the weight matrix composed using the base weight parameters and the binary bit-mask.
        :rtype: jnp.DeviceArray
        """
        # To broadcast on the post-synaptic neurons : pre, post, gate -> [(bits), post, pre, gate].T
        bits_trans = self.decode_bitmask(self.encoded_bitmask.transpose(1, 0, 2)).T
        # Restore the shape : (gate, pre, post) -> pre, post, gate
        w_rec = jnp.sum(bits_trans * self.Iw, axis=-1).transpose(1, 2, 0)
        return w_rec

    def loss_mse(
        self, parameters: Dict[str, Any], f_bound_penalty: float = 1e3
    ) -> float:
        """
        loss_mse calculates the mean square error loss between output and the target,
        given a new parameter set. Also, adds the bound violation penalty to the loss calculated.

        :param parameters: new parameter set for the autoencoder
        :type parameters: Dict[str, Any]
        :param f_bound_penalty: a factor of multiplication for bound violation penalty, defaults to 1e3
        :type f_bound_penalty: float, optional
        :return: the mean square error loss between the output and the target + bound violation penatly
        :rtype: float
        """
        # - Assign the provided parameters to the network
        net = self.ae.set_attributes(parameters)
        output, _, _ = net(self.w_flat)

        # - Calculate the loss imposing the bounds
        penalty = l.bounds_cost(parameters, self.ae.lower_bounds, {}) * f_bound_penalty
        loss = l.mse(output, self.w_flat) + penalty

        return loss

    def fit(
        self,
        n_epoch: int = 10000,
        optimizer: str = "adam",
        step_size: float = 1e-3,
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
        :type step_size: float, optional
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
            optimizer, step_size, *args, *kwargs
        )

        ## - Initialize the optimizer with the initial parameters
        params0 = deepcopy(self.ae.parameters())
        opt_state = init_fun(params0)

        ## - Get the jit compiled update and value-and-gradient functions
        loss_vgf = jit(value_and_grad(self.loss_mse))
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

        ## Update encoded_bitmask
        bitmask_flat = self.encode_bitmask(self.ae.bitmask)
        self.encoded_bitmask = (
            self.encoded_bitmask.at[self.idx_nonzero].set(bitmask_flat).astype(int)
        )

        ## Update Iws
        code = self.ae.code(self.w_flat)
        self.Iw_0 = jnp.full(self.n_post, code[0] / self.scale)
        self.Iw_1 = jnp.full(self.n_post, code[1] / self.scale)
        self.Iw_2 = jnp.full(self.n_post, code[2] / self.scale)
        self.Iw_3 = jnp.full(self.n_post, code[3] / self.scale)
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
            or __name == "encoded_bitmask"
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
        :raises ValueError: Weight matrix should be positive!
        :raises ShapeError: Weight matrix provided does not have a proper shape! It should be 3-dimensional with (pre,post,gate)!
        :return: w_flat, transforms
            w_flat: scaled, flattened, and preprocessed weight matrix
            transforms: the transforms applied to the weight matrix
        :rtype: Tuple[np.ndarray, Dict[str, Any]]
        """
        if len(weights.shape) != 3:
            raise ValueError(
                "Weight matrix provided does not have a proper shape! It should be 3-dimensional with (pre,post,gate)!"
            )

        scale = np.size(weights) / np.sum(weights)

        if scale <= 0:
            raise ValueError("Weight matrix should be positive!")

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
    def decode_bitmask(
        encoded_bitmask: Union[jnp.DeviceArray, np.ndarray]
    ) -> jnp.DeviceArray:
        """
        decode_bitmask apply 4-bit mask to numbers representing select bits and generates a bitmask 
            
            1 = 0001 -> selected bit: 0
            8 = 1000 -> selected bit: 3
            5 = 0101 -> selected bit 0 and 2

        :param bitmask: Binary mask to select, integer values representing binary numbers (shape,)
        :type bitmask: np.ndarray
        :return: an array of indices of selected bits, only binary values, (4,shape)
        :rtype: jnp.DeviceArray
        """
        bits = range(4)  # [0,1,2,3]
        bit_pattern = lambda n: (1 << n)  # 2^n

        # Indexes of the IDs to be selected in bits list
        decoded = jnp.array(
            [encoded_bitmask & bit_pattern(bit) for bit in bits], dtype=bool
        )
        return decoded

    @staticmethod
    def encode_bitmask(decoded_bitmask: jnp.DeviceArray) -> jnp.DeviceArray:
        """
        encode_bitmask converts a integer valued bitmask to 4 dimension (4-bits) bitmask representing the indexes of the selection
            
            [0,0,0,1] -> 1
            [1,0,0,0] -> 8
            [0,1,0,1] -> 5

        :param decoded_bitmask: an array of indices of selected bits, only binary values, (4,shape)
        :type decoded_bitmask: jnp.DeviceArray
        :return: a binary mask to select, integer values representing binary numbers (shape,)
        :rtype: jnp.DeviceArray
        """
        encoded = jnp.sum(decoded_bitmask.T * jnp.array([1, 2, 4, 8]), axis=-1).T
        return encoded

    @property
    def mse(self):
        """
        mse calculates the Mean Square Error loss between the reconstructed weights and the original weight matrix
        """
        w_rec, _, _ = self.ae(self.w_flat)
        loss = l.mse(w_rec, self.w_flat)
        return loss

    @property
    def Iw(self) -> jnp.DeviceArray:
        """
        Iw returns weight matrix required
        """
        return jnp.column_stack((self.Iw_0, self.Iw_1, self.Iw_2, self.Iw_3))

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

