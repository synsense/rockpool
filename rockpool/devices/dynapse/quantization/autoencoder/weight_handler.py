"""
Dynap-SE weight quantization weight handler implementation
The handler stores the non-zero indexes, flattens the recurrent and input weights and help reconstructing
Includes utility methods to restore/reshape weight matrices from compartments

* Non User Facing *
"""

from typing import Any, Optional, Tuple, Union

import jax
import numpy as np
from jax import numpy as jnp

from dataclasses import dataclass

__all__ = ["WeightHandler"]


@dataclass
class WeightHandler:
    """
    WeightHandler encapsulates the simulator weights and provides utilities to use the weight matrices in quantization pipeline.
    Also provides some utilities to restore a weight matrix from the weight parameters and CAM content.

    Examples:
        Instantitate a WeightHandler module with input and recurrent weights matrix

        >>> wh = WeightHandler(w_in, w_rec)
    """

    weights_in: Optional[np.ndarray] = None
    """input layer weights used in Dynap-SE2 simulation"""

    weights_rec: Optional[np.ndarray] = None
    """recurrent layer (in-device neurons) weights used in Dynap-SE2 simulation"""

    weights_global: Optional[np.ndarray] = None
    """global weights (input + rec) weights refered by simulation, used in quantization, if weights_in, weighs_rec provided, do not provide this!"""

    def __post_init__(self):
        """
        __post_init__ runs after __init__ and initializes the WeightHandler object, executes design rule checks

        :raises ValueError: If weights is not defined, input and recurrent weights should be given!
        :raises ValueError: Either a global weight matrix or two seperate input&recurrent weight matrices are allowed!
        :raises ValueError: Weight matrix provided does not have a proper shape! It should be 2-dimensional with (pre,post)!
        """

        self.shape_in, self.shape_rec = None, None

        if self.weights_in is not None:
            self.shape_in = self.weights_in.shape

        if self.weights_rec is not None:
            self.shape_rec = self.weights_rec.shape

        ## Make sure that the weight matrices are 2D with (pre,post)
        self.__shape_check()

        ## Mutually exclusive setting is possible between in&rec and global
        if self.weights_global is None:
            # Neither w_in nor w_rec
            if self.weights_in is None and self.weights_rec is None:
                raise ValueError(
                    "If weights is not defined, input or recurrent weights should be given!"
                )
            # Only w_rec
            elif self.weights_in is None:
                self.weights_global = self.weights_rec
            # Only w_in
            elif self.weights_rec is None:
                self.weights_global = self.weights_in
            # Both w_in and w_rec
            else:
                self.weights_global = np.vstack((self.weights_in, self.weights_rec))

        else:
            if self.weights_in is not None or self.weights_rec is not None:
                raise ValueError(
                    "Either a global weight matrix or two seperate input&recurrent weight matrices are allowed!"
                )
            if len(self.weights_global.shape) != 2:
                raise ValueError(
                    "Weight matrix provided does not have a proper shape! It should be 2-dimensional with (pre,post)!"
                )

        ## Fill property segment
        self.__nonzero_mask = self.weights_global.astype(bool)
        self.__w_flat = np.abs(self.weights_global[self.nonzero_mask].flatten())

        ### Sign
        __sign_filler = lambda w: np.sign(w) if w is not None else None
        self.__sign_mask = __sign_filler(self.weights_global)
        self.__sign_in = __sign_filler(self.weights_in)
        self.__sign_rec = __sign_filler(self.weights_rec)

    def __shape_check(self) -> None:
        """
        __shape_check checks the shapes of the given weight matrices

        :raises ValueError: Recurrent weight matrix should be square shaped!
        :raises ValueError: Weight matrices provided does not have a proper shape! They should be 2-dimensional with (pre,post)!
        :raises ValueError: The number of neurons should match!
        """

        if self.shape_rec is not None:
            if self.shape_rec[0] != self.shape_rec[1]:
                raise ValueError(
                    f"Recurrent weight matrix should be square shaped! {self.weights_rec.shape[0]} != {self.weights_rec.shape[1]}"
                )

        if self.shape_in is not None and self.shape_rec is not None:
            if len(self.shape_in) != 2 or len(self.shape_rec) != 2:
                raise ValueError(
                    "Weight matrices provided does not have a proper shape! They should be 2-dimensional with (pre,post)!"
                )

            if self.shape_in[1] != self.shape_rec[1]:
                raise ValueError(
                    f"The number of neurons should match! {self.weights_in.shape[1]} != {self.weights_rec.shape[1]}"
                )

    def reshape_flat_weights(self, w_flat: np.ndarray) -> Tuple[np.ndarray]:
        """
        reshape_flat_weights restores the original shape of the input and recurrent weights
        from the global non-zero flat weight matrix

        :param w_flat: flattened global weight matrix with only non-zero absolute values
        :type w_flat: np.ndarray
        :return: w_in, w_rec
            :w_in: input weight matrix
            :w_rec: recurrent weight matrix
        :rtype: Tuple[np.ndarray]
        """

        # Compressed matrix should match the original global matrix shape
        w_flat = np.array(w_flat)

        # Place the elements to the known non-zero indexed places
        w_shaped = np.zeros_like(self.weights_global, dtype=w_flat.dtype)
        np.place(w_shaped, self.nonzero_mask, w_flat)

        # Split the matrix into input and recurrent
        w_in = w_shaped[0 : self.shape_in[0], :] if self.shape_in is not None else None
        w_rec = (
            w_shaped[-self.shape_rec[0] :, :] if self.shape_rec is not None else None
        )

        return w_in, w_rec

    ### --- Utils --- ###

    @staticmethod
    def restore_weight_matrix(
        n_bits: Optional[int],
        code: np.ndarray,
        int_mask: np.ndarray,
        sign_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """restore_weight_matrix composes the simulated weight matrix that the given Iw vector(code), the int_mask and sign_mask
        would generate. It only provides a perspective to see the intermediate representation of the configuration.

        :param n_bits: number of bits allocated per weight, defaults to 4
        :type n_bits: Optional[int], optional
        :param code: the Iw vector functioning as the intermediate code representation [Iw_0, Iw_1, Iw_2, Iw_3]
        :type code: np.ndarray
        :param int_mask: integer values representing binary weight selecting CAM masks
        :type int_mask: np.ndarray
        :param sign_mask: the +- signs of the weight values, + means excitatory; - means inhibitory. defaults to None
        :type sign_mask: Optional[np.ndarray], optional
        :return: the simualated weight matrix
        :rtype: np.ndarray
        """

        # To broadcast on the post-synaptic neurons : pre, post -> [(bits), post, pre].T
        bits_trans = WeightHandler.int2bit_mask(n_bits, int_mask).T
        weights = np.sum(bits_trans * code, axis=-1).T
        if sign_mask is not None:
            weights *= sign_mask
        return weights

    @staticmethod
    def bit2int_mask(
        n_bits: int,
        bit_mask: Union[jax.Array, np.ndarray],
        np_back: Any = np,
    ) -> Union[jax.Array, np.ndarray]:
        """
        bit2int_mask apply 4-bit selection to binary values representing select bits and generates a compressed bit_mask

            (n_bits=4)

            [0,0,0,1] -> 1
            [1,0,0,0] -> 8
            [0,1,0,1] -> 5

        :param n_bits: number of bits reserved for representing the integer values
        :type n_bits: int
        :param bit_mask: an array of indices of selected bits, only binary values, (n_bits,shape)
        :type bit_mask: jax.Array
        :param np_back: the numpy backend to be used(jax.numpy or numpy), defaults to numpy
        :type np_back: Any
        :return: integer values representing binary numbers (shape,)
        :rtype: jax.Array
        """
        pattern = np_back.array([1 << n for n in range(n_bits)])  # [1,2,4,8, ..]
        int_mask = np_back.sum(bit_mask.T * pattern, axis=-1).T
        return int_mask.round().astype(int)

    @staticmethod
    def int2bit_mask(
        n_bits: int,
        int_mask: Union[jax.Array, np.ndarray],
        np_back: Any = np,
    ) -> Union[jax.Array, np.ndarray]:
        """
        int2bit_mask converts a integer valued bit_mask to 4 dimension (4-bits) bit_mask representing the indexes of the selection

            (n_bits=4)

            1 = 0001 -> selected bit: 0
            8 = 1000 -> selected bit: 3
            5 = 0101 -> selected bit 0 and 2

        :param n_bits: number of bits reserved for representing the integer values
        :type n_bits: int
        :param int_mask: integer values representing binary numbers to select (shape,)
        :type int_mask: jax.Array
        :param np_back: the numpy backend to be used(jax.numpy or numpy), defaults to numpy
        :type np_back: Any
        :return: an array of indices of selected bits, only binary values, (n_bits,shape)
        :rtype: jax.Array
        """

        pattern = np_back.array([1 << n for n in range(n_bits)])  # [1,2,4,8, ..]
        int_mask_ext = np_back.full((n_bits, *int_mask.shape), int_mask)

        # Indexes of the IDs to be selected in bits list
        bit_mask = np_back.bitwise_and(int_mask_ext.T, pattern).T.astype(bool)
        return bit_mask

    ### --- Properties --- ###

    @property
    def w_flat(self) -> np.ndarray:
        """flattened global weight matrix with only non-zero absolute values"""
        return self.__w_flat

    @property
    def nonzero_mask(self) -> np.ndarray:
        """the positional indexes of the matrix elements with the non-zero values"""
        return self.__nonzero_mask

    @property
    def sign_mask(self) -> np.ndarray:
        """the +- signs of global weight matrix elements"""
        return self.__sign_mask

    @property
    def sign_in(self) -> Optional[np.ndarray]:
        """the +- signs of input weight matrix elements, if input weight matrix is given"""
        return self.__sign_in

    @property
    def sign_rec(self) -> Optional[np.ndarray]:
        """the +- signs of recurrent weight matrix elements, if recurrent weight matrix is given"""
        return self.__sign_rec
