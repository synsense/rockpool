"""
Dynap-SE weight quantization weight handler implementation
The handler stores the non-zero indexes, flattens the recurrent and input weights and help reconstructing

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

first implemented inside weights/WeightParameters @220121
split from WeightParameters @220922

20/09/2022
"""
from typing import Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from jax import numpy as jnp


@dataclass
class WeightHandler:
    weights_in: Optional[np.ndarray] = None
    weights_rec: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    nonzero_mask: Optional[np.ndarray] = None
    w_flat: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.nonzero_mask is not None or self.w_flat is not None:
            raise ValueError("Leave them blank!")

        self.shape_in, self.shape_out = None, None

        if self.weights_in is not None:
            self.shape_in = self.weights_in.shape

        if self.weights_rec is not None:
            self.shape_rec = self.weights_rec.shape

        ## Make sure that the weight matrices are 2D with (pre,post)
        self.__shape_check()

        if self.weights is None:
            if self.weights_in is None or self.weights_rec is None:
                raise ValueError(
                    "If weights is not defined, input and recurrent weights should be given!"
                )
            self.weights = np.vstack((self.weights_in, self.weights_rec))

        else:
            if self.weights_in is not None or self.weights_rec is not None:
                raise ValueError(
                    "Either a big weight matrix or two seperate input&recurrent weight matrices can be given at the same time"
                )
            if len(self.weights.shape) != 2:
                raise ValueError(
                    "Weight matrix provided does not have a proper shape! It should be 2-dimensional with (pre,post)!"
                )

        self.nonzero_mask = self.weights.astype(bool)
        self.w_flat = np.abs(self.weights[self.nonzero_mask].flatten())

    def __shape_check(self) -> None:

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

    def decompress_flattened_weights(self, compressed: np.ndarray) -> Tuple[np.ndarray]:

        # Compressed matrix should match the original global matrix shape
        compressed = np.array(compressed)

        # Place the elements to the known non-zero indexed places
        decompressed = np.zeros_like(self.weights, dtype=compressed.dtype)
        np.place(decompressed, self.nonzero_mask, compressed)

        # Split the matrix into input and recurrent
        w_in = decompressed[0 : self.shape_in[0], :]
        w_rec = decompressed[self.shape_in[0] :, :]

        # Make sure that the shape is correct
        assert w_in.shape == self.shape_in
        assert w_rec.shape == self.shape_rec

        return w_in, w_rec

    @staticmethod
    def weight_matrix(
        code: np.ndarray,
        intmask: np.ndarray,
        bits_per_weight: Optional[int] = 4,
    ) -> np.ndarray:
        """
        weight_matrix generates a weight matrix for `DynapSE` modules using the base weight currents, and the intmask.
        In device, we have the opportunity to define 4 different base weight current. Then using a bit mask, we can compose a
        weight current defining the strength of the connection between two neurons. The parameters and usage explained below.

        :return: the weight matrix composed using the base weight parameters and the binary bit-mask.
        :rtype: np.ndarray
        """
        # To broadcast on the post-synaptic neurons : pre, post -> [(bits), post, pre].T
        bits_trans = WeightHandler.int2bit_mask(bits_per_weight, intmask).T
        w_rec = np.sum(bits_trans * code, axis=-1).T
        return w_rec

    @staticmethod
    def bit2int_mask(
        n_bits: int,
        bitmask: Union[jnp.DeviceArray, np.ndarray],
        np_back: Any = np,
    ) -> Union[jnp.DeviceArray, np.ndarray]:
        """
        bit2int_mask apply 4-bit selection to binary values representing select bits and generates a compressed bitmask

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
        pattern = np_back.array([1 << n for n in range(n_bits)])  # [1,2,4,8, ..]
        intmask = np_back.sum(bitmask.T * pattern, axis=-1).T
        return intmask.round().astype(int)

    @staticmethod
    def int2bit_mask(
        n_bits: int,
        intmask: Union[jnp.DeviceArray, np.ndarray],
        np_back: Any = np,
    ) -> Union[jnp.DeviceArray, np.ndarray]:
        """
        int2bit_mask converts a integer valued bitmask to 4 dimension (4-bits) bitmask representing the indexes of the selection

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

        pattern = np_back.array([1 << n for n in range(n_bits)])  # [1,2,4,8, ..]
        intmask_ext = np_back.full((n_bits, *intmask.shape), intmask)  # (n_bits,shape)

        # Indexes of the IDs to be selected in bits list
        bitmask = np_back.bitwise_and(intmask_ext.T, pattern).T.astype(bool)
        return bitmask
