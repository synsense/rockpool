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
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np


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
        self.w_flat = self.weights[self.nonzero_mask].flatten()

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
        assert compressed.shape == self.weights.shape

        # Place the elements to the known non-zero indexed places
        decompressed = np.zeros_like(self.weights)
        np.place(decompressed, self.nonzero_mask, compressed)

        # Split the matrix into input and recurrent
        w_in = decompressed[0 : self.shape_in[0], :]
        w_rec = decompressed[self.shape_in[0] :, :]

        # Make sure that the shape is correct
        assert w_in.shape == self.shape_in
        assert w_rec.shape == self.shape_rec

        return w_in, w_rec
