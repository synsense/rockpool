# -----------------------------------------------------------
# This module implements the quantizer that converts the input signal into
# python-object which can handle/simulate arbitrary register size in hardware implementation.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 27.08.2022
# -----------------------------------------------------------

import numpy as np


class Quantizer:
    """
    This module quanizes the input signals into an integer with python-object precision, which has infinite bit length.
    This makes the simulation slower but has the advantage that it avoid any numerical imprecision during the python simulation
    when the register sizes are chosen to be larger than numpy.int64 bit length.

    For example:
        - if we cast the input data into np.int64, it will work very well as far as the length of registers in the hardware are choosen
        such that the outcome does not become larger than 64 bits.
        - this works very well in general but may have issues if we use larger register length in hardware.
        - the compatibility between the hardware and software (independent of register length) is quarrenteed using this quantizer.
    """

    def quantize(self, sig_in: np.ndarray, scale: float, num_bits):
        """this function quantizes the input signal after suitable scaling.

        Args:
            sig_in (np.ndarray): input signal (single- or multi-channel)
            scale (float): scale applied before quantization.
            num_bits (int): the number of bits in the fractional part of the quantized signal.
            Typically `scale x sig_in` should have max amplitude less than 1.

        Returns:
            np.ndarray(dtype=object): the python-object quantized version of the input signal.
        """
        # save the shape of the input signal
        sig_in_shape = sig_in.shape

        # quantized version of the signal
        sig_in_q = np.array(
            [int(scale * el * 2 ** (num_bits - 1)) for el in sig_in.ravel()],
            dtype=object,
        ).reshape(*sig_in_shape)

        return sig_in_q

    # add the call version for further convenience
    def __call__(self, *args, **kwargs):
        """
        This function simply calls the `quantize()` function and is added for further convenience.
        """
        self.quantize(*args, **kwargs)
