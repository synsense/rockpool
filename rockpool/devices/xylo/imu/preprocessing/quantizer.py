from typing import Dict, Optional, Tuple, Union

import numpy as np

from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter

__all__ = ["Quantizer"]


class Quantizer(Module):
    """
    The quantizer that converts the input signal into python-object which can handle/simulate arbitrary register size in hardware implementation.
    This module quantizes the input signals into an integer with python-object precision, which has infinite bit length.
    This makes the simulation slower but has the advantage that it avoid any numerical imprecision during the python simulation
    when the register sizes are chosen to be larger than numpy.int64 bit length.

    For example:
        - if we cast the input data into np.int64, it will work very well as far as the length of registers in the hardware are chosen
        such that the outcome does not become larger than 64 bits.
        - this works very well in general but may have issues if we use larger register length in hardware.
        - the compatibility between the hardware and software (independent of register length) is guaranteed using this quantizer.
    """

    def __init__(
        self,
        shape: Optional[Union[Tuple, int]] = (3, 3),
        scale: float = 1.0,
        num_bits: int = 16,
    ) -> None:
        """Object constructor.

        Args:
            shape (Optional[Union[Tuple, int]], optional): Network shape. Defaults to (3,3).
            scale (float, optional): scale applied before quantization. Defaults to 1.0.
            num_bits (int, optional): the number of bits in the fractional part of the quantized signal. Defaults to 16.
        """
        super().__init__(shape=shape, spiking_input=False, spiking_output=False)

        self.scale = SimulationParameter(scale, shape=(1,))
        """scale factor applied before quantization"""

        self.num_bits = SimulationParameter(num_bits, shape=(1,), cast_fn=int)
        """The number of bits in the fractional part of the quantized signal"""

    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, Dict, Dict]:
        """Quantize the input signal after suitable scaling. The quantization is done using python-object precision, which has infinite bit length.

        Args:
            input_data (np.ndarray): input signal (single- or multi-channel). Typically `scale x sig_in` should have max amplitude less than 1. shape: (B X T X C)
            record (bool, optional): record flag to match with the other rockpool modules. Practically useless. Defaults to False.

        Returns:
            Tuple[np.ndarray, Dict, Dict]:
                data: the python-object quantized version of the input signal. shape:(B X T X C)
                state_dict: empty dictionary.
                record_dict: empty dictionary.
        """

        def __forward(__data: float) -> int:
            """The atomic function that quantizes the input signal.

            Args:
                __data (float): float number to be quantized.

            Returns:
                int: quantized and scaled version of the input signal.
            """
            return int(self.scale * __data * (2 ** (self.num_bits - 1)))

        input_data, _ = self._auto_batch(input_data)
        out = np.vectorize(__forward, otypes=[object])(input_data)

        return out, {}, {}
