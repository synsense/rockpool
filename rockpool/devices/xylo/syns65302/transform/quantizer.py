"""
Implements :py:class:`.AudioQuantizer`, a simulation of the XA3 audio quantization block
"""

from typing import Tuple

import numpy as np

from rockpool.devices.xylo.syns65302.afe.params import NUM_BITS_PDM_ADC
from rockpool.nn.modules import Module
from rockpool.parameters import SimulationParameter

__all__ = ["AudioQuantizer"]


class AudioQuantizer(Module):
    """
    The quantizer that converts the input signal into integer representation in hardware implementation.
    This module quantizes the input signals into an integer with defined number of bits precision.
    In XyloAudio 3, the audio signal is quantized to 14-bit by default.
    """

    def __init__(
        self,
        scale: float = 1.0,
        num_bits: int = NUM_BITS_PDM_ADC,
    ) -> None:
        """

        Args:
            shape (Optional[Union[Tuple, int]], optional): Network shape. Defaults to (1,).
            scale (float, optional): scale applied before quantization. Defaults to 1.0.
            num_bits (int, optional): the number of bits in the fractional part of the quantized signal. Defaults to 14.
        """
        super().__init__(shape=(1, 1), spiking_input=False, spiking_output=False)

        self.scale = SimulationParameter(scale, shape=(1,))
        """scaling factor applied before quantization"""

        self.num_bits = SimulationParameter(num_bits, shape=(1,), cast_fn=int)
        """The number of bits in the fractional part of the quantized signal"""

    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, dict, dict]:
        """Quantize the input signal after suitable scaling. The quantization is done using num_bits precision

        Args:
            input_data (np.ndarray): input signal (single-channel). Typically `scale x sig_in` should have max amplitude less than 1. shape: (T X C)
            record (bool, optional): record flag to match with the other rockpool modules. Practically useless. Defaults to False.

        Returns:
            Tuple[np.ndarray, Dict, Dict]:
                out: the quantized version of the input signal. shape:(T X C)
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

        out = np.vectorize(__forward)(input_data)

        return out, {}, {}
