from typing import Optional, Tuple, Union
from rockpool.devices.xylo.syns63300.transform import Quantizer

__all__ = ["AudioQuantizer"]


class AudioQuantizer(Quantizer):
    """
    The quantizer that converts the input signal into python-object which can handle/simulate arbitrary register size
    in hardware implementation. This module quantizes the input signals into an integer with python-object precision,
    which has infinite bit length. In XyloA3, the audio signal is quantized to 14-bit by default.
    """

    def __init__(
        self,
        shape: Optional[Union[Tuple, int]] = (1,),
        scale: float = 1.0,
        num_bits: int = 14,
    ) -> None:
        """

        Args:
            shape (Optional[Union[Tuple, int]], optional): Network shape. Defaults to (1,).
            scale (float, optional): scale applied before quantization. Defaults to 1.0.
            num_bits (int, optional): the number of bits in the fractional part of the quantized signal. Defaults to 14.
        """
        super().__init__(shape=shape, scale=scale, num_bits=num_bits)
