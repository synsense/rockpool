from typing import Dict, Tuple

import numpy as np

from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter

__all__ = ["RobustQuantizer"]


class RobustQuantizer(Module):
    """
    The robust quantizer that converts the input signal into specific-bits integer which can simulate arbitrary(<=64bit)
     register size in hardware implementation. In XyloA3, the audio signal is quantized to 14-bit.
    """

    def __init__(
        self,
        scale: float = 0.99,
        num_bits: int = 14,
        outlier_ratio: float = 0.01,
        eps: float = 1e-6,
    ) -> None:
        """

        Args:
            scale (float, optional): scale applied before quantization. Defaults to 0.99.
            num_bits (int, optional): the number of bits in the fractional part of the quantized signal. Defaults to 14.
            outlier_ratio(float, optional): the outlier ratio sample of the input signal. Default to 0.01
            eps(float, optional): the epsilon coefficient used in the robust quantization process. Default to 1e-6.
        """
        super().__init__(shape=(1,), spiking_input=False, spiking_output=False)

        assert num_bits <= 64, f"num_bits should <= 64, got {num_bits}."

        self.scale = SimulationParameter(scale, shape=(1,))
        self.num_bits = SimulationParameter(num_bits, shape=(1,), cast_fn=int)
        self.outlier_ratio = SimulationParameter(outlier_ratio, shape=(1,))
        self.eps = SimulationParameter(eps, shape=(1,))

    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, Dict, Dict]:
        """Quantize the input signal after suitable scaling.

        Args:
            input_data (np.ndarray): input signal (single-channel). shape: (B X T X 1)
            record (bool, optional): record flag to match with the other rockpool modules. Practically useless. Defaults to False.

        Returns:
            Tuple[np.ndarray, Dict, Dict]:
                data: the python-object quantized version of the input signal. shape:(B X T X C)
                state_dict: empty dictionary.
                record_dict: empty dictionary.
        """

        input_data, _ = self._auto_batch(input_data)
        robust_amp = self._get_robust_amplitude(input_data)
        # clamp the input signal
        input_data.setflags(write=True)
        input_data[input_data >= robust_amp] = robust_amp
        input_data[input_data <= -robust_amp] = -robust_amp
        # quantize
        q_input_data = (
            2 ** (self.num_bits - 1)
            * input_data
            * self.scale
            / (robust_amp * (1 + self.eps))
        )
        q_input_data = q_input_data.astype(np.int64)

        return q_input_data, {}, {}

    def _get_robust_amplitude(self, input_data: np.ndarray) -> float:
        """
        Get the robust amplitude of the given signal.
        :param input_data(np.ndarray): in shape [B x T x 1]
        :return: robust_amplitude (float)
        """
        sorted_amp = np.sort(np.abs(input_data), axis=1).squeeze()
        robust_amplitude = sorted_amp[
            int((len(sorted_amp) - 1) * (1 - self.outlier_ratio))
        ]
        return robust_amplitude
