"""
Implements :py:class:`.AmplitudeNormalizer`, a robust signal normalization block used by AFESim3
"""

from typing import Dict, Tuple, Optional, Union

import numpy as np

from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter

__all__ = ["AmplitudeNormalizer"]


class AmplitudeNormalizer(Module):
    """
    Robust amplitude normalizer. To find the robust amplitude of the signal, we first sort the absolute value of the
    signal:
        X_sorted[0] <= X_sorted[1] <= ... <= X_sorted[N -1].
    And we assume the percentage of the outlier p:
        0.0 <= p <= 0.5
    Then the robust amplitude is defined as:
        A_robust = X_sorted[(N-1)(1-p)]
    """

    def __init__(
        self,
        shape: Optional[Union[Tuple, int]] = (1,),
        p: float = 0.01,
    ) -> None:
        """Object constructor

        Args:
            shape (Optional[Union[Tuple, int]]): Network shape. Defaults to (1, ).
            p (float): The outlier sample ratio of the signal. Defaults to 0.01.

        """
        super().__init__(shape=shape, spiking_input=False, spiking_output=False)

        self.outlier_ratio = SimulationParameter(p, shape=(1,))
        """The percentage of the outlier sample in the input signal"""

    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, Dict, Dict]:
        """Normalize the input signal.

        Args:
            input_data (np.ndarray): input signal (single- or multi-channel) in shape: (B X T X C)
            record (bool, optional): record flag to match with the other rockpool modules.
                Practically useless. Defaults to False.

        Returns:
            Tuple[np.ndarray, Dict, Dict]:
                data: the python-object quantized version of the input signal. shape:(B X T X C)
                state_dict: empty dictionary.
                record_dict: empty dictionary.
        """

        input_data, _ = self._auto_batch(input_data)
        robust_amp = self._get_robust_amplitude(input_data)

        if robust_amp == 0.0:
            raise ValueError(f"Got 0 as robust amplitude!")

        input_data.setflags(write=True)
        input_data[input_data >= robust_amp] = robust_amp
        input_data[input_data <= -robust_amp] = -robust_amp
        """ clamp the input signal """

        normalized_input_data = input_data / robust_amp
        """ normalize the amplitude """

        return normalized_input_data, {}, {}

    def _get_robust_amplitude(self, input_data: np.ndarray) -> float:
        """
        Get the robust amplitude of the given signal.
        Robust amplitude protects normalization from potential precision loss caused by the outlier(extreme large sample)

        it first sort the absolute value of the signal:
            X_sorted[0] <= X_sorted[1] <= ... <= X_sorted[N -1].
        And we assume the percentage of the outlier p:
            0.0 <= p_outlier <= 0.5
        Then the robust amplitude is defined as:
            amp_robust = X_sorted[(N-1)(1-p_outlier)]
        Args:
            input_data(np.ndarray): in shape [B x T x 1]

        Returns:
            robust_amplitude (float)
        """
        sorted_amp = np.sort(np.abs(input_data), axis=1).squeeze()
        robust_amplitude = sorted_amp[
            int((len(sorted_amp) - 1) * (1 - self.outlier_ratio))
        ]
        return robust_amplitude
