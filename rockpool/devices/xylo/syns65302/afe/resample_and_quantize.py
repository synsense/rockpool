"""
Defines the audio signal path for the AFE simulation skipping the PDM pathway for XyloAudio 3.

Defines the class :py:class:`.ResampleAndQuantize`.
"""

from rockpool.devices.xylo.syns65302.transform import AudioQuantizer, ResampleAudio
from rockpool.nn.combinators.sequential import ModSequential
from rockpool.devices.xylo.syns65302.afe.params import (
    AUDIO_SAMPLING_RATE,
    NUM_BITS_PDM_ADC,
)

__all__ = ["ResampleAndQuantize"]


class ResampleAndQuantize(ModSequential):
    """
    Simulates the input directly to the filter banks on XyloAudio 3 skipping the PDM pathway.

    For this, the input needs to be resampled and quantized.
    """

    def __init__(
        self,
        fs: float = AUDIO_SAMPLING_RATE,
        scale: float = 1.0,
        num_bits: int = NUM_BITS_PDM_ADC,
    ) -> None:
        """
        _summary_

        Args:
            fs (float, optional): The expected audio sampling rate.
                Defaults to AUDIO_SAMPLING_RATE.
                If the input sampling rate is different, it will AUTOMATICALLY be resampled.
            scale (float, optional): the input signal amplitude scaling.
                Usually, it's OK to leave it as 1.0. Defaults to 1.0.
            num_bits (int, optional): The number of bits devoted to the final sampled audio.
                Defaults to NUM_BITS_PDM_ADC.
        """
        super().__init__(
            ResampleAudio(fs_target=fs), AudioQuantizer(scale=scale, num_bits=num_bits)
        )
