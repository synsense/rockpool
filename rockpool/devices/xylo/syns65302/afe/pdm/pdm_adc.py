"""
This module contains the PDM-based ADC for sampling the input audion signal.
It consists of two main parts:
  (i)     a deltasigma based PDM microphone that converts the input analog audio into a PDM bit stream where the
          relative frequency of 1-vs-0 depends on the amplitude of the signal
  (ii)    a low-pass filter follows by decimation stage that processes the PDM bit stream and recovers the sampled
          audio upto a given bit precision.

In brief, PDM microphone with its internal deltasigma modulation followed by low-pass filtering + decimation
module implemented here yield an ADC for the input analog audio signal.

The low-pass filtering is implemented as a **polyphase** filter structure to consume as less power as possible.
"""

from rockpool.devices.xylo.syns65302.afe.params import (
    AUDIO_CUTOFF_FREQUENCY,
    AUDIO_CUTOFF_FREQUENCY_WIDTH,
    DECIMATION_FILTER_LENGTH,
    DELTA_SIGMA_ORDER,
    NUM_BITS_FILTER_Q,
    NUM_BITS_PDM_ADC,
    PDM_FILTER_DECIMATION_FACTOR,
    PDM_SAMPLING_RATE,
)
from rockpool.nn.combinators.sequential import ModSequential

from .microphone_pdm import MicrophonePDM
from .poly_phase_fir import PolyPhaseFIR

__all__ = ["PDMADC"]


class PDMADC(ModSequential):
    """
    Pulse-density modulatio Analog-to-Digital (ADC) module for XyloAudio 3 chip consisting of
        (i)  PDM microphone converting the input analog audio signal into PDM bit-stream.
        (ii) low-pass filtering and decimation module converting the binary PDM stream into the target sampled audio signal.
    """

    def __init__(self) -> None:
        """
        Instantiate a `PDMADC` object
        """

        self.sdm_order: int = DELTA_SIGMA_ORDER
        """order of deltasigma modulator. Defaults to DELTA_SIGMA_ORDER."""

        self.sdm_OSR: int = PDM_FILTER_DECIMATION_FACTOR
        """
        oversampling rate of PDM microphone : ratio between PDM clock and rate of target audio signal. Defaults to PDM_FILTER_DECIMATION_FACTOR.
        This is the same as decimation factor in PDM low-pass filter. Defaults to PDM_FILTER_DECIMATION_FACTOR.
        """

        self.fs: int = PDM_SAMPLING_RATE
        """sampling rate/clock rate of PDM module. Defaults to PDM_SAMPLING_RATE."""

        self.cutoff: float = AUDIO_CUTOFF_FREQUENCY
        """cutoff frequency of the low-pass filter used for recovery of target sampled audio from PDM bit-stream. Defaults to AUDIO_CUTOFF_FREQUENCY."""

        self.cutoff_width: float = AUDIO_CUTOFF_FREQUENCY_WIDTH
        """transition withs of the low-pass filter. Defaults to AUDIO_CUTOFF_FREQUENCY_WIDTH (is set to 20% of the cutoff frequency)"""

        self.filt_length: int = DECIMATION_FILTER_LENGTH
        """length of the FIR low-pass filter. Defaults to DECIMATION_FILTER_LENGTH."""

        self.num_bits_filter_Q: int = NUM_BITS_FILTER_Q
        """number of bits used for quantizing the filter taps. Defaults to NUM_BITS_FILTER_Q."""

        self.num_bits_output: int = NUM_BITS_PDM_ADC
        """
        target number of bits in the final sampled audio obtained after low-pass filtering and decimation.
        This is equivalent to the number of bits in the equivalent ADC. Defaults to NUM_BITS_PDM_ADC.
        """

        super().__init__(
            MicrophonePDM(
                sdm_order=self.sdm_order,
                sdm_OSR=self.sdm_OSR,
                bandwidth=self.cutoff,
                fs=self.fs,
            ),
            PolyPhaseFIR(
                decimation_factor=self.sdm_OSR,
                cutoff=self.cutoff,
                cutoff_width=self.cutoff_width,
                filt_length=self.filt_length,
                num_bits_filter_Q=self.num_bits_filter_Q,
                num_bits_output=self.num_bits_output,
                fs=self.fs,
            ),
        )
