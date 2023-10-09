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

from rockpool.nn.combinators import Sequential

from .microphone_pdm import MicrophonePDM
from .poly_phase_fir import PolyPhaseFIR


from rockpool.devices.xylo.syns65302.afe.params import (
    AUDIO_CUTOFF_FREQUENCY,
    AUDIO_CUTOFF_FREQUENCY_WIDTH,
    PDM_FILTER_DECIMATION_FACTOR,
    PDM_SAMPLING_RATE,
    DELTA_SIGMA_ORDER,
    DECIMATION_FILTER_LENGTH,
    NUM_BITS_FILTER_Q,
    NUM_BITS_PDM_ADC,
)

__all__ = ["PDM_ADC"]


def PDM_ADC(
    sdm_order: int = DELTA_SIGMA_ORDER,
    sdm_OSR: int = PDM_FILTER_DECIMATION_FACTOR,
    fs: int = PDM_SAMPLING_RATE,
    cutoff: float = AUDIO_CUTOFF_FREQUENCY,
    cutoff_width: float = AUDIO_CUTOFF_FREQUENCY_WIDTH,
    filt_length: int = DECIMATION_FILTER_LENGTH,
    num_bits_filter_Q: int = NUM_BITS_FILTER_Q,
    num_bits_output: int = NUM_BITS_PDM_ADC,
):
    """
    Analog-to-Digital (ADC) module for Xylo-A3 chip consisting of
        (i)  PDM microphone converting the input analog audio signal into PDM bit-stream.
        (ii) low-pass filtering and decimation module converting the binary PDM stream into the target sampled audio signal.

    Args:
        sdm_order (int, optional): order of deltasigma modulator. Defaults to DELTA_SIGMA_ORDER.
        sdm_OSR (int, optional): oversampling rate of PDM microphone : ratio between PDM clock and rate of target audio signal. Defaults to PDM_FILTER_DECIMATION_FACTOR.
        This is the same as decimation factor in PDM low-pass filter. Defaults to PDM_FILTER_DECIMATION_FACTOR.
        fs (int, optional): sampling rate/clock rate of PDM module. Defaults to PDM_SAMPLING_RATE.
        cutoff (float, optional): cutoff frequency of the low-pass filter used for recovery of target sampled audio from PDM bit-stream. Defaults to AUDIO_CUTOFF_FREQUENCY.
        cutoff_width (float, optional): transition withs of the low-pass filter. Defaults to AUDIO_CUTOFF_FREQUENCY_WIDTH (is set to 20% of the cutoff frequency).
        filt_length (int, optional): length of the FIR low-pass filter. Defaults to DECIMATION_FILTER_LENGTH.
        num_bits_filter_Q (int, optional): number of bits used for quantizing the filter taps. Defaults to NUM_BITS_FILTER_Q.
        num_bits_output (int, optional): target number of bits in the final sampled audio obtained after low-pass filtering and decimation.
        This is equivalent to the number of bits in the equivalent ADC. Defaults to NUM_BITS_PDM_ADC.
    """
    # two modules of equivalent ADC
    return Sequential(
        MicrophonePDM(
            sdm_order=sdm_order,
            sdm_OSR=sdm_OSR,
            bandwidth=cutoff,
            fs=fs,
        ),
        PolyPhaseFIR(
            decimation_factor=sdm_OSR,
            cutoff=cutoff,
            cutoff_width=cutoff_width,
            filt_length=filt_length,
            num_bits_filter_Q=num_bits_filter_Q,
            num_bits_output=num_bits_output,
            fs=fs,
        ),
    )
