# Simulator of Xylo-A3 Audio front-end
This module contains a full simulator for the Xylo-A3 audio front-end, from the input audio up to the produced spikes.

In current version of Xylo-A3, there are two input signal paths which start from the input analog signal and end in a 14-bit quantized signal, which is used for further processing and feature extraction for the SNN within Xylo-A3.

These two input signal paths are as follows:
- **ADC with AGC (automatic gain control) capability**: In this signal path, the audio signal is amplified with a fixed-gain amplifier followed by a programmable gain amplifier whose amplification level is adjusted by
an AGC module. The output of this signal path is a 10-bit quantized output of AGC.
- **PDM quantization module**: This module takes the input audio signal and applies sigma-delta modultion on it. This transformation is applied within a microphone. The result is a 1-bit stream of modulated signal. The resulting 1-bit signal is further processed by a low-pass filter and decimator to produce the sampled audio signal. The PDM microphone followed by the implemented low-pass filtering can be seen as an equivalent ADC that takes the input audio signal and converts it into sampled\& quantized audio signal.
The output of this signal path is a 14-bit quantized signal.

In Xylo-A3, we have the option to choose from these two input signal paths where the output of 10-bit AGC-ADC is left-bit-shifted by 4 bits to yield a 14-bit quantized signal as in PDM-ADC.

The resulting 14-bit signal is then processed by the following modules to produce spike features:
- **Digital filterbanks**: The sampled audio signal is then porcessed by a collection of 16 band-pass filters. In this version of Xylo, all filters have been replaced with the digital ones and do not suffer the mismatch existing in the analog filters in the previous versions.
- **Divisive Normalization/Spike Generation**: The filtered signal is applied to a divisive normalization module (DN) to normalize its energy. In the previous version, spikes were produced by the analog AFE and one had to apply DN in the spikes domain. In this version, however, the output of the filters is directly available. So we decided to apply DN to the filter outputs rather than to the spikes. This provided a much better DN by merging the spike generation and DN in a single module.
- **Spike Generation**: There is an option to deactivate the DN module where in that case one can provide fixed thresholds (rather than adaptive thresholds obtained through DN) for spike generation.