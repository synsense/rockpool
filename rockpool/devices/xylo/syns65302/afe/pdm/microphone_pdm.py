"""
A deltasigma based PDM microphone that converts the input analog audio into a PDM bit stream 
where the relative frequency of 1-vs-0 depends on the amplitude of the signal
"""

import warnings
from numbers import Number
from typing import Any, Tuple, Union

import numpy as np

from rockpool.devices.xylo.syns65302.afe.params import (
    AUDIO_CUTOFF_FREQUENCY,
    DELTA_SIGMA_ORDER,
    PDM_FILTER_DECIMATION_FACTOR,
    PDM_SAMPLING_RATE,
)
from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter
from rockpool.typehints import P_float, P_int

from .delta_sigma import DeltaSigma

__all__ = ["MicrophonePDM"]


class MicrophonePDM(Module):
    """
    This class simulates a PDM microphone which applies deltasigma modulation on the input audio signal

    The input to microphone is an analog audio signal and the output is a PDM bit-stream in which the relative
    frequency of 1-vs-0 depends on the instantaneous amplitude of the signal.
    """

    def __init__(
        self,
        sdm_order: int = DELTA_SIGMA_ORDER,
        sdm_OSR: int = PDM_FILTER_DECIMATION_FACTOR,
        bandwidth: float = AUDIO_CUTOFF_FREQUENCY,
        fs: float = PDM_SAMPLING_RATE,
    ):
        """
        Initialise a MicrophonePDM module

        Args:
            sdm_order (int): order of the deltasigma modulator (conventional ones are 2 or 3). Defaults to DELTA_SIGMA_ORDER.
            sdm_OSR (int): oversampling rate in deltasigma modulator. Defaults to PDM_FILTER_DECIMATION_FACTOR.
            fs (int): rate of the clock used for deriving the PDM microphone.
                    NOTE: PDM microphone can be derived by various clock rates. By changing the clock rate and sdm_OSR we can
                    keep the sampling rate of the audio fixed.
        """
        super().__init__(shape=1, spiking_output=True)

        # deltasigma modulator parameters
        self.sdm_order: P_int = SimulationParameter(sdm_order)
        """ int: Order of the deltasigma modulator """

        self.sdm_OSR: P_int = SimulationParameter(sdm_OSR)
        """ int: Oversampling rate in deltasigma modulator """

        self.fs: P_float = SimulationParameter(fs)
        """ float: Sampling rate of this module in Hz """

        target_audio_sample_rate = self.fs / self.sdm_OSR
        if bandwidth > target_audio_sample_rate / 2.0:
            raise ValueError(
                f"PDM microphone with clock rate {self.fs} and oversampling factor {self.sdm_OSR} is targeted to process audio with sample rate {target_audio_sample_rate}.\n"
                + f"Therefore its deltasigma analog circuits should have a bandwidth less than half target audio sample rate, i.e., < {target_audio_sample_rate/2} Hz.\n"
            )

        self.bandwidth: P_float = SimulationParameter(bandwidth)

        # build the deltasigma module
        # self._sdm_module = ds.synthesizeNTF(self.sdm_order, self.sdm_OSR, opt=1)
        self._sdm_module = DeltaSigma(
            amplitude=1.0,
            bandwidth=self.bandwidth,
            order=self.sdm_order,
            fs=self.fs,
        )

    def evolve(
        self,
        audio_in: Union[np.ndarray, Tuple[np.ndarray, float]],
        record: bool = False,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """
            This function takes the input audio signal and produces a PDM bit-stream

            NOTE: In reality the input signal to deltasigma modulator in PDM microphone is the analog audio signal. In simulation, however, we have to still use a sampled version of this analog signal as representative.

            NOTE: the audio signal should be normalized to the valid range of deltasigma modulator [-1.0, 1.0]. If not in this range, clipping should be applied manually to limit the signal into this range.
            If the signal amplitue is very close to +1 or -1, there is a higher chance that the block-diagram version we use to simulate the signal diverges. In such cases, it is better to reduce the signal amplitude slightly.

            Resampling is performed if the sample rate of the input signal is less that the clock rate of PDM bitstream.

        Args:
            audio_in (Tuple[np.ndarray), float]: a tuple containing the input audio signal and its sampling rate.
            record (bool): Record the inner states of the deltasigma module used for PDM modulation. Default: ``False``

        Raises:
            ValueError: if the amplitude is not scaled properly and is not in the valid range [-1.0, 1.0]

        Returns:
            np.ndarray: array containing PDM bit-stream at the output of the microphone.
        """

        if isinstance(audio_in, tuple):
            try:
                audio, sample_rate = audio_in

                if isinstance(audio, np.ndarray) and isinstance(sample_rate, Number):
                    pass
            except:
                raise TypeError(
                    "`audio_in` should be a tuple consisting of a numpy array containing the audio and its sampling rate."
                )
        elif isinstance(audio_in, np.ndarray):
            audio = audio_in
            sample_rate = self.fs
        else:
            raise TypeError(
                "`audio_in` must be either a numpy array or a tuple of a numpy array and the sampling rate."
            )

        audio, _ = self._auto_batch(audio)
        Nb, Nt, Nc = audio.shape

        if Nb > 1 or Nc > 1:
            raise ValueError(
                "only single-batch single-channel audio signals can be processed by the deltasigma modulator in PDM microphone."
            )

        audio = audio[0, :, 0]

        if np.max(np.abs(audio)) > 1.0:
            raise ValueError(
                "Some of the signal samples have an amplitude larger than 1.0.\n"
                + "Sigma-delta modulator is designed to work with signal values normalized in the range [-1.0, 1.0].\n"
                + "Normalize the signal or clip it to the range [-1.0, 1.0] manually before applying it to PDM microhpne."
            )

        if sample_rate != self.fs:
            warnings.warn(
                "\n\n"
                + " warnings ".center(120, "+")
                + "\n"
                + f"In practice, the input to the PDM microphone (fed by a clock of rate:{self.fs}) is the analog audio signal.\n"
                + "In simulations, however, we have to use sampled audio signal at the input to mimic this analog signal.\n"
                + f"Here we resample the input audio to the higher sampling rate of PDM microphone ({self.fs}).\n"
                + "For a more realistic simulation, it is better to provide an audio signal which is originally sampled with a higher rate.\n"
                + "+" * 120
                + "\n\n"
            )

        # compute the deltasigma modulation
        (
            audio_pdm,
            audio_pdm_pre_Q,
            sig_resampled,
            deltasigma_filter_states,
        ) = self._sdm_module.evolve(
            sig_in=audio, sample_rate=sample_rate, record=record
        )

        if record:
            recording = {
                "deltasigma_signal_pre_Q": audio_pdm_pre_Q,
                "deltasigma_filter_states": deltasigma_filter_states,
            }
        else:
            recording = {}

        # validate the signal to make sure that it is reasonably converged
        if not self._sdm_module.validate(sig_in=sig_resampled, bin_in=audio_pdm):
            warnings.warn(
                "It seems that the simulator used for deltasigma modulation has not converged properly!\n"
                + "To solve this issue, try reducing the amplitude of the signal. Also try feeding the input with a little bit of noise, which may help the convergence!\n"
            )

        # use the integer format for the final {-1,+1}-valued binary PDM data
        if audio_pdm.dtype != np.int64:
            audio_pdm = audio_pdm.astype(np.int64)

        unique_vals = set(np.unique(audio_pdm))
        if unique_vals != {-1, 1} and unique_vals != {1} and unique_vals != {-1}:
            raise ValueError(
                "The output of deltasigma modulator should be a {-1,+1}-valued signal.\n"
                + "This problem may arise when the deltasigma simulator is unstable!\n"
                + "To solve this issue, try reducing the amplitude of the signal. Also try feeding the input with a little bit of noise, which may help the convergence!\n"
            )

        return audio_pdm, self.state(), recording

    def __setattr__(self, name: str, val: Any):
        # - Use super-class setattr to assign attribute
        super().__setattr__(name, val)

        # - Re-generate SDM module
        if not self._in_Module_init and name != "_sdm_module":
            self._sdm_module = DeltaSigma(
                amplitude=1.0,
                bandwidth=self.bandwidth,
                order=self.sdm_order,
                fs=self.fs,
            )

    def _info(self) -> str:
        string = (
            "This is the module for simulating PDM microphone which uses deltasigma modulation.\n"
            + "The input analog audio signal is mapped to a binary stream of modulated data which is them interpolated to recover a sampled version of the analog input.\n"
            + "Parameters:\n"
            + f"Sigma-Delta modulation order: {self.sdm_order}\n"
            + f"Sigma-Delta oversampling rate (ratio between the rate of PDM clock and target audio sampling rate): {self.sdm_OSR}\n"
            + f"Sigma-Delta clock rate: {self.fs}\n"
        )
        return string
