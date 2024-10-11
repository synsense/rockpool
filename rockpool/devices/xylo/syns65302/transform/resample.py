"""
Implements Audio resampling module :py:class:`.ResampleAudio` to make sure that the XyloAudio 3 AFESim modules works with all possible sampling rates with no problem
"""

import logging
import warnings
from typing import Tuple, Union

import numpy as np

from rockpool.nn.modules import Module
from rockpool.parameters import SimulationParameter

__all__ = ["ResampleAudio"]


class ResampleAudio(Module):
    """
    Time resampling module, working perfectly with the XyloAudio 3 front-end modules
    """

    def __init__(self, fs_target: float) -> None:
        """
        Args:
            fs_target (float): Target sampling rate of the signal
        """
        super().__init__(shape=(1, 1), spiking_input=False, spiking_output=False)
        self.fs_target = SimulationParameter(fs_target, shape=())
        """ (float) The target output sampling rate for the audio signal."""

    def evolve(
        self, signal: Union[np.ndarray, Tuple[np.ndarray, float]], record: bool = False
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        Apply linear interpolation to the input audio signal and resample

        Args:
            signal (Union[np.ndarray, Tuple[np.ndarray, float]]): Either a numpy array, assumed to already be at the target sampling rate, or a tuple of the audio signal and the actual input sampling rate.
            record (bool): Record internal state during evoltion. Defaults to ``False``. Note: This module does not have internal state.

        Returns:
            Tuple[np.ndarray, dict, dict]:
                out: the resampled signal
                state_dict: empty dictionary
                record_dict : empty dictionary
        """
        if isinstance(signal, tuple):
            try:
                audio, sample_rate = signal

            except:
                raise TypeError(
                    "`signal` should be a tuple consisting of a numpy array containing the audio and its sampling rate."
                )
        elif isinstance(signal, np.ndarray):
            audio = signal
            sample_rate = self.fs_target
        else:
            raise TypeError(
                "`signal` must be either a numpy array or a tuple of a numpy array and a sampling rate."
            )

        # - Verify sampling rate
        if not isinstance(sample_rate, (int, float)):
            raise TypeError(
                "The given sampling rate must be an integer or float number."
            )

        # - perform auto-batching, verify dimensions
        audio, _ = self._auto_batch(audio)
        Nb, Nt, Nc = audio.shape

        if Nb > 1 or Nc > 1:
            raise ValueError(
                "Only single-batch, single-channel audio signals can be processed by this module."
            )

        audio = audio[0, :, 0]

        if sample_rate != self.fs_target:
            warnings.warn(
                f"Resampling the signal."
                + f"\nInput sampling rate = {sample_rate}, target sampling rate = {self.fs_target}"
            )
            duration = (len(audio) - 1) / sample_rate
            time_in = np.arange(len(audio)) / sample_rate
            time_target = np.arange(0, duration, step=1 / self.fs_target)
            audio = np.interp(time_target, time_in, audio)
            logging.info("Resampling done!")

        return audio, {}, {}
