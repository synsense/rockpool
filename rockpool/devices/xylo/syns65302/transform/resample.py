"""
Implements Audio resampling module to make sure that the XyloAudio 3 AFESim modules works with all possible sampling rates with no problem
"""
import logging
import warnings
from typing import Tuple

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

    def evolve(
        self, signal: Tuple[np.ndarray, float], record: bool = False
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        Apply linear interpolation to the input audio signal and resample

        Args:
            signal (Tuple[np.ndarray, float]): A tuple of the actual audio signal and the sampling rate
            record (bool, optional): Dummy variable, to meet the rockpool conventions. Defaults to False.

        Returns:
            Tuple[np.ndarray, dict, dict]:
                out: the resampled signal
                state_dict: empty dictionary
                record_dict : empty dictionary
        """
        try:
            audio, sample_rate = signal

        except:
            raise TypeError(
                "`signal` should be a tuple consisting of a numpy array containing the audio and its sample rate!"
            )

        if not isinstance(audio, np.ndarray):
            raise TypeError("The given input audio is not a numpy array!")
        if not isinstance(sample_rate, (int, float)):
            raise TypeError("The given sample rate is not a number!")
        if audio.ndim != 1:
            raise ValueError(
                "only single-channel audio signals can be processed by this module!"
            )

        if sample_rate != self.fs_target:
            warnings.warn(
                f"Resampling the signal!"
                + f"\nSample rate given = {sample_rate}, sample rate required = {self.fs_target}"
            )
            duration = (len(audio) - 1) / sample_rate
            time_in = np.arange(len(audio)) / sample_rate
            time_target = np.arange(0, duration, step=1 / self.fs_target)
            audio = np.interp(time_target, time_in, audio)
            logging.info("Resampling done!")

        return audio, {}, {}
