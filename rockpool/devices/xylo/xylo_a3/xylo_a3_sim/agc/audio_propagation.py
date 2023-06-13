# -----------------------------------------------------------
# This module implements a simple audio propagation model depending on the environment.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 28.03.2023
# -----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from rockpool.timeseries import TSContinuous
from typing import Any


# ===========================================================================
# *    some constants defined according to Xylo-A3 specficiations
# ===========================================================================
from agc.amplifier import XYLO_MAX_AMP, AUDIO_SAMPLING_RATE


class AudioWave:
    def __init__(
        self,
        power_decay_factor: float = 2.0,
        ref_distance: float = 1.0,
        max_audio_amplitude: float = XYLO_MAX_AMP,
        fs: float = AUDIO_SAMPLING_RATE,
    ):
        """this class takes an input audio signal and simulates how it is received as an audio wave
        when the distance from microphone starts to change.

        Args:
            power_decay_factor (float, optional): decay factor of the audio wave power as it propagates through the environment. Defaults to 2.0 for spherical propagation.
            ref_distance (float, optional): reference distance for the audio signal. At this distance the audio is received with max amplitude.
            max_audio_amplitude (float, optional): maximum amplitude of the bipolar audio signal.
            fs (float, optional): the sampling rate of the produced time-series (this can be larger than the audio sampling rate in case needed).
        """
        self.power_decay_factor = power_decay_factor
        self.ref_distance = ref_distance
        self.max_audio_amplitude = max_audio_amplitude
        self.fs = fs

    def evolve(
        self, audio: np.ndarray, audio_sample_rate: float, distance: TSContinuous
    ):
        """this module takes the input signal and simulates how this signal is received

        Args:
            audio (np.ndarray): input audio signal.
            audio_sample_rate (float): sampling rate of the audio.
            distance (TSContinuous): time-series illustrating how the distance of the audio source to the microphone changes with time.
        """
        audio_duration = len(audio) / audio_sample_rate

        time_vec = np.arange(0, audio_duration, step=1 / self.fs)

        # sample the audio signal and also distance function with the sampling rate of the module
        audio_resampled = TSContinuous.from_clocked(
            audio, dt=1 / audio_sample_rate, periodic=True
        )(time_vec).ravel()
        distance_resampled = distance(time_vec).ravel()

        ## compute the updated audio
        # scale the audio so that at reference distance we have max amplitude
        audio_resampled = (
            audio_resampled / np.max(np.abs(audio_resampled)) * self.max_audio_amplitude
        )

        # NOTE:
        # amplitude decay factor is half the power decay factor.
        # This is the reason we have division by 2.0.

        EPS = 0.0001
        audio_received = audio_resampled / (
            EPS
            + np.power(
                distance_resampled / self.ref_distance, self.power_decay_factor / 2.0
            )
        )

        # truncate the audio amplitude if it goes beyond voltage source
        audio_received[
            audio_received > self.max_audio_amplitude
        ] = self.max_audio_amplitude
        audio_received[
            audio_received < -self.max_audio_amplitude
        ] = -self.max_audio_amplitude

        return audio_received

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        this function is the same as evolve.
        """
        return self.evolve(*args, **kwargs)
