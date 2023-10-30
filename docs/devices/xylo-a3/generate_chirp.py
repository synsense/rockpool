import os
from typing import Optional, Tuple

import numpy as np
import scipy.io.wavfile

from rockpool.devices.xylo.syns65302.afe import (
    AUDIO_SAMPLING_RATE,
    PDM_FILTER_DECIMATION_FACTOR,
)

file_path = os.path.dirname(os.path.realpath(__file__))


def generate_chirp(
    filename: Optional[str] = "freq_sweep.wav",
    start_freq: float = 20,
    end_freq: float = 20000,
    duration: float = 4.0,
    fs: float = AUDIO_SAMPLING_RATE * PDM_FILTER_DECIMATION_FACTOR,
) -> Tuple[np.ndarray, float]:
    """
    Generate a frequency sweep signal and save it to a WAV file.
    Increase the frequency from `start_freq` to `end_freq` over `duration` seconds.

    Args:
        filename (Optional[str], optional): Name of the WAV file, it does not save as a `.wav` if None. Defaults to "freq_sweep.wav".
        start_freq (float, optional): The starting frequency of the sweep. Defaults to 20.
        end_freq (float, optional): The end frequency of the sweep. Defaults to 20000.
        duration (float, optional): the total duration of the audio. Defaults to 4.0.
        fs (float, optional): The sampling rate of the audio. Defaults to AUDIO_SAMPLING_RATE ~= 48.8k.

    Returns:
        Tuple[np.ndarray, float]:
            audio (np.ndarray): The audio signal as a numpy array.
            fs (float): The sampling rate of the audio.
    """
    # - Use the half duration, because we will concatenate the signal with its reverse
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)  # time variable
    freq_sweep = np.linspace(start_freq, end_freq, int(duration * fs), endpoint=False)

    # - Concatenate the signal with its reverse
    phi_inst = 2 * np.pi * np.cumsum(freq_sweep) * (1 / fs)
    signal = np.sin(phi_inst)

    # - Ensure that highest values are in 16-bit range
    audio = np.int16(signal / np.max(np.abs(signal)) * np.int16(2**15 - 1))

    if filename is not None:
        filename = os.path.join(file_path, filename)
        scipy.io.wavfile.write(filename, int(fs), audio)
        np.save(filename.replace(".wav", ".npy"), freq_sweep)
    return audio, fs


if __name__ == "__main__":
    generate_chirp()
