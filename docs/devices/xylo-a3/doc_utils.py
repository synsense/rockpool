import os
from typing import Optional, Tuple

import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
from matplotlib.figure import Figure

from rockpool.devices.xylo.syns65302.afe import (
    AUDIO_SAMPLING_RATE_PDM,
    PDM_FILTER_DECIMATION_FACTOR,
)
from rockpool.timeseries import TSContinuous, TSEvent

file_path = os.path.dirname(os.path.realpath(__file__))

__all__ = [
    "generate_chirp",
    "plot_input_signal",
    "plot_filter_bank_output",
    "plot_divisive_normalization_output",
    "plot_raster_output",
]


def generate_chirp(
    filename: Optional[str] = "freq_sweep.wav",
    start_freq: float = 20,
    end_freq: float = 20000,
    duration: float = 4.0,
    fs: float = AUDIO_SAMPLING_RATE_PDM * PDM_FILTER_DECIMATION_FACTOR,
) -> Tuple[np.ndarray, float]:
    """
    Generate a frequency sweep signal and save it to a WAV file.
    Increase the frequency from `start_freq` to `end_freq` over `duration` seconds.
    Also saves the parameters used to generate the signal as a JSON file.

    Args:
        filename (Optional[str], optional): Name of the WAV file, it does not save as a `.wav` if None. Defaults to "freq_sweep.wav".
        start_freq (float, optional): The starting frequency of the sweep. Defaults to 20.
        end_freq (float, optional): The end frequency of the sweep. Defaults to 20000.
        duration (float, optional): the total duration of the audio. Defaults to 4.0.
        fs (float, optional): The sampling rate of the audio. Defaults to AUDIO_SAMPLING_RATE_PDM * PDM_FILTER_DECIMATION_FACTOR ~= 1.56 MHz.

    Returns:
        Tuple[np.ndarray, float]:
            audio (np.ndarray): The audio signal as a numpy array.
            fs (float): The sampling rate of the audio.
    """
    if filename is not None:
        if not filename.endswith(".wav"):
            filename = filename + ".wav"
        args = {
            "start_freq": start_freq,
            "end_freq": end_freq,
            "duration": duration,
            "fs": fs,
        }

    if fs < 2 * max(start_freq, end_freq):
        raise ValueError("Sampling rate must be at least twice the maximum frequency")

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
        with open(filename.replace(".wav", ".json"), "w") as f:
            json.dump(args, f)
    return audio, fs


def plot_chirp_signal(
    signal: np.ndarray,
    sr: float,
    t_cut: Optional[float] = 0.1,
    start_freq: float = 20,
    end_freq: float = 20000,
) -> Figure:
    """
    Plots the chirp signal with time and frequency axes.

    Args:
        signal (np.ndarray): the audio signal, most probably a `librosa.load` output.
        sr (float): the sampling rate of the audio
        t_cut (Optional[float], optional): first `t_cut` seconds of the signal is being processed. If None, the full signal is being plotted. Defaults to 0.1.
        start_freq (float, optional): The starting frequency of the sweep. Defaults to 20.
        end_freq (float, optional): The end frequency of the sweep. Defaults to 20000.

    Returns:
        Figure: _description_
    """
    # - Get the cut index
    if t_cut is None:
        t_cut = len(signal) / sr

    T = int(t_cut * sr)

    # - Set the plot and x-axis ticks
    fig, ax = plt.subplots(figsize=(16, 6))
    __freq_sw = np.linspace(start_freq, end_freq, len(signal))[:T]
    __time_sw = np.linspace(0, t_cut, T)

    # - Plot the time axis
    ax.plot(__time_sw, signal[:T])
    ax.set_xlabel("Time (s)")

    # - Plot the frequency axis
    ax_twin = ax.twiny()
    ax_twin.plot(__freq_sw[:T], signal[:T], alpha=0)
    ax_twin.set_xlabel("Frequency (Hz)")

    plt.title("Chirp Signal")
    plt.tight_layout()
    return fig


def time_to_frequency(
    time_ticks: list, start_frequency=20, end_frequency=20_000
) -> np.ndarray:
    """
    Convert time ticks to frequency ticks
    """
    freq_sw = np.linspace(start_frequency, end_frequency, len(time_ticks))
    return freq_sw


def plot_filter_bank_output(
    filtered_signal: np.ndarray,
    __sr: float,
    start_frequency=20,
    end_frequency=20000,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 12))

    TSContinuous.from_clocked(filtered_signal, dt=1 / __sr).plot(stagger=1e7)

    # - Annotate the frequency sweep
    ax_twin = ax.twiny()
    ax_twin.set_xlim(ax.get_xlim())

    # Get the current time ticks and convert them to frequency
    time_ticks = ax.get_xticks()
    frequency_ticks = time_to_frequency(time_ticks)

    # Apply the converted frequency ticks to the frequency axis
    ax_twin.set_xticks(time_ticks)
    ax_twin.set_xticklabels([f"{f:.1f}" for f in frequency_ticks])
    ax_twin.set_xlabel("Frequency (Hz)")

    # - Plot
    plt.title("Filter Bank Output")
    plt.tight_layout()
    plt.grid()
    plt.show()


def plot_divisive_normalization_output(spike_out: np.ndarray, sr: float) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))

    TSEvent.from_raster(spike_out, dt=1 / sr).plot()

    # - Annotate the frequency sweep
    ax_twin = ax.twiny()
    ax_twin.set_xlim(ax.get_xlim())

    # - Get the current time ticks and convert them to frequency
    time_ticks = ax.get_xticks()
    frequency_ticks = time_to_frequency(time_ticks)

    # - Apply the converted frequency ticks to the frequency axis
    ax_twin.set_xticks(time_ticks)
    ax_twin.set_xticklabels([f"{f:.1f}" for f in frequency_ticks])
    ax_twin.set_xlabel("Frequency (Hz)")

    # - Plot
    plt.title("Divisive Normalization Output")
    plt.tight_layout()
    plt.grid()
    plt.show()


def plot_raster_output(out: np.ndarray, dt: float) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(16, 12))

    # - Plot the raster
    plt.sca(axs[0])
    TSEvent.from_raster(out, dt=dt).plot()

    # - Plot the 3D image of the raster, color encoding the number of spikes
    axs[1].imshow(out.T, aspect="auto", origin="lower")
    axs[1].set_xlabel("Sample")
    axs[1].set_title("3D Spike Raster")

    plt.title("Accumulated Spike Output")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    generate_chirp()
