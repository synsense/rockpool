# -----------------------------------------------------------
# This module provides some test cases to make sure that PDM microphone is working as it should.
#
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 18.01.2023
# -----------------------------------------------------------
import numpy as np
from xylo_a3_sim.pdm_adc import PDM_Microphone
import matplotlib.pyplot as plt
import ot


## some test cases
def test_pdm_microphone():
    # PDM microphone
    mic = PDM_Microphone()

    # produce an audio signal
    freq = 8_000
    audio_fs = 50_000
    num_periods = 1
    duration = num_periods / freq

    audio_time_vec = np.arange(0, duration, step=1 / audio_fs)
    audio = 0.99 * np.sin(2 * np.pi * freq * audio_time_vec)

    audio_time_vec_high_prec = np.arange(0, duration, step=1 / mic.fs)
    audio_high_prec = 0.99 * np.sin(2 * np.pi * freq * audio_time_vec_high_prec)

    # pdm signal
    audio_pdm = mic.evolve(audio=audio, audio_sampling_rate=audio_fs)
    audio_pdm_high_prec = mic.evolve(audio=audio_high_prec, audio_sampling_rate=mic.fs)

    pdm_time_vec = np.arange(0, duration, step=1 / mic.fs)

    # measure the distortion between two spikes
    spikes1 = audio_pdm > 0.0
    spikes1 = spikes1 / np.sum(spikes1)

    spikes2 = audio_pdm_high_prec > 0.0
    spikes2 = spikes2 / np.sum(spikes2)

    dist1 = np.cumsum(np.ones(len(spikes1))).reshape(-1, 1)
    dist2 = np.cumsum(np.ones(len(spikes2))).reshape(1, -1)
    distortion_mat = np.abs(dist1 - dist2)

    # apply wasserstein distance metric between two PDM binary processes
    ws_opt_mat = ot.emd(spikes1, spikes2, distortion_mat)
    ws_dist = np.sum(ws_opt_mat * distortion_mat)

    # how much on average relative movement in time is needed to match two PDM sequences
    ws_dist_norm = ws_dist / min(len(spikes1), len(spikes2))

    plt.figure(figsize=(10, 10))

    plt.subplot(211)
    plt.plot(pdm_time_vec, audio_pdm[: len(pdm_time_vec)])
    plt.plot(audio_time_vec, audio)
    plt.grid(True)
    plt.ylabel("low-prec audio signal")
    plt.legend(["pdm", "audio"])
    plt.title(f"audio signal [freq:{freq} Hz] and its binary PDM modulation")

    plt.subplot(212)
    plt.plot(pdm_time_vec, audio_pdm_high_prec)
    plt.plot(audio_time_vec_high_prec, audio_high_prec)
    plt.grid(True)
    plt.xlabel("time (sec)")
    plt.ylabel("high-prec audio signal")
    plt.legend(["pdm", "audio"])
    plt.draw()

    plt.figure(figsize=(10, 10))
    plt.plot(np.cumsum(spikes1))
    plt.plot(np.cumsum(spikes2))
    plt.grid(True)
    plt.legend(["low-prec", "high-prec"])
    plt.title(f"Variation in PDM bit-sequence: WS-dist:{ws_dist_norm:0.5f}/1.0")

    plt.show()


def main():
    test_pdm_microphone()


if __name__ == "__main__":
    main()
