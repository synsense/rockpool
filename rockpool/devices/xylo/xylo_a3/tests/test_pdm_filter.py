# -----------------------------------------------------------
# This module provides some test cases to make sure that low-pass filter + decimation part is working well.
#
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 18.01.2023
# -----------------------------------------------------------

from xylo_a3_sim.pdm_adc import PDM_Microphone, PolyPhaseFIR_DecimationFilter

import numpy as np
import matplotlib.pyplot as plt
from rockpool.timeseries import TSContinuous


def test_filter():
    # use default values for microphone and decimation filter
    mic = PDM_Microphone()
    decimation_filt = PolyPhaseFIR_DecimationFilter()

    # create a simple audio signal
    freq = 100
    num_periods = 4
    duration = num_periods / freq

    # we use maximum sampling frequency for a high-prec simulation
    audio_sampling_rate = mic.fs

    # NOTE: case 1: sinusoid signal
    time_vec = np.arange(0, duration, step=1 / mic.fs)
    audio_sin = np.sin(2 * np.pi * freq * time_vec)

    # NOTE: case 2: worst-case audio that may reach very close the dynamic range specified by num_bits_pre_Q
    # In general, this incurs a loss in dynamic range of the audio signal BUT
    # we should not worry since we have devoted 10 + 4 (4 additional) bits and those bits will compensate this loss.
    audio_worst = TSContinuous.from_clocked(
        np.sign(decimation_filt.h[::-1]), dt=1 / (mic.fs / mic.sdm_OSR), periodic=True
    )(time_vec).ravel()

    audio_list = [audio_worst, audio_sin]

    for audio in audio_list:
        # obtain the binary pdm encoding
        audio_pdm = mic(audio=audio, audio_sampling_rate=audio_sampling_rate)

        # interpolate to recover the sampled audio
        audio_rec = decimation_filt(sig_in=audio_pdm)

        # for sanity-check we also use the undecimated version
        audio_rec_undecimated = decimation_filt.fullevolve(sig_in=audio_pdm)

        # scale the recovered audio
        audio_rec_scaled = audio_rec / (2 ** (decimation_filt.num_bits_output - 1) - 1)
        audio_rec_undecimated_scaled = audio_rec_undecimated / (
            2 ** (decimation_filt.num_bits_output - 1) - 1
        )

        MAX_VAL = np.max(np.abs(audio_rec_undecimated_scaled))

        plt.figure(figsize=(10, 10))
        audio_dt = 1 / mic.fs
        audio_rec_dt = 1 / (decimation_filt.fs / decimation_filt.decimation_factor)
        audio_rec_undecimated_dt = 1 / decimation_filt.fs

        plt.plot(np.arange(0, len(audio)) * audio_dt, audio)
        plt.plot(
            np.arange(0, len(audio_rec_undecimated_scaled)) * audio_rec_undecimated_dt,
            audio_rec_undecimated_scaled,
        )
        plt.plot(
            np.arange(0, len(audio_rec_scaled)) * audio_rec_dt, audio_rec_scaled, ".-"
        )

        plt.xlabel("time (sec)")
        plt.ylabel("PDM ADC")
        plt.grid(True)
        plt.legend(["original", "recovered-undecimated", "pdm-adc"])
        plt.title(
            f"Dynamic range: input signal covers a {MAX_VAL:0.8f} of the dynamic range in {decimation_filt.num_bits_output} output bits"
        )

        plt.draw()

    plt.show()


def main():
    test_filter()


if __name__ == "__main__":
    main()
