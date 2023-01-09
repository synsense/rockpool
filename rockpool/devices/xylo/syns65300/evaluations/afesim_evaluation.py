# -----------------------------------------------------------
# This module runs multiple tests on AFESim to check:
#   (i)     typical ranges of spike rate for a limited amplitude signal
#   (ii)    how the spike rate is affected by the system parameter.
#
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 05.01.2023
# -----------------------------------------------------------
from afe_audio.afe_sinusoid_probing import SinWaveGen
from rockpool.devices.xylo.syns65300.afe_sim import AFESim

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import sosfreqz



class AFESim_Evaluation:
    def __init__(self):
        # sampling frequency needed for simulation
        self.fs = 48_000
        

        # parameters of the filterbank as designed currently
        self.hardware_params = {
            "fs" : self.fs,
            "fc1" : 40,
            "f_factor" : 1.41,
            "LNA_gain" : 0,
            "Q" : 4,
            "digital_counter" : 1,
            "add_noise" : False,
            "thr_up" : 0.5,
            "leakage" : 1.0,
        }


        # create a afe module
        num_workers = 4
        self.afe = AFESim(**self.hardware_params, num_workers=num_workers)


        # sin wave generator
        num_bits_Q = 16
        self.sinwave = SinWaveGen(
            num_bits_Q=num_bits_Q,
            fs=self.fs,
        )

    
    def frequency_analysis_empirical(self):
        """
        This function computes the frequency response of the filters by computing the spike rate at the output of filters.
        """
        # duration of the sinusoid signal
        duration = 5.0

        # frequency range of the sinusoid signal
        fmin=100
        fmax=8000

        num_freq = 500

        scale_ratio = (fmax/fmin)**(1/(num_freq-1))
        freq_vec = fmin * (scale_ratio ** np.arange(num_freq))

        spike_rate_vec = []

        for freq in freq_vec:
            sig_in, _, _ = self.sinwave.evolve(
                freq=freq,
                duration=duration,
            )

            # compute output spike
            spike_out, *_ = self.afe.evolve(input=sig_in)

            # compute the rate in each channel
            spike_rate = spike_out.mean(axis=0) * self.fs

            spike_rate_vec.append(spike_rate)

            """
            # plot spike rate for each frequency
            plt.figure(figsize=(10,10))
            plt.plot(spike_rate)
            plt.xlabel("filter idx")
            plt.ylabel("spike rate (spike/sec")
            plt.grid(True)
            plt.title(f"AFE spike rate for a sinusoid with frequency {freq}")
            plt.draw()
            """
        
        # plot the spike rate as well
        spike_rate_vec = np.asarray(spike_rate_vec)
        spike_rate_vec = (spike_rate_vec + 1)/spike_rate_vec.max()


        plt.figure(figsize=(16,10))
        plt.semilogx(freq_vec, 20 * np.log10(spike_rate_vec))
        plt.grid(True)
        plt.xlabel("frequency (Hz)")
        plt.ylabel("empirical frequency response [dB]")
        plt.tight_layout()
        plt.ylim([-10, 0.3])
        plt.draw()

        plt.show()


    def plot_filters(self):
        """
        this function plots the frequency response of the filters in AFESim module.
        """
        from matplotlib import cm

        filters = self.afe.butter_filterbank._filters
        fcs = self.afe.butter_filterbank.frequency

        colors = cm.Blues(np.linspace(0.5, 1, len(filters)))
        plt.figure(figsize=(16, 10))
        for i, filt in enumerate(filters):
            sos_freqz = sosfreqz(filt, worN=2**15)
            db = 20 * np.log10(np.maximum(np.abs(sos_freqz[1]), 1e-5))
            plt.semilogx((self.fs / 2) * sos_freqz[0] / np.pi, db, color=colors[i])

        
        fmax = 8_000
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Gain (db)")
        plt.title("frequency response of filters as implemented in AFESim")
        plt.grid(True)
        plt.ylim([-10, 0.3])
        plt.xlim([20, fmax])
        plt.legend([f'fc = {f:0.2f}' for f in fcs])
        plt.tight_layout()
        plt.show()


def main():
    afesim_eval = AFESim_Evaluation()
    afesim_eval.frequency_analysis_empirical()
    afesim_eval.plot_filters()


if __name__ == '__main__':
    main()
