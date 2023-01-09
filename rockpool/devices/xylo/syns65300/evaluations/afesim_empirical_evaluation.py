# -----------------------------------------------------------
# This module runs multiple tests on AFESimEmpirical to check:
#   (i)     typical ranges of spike rate for a limited amplitude signal
#   (ii)    how the spike rate is affected by the system parameter.
#
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 09.01.2023
# -----------------------------------------------------------
from afe_audio.afe_sinusoid_probing import SinWaveGen
from rockpool.devices.xylo.syns65300.afe_sim_empirical import AFESimEmprirical

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import sosfreqz
from tqdm import tqdm



class AFESimEmpirical_Evaluation:
    def __init__(self):
        # sampling frequency needed for simulation
        self.fs = 110_000
        self.raster_period = 10e-3
        self.max_spike_per_period = 15
        

        # parameters of the filterbank as designed currently
        self.params = {
            "add_noise" : True,
            "add_offset" : True,
            "add_mismatch" : True,
            "one_shot_nonideality" : True,
            "num_workers" : 4,
            "fs" : self.fs,
            "raster_period": self.raster_period,
            "max_spike_per_period": self.max_spike_per_period,
        }


        # create a afe module
        self.afe = AFESimEmprirical(**self.params)


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

        for freq in tqdm(freq_vec):
            sig_in, _, _ = self.sinwave.evolve(
                freq=freq,
                duration=duration,
            )

            # scale the input signal to match its amplitude to the linear regime
            voltage_amplitude = 60e-3     # 60 mv
            sig_in = voltage_amplitude * sig_in/np.max(np.abs(sig_in))

            # compute output spike
            spike_out, *_ = self.afe.evolve(input=sig_in)

            # raster the spikes to the given period: this can also be used to compute the rate
            spike_rastered = self.afe.raster(spike_out)

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
    afesim_eval = AFESimEmpirical_Evaluation()
    afesim_eval.frequency_analysis_empirical()
    afesim_eval.plot_filters()


if __name__ == '__main__':
    main()
