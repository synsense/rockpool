"""
Simulation of an analog audio filtering front-end
"""

# - Rockpool imports
from rockpool.nn.modules.timed_module import TimedModule
from rockpool.timeseries import TSEvent
from rockpool.parameters import Parameter, State, SimulationParameter, ParameterBase

# - Matplotlib
from matplotlib import mlab
from matplotlib import pyplot as plt

# - Other imports
import numpy as np
from scipy.signal import butter, lfilter, freqz
from scipy import signal, fftpack

from typing import Union

P_int = Union[int, ParameterBase]
P_float = Union[float, ParameterBase]
P_array = Union[np.array, ParameterBase]


class AFE(TimedModule):
    def __init__(
        self,
        shape: tuple = (16,),
        Q: int = 5,  # 3-5
        fc1: float = 100.0,
        f_factor: float = 1.325,
        thr_up: float = 0.5,
        leakage: float = 1.0,  # 0.5-20 nA
        digital_counter: int = 1,  # keep 1 spike every xxx spikes
        LNA_gain: float = 0.0,  # ~ +6db  6db/steps
        fs: int = 16000,
        manual_scaling: float = None,
        add_noise: bool = True,
        seed: int = None,
    ):
        """
        The Analog Frontend simulates the analog hardware for preprocessing audio.

        Parameters
        ----------
        Q: int
           Quality factor 
        fc1: float 
            Center frequency of the first band-pass filter 
        f_factor: float
            Logarithmic distribution of the center frequencies is based on f_factor
        thr_up: float 
            Spiking threshold for spike conversion
        leakage: float
            Leakage for spike conversion 
        digital counter: int
            Digital counter for spike conversion - lets only every nth spike pass
        LNA_gain: float
            Gain of the low-noise amplification
        fs: int
            Sampling frequency of the input data
        num_filters: int
            Number of filters
        manual_scaling: float
            Disables automatic scaling from the LNA and instead scales the input by this factor
        add_noise: bool
            Enables / disables the simulated noise generated be the AFE.
        seed: bool
            The AFE is subject to mismatch, this can be seeded.      
        """

        # - Check shape argument
        if not np.isscalar(shape):
            raise ValueError(
                "The `shape` argument to AFE must only specify one dimension."
            )

        # - Initialise superclass
        super().__init__(shape=shape, dt=1 / fs)

        # - Provide pRNG seed
        self.seed: P_int = SimulationParameter(seed)
        if self.seed is not None:
            self._rng_state = np.random.get_state()
            np.random.seed(self.seed)

        # - Max input from microphone
        self.INPUT_AMP_MAX: P_float = Parameter(320e-3)  # 100mV
        """ Maximum input amplitude from the microphone in Volts (Default: 320mV) """

        #
        self.C_IAF: P_float = Parameter(5e-12)  # 2 pF
        """ Integrator Capacitance for IAF (Default: 5e-12)"""

        # - Parameters for BPF
        self.Q: P_int = Parameter(Q)
        """ Quality parameter for band-pass filters"""

        # - Center frequency for 1st BPF in filter bank
        self.FC1: P_float = Parameter(fc1)
        """ Centre frequnecy of first filter, in Hz. """

        self.Fs: P_float = SimulationParameter(fs)
        """ Sample frequency of input data """

        # Frequency f_bp1 = fc1     f_bp2 = fc1*f_factor   f_bp3 = fc1*f_factor^2...
        self.f_factor: P_float = Parameter(f_factor)  # 16 channel  100Hz - 8KHz
        """ Centre-frequency scale-up factor per channel.
        
            Centre freq. F1 = FC1
            Centre freq. F2 = FC1 * f_factor
            Centre freq. F3 = FC1 * f_factor**2
            ...
        """

        self.ORDER_BPF: P_int = Parameter(2)
        """ Band-pass filter order (Default: 2)"""

        # Non-ideal
        self.MAX_INPUT_OFFSET: P_float = Parameter(0)  # form microphone
        """ Maxmimum input offset from microphone (Default: 0)"""

        self.MAX_LNA_OFFSET: P_float = Parameter(5)  # +/-5mV random
        """ Maxmimum low-noise amplifier offset in mV (Default: 5mV) """

        self.MAX_BPF_OFFSET: P_float = Parameter(5)  # +/-5mV random
        """ Maxmum band-pass filter offset in mV (Default: 5mV)"""

        self.DISTORTION: P_float = Parameter(0.1)  # 0-1
        """ Distortion parameter (0..1) Default: 0.1"""

        self.BPF_FC_SHIFT: P_float = Parameter(
            -5
        )  # 5 for +5%    -5 for -5%  ------- 16 channels center freq shift for same direction
        """ Centre frequency band-pass filter shift in % (Default: -5%) """

        self.Q_MIS_MATCH: P_float = Parameter(10)  # +/-10% random
        """ Mismatch in Q in % (Default: 10%) """

        self.FC_MIS_MATCH: P_float = Parameter(5)  # +/-5% random
        """ Mismatch in centre freq. in % (Default: 5%)"""

        ## Threshold for delta modulation in up and down direction
        self.THR_UP: P_float = Parameter(thr_up)  # 0.1-0.8 V
        """ Threshold for delta modulation in V (0.1--0.8) (Default: 0.5V)"""

        self.LEAKAGE: P_float = Parameter(leakage)
        """ Leakage for LIF neuron in nA. Default: 1nA """

        self.DIGITAL_COUNTER: P_int = Parameter(digital_counter)
        """ Digital counter factor to reduce output spikes by. Default: 1 (no reduction) """

        ########### Macro defnitions related to noise ###
        self.VRMS_SQHZ_LNA: P_float = Parameter(70e-9)
        self.F_KNEE_LNA: P_float = Parameter(70e3)
        self.F_ALPHA_LNA: P_float = Parameter(1)

        self.VRMS_SQHZ_BPF: P_float = Parameter(1e-9)
        self.F_KNEE_BPF: P_float = Parameter(100e3)
        self.F_ALPHA_BPF: P_float = Parameter(1)

        self.VRMS_SQHZ_FWR: P_float = Parameter(700e-9)
        self.F_KNEE_FWR: P_float = Parameter(158)
        self.F_ALPHA_FWR: P_float = Parameter(1)

        self.F_CORNER_HIGHPASS: P_float = Parameter(100)
        """ High pass corner frequency due to AC Coupling from BPF to FWR in Hz. (Default: 100Hz)"""

        # LNA
        self.lna_gain_db: P_float = Parameter(LNA_gain)  # in dB
        """ Low-noise amplifer gain in dB (Default: 0.) """

        self.lna_offset: P_float = State(
            np.random.randint(-self.MAX_LNA_OFFSET, self.MAX_LNA_OFFSET) * 0.001
        )
        """ Mismatch offset in low-noise amplifier """

        self.bpf_offset: P_array = State(
            np.random.randint(-self.MAX_BPF_OFFSET, self.MAX_BPF_OFFSET, self.size_out)
        )
        """ Mismatch offset in band-pass filters """

        self.Q_mismatch: P_array = State(
            np.random.randint(-self.Q_MIS_MATCH, self.Q_MIS_MATCH, self.size_out)
        )
        """ Mismatch in Q over band-pass filters """

        self.fc_mismatch: P_array = State(
            np.random.randint(-self.FC_MIS_MATCH, self.FC_MIS_MATCH, self.size_out)
        )
        """ Mismatch in centre frequency for band-pass filters """

        fc1 = self.FC1 * (1 + self.BPF_FC_SHIFT / 100) * (1 + self.fc_mismatch[0] / 100)
        self.fcs: P_array = State(
            [fc1]
            + [
                fc1 * (self.f_factor ** i) * (1 + self.fc_mismatch[i] / 100)
                for i in np.arange(1, self.size_out)
            ]
        )
        """ Centre frequency of each band-pass filter in Hz """

        # Add the non-ideality of the filter - shift in the centre frequency of the BPF by mismatch
        self.bws: P_array = State(
            [
                self.fcs[i] / (self.Q * (1 + self.Q_mismatch[i] / 100))
                for i in range(self.size_out)
            ]
        )
        """ Shift in centre frequencies due to mismatch """

        self.manual_scaling: P_float = SimulationParameter(manual_scaling)
        """ Manual scaling of low-noise amplifier gain. Default: `None` (use automatic scaling) """

        self.add_noise: Union[bool, ParameterBase] = SimulationParameter(add_noise)
        """ Flag indicating that noise should be simulated during operation. Default: `True` """

    def _butter_bandpass(
        self, lowcut: float, highcut: float, fs: float, order: int = 2
    ) -> (float, float):
        """
        Build a Butterworth bandpass filter from specification

        Args:
            lowcut (float): Low-cut frequency in Hz
            highcut (float): High-cut frequency in Hz
            fs (float): Sampling frequecy in Hz
            order (int): Order of the filter

        Returns: (float, float): b, a
            Parameters for the bandpass filter
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        return b, a

    def _butter_bandpass_filter(
        self, data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 2
    ) -> np.ndarray:
        """
        Filter data with a bandpass Butterworth filter, according to specifications

        Args:
            data (np.ndarray): Input data with shape ``(T, N)``
            lowcut (float): Low-cut frequency in Hz
            highcut (float): High-cut frequency in Hz
            fs (float): Sampling frequency in Hz
            order (int): Order of the filter

        Returns: np.ndarray: Filtered data with shape ``(T, N)``
        """
        b, a = self._butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def _butter_highpass(
        self, cutoff: float, fs: float, order: int = 1
    ) -> (float, float):
        """
        Build a Butterworth high-pass filter from specifications

        Args:
            cutoff (float): High-pass cutoff frequency in Hz
            fs (float): Sampling rate in Hz
            order (int): Order of the filter

        Returns: (float, float): b, a
            Parameters for the high-pass filter
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="high", analog=False)
        return b, a

    def _butter_highpass_filter(
        self, data: np.ndarray, cutoff: float, fs: float, order: int = 1
    ) -> np.ndarray:
        """
        Filter some data with a Butterworth high-pass filter from specifications
        
        Args:
            data (np.ndarray): Array of input data to filter, with shape ``(T, N)``
            cutoff (float): Cutoff frequency of the high-pass filter, in Hz
            fs (float): Sampling frequency of ``data``, in Hz
            order (int): Order of the Butterwoth filter

        Returns: np.ndarray: Filtered output data with shape ``(T, N)``
        """
        b, a = self._butter_highpass(cutoff, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y

    #################  Function to Generate Noise  ########################
    def generateNoise(
        self, x, Fs=16e3, VRMS_SQHZ=1e-6, F_KNEE=1e3, F_ALPHA=1.4, PLOT_SPECTRUM=False
    ):

        if not self.add_noise:
            return np.zeros_like(x)

        def one_over_f(f, knee, alpha):
            d = np.ones_like(f)
            d[f < knee] = np.abs(((knee / f[f < knee]) ** (alpha)))
            d[0] = 1
            return d

        W_NOISE_SIGMA = VRMS_SQHZ * np.sqrt(Fs / 2)  # Noise in the bandwidth 0 - Fs/2

        N = len(x)
        wn = np.random.normal(0, W_NOISE_SIGMA, N)
        Ts = 1 / Fs
        t = np.arange(0, len(x) * Ts, Ts)
        s = fftpack.rfft(wn)
        f = fftpack.rfftfreq(len(s)) * Fs
        ff = s * one_over_f(f, F_KNEE, F_ALPHA)
        x_t = fftpack.irfft(ff)
        s, f = mlab.psd(x_t, NFFT=N, Fs=Fs, scale_by_freq=True)

        if PLOT_SPECTRUM == True:
            plt.figure()
            plt.loglog(f, np.sqrt(s), "b-", label="simulated noise")
            plt.loglog(
                f,
                one_over_f(f, F_KNEE, F_ALPHA) * W_NOISE_SIGMA / np.sqrt(Fs / 2),
                "r",
                label="noise asymptote",
            )
            plt.vlines(F_KNEE, *plt.ylim())
            plt.grid()
            plt.xlabel("Frequency")
            plt.title("Amplitude spectrum of noise generated")
            plt.legend()

            plt.figure()
            plt.plot(t, x_t, "b-")
            plt.xlabel("Time (s)")
            plt.ylabel("Volts(V)")
            plt.grid()
            plt.title("Noise voltage in time domain")

        return x_t

    def encode_spikes(self, t, filter_out, thr_up, c_iaf, leakage):
        cdc = filter_out[0]
        td = t[0]
        data_up = []
        data = filter_out * 1e-6  # convert voltage to current for V2I module
        for i in range(len(data)):
            dt = t[i] - td
            lk = leakage * cdc * 1e-9
            dq_lk = lk * dt
            dv = (dt * data[i] - dq_lk) / c_iaf
            cdc = cdc + dv if (cdc + dv) >= 0 else 0
            if cdc < thr_up:
                data_up.append(0)
                td = t[i]
            elif cdc > thr_up:
                data_up.append(1)
                cdc = 0
                td = t[i]
            else:
                data_up.append(0)
                td = t[i]
        return data_up

    def sampling_signal(self, ch1, count):

        sam_count = 1
        sampled = []

        for i in range(len(ch1)):
            if (ch1[i] == 1) & (sam_count < count):
                sam_count = sam_count + 1
                sampled.append(0)
            elif (ch1[i] == 1) & (sam_count == count):
                sam_count = 1
                sampled.append(1)
            else:
                sampled.append(0)

        return sampled

    def evolve(
        self,
        ts_input=None,
        duration=None,
        num_timesteps=None,
        kwargs_timeseries=None,
        record: bool = False,
        *args,
        **kwargs,
    ):

        t = np.arange(0, num_timesteps) * self.dt
        y = ts_input(t)[:, 0]

        input_offset = self.MAX_INPUT_OFFSET
        if self.manual_scaling:
            y_scaled = self.manual_scaling * y * self.INPUT_AMP_MAX + input_offset
        else:
            y_scaled = (y / max(y)) * self.INPUT_AMP_MAX + input_offset

        #######   LNA - Gain  ##########
        lna_nonlinearity = self.DISTORTION / self.INPUT_AMP_MAX
        lna_distortion = (y_scaled ** 2) * lna_nonlinearity
        lna_gain_v = 2 ** (self.lna_gain_db / 6)
        lna_out = y_scaled * (1 + lna_distortion) * lna_gain_v + self.lna_offset

        #######  Add Noise ###############
        lna_out = lna_out + self.generateNoise(
            lna_out,
            self.Fs,
            self.VRMS_SQHZ_LNA,
            self.F_KNEE_LNA,
            self.F_ALPHA_LNA,
            False,
        )

        bpfs = [
            self._butter_bandpass_filter(
                lna_out + self.bpf_offset[i] * 0.001,
                self.fcs[i] - self.bws[i] / 2,
                self.fcs[i] + self.bws[i] / 2,
                self.Fs,
                order=self.ORDER_BPF,
            )
            for i in range(self.size_out)
        ]

        # add noise
        bpfs = [
            bpfs[i]
            + self.generateNoise(
                bpfs[i],
                self.Fs,
                self.VRMS_SQHZ_BPF,
                self.F_KNEE_BPF,
                self.F_ALPHA_BPF,
                False,
            )
            for i in range(self.size_out)
        ]

        rcs = [
            abs(
                self._butter_highpass_filter(
                    bpfs[i], self.F_CORNER_HIGHPASS, self.Fs, order=1
                )
                + self.generateNoise(
                    bpfs[i],
                    self.Fs,
                    self.VRMS_SQHZ_FWR,
                    self.F_KNEE_FWR,
                    self.F_ALPHA_FWR,
                    False,
                )
            )
            for i in range(self.size_out)
        ]

        # Encoding to spike by integrating the FWR output for positive going(UP)
        spikes = [
            self.encode_spikes(t, rcs[i], self.THR_UP, self.C_IAF, self.LEAKAGE)
            for i in range(self.size_out)
        ]

        spikes = np.array(
            [
                self.sampling_signal(spikes[i], self.DIGITAL_COUNTER)
                for i in range(self.size_out)
            ]
        )

        if record:
            recording = {
                "LNA_out": lna_out,
                "BPF": bpfs,
                "rect": rcs,
                "spks_out": spikes,
            }
        else:
            recording = {}

        spikes = TSEvent.from_raster(spikes.T, dt=self.dt)

        return spikes, self.state(), recording
