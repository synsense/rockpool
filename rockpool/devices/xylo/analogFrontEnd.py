"""
Simulation of an analog audio filtering front-end
"""

# - Rockpool imports
from rockpool.nn.modules.module import Module
from rockpool.nn.modules.native.filter_bank import ButterFilter
from rockpool.timeseries import TSEvent, TSContinuous
from rockpool.parameters import Parameter, State, SimulationParameter, ParameterBase

# - Other imports
import numpy as np
from scipy.signal import butter, lfilter
from scipy import signal, fftpack

from typing import Union

P_int = Union[int, ParameterBase]
P_float = Union[float, ParameterBase]
P_array = Union[np.array, ParameterBase]

# - Define exports
__all__ = ["AFE"]

# - Try to use Jax as speedup
try:
    import jax
    import jax.numpy as jnp

    enable_jax = True

    @jax.jit
    def _encode_spikes(
        inital_state: np.ndarray,
        dt: float,
        data: np.ndarray,
        thr_up: float,
        c_iaf: float,
        leakage: float,
    ) -> (np.ndarray, np.ndarray):
        """
        Encode a signal as events, using an LIF neuron membrane

        Args:
            inital_state (np.ndarray): Initial state of the LIF neurons
            dt (float): Time-step in seconds
            data (np.ndarray): Array ``(T,N)`` containing data to convert to events
            thr_up (float): Firing threshold voltage
            c_iaf (float): Membrane capacitance
            leakage (float): Leakage factor per time step

        Returns: np.ndarray: Raster of output events ``(T,N)``, where ``True`` indicates a spike
        """
        data *= 1e-6  # convert voltage to current for V2I module

        def forward(cdc, data_t):
            lk = leakage * cdc * 1e-9
            dq_lk = lk * dt
            dv = (dt * data_t - dq_lk) / c_iaf

            # - Accumulate membrane voltage, clip to zero
            cdc += dv
            cdc *= cdc > 0.0

            spikes = cdc > thr_up
            return cdc * (1 - spikes), spikes

        # - Evolve over the data array
        final_state, data_up = jax.lax.scan(forward, inital_state, data)

        return jnp.array(data_up), final_state

except:

    def _encode_spikes(
        inital_state: np.ndarray,
        dt: float,
        data: np.ndarray,
        thr_up: float,
        c_iaf: float,
        leakage: float,
    ) -> (np.ndarray, np.ndarray):
        """
        Encode a signal as events, using an LIF neuron membrane

        Args:
            inital_state (np.ndarray): Initial state of the LIF neurons
            dt (float): Time-step in seconds
            data (np.ndarray): Array ``(T,N)`` containing data to convert to events
            thr_up (float): Firing threshold voltage
            c_iaf (float): Membrane capacitance
            leakage (float): Leakage factor per time step

        Returns: np.ndarray: Raster of output events ``(T,N)``, where ``True`` indicates a spike
        """
        cdc = inital_state
        data_up = []
        data *= 1e-6  # convert voltage to current for V2I module
        for i in range(len(data)):
            lk = leakage * cdc * 1e-9
            dq_lk = lk * dt
            dv = (dt * data[i] - dq_lk) / c_iaf

            # - Accumulate membrane voltage, clip to zero
            cdc += dv
            cdc *= cdc > 0.0

            spikes = cdc > thr_up
            data_up.append(spikes)
            cdc = cdc * (1 - spikes)

        return np.array(data_up), cdc


class AFE(Module):
    """
    A :py:class:`.Module` that simulates analog hardware for preprocessing audio

    This module simulates the Xylo audio front-end stage. This is a signal-to-event core that provides a number of band-pass filters, followed by rectifying event production simulating a spiking LIF neuron. The event rate in each channel is roughly correlated to the energy in each filter band.

    Notes:
        - The AFE contains frequency tripling internally. For accurate simulation, the sampling frequency must be at least 6 times higher than the highest frequency component in the filtering chain. This would be the centre frequency of the highest filter, plus half the BW of that signal. To prevent signal aliasing, you should apply a low-pass filter to restrict the bandwidth of the input, to ensure you don't exceed this target highest frequency.

        - Input to the module is in Volts. Input amplitude should be scaled to a maximum of 112mV RMS.

    See Also:
        For example usage of the :py:class:`.AFE` Module, see :ref:`/devices/analog-frontend-example.ipynb`
    """

    def __init__(
        self,
        shape: Union[tuple, int] = (1, 16),
        Q: int = 5,  # 3-5
        fc1: float = 100.0,
        f_factor: float = 1.325,
        thr_up: float = 0.5,
        leakage: float = 1.0,  # 0.5-20 nA
        digital_counter: int = 1,  # keep 1 spike every xxx spikes
        LNA_gain: float = 0.0,  # ~ +6db  6db/steps
        fs: int = 48000,
        manual_scaling: float = None,
        add_noise: bool = True,
        seed: int = np.random.randint(2**32 - 1),
        num_workers: int = 1,
        *args,
        **kwargs,
    ):
        """

        Parameters
        ----------
        Q: int
           Quality factor (sharpness of filters). Default: 5
        fc1: float
            Center frequency of the first band-pass filter, in Hz. Default: 100Hz
        f_factor: float
            Logarithmic distribution of the center frequencies is based on ``f_factor``. Default: 1.325
        thr_up: float
            Spiking threshold for spike conversion. Default: 0.5
        leakage: float
            Leakage for spike conversion, in nA. Default: 1.0
        digital counter: int
            Digital counter for spike conversion - lets only every nth spike pass. Default: 1 (let every spike pass)
        LNA_gain: float
            Gain of the first stage low-noise amplifier, in dB. Default: 0.
        fs: int
            Sampling frequency of the input data, in Hz. Default: 16kHz. **Note that the AFE contains frequency tripling, so the maximum representable frequency is ``fs/6``.**
        shape: int
            Number of filters / output channels. Default: ``(16,)``
        manual_scaling: float
            Disables automatic scaling from the LNA and instead scales the input by this factor. Default: ``None`` (Automatic scaling)
        add_noise: bool
            Enables / disables the simulated noise generated be the AFE. Default: ``True``, include noise
        seed: int
            The AFE is subject to mismatch, this can be seeded by providing an integer seed. Default: random seed. Provide ``None`` to prevent seeding.
        """

        # - Check shape argument
        if np.size(shape) == 1:
            shape = (1, np.array(shape).item())

        # - Initialise superclass
        super().__init__(shape=shape, spiking_output=True, *args, **kwargs)

        # - Provide pRNG seed
        self.seed: P_int = SimulationParameter(seed, shape=(), init_func=lambda _: None)
        if self.seed is not None:
            np.random.seed(self.seed)

        # - Max input from microphone
        self.INPUT_AMP_MAX: P_float = Parameter(320e-3)  # 100mV
        """ float: Maximum input amplitude from the microphone in Volts (Default 320mV) """

        self.C_IAF: P_float = Parameter(5e-12)  # 2 pF
        """ float: Integrator Capacitance for IAF (Default 5e-12)"""

        # - Parameters for BPF
        self.Q: P_int = Parameter(Q, shape=())
        """ int: Quality parameter for band-pass filters"""

        # - Center frequency for 1st BPF in filter bank
        self.FC1: P_float = Parameter(fc1, shape=())
        """ float: Centre frequnecy of first filter, in Hz. """

        self.Fs: P_float = SimulationParameter(fs, shape=())
        """ float: Sample frequency of input data """

        # Frequency f_bp1 = fc1     f_bp2 = fc1*f_factor   f_bp3 = fc1*f_factor^2...
        self.f_factor: P_float = Parameter(
            f_factor, shape=()
        )  # 16 channel  100Hz - 8KHz
        """ float: Centre-frequency scale-up factor per channel.
        
            Centre freq. F1 = FC1
            Centre freq. F2 = FC1 * f_factor
            Centre freq. F3 = FC1 * f_factor**2
            ...
        """

        self.ORDER_BPF: P_int = Parameter(2)
        """ int: Band-pass filter order (Default 2)"""

        # Non-ideal
        self.MAX_INPUT_OFFSET: P_float = Parameter(0.0)  # from microphone
        """ float: Maxmimum input offset from microphone (Default 0.) """

        self.MAX_LNA_OFFSET: P_float = Parameter(5.0)  # +/-5mV random
        """ float: Maxmimum low-noise amplifier offset in mV (Default 5mV) """

        self.MAX_BPF_OFFSET: P_float = Parameter(5.0)  # +/-5mV random
        """ float: Maxmum band-pass filter offset in mV (Default 5mV)"""

        self.DISTORTION: P_float = Parameter(0.1)  # 0-1
        """ float: Distortion parameter (0..1) Default 0.1"""

        self.BPF_FC_SHIFT: P_float = Parameter(
            -5
        )  # 5 for +5%    -5 for -5%  ------- 16 channels center freq shift for same direction
        """ float: Centre frequency band-pass filter shift in % (Default -5%) """

        self.Q_MIS_MATCH: P_float = Parameter(10)  # +/-10% random
        """ float: Mismatch in Q in % (Default 10%) """

        self.FC_MIS_MATCH: P_float = Parameter(5)  # +/-5% random
        """ float: Mismatch in centre freq. in % (Default 5%)"""

        ## Threshold for delta modulation in up and down direction
        self.THR_UP: P_float = Parameter(thr_up, shape=())  # 0.1-0.8 V
        """ float: Threshold for delta modulation in V (0.1--0.8) (Default 0.5V)"""

        self.LEAKAGE: P_float = Parameter(leakage, shape=())
        """ float: Leakage for LIF neuron in nA. Default: 1nA """

        self.DIGITAL_COUNTER: P_int = Parameter(digital_counter, shape=())
        """ int: Digital counter factor to reduce output spikes by. Default 1 (no reduction) """

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
        """ float: High pass corner frequency due to AC Coupling from BPF to FWR in Hz. (Default 100Hz)"""

        # LNA
        self.lna_gain_db: P_float = Parameter(LNA_gain, shape=())  # in dB
        """ float: Low-noise amplifer gain in dB (Default 0.) """

        self.lna_offset: P_float = Parameter(
            np.random.randint(-self.MAX_LNA_OFFSET, self.MAX_LNA_OFFSET) * 0.001
        )
        """ float: Mismatch offset in low-noise amplifier """

        self.bpf_offset: P_array = Parameter(
            np.random.randint(-self.MAX_BPF_OFFSET, self.MAX_BPF_OFFSET, self.size_out)
        )
        """ float: Mismatch offset in band-pass filters """

        self.Q_mismatch: P_array = Parameter(
            np.random.randint(-self.Q_MIS_MATCH, self.Q_MIS_MATCH, self.size_out)
        )
        """ float: Mismatch in Q over band-pass filters """

        self.fc_mismatch: P_array = Parameter(
            np.random.randint(-self.FC_MIS_MATCH, self.FC_MIS_MATCH, self.size_out)
        )
        """ float: Mismatch in centre frequency for band-pass filters """

        fc1 = self.FC1 * (1 + self.BPF_FC_SHIFT / 100) * (1 + self.fc_mismatch[0] / 100)
        self.fcs: P_array = Parameter(
            [fc1]
            + [
                fc1 * (self.f_factor**i) * (1 + self.fc_mismatch[i] / 100)
                for i in np.arange(1, self.size_out)
            ]
        )
        """ np.ndarray: Centre frequency of each band-pass filter in Hz """

        # - Check the filters w.r.t the sampling frequency
        # if self.Fs < (6 * np.max(self.fcs)):
        #     raise ValueError(
        #         f"Sampling frequency must be at least 6 times highest BPF centre freq. (>{6 * np.max(self.fcs)} Hz)"
        #     )

        # Bandwidths of the filters
        self.bws: P_array = Parameter(
            [
                self.fcs[i] / (self.Q * (1 + self.Q_mismatch[i] / 100))
                for i in range(self.size_out)
            ]
        )
        """ np.ndarray: Bandwidths of each filter in Hz """

        self.manual_scaling: P_float = SimulationParameter(
            manual_scaling, shape=(), init_func=lambda s: -1
        )
        """ float: Manual scaling of low-noise amplifier gain. Default `None` (use automatic scaling) """

        self.add_noise: Union[bool, ParameterBase] = SimulationParameter(
            add_noise, shape=()
        )
        """ bool: Flag indicating that noise should be simulated during operation. Default `True` """

        # - Generate the sub-modules
        self.butter_filterbank = ButterFilter(
            frequency=self.fcs,
            bandwidth=self.bws,
            fs=fs,
            order=self.ORDER_BPF,
            num_workers=num_workers,
            use_lowpass=False,
        )

        # - High-pass filter parameters
        self._HP_filt = self._butter_highpass(self.F_CORNER_HIGHPASS, self.Fs, order=1)
        """ High-pass filter on input """

        # - Internal neuron state
        self.lif_state: Union[np.ndarray, State] = State(np.zeros(self.size_out))
        """ (np.ndarray) Internal state of the LIF neurons used to generate events """

        self._last_input = None
        """ (np.ndarray) The last chunk of input, to avoid artefacts at the beginning of an input chunk """

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
        b, a = butter(order, [low, high], btype="band", output="ba")
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

    def _generateNoise(
        self,
        T,
        Fs: float = 16e3,
        VRMS_SQHZ: float = 1e-6,
        F_KNEE: float = 1e3,
        F_ALPHA: float = 1.4,
    ) -> np.ndarray:
        """
        Generate band-limited noise, for use in simulating the AFE architecture

        Args:
            x (np.ndarray): Input signal defining desired shape of noise ``(T,)``
            Fs (float): Sampling frequency in Hz
            VRMS_SQHZ (float):
            F_KNEE (float):
            F_ALPHA (float):

        Returns: np.ndarray: Generated noise with shape ``(T,)``
        """

        def one_over_f(f: np.ndarray, knee: float, alpha: float) -> np.ndarray:
            d = np.ones_like(f)
            f = np.clip(f, 1e-12, np.inf)
            d[f < knee] = np.abs(((knee / f[f < knee]) ** (alpha)))
            d[0] = 1
            return d

        W_NOISE_SIGMA = VRMS_SQHZ * np.sqrt(Fs / 2)  # Noise in the bandwidth 0 - Fs/2

        wn = np.random.normal(0, W_NOISE_SIGMA, T)
        s = fftpack.rfft(wn)
        f = fftpack.rfftfreq(len(s)) * Fs
        ff = s * one_over_f(f, F_KNEE, F_ALPHA)
        x_t = fftpack.irfft(ff)

        return x_t

    def _sampling_signal(self, spikes: np.ndarray, count: int) -> np.ndarray:
        """
        Down-sample events in a signal, by passing one in every ``N`` events

        Args:
            spikes (np.ndarray): Raster ``(T, N)`` of events
            count (int): Number of events to ignore before passing one event

        Returns: np.ndarray: Raster ``(T, N)`` of down-sampled events
        """

        return (np.cumsum(spikes, axis=0) % count * spikes) == (count - 1)

        # sam_count = 1
        # sampled = []
        #
        # for i in range(len(ch1)):
        #     if (ch1[i] == 1) & (sam_count < count):
        #         sam_count = sam_count + 1
        #         sampled.append(0)
        #     elif (ch1[i] == 1) & (sam_count == count):
        #         sam_count = 1
        #         sampled.append(1)
        #     else:
        #         sampled.append(0)
        #
        # return sampled

    def evolve(
        self,
        input: np.ndarray = None,
        record: bool = False,
        *args,
        **kwargs,
    ):
        # - Make sure input is 1D
        if np.ndim(input) > 1:
            input = input[:, 0]

        # - Set up the previous input chunk
        if self._last_input is None:
            self._last_input = np.zeros_like(input)

        # - Augment input data to avoid artefacts, and save for next time
        input_length = input.shape[0]
        this_input = input
        input = np.concatenate((self._last_input, input))
        self._last_input = this_input

        input_offset = self.MAX_INPUT_OFFSET
        if self.manual_scaling > 0.0:
            y_scaled = self.manual_scaling * input * self.INPUT_AMP_MAX + input_offset
        else:
            y_scaled = (input / np.max(input)) * self.INPUT_AMP_MAX + input_offset

        #######   LNA - Gain  ##########
        lna_nonlinearity = self.DISTORTION / self.INPUT_AMP_MAX
        lna_distortion = (y_scaled**2) * lna_nonlinearity
        lna_gain_v = 2 ** (self.lna_gain_db / 6)
        lna_out = y_scaled * (1 + lna_distortion) * lna_gain_v + self.lna_offset

        #######  Add Noise ###############
        noise = self._generateNoise(
            input.shape[0],
            self.Fs,
            self.VRMS_SQHZ_LNA,
            self.F_KNEE_LNA,
            self.F_ALPHA_LNA,
        )

        lna_out = lna_out + noise

        # - Expand lna_output dimensions
        lna_out = np.tile(np.atleast_2d(lna_out).T, (1, self.size_out))

        # - Perform the filtering
        filtered, _, _ = self.butter_filterbank(lna_out + self.bpf_offset * 0.001)

        # bpfs = [
        #     self._butter_bandpass_filter(
        #         lna_out + self.bpf_offset[i] * 0.001,
        #         self.fcs[i] - self.bws[i] / 2,
        #         self.fcs[i] + self.bws[i] / 2,
        #         self.Fs,
        #         order=self.ORDER_BPF,
        #     )
        #     for i in range(self.size_out)
        # ]

        # add noise
        if self.add_noise:
            for i in range(self.size_out):
                filtered[:, i] += self._generateNoise(
                    input.shape[0],
                    self.Fs,
                    self.VRMS_SQHZ_BPF,
                    self.F_KNEE_BPF,
                    self.F_ALPHA_BPF,
                )

        # bpfs = [
        #     bpfs[i]
        #     + self._generateNoise(
        #         bpfs[i], self.Fs, self.VRMS_SQHZ_BPF, self.F_KNEE_BPF, self.F_ALPHA_BPF,
        #     )
        #     for i in range(self.size_out)
        # ]

        # - High-pass filter
        # rectified = signal.filtfilt(*self._HP_filt, filtered)

        # - HP filt, additional noise, rectify
        rectified = np.zeros_like(filtered)
        for i in range(self.size_out):
            rectified[:, i] = abs(
                signal.filtfilt(*self._HP_filt, filtered[:, i])
                # rectified[:, i]
                + self._generateNoise(
                    input.shape[0],
                    self.Fs,
                    self.VRMS_SQHZ_FWR,
                    self.F_KNEE_FWR,
                    self.F_ALPHA_FWR,
                )
            )

        # Encoding to spike by integrating the FWR output for positive going(UP)
        spikes, new_state = _encode_spikes(
            self.lif_state,
            1 / self.Fs,
            rectified,
            self.THR_UP,
            self.C_IAF,
            self.LEAKAGE,
        )

        # - Keep a record of the LIF neuron states
        self.lif_state = new_state

        # spikes = np.array(
        #     [
        #         self._sampling_signal(spikes[i], self.DIGITAL_COUNTER)
        #         for i in range(self.size_out)
        #     ]
        # )

        if self.DIGITAL_COUNTER > 1:
            spikes = self._sampling_signal(spikes, self.DIGITAL_COUNTER)

        # - Trim data to this chunk
        spikes = spikes[-input_length:, :]

        if record:
            # - Trim data to this chunk
            lna_out = lna_out[-input_length:, :]
            filtered = filtered[-input_length:, :]
            rectified = rectified[-input_length:, :]

            recording = {
                "LNA_out": lna_out,
                "BPF": filtered,
                "rect": rectified,
                "spks_out": spikes,
            }
        else:
            recording = {}

        return spikes, self.state(), recording

    @property
    def dt(self) -> float:
        """
        Simulation time-step in seconds

        Returns:
            float: Simulation time-step
        """
        return 1 / self.Fs

    def _wrap_recorded_state(self, state_dict: dict, t_start: float = 0.0) -> dict:
        args = {"dt": self.dt, "t_start": t_start}

        return {
            "LNA_out": TSContinuous.from_clocked(
                state_dict["LNA_out"], name="LNA", **args
            ),
            "BPF": TSContinuous.from_clocked(state_dict["BPF"], name="BPF", **args),
            "rect": TSContinuous.from_clocked(state_dict["rect"], name="Rect", **args),
            "spks_out": TSEvent.from_raster(
                state_dict["spks_out"],
                name="Spikes",
                num_channels=self.size_out,
                **args,
            ),
        }
