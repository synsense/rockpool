from rockpool.nn.modules.timed_module import TimedModule
from rockpool.timeseries import TSEvent
from rockpool.parameters import Parameter, State, SimulationParameter
import numpy as np
from scipy.signal import butter, lfilter, freqz
from scipy import signal, fftpack
import random
from scipy.interpolate import interp1d
from random import randint
from matplotlib import mlab



class AFE(TimedModule):
    def __init__(self,
                 Q: int = 5,# 3-5
                 fc1: float = 100.,
                 f_factor: float = 1.325,
                 thr_up: float = 0.5,
                 leakage: float = 1.0,# 0.5-20 nA
                 digital_counter = 1, # keep 1 spike every xxx spikes
                 LNA_gain: float = 0.0,# ~ +6db  6db/steps
                 fs: int = 16000,
                 num_filters: int = 16,
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
        digital counter: float
            Digital counter for spike converison - lets only every nth spike pass 
        LNA_gain: float
            Gain of the low-noise amplification
        fs: int
            Sampling frequency of the input data
        num_filters: int
            Number of filters
        manual_scaling: float
            Disables automatical scaling from the LNA and instead scales the input by this factor
        add_noise: bool
            Enables / disables the simulated noise generated be the AFE.
        seed: bool
            The AFE is subject to mismatch, this can be seeded.      
        """

        super().__init__(dt=1/fs)

        self.seed = seed
        if not self.seed is None:
            self.rng_state = np.random.get_state()
            np.random.seed(self.seed)

        # Max input from microphone
        self.INPUT_AMP_MAX = Parameter(320e-3)   # 100mV
        
        # Integrator Capacitance for IAF
        self.C_IAF = Parameter(5e-12)            # 2 pF
        
        # Parameters for BPF
        self.Q = Parameter(Q)
        
        # Center frequency for 1st BPF in filter bank
        self.FC1 = Parameter(fc1)

        self.Fs = SimulationParameter(fs)
        self.num_filters = SimulationParameter(num_filters)
        
        # Frequency f_bp1 = fc1     f_bp2 = fc1*f_factor   f_bp3 = fc1*f_factor^2...
        self.f_factor =  Parameter(f_factor) # 16 channel  100Hz - 8KHz
        self.ORDER_BPF = Parameter(2)
        
        # Non-ideal
        self.MAX_INPUT_OFFSET = Parameter(0)     # form microphone
        self.MAX_LNA_OFFSET = Parameter(5)      # +/-5mV random
        self.MAX_BPF_OFFSET = Parameter(5)       # +/-5mV random
        self.DISTORTION = Parameter(0.1)         # 0-1
        self.BPF_FC_SHIF = Parameter(-5)         # 5 for +5%    -5 for -5%  ------- 16 channels center freq shift for same direction
        self.Q_MIS_MATCH = Parameter(10)         # +/-10% random
        self.FC_MIS_MATCH = Parameter(5)         # +/-5% random

        ## Threshold for delta modulation in up and down direction
        self.THR_UP = Parameter(thr_up)             # 0.1-0.8 V
        self.LEAKAGE = Parameter(leakage)
        self.DIGITAL_COUNTER = Parameter(digital_counter)

        ########### Macro defnitions related to noise ###
        self.VRMS_SQHZ_LNA = Parameter(70e-9)
        self.F_KNEE_LNA = Parameter(70e3)
        self.F_ALPHA_LNA = Parameter(1)

        self.VRMS_SQHZ_BPF = Parameter(1e-9)
        self.F_KNEE_BPF = Parameter(100e3)
        self.F_ALPHA_BPF = Parameter(1)

        self.VRMS_SQHZ_FWR = Parameter(700e-9)
        self.F_KNEE_FWR = Parameter(158)
        self.F_ALPHA_FWR = Parameter(1)

        self.F_CORNER_HIGHPASS = Parameter(100)    # High pass corner frequency due to AC Coupling from BPF to FWR


        # LNA
        self.lna_gain_db = Parameter(LNA_gain)   # in dB
        self.lna_gain_v = Parameter(2 ** (self.lna_gain_db / 6))
        self.lna_offset = Parameter(np.random.randint(-self.MAX_LNA_OFFSET,self.MAX_LNA_OFFSET) * 0.001)
        self.lna_nonlinearity = Parameter(self.DISTORTION/self.INPUT_AMP_MAX)

        self.bpf_offset = State(np.random.randint(-self.MAX_BPF_OFFSET,self.MAX_BPF_OFFSET, self.num_filters))
        self.Q_mismatch = State(np.random.randint(-self.Q_MIS_MATCH,self.Q_MIS_MATCH, self.num_filters))
        self.fc_mismatch = State(np.random.randint(-self.FC_MIS_MATCH,self.FC_MIS_MATCH, self.num_filters))

        fc1 = self.FC1 * (1 + self.BPF_FC_SHIF/100) * (1 + self.fc_mismatch[0]/100)
        self.fcs = State([fc1] + [fc1 * (self.f_factor ** i) * (1 + self.fc_mismatch[i]/100) for i in np.arange(1, self.num_filters)])
 
        # Add the non-ideality of the filter - shift in the centre frequency of the BPF by mismatch
        self.bws = State([self.fcs[i] / (self.Q * (1 + self.Q_mismatch[i] / 100)) for i in range(self.num_filters)])
       
        self.recordings = {"LNA_out": [], "BPF": [], "rect": [], "spks_out": []}

        self.manual_scaling = manual_scaling
        self.add_noise = add_noise

        if not self.seed is None:
            np.random.set_state(self.rng_state)

        

    def butter_bandpass(self, lowcut, highcut, fs, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=2):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def butter_highpass(self, cutoff, fs, order=1):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a
    
    def butter_highpass_filter(self, data, cutoff, fs, order=1):
        b, a = self.butter_highpass(cutoff, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y

    #################  Function to Generate Noise  ########################
    def generateNoise(self, 
                      x, 
                      Fs = 16e3, 
                      VRMS_SQHZ = 1e-6, 
                      F_KNEE = 1e3,
                      F_ALPHA = 1.4, 
                      PLOT_SPECTRUM = False):

        if not self.add_noise:
            return np.zeros_like(x) 
        
        def one_over_f(f, knee, alpha):
            d        = np.ones_like(f)
            d[f<knee]= np.abs(((knee/f[f<knee])**(alpha)))
            d[0]     = 1
            return d
        
        W_NOISE_SIGMA = VRMS_SQHZ*np.sqrt(Fs/2) # Noise in the bandwidth 0 - Fs/2 
        
        N   = len(x)
        wn  = np.random.normal(0,W_NOISE_SIGMA,N)
        Ts  = 1/Fs
        t   = np.arange(0,len(x)*Ts,Ts)
        s   = fftpack.rfft(wn)
        f   = fftpack.rfftfreq(len(s)) * Fs
        ff  = s * one_over_f(f, F_KNEE, F_ALPHA)
        x_t = fftpack.irfft(ff)
        s,f = mlab.psd(x_t, NFFT=N, Fs=Fs, scale_by_freq=True);
        
        if(PLOT_SPECTRUM == True):
            plt.figure()
            plt.loglog(f, np.sqrt(s),'b-', label='simulated noise')
            plt.loglog(f, one_over_f(f, F_KNEE, F_ALPHA) * W_NOISE_SIGMA /np.sqrt(Fs/2), 'r',label='noise asymptote')
            plt.vlines(F_KNEE,*plt.ylim())
            plt.grid()
            plt.xlabel('Frequency')
            plt.title('Amplitude spectrum of noise generated')
            plt.legend()
        
            plt.figure()
            plt.plot(t,x_t,'b-')
            plt.xlabel('Time (s)')
            plt.ylabel('Volts(V)')
            plt.grid()
            plt.title('Noise voltage in time domain')
            
        return(x_t)


       
    def encode_spikes(self, t, filter_out,thr_up,c_iaf,leakage):
        cdc = filter_out[0]
        td  = t[0]
        data_up  = []
        data = filter_out*1e-6     # convert voltage to current for V2I module
        for i in range(len(data)):
            dt = t[i]-td
            lk = leakage * cdc * 1e-9
            dq_lk = (lk * dt)
            dv = (dt * data[i] - dq_lk)/c_iaf
            cdc = cdc + dv if (cdc + dv) >= 0 else 0
            if (cdc < thr_up):
                data_up.append(0)
                td = t[i]
            elif (cdc > thr_up):
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
            if (ch1[i] == 1) & (sam_count<count):
                sam_count = sam_count + 1
                sampled.append(0)
            elif (ch1[i] == 1) & (sam_count == count):
                sam_count = 1  
                sampled.append(1)
            else:
                sampled.append(0)
                
        return sampled

    def evolve(self,
               ts_input=None,
               duration=None,
               num_timesteps=None,
               kwargs_timeseries=None,
               record: bool = False,
               *args,
               **kwargs,):


        t = np.arange(0, num_timesteps) * self.dt
        y = ts_input(t)[:, 0] 

        input_offset = self.MAX_INPUT_OFFSET
        if self.manual_scaling:
            y_scaled = self.manual_scaling * y * self.INPUT_AMP_MAX + input_offset 
        else:
            y_scaled = (y / max(y))*self.INPUT_AMP_MAX + input_offset
        
        #######   LNA - Gain  ########## 
        lna_distortion = (y_scaled **2) * self.lna_nonlinearity
        lna_out = y_scaled * (1+lna_distortion) * self.lna_gain_v + self.lna_offset

        #######  Add Noise ###############
        lna_out = lna_out + self.generateNoise(lna_out,
                                               self.Fs,
                                               self.VRMS_SQHZ_LNA,
                                               self.F_KNEE_LNA,
                                               self.F_ALPHA_LNA,
                                               False)

        bpfs = [self.butter_bandpass_filter(lna_out + self.bpf_offset[i] * 0.001, 
                                            self.fcs[i] - self.bws[i] / 2,
                                            self.fcs[i] + self.bws[i] / 2,
                                            self.Fs,
                                            order=self.ORDER_BPF) for i in range(self.num_filters)]

        # add noise
        bpfs = [bpfs[i] + self.generateNoise(bpfs[i],
                                             self.Fs,
                                             self.VRMS_SQHZ_BPF,
                                             self.F_KNEE_BPF,
                                             self.F_ALPHA_BPF,
                                             False) for i in range(self.num_filters)]
        

        rcs = [abs(self.butter_highpass_filter(bpfs[i],
                                               self.F_CORNER_HIGHPASS,
                                               self.Fs,
                                               order=1) + self.generateNoise(bpfs[i],
                                                                             self.Fs,
                                                                             self.VRMS_SQHZ_FWR,
                                                                             self.F_KNEE_FWR,
                                                                             self.F_ALPHA_FWR,
                                                                             False)) for i in range(self.num_filters)]
        
        # Encoding to spike by integrating the FWR output for positive going(UP)
        spikes = [self.encode_spikes(t, 
                                     rcs[i],
                                     self.THR_UP,
                                     self.C_IAF,
                                     self.LEAKAGE) for i in range(self.num_filters)]


        spikes = np.array([self.sampling_signal(spikes[i],
                                                self.DIGITAL_COUNTER) for i in range(self.num_filters)])


        if record:
            self.recordings["LNA_out"] = lna_out
            self.recordings["BPF"] = bpfs
            self.recordings["rect"] = rcs
            self.recordings["spks_out"] = spikes
        else:
            self.recordings["LNA_out"] = [] 
            self.recordings["BPF"] = [] 
            self.recordings["rect"] = [] 
            self.recordings["spks_out"] = [] 
        
        spikes = TSEvent.from_raster(spikes.T, dt=self.dt)
        
        return spikes, self.state(), self.recordings 



