import json
from itertools import product
import numpy as np
from rockpool.timeseries import TSContinuous, TSEvent
from rockpool.layers import Layer
from typing import Optional, Union, Tuple, List
from scipy.signal import butter, sosfilt
from multiprocessing import Pool


class ButterMelFilter(Layer):
    """
    Define a Butterworth filter bank (mel spacing) filtering layer with continuous time series output
    """

    ## - Constructor
    def __init__(
        self,
        fs: float,
        dt: float = None,
        name: str = "unnamed",
        cutoff_fs: float = 100,
        num_filters: int = 64,
        mean_subtraction = False,
        normalize = False,
        order:int = 2,
        num_workers: int = 1,
    ):
        """
        Layer which applies the butterworth filter in MEL scale to a one-dimensional input signal. Further dimensions can be passed through the layer without being filtered.

        :param fs:              float sampling frequency of input signal
        :param dt:             float Time-step. Default: 0.1 ms
        :param name:         str Name for the layer. Default: 'unnamed'
        :param cutoff_fs:        float frequency to lowpass the output of the filters. Default: 100 Hz
        :param num_filters:            int number of filters Default: 64
        :param mean_subtraction: bool True in fft filterbank subtract the mean per channel False otherwise
        :param order: int order of the butterworth filter. Default: 2
        :param num_workers: int number of workers for running the filters. Default: 1
        """

        # - Call super constructor (`asarray` is used to strip units)
        super().__init__(
            weights=np.ones([1, num_filters]),
            dt=np.asarray(dt),
            name=name,
        )

        self.fs = fs
        self._timestep = 0
        self.cutoff_fs = cutoff_fs
        self.mean_subtraction = mean_subtraction
        self.num_filters = num_filters
        self.normalize = normalize
        self.order = order
        self.num_workers = num_workers

        if dt == None:
            self.dt = 1 / fs
        else:
            self.dt = dt

        filter_bandwidth = 5 / self.num_filters
        nyquist = self.fs / 2
        self.downsample = int(self.fs / self.cutoff_fs)

        low_freq = ButterMelFilter.hz2mel(self.cutoff_fs)
        high_freq = ButterMelFilter.hz2mel(self.fs / 2 / (1 + filter_bandwidth) - 1)
        freqs = ButterMelFilter.mel2hz(np.linspace(low_freq, high_freq, self.num_filters))

        freq_bands = np.array([freqs, freqs * (1 + filter_bandwidth)]) / nyquist
        filters = list(map(lambda fb: butter(self.order, fb, analog=False, btype="band", output="sos"), freq_bands.T))
        self.filter_lowpass = butter(3, self.cutoff_fs / nyquist, analog=False, btype="low", output="sos")

        chunk_size = int(np.ceil(self.num_filters / num_workers))
        self.chunks = ButterMelFilter.generate_chunks(filters, chunk_size)

        self.pool = None

    def reset_all(self):
        self._timestep = 0

    def hz2mel(x: float):
        """
        Takes value from hz and returns mel
        """
        return 2595 * np.log10(1 + x / 700)

    def terminate(self):
        if not self.pool == None:
            self.pool.close()

    def mel2hz(x: float):
        """
        Takes value from mel and returns hz
        """
        return 700 * (10 ** (x / 2595) - 1)

    def generate_chunks(l, n):
        chunks = []
        for i in range(0, len(l), n):
            if i + n > len(l):
                chunks.append(l[i:])
            else:
                chunks.append(l[i : i + n])
        return chunks

    def process_filters(args):
        filters, params = args
        signal, filter_lowpass, downsample = params
        filters_output = []
        for f in filters:
            sig = sosfilt(f, signal)
            sig = np.abs(sig)
            sig = sosfilt(filter_lowpass, sig)[::downsample]
            filters_output.append(sig)
        return filters_output

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False
    ) ->TSContinuous:

        # - Prepare time base
        time_base, input_step, num_time_steps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        args = list(product(self.chunks, [(input_step.T[0], self.filter_lowpass, self.downsample)]))

        if self.pool == None:
            self.pool = Pool(self.num_workers)

        res = self.pool.map(ButterMelFilter.process_filters, args)

        filtOutput = np.concatenate(res).T

        vtTimeBase = time_base[0] + np.arange(len(filtOutput)) / self.cutoff_fs
        self._timestep += input_step.shape[0] - 1

        if self.normalize:
            filtOutput /= np.max(np.abs(filtOutput))

        if self.mean_subtraction:
            filtOutput -= np.mean(filtOutput)



        return TSContinuous(
            vtTimeBase,
            filtOutput,
            name="filteredInput")


    def to_dict(self):

        config = {}
        config["fs"] = self.fs if type(self.fs) in (float, int) else self.fs.tolist()
        config["dt"] = self.dt if type(self.dt) is float else self.dt.tolist()
        config["name"] = self.name
        config['normalize'] = self.normalize
        config['num_filters'] = self.num_filters
        config['cutoff_fs'] = self.cutoff_fs
        config['mean_subtraction'] = self.mean_subtraction
        config['order'] = self.order
        config['num_workers'] = self.num_workers
        config["class_name"] = "ButterMelFilter"

        return config

    @staticmethod
    def load_from_dict(config):

        return ButterMelFilter(
            dt=config["dt"],
            fs=config["fs"],
            name=config["name"],
            normalize=config["normalize"],
            num_filters=config["num_filters"],
            cutoff_fs=config["cutoff_fs"],
            mean_subtraction=config["mean_subtraction"],
            order=config["order"],
            num_workers=config["num_workers"],
        )

    @staticmethod
    def load_from_file(filename):
        with open(filename, "r") as f:
            config = json.load(f)

        return ButterMelFilter(
            dt=config["dt"],
            fs=config["fs"],
            name=config["name"],
            normalize=config["normalize"],
            num_filters=config["num_filters"],
            cutoff_fs=config["cutoff_fs"],
            mean_subtraction=config["mean_subtraction"],
            order=config["order"],
            num_workers=config["num_workers"],
        )
