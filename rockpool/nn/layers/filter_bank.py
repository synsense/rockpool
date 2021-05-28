from abc import ABC, abstractmethod
from typing import Optional, Union
from itertools import product
from multiprocessing import Pool

import numpy as np
from scipy.signal import butter, sosfilt, sosfreqz

from rockpool.timeseries import TSContinuous
from rockpool.nn.layers.layer import Layer
from rockpool.nn.modules.timed_module import astimedmodule


class FilterBank(Layer, ABC):
    """
    Super-class to create a filter bank layer.
    This class actually does not build any filter, but it contains shared instances.
    """

    ## - Constructor
    def __init__(
        self,
        fs: float,
        name: str = "unnamed",
        cutoff_fs: float = 100.0,
        num_filters: int = 64,
        order: int = 2,
        mean_subtraction: bool = False,
        normalize: bool = False,
        num_workers: int = 1,
    ):
        """
        :param float fs:                input signal sampling frequency
        :param str name:                name of the layer. Default ``"unnamed``
        :param float cutoff_fs:         lowpass frequency to get only the enveloppe
                                        of filters output. Default: ``100 Hz``
        :param int num_filters:         number of filters. Default: ``64``
        :param int order:               filter order. Default: ``2``
        :param bool mean_subtraction:   subtract the mean of output signals (per channel).
                                        Default ``False``
        :param bool normalize:          divide output signals by their maximum value (i.e. filter
                                        responses in the range [-1, 1]). Default: ``False``
        :param int num_workers:         Number of CPU cores to use in simulation. Default: ``1``
        """

        self.name = name

        assert fs > 0.0 and isinstance(fs, (int, float)), (
            self.start_print + f"`fs` must be a strictly positive float (given: {fs})"
        )
        self._fs = fs

        assert (
            cutoff_fs > 0.0
            and cutoff_fs < self.nyquist
            and isinstance(cutoff_fs, (int, float))
        ), (
            self.start_print
            + f"`cutoff_fs` must be greater than 0 and lesser than `fs`/2 (given: {cutoff_fs})"
        )
        self._cutoff_fs = cutoff_fs

        assert order > 0 and isinstance(order, int), (
            self.start_print
            + f"`order` must be a strictly positive integer (given: {order})"
        )
        self._order = order

        assert num_workers > 0 and isinstance(num_workers, int), (
            self.start_print
            + f"`num_workers` must be a strictly postive integer (given: {num_workers})"
        )
        self._num_workers = num_workers

        self.mean_subtraction = mean_subtraction
        self.normalize = normalize

        super().__init__(weights=np.ones([1, num_filters]), dt=1 / self.fs, name=name)

        self.filter_lowpass = butter(
            self.order,
            self.cutoff_fs / self.nyquist,
            analog=False,
            btype="low",
            output="sos",
        )

        self.pool = None

    def terminate(self):
        """ Terminates all processes in the worker pool """

        if self.pool is not None:
            self.pool.close()

    @staticmethod
    def generate_chunks(l, n) -> list:
        """ Generates chunks of data"""

        chunks = []
        for i in range(0, len(l), n):
            if i + n > len(l):
                chunks.append(l[i:])
            else:
                chunks.append(l[i : i + n])
        return chunks

    @staticmethod
    def process_filters(args) -> list:
        """ Method for processing the filters each worker executes """

        filters, params = args
        signal, filter_lowpass = params
        filters_output = []
        for f in filters:
            sig = sosfilt(f, signal)
            sig = np.abs(sig)
            sig = sosfilt(filter_lowpass, sig)
            filters_output.append(sig)
        return filters_output

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSContinuous:
        """
        Evolve the state of the filterbanks, given an input

        :param Optional[TSContinuous] ts_input:   Raw input signal
        :param Optional[float] duration:          Duration of evolution, in s
        :param Optional[int] num_timesteps:       Number of time steps to evolve
        :param bool verbose:                      Currently unused

        :return TSContinuous:                     Output of the filterbanks
        """

        # - Prepare time base
        time_base, input_step, num_time_steps = self._prepare_input(
            ts_input=ts_input, duration=duration, num_timesteps=num_timesteps
        )

        args = list(product(self.chunks, [(input_step.T[0], self.filter_lowpass)]))

        if self.pool is None:
            self.pool = Pool(self.num_workers)

        res = self.pool.map(ButterMelFilterV1.process_filters, args)
        filtOutput = np.concatenate(res).T

        vtTimeBase = time_base[0] + np.arange(len(filtOutput)) / self.fs
        self._timestep += input_step.shape[0] - 1

        if self.normalize:
            filtOutput /= np.max(np.abs(filtOutput))

        if self.mean_subtraction:
            filtOutput -= np.mean(filtOutput)

        return TSContinuous(vtTimeBase, filtOutput, name="filteredInput")

    def reset_all(self):
        """ Override `reset_all` method """
        self.reset_time()

    @abstractmethod
    def to_dict(self) -> dict:
        """ Return the parameters of this layer as a dict, for saving """
        return {
            "fs": self.fs,
            "name": self.name,
            "cutoff_fs": self.cutoff_fs,
            "num_filters": self.num_filters,
            "order": self.order,
            "mean_subtraction": self.mean_subtraction,
            "normalize": self.normalize,
            "num_workers": self.num_workers,
            "class_name": "FilterBankBase",
        }

    def get_analytical_filter_response(self, n_pts: int) -> (np.array, np.array):
        """
        Compute the analytical response of each filter as a function of frequency (for plotting)

        :param int n_pts:           number of elements of returned arrays

        :return array freqs:        frequency array
        :return array responses:    filters responses
        """
        responses = []
        for sos in self.filters:
            w, h = sosfreqz(sos, worN=n_pts)
            responses.append(np.abs(h))
        freqs = w / np.pi * self.fs / 2

        return freqs, np.array(responses)

    @property
    def fs(self) -> float:
        """ (float) return the sampling frequency of the input signal """
        return self._fs

    @property
    def nyquist(self) -> float:
        """ (float) Nyquist frequency """
        return self.fs / 2

    @property
    def num_filters(self) -> int:
        """ (int) return the size of the filter bank """
        return self._size

    @property
    def cutoff_fs(self) -> float:
        """ lowpass filter cutoff frequency """
        return self._cutoff_fs

    @property
    def order(self) -> int:
        """ (int) filter order """
        return self._order

    @property
    def num_workers(self) -> int:
        return self._num_workers


class ButterMelFilterV1(FilterBank):
    """
    Define a Butterworth filter bank (mel spacing) filtering layer with continuous time series output
    """

    ## - Constructor
    def __init__(
        self,
        fs: float,
        name: str = "unnamed",
        cutoff_fs: float = 100.0,
        num_filters: int = 64,
        filter_width: float = 2.0,
        mean_subtraction: bool = False,
        normalize: bool = False,
        order: int = 2,
        num_workers: int = 1,
        plot: bool = False,
    ):
        """
        Layer which applies the butterworth filter in MEL scale to a one-dimensional input signal.
        Further dimensions can be passed through the layer without being filtered.

        :param float fs:                input signal sampling frequency
        :param str name:                name of the layer. Default ``"unnamed"``
        :param float cutoff_fs:         lowpass frequency to get only the enveloppe of filters output.
                                        Also the lowest frequency of the filter bank. Default: ``100 Hz``
                                        Don't set it yourself unless you know what you're doing.
        :param int num_filters:         number of filters. Default: ``64``
        :param float filter_width:      The width of the filters which is scaled with the number of filters. This
                                        determines the overlap between channels. Default: 2.
        :param int order:               filter order. Default: ``2``
        :param bool mean_subtraction:   subtract the mean of output signals (per channel).
                                        Default ``False``
        :param bool normalize:          divide output signals by their maximum value (i.e. filter
                                        responses in the range [-1, 1]). Default: ``False``
        :param int num_workers:         Number of CPU cores to use in simulation. Default: ``1``
        :param bool plot:               Plots the filter response
        """

        # - Call super constructor (`asarray` is used to strip units)
        super().__init__(
            fs=fs,
            name=name,
            cutoff_fs=cutoff_fs,
            num_filters=num_filters,
            order=order,
            mean_subtraction=mean_subtraction,
            normalize=normalize,
            num_workers=num_workers,
        )

        filter_bandwidth = filter_width / self.num_filters
        low_freq = ButterMelFilterV1.hz2mel(self.cutoff_fs)
        high_freq = ButterMelFilterV1.hz2mel(self.nyquist / (1 + filter_bandwidth) - 1)
        freqs = ButterMelFilterV1.mel2hz(
            np.linspace(low_freq, high_freq, self.num_filters)
        )

        if np.max(freqs * (1 + filter_bandwidth) / self.nyquist) >= 1.0:
            raise ValueError(
                "{} `{}`: `cutoff_fs` is too large (given: {})".format(
                    self.__class__.__name__, self.name, self.cutoff_fs
                )
            )

        freq_bands = np.array([freqs, freqs * (1 + filter_bandwidth)]) / self.nyquist
        self.filters = list(
            map(
                lambda fb: butter(
                    self.order, fb, analog=False, btype="band", output="sos"
                ),
                freq_bands.T,
            )
        )

        chunk_size = int(np.ceil(self.num_filters / num_workers))
        self.chunks = ButterMelFilterV1.generate_chunks(self.filters, chunk_size)

        if plot:
            import matplotlib.pyplot as plt
            from matplotlib import cm

            colors = cm.Blues(np.linspace(0.5, 1, len(self.filters)))
            plt.figure(figsize=(16, 10))
            for i, filt in enumerate(self.filters):
                sos_freqz = sosfreqz(filt, worN=1024)
                db = 20 * np.log10(np.maximum(np.abs(sos_freqz[1]), 1e-5))
                plt.plot(self.nyquist * sos_freqz[0] / np.pi, db, color=colors[i])

            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Gain (db)")
            plt.ylim([-10, 2])
            plt.xlim([0, self.nyquist])
            plt.tight_layout()
            plt.show(block=True)

    def hz2mel(x: Union[float, np.array]) -> Union[float, np.array]:
        """Takes value from hz and returns mel"""
        return 2595 * np.log10(1 + x / 700)

    def mel2hz(x: Union[float, np.array]) -> Union[float, np.array]:
        """
        Takes value from mel and returns hz
        """
        return 700 * (10 ** (x / 2595) - 1)

    def to_dict(self) -> dict:
        """ Return the parameters of this layer as a dict, for saving """
        config = super().to_dict()
        config["class_name"] = "ButterMelFilter"
        return config


class ButterFilterV1(FilterBank):
    """
    Define a Butterworth filter bank filtering layer with continuous time series output
    """

    ## - Constructor
    def __init__(
        self,
        fs: float,
        frequency: Union[float, np.ndarray],
        bandwidth: Union[float, np.ndarray],
        name: str = "unnamed",
        order: int = 2,
        mean_subtraction=False,
        normalize=False,
        num_workers: int = 1,
    ):
        """
        Layer which applies the butterworth filter to a one-dimensional input signal.

        :param float fs:                    input signal sampling frequency
        :param array frequency:             frequency center positions of filters
                                            (low bound: where the filter response start to be maximal)
                                            the size determines the number of filters
        :param (float, array) bandwidth:    filters response bandwidth
                                            (high bound: frequency + bandwidth)
        :param str name:                    name of the layer. Default ``"unnamed"``
        :param int order:                   filter order. Default: ``2``
        :param bool mean_subtraction:       subtract the mean of output signals (per channel).
                                            Default ``False``
        :param bool normalize:              divide output signals by their maximum absolute value.
                                            Default: ``False``
        :param int num_workers:             number of CPU cores to use in simulation. Default: ``1``
        """

        self.name = name
        self._frequency = np.array(frequency).reshape((np.size(frequency),))

        if np.size(bandwidth) == 1:
            self._bandwidth = np.ones(self.frequency.shape) * bandwidth
        else:
            self._bandwidth = np.asarray(bandwidth)

        assert np.size(self.frequency) == np.size(self.bandwidth), (
            self.start_print
            + "`bandwidth` must be either a scalar or of the same size than `frequency`"
        )

        assert (self.frequency - self.bandwidth / 2 > 0.0).any(), (
            self.start_print + "`frequency` must be greater than `bandwidth` / 2"
        )

        assert (self.frequency + self.bandwidth / 2 < fs / 2).any(), (
            self.start_print
            + "`frequency` must be lesser than (`fs` - `bandwidth`) / 2"
        )

        # idx = np.argmin(self.frequency)
        # cutoff_fs = self.frequency[idx] - self.bandwidth[idx] / 2
        cutoff_fs = 100.0

        # - Call super constructor
        super().__init__(
            fs=fs,
            name=name,
            cutoff_fs=cutoff_fs,
            num_filters=np.size(self.frequency),
            order=order,
            mean_subtraction=mean_subtraction,
            normalize=normalize,
            num_workers=num_workers,
        )

        freq_bands = (
            np.array(
                [
                    self.frequency - self.bandwidth / 2,
                    self.frequency + self.bandwidth / 2,
                ]
            )
            / self.nyquist
        )

        self.filters = list(
            map(
                lambda fb: butter(
                    self.order, fb, analog=False, btype="band", output="sos"
                ),
                freq_bands.T,
            )
        )

        chunk_size = int(np.ceil(self.num_filters / num_workers))
        self.chunks = ButterFilterV1.generate_chunks(self.filters, chunk_size)
        self.pool = None

    def to_dict(self) -> dict:
        """ Return the parameters of this layer as a dict, for saving """
        config = super().to_dict()
        config.pop("num_filters")
        config.pop("cutoff_fs")
        config["frequency"] = self.frequency.tolist()
        config["bandwidth"] = self.bandwidth.tolist()
        config["class_name"] = "ButterFilter"
        return config

    @property
    def frequency(self) -> Union[float, np.array]:
        """ return the frequency center positions of filters """
        return self._frequency

    @property
    def bandwidth(self) -> Union[float, np.array]:
        """ return filters bandwidth """
        return self._bandwidth


ButterMelFilter = astimedmodule(
    ButterMelFilterV1,
    parameters=[],
    simulation_parameters=[
        "fs",
        "cutoff_fs",
        "num_filters",
        "mean_subtraction",
        "normalize",
        "order",
        "num_workers",
        "name",
    ],
    states=[],
)

ButterFilter = astimedmodule(
    ButterFilterV1,
    parameters=[],
    simulation_parameters=[
        "fs",
        "frequency",
        "bandwidth",
        "mean_subtraction",
        "normalize",
        "order",
        "num_workers",
        "name",
    ],
    states=[],
)
