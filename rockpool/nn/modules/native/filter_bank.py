"""
Modules implementing filter banks
"""

from typing import Union, Iterable
from itertools import product
from multiprocessing import Pool

import numpy as np
from scipy.signal import butter, sosfilt, sosfreqz

from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter

from typing import Optional
from rockpool.typehints import P_int, P_float, P_bool

__all__ = ["ButterFilter", "ButterMelFilter"]


class FilterBankBase(Module):
    """
    Super-class to create a filter bank layer.
    This class actually does not build any filter, but it contains shared instances.
    """

    ## - Constructor
    def __init__(
        self,
        shape: Union[tuple, int] = (1, 64),
        fs: float = 44100.0,
        cutoff_fs: Optional[float] = 100.0,
        order: int = 2,
        mean_subtraction: bool = False,
        normalize: bool = False,
        num_workers: int = 1,
        use_lowpass: bool = True,
        *args,
        **kwargs,
    ):
        """
        :param tuple shape:             number of filters. Default: ``64``
        :param float fs:                input signal sampling frequency
        :param float cutoff_fs:         lowpass frequency to get only the enveloppe
                                        of filters output. Default: ``100 Hz``
        :param int order:               filter order. Default: ``2``
        :param bool mean_subtraction:   subtract the mean of output signals (per channel).
                                        Default ``False``
        :param bool normalize:          divide output signals by their maximum value (i.e. filter
                                        responses in the range [-1, 1]). Default: ``False``
        :param int num_workers:         Number of CPU cores to use in simulation. Default: ``1``
        :param bool use_lowpass:        Iff ``True``, use a low-pass filter following the band-pass filters. Default: ``True``
        """

        # - Correct the shape, if passed as an integer
        if not isinstance(shape, Iterable):
            shape = (1, shape)

        if shape[0] != 1:
            raise ValueError("The input dimension (`shape[0]`) must be `1`")

        # - Initialise the superclass
        super().__init__(
            shape=shape,
            *args,
            **kwargs,
        )

        assert fs > 0.0 and isinstance(
            fs, (int, float)
        ), f"`fs` must be a strictly positive float (given: {fs})"
        self.fs: P_float = SimulationParameter(fs, shape=())
        """ (float) Input sampling frequency in Hz """

        if cutoff_fs is not None:
            assert 0.0 < cutoff_fs < self.fs / 2 and isinstance(
                cutoff_fs, (int, float)
            ), f"`cutoff_fs` must be greater than 0 and lesser than `fs`/2 (given: {cutoff_fs})"
        self.cutoff_fs: P_float = SimulationParameter(cutoff_fs, shape=())
        """ (float) Post-filtering output low-pass cutoff frequency in Hz """

        assert order > 0 and isinstance(
            order, int
        ), f"`order` must be a strictly positive integer (given: {order})"
        self.order: P_int = SimulationParameter(order, shape=())
        """ (int) Filter order """

        assert num_workers > 0 and isinstance(
            num_workers, int
        ), f"`num_workers` must be a strictly postive integer (given: {num_workers})"
        self.num_workers: P_int = SimulationParameter(num_workers, shape=())
        """ (int) Number of workers to use in filtering """

        self.mean_subtraction: P_bool = SimulationParameter(mean_subtraction, shape=())
        """ (bool) Iff ``True``, subtract the mean filter output value from each output """

        self.normalize: P_bool = SimulationParameter(normalize, shape=())
        """ (bool) Iff ``True``, collectively normalise the filter outputs [-1, 1] """

        self.use_lowpass: P_bool = SimulationParameter(use_lowpass, shape=())
        """ (bool) Iff ``True``, perform a low-pass filter after filtering """

        # - Build low-pass filter
        self._filter_lowpass = (
            butter(
                self.order,
                self.cutoff_fs / (self.fs / 2),
                analog=False,
                btype="low",
                output="sos",
            )
            if use_lowpass
            else None
        )

        # - Initialise chunks and filters
        self._chunks: list = []
        self._filters: list = []

        # - Initialise worker pool
        self._pool = Pool(self.num_workers)

    def _terminate(self):
        """Terminates all processes in the worker _pool"""

        if self._pool is not None:
            self._pool.close()

    @staticmethod
    def _generate_chunks(l, n) -> list:
        """Generates chunks of data"""

        chunks = []
        for i in range(0, len(l), n):
            if i + n > len(l):
                chunks.append(l[i:])
            else:
                chunks.append(l[i : i + n])
        return chunks

    @staticmethod
    def _process_filters(args) -> list:
        """Method for processing the filters each worker executes"""

        filters, params = args
        signal, filter_lowpass = params
        filters_output = []
        for f in filters:
            sig = sosfilt(f, signal)
            if filter_lowpass is not None:
                sig = np.abs(sig)
                sig = sosfilt(filter_lowpass, sig)
            filters_output.append(sig)
        return filters_output

    def evolve(
        self,
        input: np.ndarray,
        *args,
        **kwargs,
    ) -> (np.ndarray, dict, dict):
        """
        Evolve the state of the filterbanks, given an input

        :param np.ndarray input:   Raw input signal
        """

        # - Build arguments to map filters over input
        args = list(product(self._chunks, [(input.T[0], self._filter_lowpass)]))

        # - Map the filtering process over the worker pool
        res = self._pool.map(self._process_filters, args)

        # - Combine the results
        filtOutput = np.concatenate(res).T

        # - Normalise the filter outputs
        if self.normalize:
            filtOutput /= np.max(np.abs(filtOutput))

        # - Mean-subtract the filter outputs
        if self.mean_subtraction:
            filtOutput -= np.mean(filtOutput)

        # - Return outputs
        return filtOutput, {}, {}


class ButterMelFilter(FilterBankBase):
    """
    Define a Butterworth filter bank (mel spacing) filtering layer with continuous sampled output
    """

    ## - Constructor
    def __init__(
        self,
        shape: Union[tuple, int] = (1, 64),
        fs: float = 44100.0,
        cutoff_fs: float = 100.0,
        filter_width: float = 2.0,
        mean_subtraction: bool = False,
        normalize: bool = False,
        order: int = 2,
        num_workers: int = 1,
        plot: bool = False,
        use_lowpass: bool = True,
        *args,
        **kwargs,
    ):
        """
        Layer which applies the butterworth filter in MEL scale to a one-dimensional input signal.
        Further dimensions can be passed through the layer without being filtered.

        :param tuple shape:             Module shape ``(1, N)``
        :param float fs:                input signal sampling frequency
        :param str name:                name of the layer. Default ``"unnamed"``
        :param float cutoff_fs:         lowpass frequency to get only the enveloppe of filters output.
                                        Also the lowest frequency of the filter bank. Default: ``100 Hz``
                                        Don't set it yourself unless you know what you're doing.
        :param float filter_width:      The width of the filters which is scaled with the number of filters. This determines the overlap between channels. Default: 2.
        :param int order:               filter order. Default: ``2``
        :param bool mean_subtraction:   subtract the mean of output signals (per channel). Default ``False``
        :param bool normalize:          divide output signals by their maximum value (i.e. filter
                                        responses in the range [-1, 1]). Default: ``False``
        :param int num_workers:         Number of CPU cores to use in simulation. Default: ``1``
        :param bool use_lowpass:        Iff ``True``, return the filtered rectified smoothed signal. Default: ``True``. If ``False``, simply perform the band-pass filtering.
        :param bool plot:               Plots the filter response. Default: ``False``
        """

        # - Call super constructor (`asarray` is used to strip units)
        super().__init__(
            shape=shape,
            fs=fs,
            cutoff_fs=cutoff_fs,
            order=order,
            mean_subtraction=mean_subtraction,
            normalize=normalize,
            num_workers=num_workers,
            use_lowpass=use_lowpass,
            *args,
            **kwargs,
        )

        def hz2mel(x: Union[float, np.array]) -> Union[float, np.array]:
            """Takes value from hz and returns mel"""
            return 2595 * np.log10(1 + x / 700)

        def mel2hz(x: Union[float, np.array]) -> Union[float, np.array]:
            """
            Takes value from mel and returns hz
            """
            return 700 * (10 ** (x / 2595) - 1)

        filter_bandwidth = filter_width / self.shape[-1]
        low_freq = hz2mel(self.cutoff_fs)
        high_freq = hz2mel((self.fs / 2) / (1 + filter_bandwidth) - 1)
        freqs = mel2hz(np.linspace(low_freq, high_freq, self.shape[-1]))

        if np.max(freqs * (1 + filter_bandwidth) / (self.fs / 2)) >= 1.0:
            raise ValueError(
                "{} `{}`: `cutoff_fs` is too large (given: {})".format(
                    self.__class__.__name__, self.name, self.cutoff_fs
                )
            )

        freq_bands = np.array([freqs, freqs * (1 + filter_bandwidth)]) / (self.fs / 2)
        self._filters = list(
            map(
                lambda fb: butter(
                    self.order, fb, analog=False, btype="band", output="sos"
                ),
                freq_bands.T,
            )
        )

        # - Generate chunks
        chunk_size = int(np.ceil(self.shape[-1] / num_workers))
        self._chunks = self._generate_chunks(self._filters, chunk_size)

        if plot:
            import matplotlib.pyplot as plt
            from matplotlib import cm

            colors = cm.Blues(np.linspace(0.5, 1, len(self._filters)))
            plt.figure(figsize=(16, 10))
            for i, filt in enumerate(self._filters):
                sos_freqz = sosfreqz(filt, worN=1024)
                db = 20 * np.log10(np.maximum(np.abs(sos_freqz[1]), 1e-5))
                plt.plot((self.fs / 2) * sos_freqz[0] / np.pi, db, color=colors[i])

            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Gain (db)")
            plt.ylim([-10, 2])
            plt.xlim([0, self.fs / 2])
            plt.tight_layout()
            plt.show(block=True)


class ButterFilter(FilterBankBase):
    """
    Define a Butterworth filter bank filtering layer with continuous output
    """

    ## - Constructor
    def __init__(
        self,
        frequency: Union[float, np.ndarray],
        bandwidth: Union[float, np.ndarray],
        fs: float = 44100.0,
        order: int = 2,
        mean_subtraction: bool = False,
        normalize: bool = False,
        num_workers: int = 1,
        use_lowpass: bool = True,
        *args,
        **kwargs,
    ):
        """
        Layer which applies the butterworth filter to a one-dimensional input signal.

        :param array frequency:             frequency center positions of filters
                                            (low bound: where the filter response start to be maximal)
                                            the size determines the number of filters
        :param (float, array) bandwidth:    filters response bandwidth
                                            (high bound: frequency + bandwidth)
        :param float fs:                    input signal sampling frequency in Hz. Default: 44100.
        :param str name:                    name of the layer. Default ``"unnamed"``
        :param int order:                   filter order. Default: ``2``
        :param bool mean_subtraction:       subtract the mean of output signals (per channel).
                                            Default ``False``
        :param bool normalize:              divide output signals by their maximum absolute value.
                                            Default: ``False``
        :param int num_workers:             number of CPU cores to use in simulation. Default: ``1``
        """

        # - Check input arguments
        frequency = np.array(frequency).reshape((np.size(frequency),))

        if np.size(bandwidth) == 1:
            bandwidth = np.ones(frequency.shape) * bandwidth
        else:
            bandwidth = np.asarray(bandwidth)

        if np.size(frequency) != np.size(bandwidth):
            raise ValueError(
                f"`bandwidth` must be either a scalar or of the same size than `frequency`. Got {np.size(frequency)} and {np.size(bandwidth)}"
            )

        if np.any(frequency - bandwidth / 2 <= 0.0):
            raise ValueError("`frequency` must be greater than `bandwidth` / 2")

        if np.any(frequency + bandwidth / 2 > fs / 2):
            raise ValueError("`frequency` must be lesser than (`fs` - `bandwidth`) / 2")

        idx = np.argmin(frequency)
        cutoff_fs = frequency[idx] - bandwidth[idx] / 2

        # - Call super constructor
        super().__init__(
            shape=np.size(frequency),
            fs=fs,
            cutoff_fs=cutoff_fs,
            order=order,
            mean_subtraction=mean_subtraction,
            normalize=normalize,
            num_workers=num_workers,
            use_lowpass=use_lowpass,
            *args,
            **kwargs,
        )

        # - Add parameters
        self.frequency: P_float = SimulationParameter(frequency)
        """ (np.ndarray) Vector of centre frequencies for the filters, in Hz """

        self.bandwidth: P_float = SimulationParameter(bandwidth)
        """ (np.ndarray) Vector of bandwidths of each filter, in Hz"""

        freq_bands = np.array(
            [
                self.frequency - self.bandwidth / 2,
                self.frequency + self.bandwidth / 2,
            ]
        ) / (self.fs / 2)

        # - Build the filters
        self._filters = list(
            map(
                lambda fb: butter(
                    self.order, fb, analog=False, btype="band", output="sos"
                ),
                freq_bands.T,
            )
        )

        # - Generate chunks
        chunk_size = int(np.ceil(self.shape[-1] / num_workers))
        self._chunks = self._generate_chunks(self._filters, chunk_size)
