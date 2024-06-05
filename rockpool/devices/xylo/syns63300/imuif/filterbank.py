"""
Hardware butterworth filter implementation for the Xylo IMU.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from rockpool.devices.xylo.syns63300.imuif.utils import (
    type_check,
    unsigned_bit_range_check,
    signed_bit_range_check,
)

from rockpool.devices.xylo.syns63300.imuif.params import (
    NUM_BITS,
    B_WORST_CASE,
    FILTER_ORDER,
    CLOCK_RATE,
    DEFAULT_FILTER_BANDS,
)

from rockpool.nn.modules.module import Module
from scipy.signal import butter
from numpy.linalg import norm

EPS = 0.001
"""Epsilon for floating point comparison"""

__all__ = ["FilterBank", "BandPassFilter"]


@dataclass
class BandPassFilter:
    """
    Class that instantiates a single quantised band-pass filter, as implemented on Xylo IMU hardware

    This class will design the best filter that meets a pass-band specification, using the class method :py:meth:`~.BandPassFilter.from_specification`.
    Alternatively, you can specify the `a1` and `a2` taps directly as a signed integer representation, as well as specifying the number of bits needed for specifying the filter.
    """

    B_b: int = 6
    """Bits needed for scaling b0"""

    B_wf: int = 8
    """Bits needed for fractional part of the filter output"""

    B_af: int = 9
    """Bits needed for encoding the fractional parts of taps"""

    a1: int = -36565
    """Integer representation of a1 tap"""

    a2: int = 31754
    """Integer representation of a2 tap"""

    scale_out: Optional[float] = None
    """A virtual scaling factor that is applied to the output of the filter, NOT IMPLEMENTED ON HARDWARE!
    That shows the surplus scaling needed in the output (accepted range is [0.5, 1.0])"""

    def __post_init__(self) -> None:
        """
        Check the validity of the parameters.
        """
        self.B_w = NUM_BITS + B_WORST_CASE + self.B_wf
        """Total number of bits devoted to storing the values computed by the AR-filter."""

        self.B_out = NUM_BITS + B_WORST_CASE
        """Total number of bits needed for storing the values computed by the WHOLE filter."""

        if self.scale_out is not None:
            if self.scale_out < 0.5 or self.scale_out > 1.0:
                raise ValueError(
                    f"scale_out should be in the range [0.5, 1.0]. Got {self.scale_out}"
                )

        unsigned_bit_range_check(self.B_b, n_bits=4)
        unsigned_bit_range_check(self.B_wf, n_bits=4)
        unsigned_bit_range_check(self.B_af, n_bits=4)
        signed_bit_range_check(self.a1, n_bits=17)
        signed_bit_range_check(self.a2, n_bits=17)

    @type_check
    def compute_AR(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute the AR part of the filter in the block-diagram with the given parameters.

        Args:
            signal (np.ndarray): the quantized input signal in python-object integer format.

        Raises:
            OverflowError: if any overflow happens during the filter computation.

        Returns:
            np.ndarray: the output signal of the AR filter.
        """

        # check that the input is within the valid range of block-diagram

        unsigned_bit_range_check(np.max(np.abs(signal)), n_bits=NUM_BITS - 1)

        output = []

        # w[n], w[n-1], w[n-2]
        w = [0, 0, 0]

        for val in signal:
            # Computation after the clock edge
            w_new = (val << self.B_wf) + (
                (-self.a2 * w[2] - self.a1 * w[1]) >> self.B_af
            )
            w_new = w_new >> self.B_b

            w[0] = w_new

            # register shift at the rising edge of the clock
            w[1], w[2] = w[0], w[1]

            output.append(w[0])

            # check the overflow: here we have the integer version

        unsigned_bit_range_check(np.max(np.abs(output)), n_bits=self.B_w - 1)
        # convert into numpy
        return np.array(output, dtype=np.int64).astype(object)

    @type_check
    def compute_MA(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute the MA part of the filter in the block-diagram representation.

        Args:
            signal (np.ndarray): input signal (in this case output of AR part) of datatype `pyton.object`.

        Raises:
            OverflowError: if any overflow happens during the filter computation.

        Returns:
            np.ndarray: quantized filtered output signal.
        """

        # check dimension
        if signal.ndim > 1:
            raise ValueError("input signal should be 1-dim.")

        sig_out = np.copy(signal)
        sig_out[2:] = sig_out[2:] - signal[:-2]

        # apply the last B_wf bitshift to get rid of additional scaling needed to avoid dead-zone in the AR part
        sig_out = sig_out >> self.B_wf

        # check the validity of the computed output
        unsigned_bit_range_check(np.max(np.abs(sig_out)), n_bits=self.B_out - 1)
        return np.array(sig_out, dtype=np.int64).astype(object)

    @type_check
    def __call__(self, signal: np.ndarray):
        """
        Combine the filtering done in the AR and MA part of the block-diagram representation.

        Args:
            sig_in (np.ndarray): quantized input signal of python.object integer type.

        Raises:
            OverflowError: if any overflow happens during the filter computation.

        Returns:
            np.nadarray: quantized filtered output signal.
        """
        signal = self.compute_AR(signal)
        signal = self.compute_MA(signal)
        return signal

    @classmethod
    def from_specification(
        cls, low_cut_off: float, high_cut_off: float, fs: float = CLOCK_RATE
    ) -> "BandPassFilter":
        """
        Create a filter with the given upper and lower cut-off frequencies.
        Note that the hardware filter WOULD NOT BE EXACTLY THE SAME as the one specified here.
        This script finds the closest one possible

        Args:
            low_cut_off (float): The low cut-off frequency of the band-pass filter.
            high_cut_off (float): The high cut-off frequency of the band-pass filter.
            fs (float): The clock rate of the chip running the filters (in Hz). Defaults to 200.

        Returns:
            BandPassFilter: the filter with the given cut-off frequencies.
        """
        if low_cut_off >= high_cut_off:
            raise ValueError(
                f"Low cut-off frequency should be smaller than the high cut-off frequency."
            )
        elif low_cut_off <= 0:
            raise ValueError(f"Low cut-off frequency should be positive.")

        # IIR filter coefficients
        b, a = butter(
            N=FILTER_ORDER,
            Wn=(low_cut_off, high_cut_off),
            btype="bandpass",
            analog=False,
            output="ba",
            fs=fs,
        )

        # --- Sanity Check --- #

        if np.max(np.abs(b)) >= 1:
            raise ValueError(
                "all the coefficients of MA part `b` should be less than 1!"
            )

        if a[0] != 1:
            raise ValueError(
                "AR coefficients: `a` should be in standard format with a[0]=1!"
            )

        if np.max(np.abs(a)) >= 2:
            raise ValueError(
                "AR coefficients seem to be invalid: make sure that all values a[.] are in the range (-2,2)!"
            )

        b_norm = b / abs(b[0])
        b_norm_expected = np.array([1, 0, -1])

        if (
            norm(b_norm - b_norm_expected) > EPS
            and norm(b_norm + b_norm_expected) > EPS
        ):
            raise ValueError(
                "in Butterworth filters used in Xylo-IMU the normalize MA part should be of the form [1, 0, -1]!"
            )

        # compute the closest power of 2 larger that than b[0]
        B_b = int(np.log2(1 / abs(b[0])))
        B_wf = B_WORST_CASE - 1
        B_af = NUM_BITS - B_b - 1

        # quantize the a-taps of the filter
        a_taps = (2 ** (B_af + B_b) * a).astype(np.int64)
        a1 = a_taps[1]
        a2 = a_taps[2]
        scale_out = b[0] * (2**B_b)

        return cls(B_b=B_b, B_wf=B_wf, B_af=B_af, a1=a1, a2=a2, scale_out=scale_out)


class FilterBank(Module):
    """
    This class builds the block-diagram version of the filters, which is exactly as it is done in HW.

    NOTE: Here we have considered a collection of `candidate` band-pass filters that have the potential to be chosen and implemented by the algorithm team.
    Here we make sure that all those filters work properly.
    """

    def __init__(
        self,
        shape: Optional[Union[Tuple, int]] = (3, 15),
        *args: Union[List[BandPassFilter], List[Tuple[float]]],
    ) -> None:
        """Build a FilterBank simulation by specifying pass bands for individual filters

        .. Examples::

            # - Generate a default filterbank
            >>> FilterBank()

            # - Specify three filters, single input channel
            >>> Filterbank(3, (0.1, 5), (5, 10), (10, 20))

            # - Combine existing band-pass filters into a filter bank
            >>> bpf1 = BandPassFilter(...)
            >>> bpf2 = BandPassFilter(...)
            >>> bpf3 = BandPassFilter(...)
            >>> FilterBank(3, bpf1, bpf2, bpf3)

        Args:
            shape (Optional[Union[Tuple, int]], optional): The number of input and output channels. Defaults to `(3, 15)`.
            *args: A list of `BandPassFilter`s to register to the filterbank. Defaults to `None`; use a default filterbank configuration.
        """

        # - If an integer number of filters is provided, use one input channel only
        if isinstance(shape, int):
            shape = (1, shape)

        # - Check that input and output shapes match
        if shape[1] % shape[0] != 0:
            raise ValueError(
                f"The number of output channels should be a multiple of the number of input channels."
            )

        super().__init__(shape=shape, spiking_input=False, spiking_output=False)

        if not args:
            args = [
                BandPassFilter.from_specification(*band)
                for band in DEFAULT_FILTER_BANDS
            ]

        self._filters = []
        for arg in args:
            if isinstance(arg, BandPassFilter):
                self._filters.append(arg)
            else:
                raise TypeError(
                    f"Expected `BandPassFilter` or specifications, got {type(arg)} instead."
                )

        if shape[1] != len(self._filters):
            raise ValueError(
                f"The output size ({shape[1]}) must match the number of filters {len(self._filters)} to compute filtered output!"
            )

        self._channel_mapping = np.sort(
            [i % self.size_in for i in range(self.size_out)]
        )
        """ Mapping from IMU channels to filter channels. Equal number of filters per channel by default, with all filters for each input channel in order of input channel. e.g. [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2] or similar"""

    @classmethod
    def from_specification(
        cls, shape: Tuple[int] = (3, 15), *args: List[Tuple[float]]
    ) -> "FilterBank":
        """
        Create a filter bank with the given frequency bands.

        Args:
            *args (List[Tuple[float]]): A list of tuples containing the lower and upper cut-off frequencies of the filters.

        Returns:
            FilterBank: the filter bank with the given frequency bands.
        """
        if not args:
            args = DEFAULT_FILTER_BANDS

        for arg in args:
            if not isinstance(arg, tuple):
                raise TypeError(f"Expected tuple, got {type(arg)} instead.")
            elif not (len(arg) == 2 or len(arg) == 3):
                raise ValueError(
                    f"Expected tuple of length 2 or 3, got {len(arg)} instead."
                )

        filter_list = [BandPassFilter.from_specification(*band) for band in args]

        return cls(shape, *filter_list)

    @type_check
    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """
        Compute the output of all filters for an input signal.
        Combine the filtering done in the `AR` and `MA` part of the block-diagram representation.

        Args:
            input_data (np.ndarray): the quantized input signal of datatype python.object integer. (BxTxC)

        Returns:
            Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
                np.ndarray: the filtered output signal of all filters (BxTxC)
                dict: empty record dictionary.
                dict: empty state dictionary.
        """

        # -- Batch processing
        input_data, _ = self._auto_batch(input_data)
        input_data = np.array(input_data, dtype=np.int64).astype(object)

        # -- Revert and repeat the input signal in the beginning to avoid boundary effects
        __B, __T, __C = input_data.shape
        __input_data_rev = np.flip(input_data, axis=1)
        input_data = np.concatenate((__input_data_rev, input_data), axis=1)

        # -- Filter
        data_out = []

        # iterate over batches
        for signal in input_data:
            channel_out = []
            for __filter, __ch in zip(self._filters, self._channel_mapping):
                out = __filter(signal.T[__ch])
                channel_out.append(out)

            data_out.append(channel_out)

        # convert into numpy
        data_out = np.asarray(data_out, dtype=object)
        data_out = data_out.transpose(0, 2, 1)  # BxTxC

        # -- Cut the margin
        data_out = data_out[:, __T:, :]

        return data_out, {}, {}

    @property
    def B_b_list(self) -> List[int]:
        """List of B_b values of all filters"""
        return [f.B_b for f in self._filters]

    @property
    def B_wf_list(self) -> List[int]:
        """List of B_wf values of all filters"""
        return [f.B_wf for f in self._filters]

    @property
    def B_af_list(self) -> List[int]:
        """List of B_af values of all filters"""
        return [f.B_af for f in self._filters]

    @property
    def a1_list(self) -> List[int]:
        """List of a1 values of all filters"""
        return [f.a1 for f in self._filters]

    @property
    def a2_list(self) -> List[int]:
        """List of a2 values of all filters"""
        return [f.a2 for f in self._filters]
