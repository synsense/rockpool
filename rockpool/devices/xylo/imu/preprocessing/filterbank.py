"""
Hardware butterworth filter implementation for the Xylo IMU.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from rockpool.devices.xylo.imu.preprocessing.utils import (
    type_check,
    unsigned_bit_range_check,
    signed_bit_range_check,
)
from rockpool.parameters import SimulationParameter
from rockpool.nn.modules.module import Module
from scipy.signal import butter
from numpy.linalg import norm

B_IN = 16
"""Number of input bits that can be processed with the block diagram"""

B_WORST_CASE: int = 9
"""Number of additional bits devoted to storing filter taps such that no over- and under-flow can happen"""

FILTER_ORDER = 1
"""HARD_CODED: Filter order of the Xylo-IMU filters"""

EPS = 0.001
"""Epsilon for floating point comparison"""

DEFAULT_FILTER_BANDS = [(1e-6, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 10.0)]
"""Default filter bands for the Xylo-IMU"""

__all__ = ["FilterBank", "BandPassFilter"]


@dataclass(eq=False, repr=False)
class BandPassFilter:
    """
    Class containing the parameters of the filter in state-space representation
    This is the block-diagram structure proposed for implementation.
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
        self.B_w = B_IN + B_WORST_CASE + self.B_wf
        """Total number of bits devoted to storing the values computed by the AR-filter."""

        self.B_out = B_IN + B_WORST_CASE
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

        unsigned_bit_range_check(np.max(np.abs(signal)), n_bits=B_IN - 1)

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
        cls, low_cut_off: float, high_cut_off: float, fs: float = 200
    ) -> "BandPassFilter":
        """
        Create a filter with the given upper and lower cut-off frequencies.
        Note that the hardware filter WOULD NOT BE EXACTLY the same as the one specified here.
        This script finds the closest one possible

        Args:
            low_cut_off (float): The low cut-off frequency of the band-pass filter.
            high_cut_off (float): The high cut-off frequency of the band-pass filter.
            fs (float, optional): The clock rate of the chip running the filters (in Hz). Defaults to 200.
        Raises:
            ValueError: if the low cut-off frequency is larger than the high cut-off frequency.

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
        B_af = B_IN - B_b - 1

        # quantize the a-taps of the filter
        a_taps = (2 ** (B_af + B_b) * a).astype(np.int64)
        a1 = a_taps[1]
        a2 = a_taps[2]
        scale_out = b[0] * (2**B_b)

        return cls(B_b=B_b, B_wf=B_wf, B_af=B_af, a1=a1, a2=a2, scale_out=scale_out)


class FilterBank(Module):
    """
    This class builds the block-diagram version of the filters, which is exactly as it is done in FPGA.

    NOTE: Here we have considered a collection of `candidate` band-pass filters that have the potential to be chosen and implemented by the algorithm team.
    Here we make sure that all those filters work properly.
    """

    def __init__(
        self,
        shape: Optional[Union[Tuple, int]] = (3, 15),
        filter_0: Optional[BandPassFilter] = None,
        filter_1: Optional[BandPassFilter] = None,
        filter_2: Optional[BandPassFilter] = None,
        filter_3: Optional[BandPassFilter] = None,
        filter_4: Optional[BandPassFilter] = None,
        filter_5: Optional[BandPassFilter] = None,
        filter_6: Optional[BandPassFilter] = None,
        filter_7: Optional[BandPassFilter] = None,
        filter_8: Optional[BandPassFilter] = None,
        filter_9: Optional[BandPassFilter] = None,
        filter_10: Optional[BandPassFilter] = None,
        filter_11: Optional[BandPassFilter] = None,
        filter_12: Optional[BandPassFilter] = None,
        filter_13: Optional[BandPassFilter] = None,
        filter_14: Optional[BandPassFilter] = None,
    ) -> None:
        """Object Constructor

        Args:
            shape (Optional[Union[Tuple, int]], optional): The number of input and output channels. Defaults to (3,15).
            filter_0 (Optional[BandPassFilter], optional): The zeroth filter, processes the most significant channels(ch0) input. Defaults to None.
            filter_1 (Optional[BandPassFilter], optional): The first filter, processes the most significant channels(ch0) input. Defaults to None.
            filter_2 (Optional[BandPassFilter], optional): The second filter, processes the most significant channels(ch0) input. Defaults to None.
            filter_3 (Optional[BandPassFilter], optional): The third filter, processes the most significant channels(ch0) input. Defaults to None.
            filter_4 (Optional[BandPassFilter], optional): The fourth filter, processes the most significant channels(ch0) input. Defaults to None.
            filter_5 (Optional[BandPassFilter], optional): The fifth filter, processes the second most significant channels(ch1) input. Defaults to None.
            filter_6 (Optional[BandPassFilter], optional): The sixth filter, processes the second most significant channels(ch1) input. Defaults to None.
            filter_7 (Optional[BandPassFilter], optional): The seventh filter, processes the second most significant channels(ch1) input. Defaults to None.
            filter_8 (Optional[BandPassFilter], optional): The eighth filter, processes the second most significant channels(ch1) input. Defaults to None.
            filter_9 (Optional[BandPassFilter], optional): The ninth filter, processes the second most significant channels(ch1) input. Defaults to None.
            filter_10 (Optional[BandPassFilter], optional): The tenth filter, processes the least significant channels(ch2) input. Defaults to None.
            filter_11 (Optional[BandPassFilter], optional): The eleventh filter, processes the least significant channels(ch2) input. Defaults to None.
            filter_12 (Optional[BandPassFilter], optional): The twelfth filter, processes the least significant channels(ch2) input. Defaults to None.
            filter_13 (Optional[BandPassFilter], optional): The thirteenth filter, processes the least significant channels(ch2) input. Defaults to None.
            filter_14 (Optional[BandPassFilter], optional): The fourteenth filter, processes the least significant channels(ch2) input. Defaults to None.
        """

        if shape[1] // shape[0] != shape[1] / shape[0]:
            raise ValueError(
                f"The number of output channels should be a multiple of the number of input channels."
            )

        super().__init__(shape=shape, spiking_input=False, spiking_output=False)

        # First channel filters
        if filter_0 is None:
            filter_0 = BandPassFilter.from_specification(*DEFAULT_FILTER_BANDS[0])

        if filter_1 is None:
            filter_1 = BandPassFilter.from_specification(*DEFAULT_FILTER_BANDS[1])

        if filter_2 is None:
            filter_2 = BandPassFilter.from_specification(*DEFAULT_FILTER_BANDS[2])

        if filter_3 is None:
            filter_3 = BandPassFilter.from_specification(*DEFAULT_FILTER_BANDS[3])

        if filter_4 is None:
            filter_4 = BandPassFilter.from_specification(*DEFAULT_FILTER_BANDS[4])

        # Second channel filters
        if filter_5 is None:
            filter_5 = BandPassFilter.from_specification(*DEFAULT_FILTER_BANDS[0])

        if filter_6 is None:
            filter_6 = BandPassFilter.from_specification(*DEFAULT_FILTER_BANDS[1])

        if filter_7 is None:
            filter_7 = BandPassFilter.from_specification(*DEFAULT_FILTER_BANDS[2])

        if filter_8 is None:
            filter_8 = BandPassFilter.from_specification(*DEFAULT_FILTER_BANDS[3])

        if filter_9 is None:
            filter_9 = BandPassFilter.from_specification(*DEFAULT_FILTER_BANDS[4])

        # Third channel filters
        if filter_10 is None:
            filter_10 = BandPassFilter.from_specification(*DEFAULT_FILTER_BANDS[0])

        if filter_11 is None:
            filter_11 = BandPassFilter.from_specification(*DEFAULT_FILTER_BANDS[1])

        if filter_12 is None:
            filter_12 = BandPassFilter.from_specification(*DEFAULT_FILTER_BANDS[2])

        if filter_13 is None:
            filter_13 = BandPassFilter.from_specification(*DEFAULT_FILTER_BANDS[3])

        if filter_14 is None:
            filter_14 = BandPassFilter.from_specification(*DEFAULT_FILTER_BANDS[4])

        self.filter_list = [
            filter_0,
            filter_1,
            filter_2,
            filter_3,
            filter_4,
            filter_5,
            filter_6,
            filter_7,
            filter_8,
            filter_9,
            filter_10,
            filter_11,
            filter_12,
            filter_13,
            filter_14,
        ]

        if shape[1] != len(self.filter_list):
            raise ValueError(
                f"The output size should be {len(self.filter_list)} to compute filtered output!"
            )

        self.channel_mapping = np.sort([i % self.size_in for i in range(self.size_out)])
        """Mapping from IMU channels to filter channels. [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2] by default"""

    @classmethod
    def from_specification(
        self,
        band_0: Tuple[float] = DEFAULT_FILTER_BANDS[0],
        band_1: Tuple[float] = DEFAULT_FILTER_BANDS[1],
        band_2: Tuple[float] = DEFAULT_FILTER_BANDS[2],
        band_3: Tuple[float] = DEFAULT_FILTER_BANDS[3],
        band_4: Tuple[float] = DEFAULT_FILTER_BANDS[4],
        band_5: Tuple[float] = DEFAULT_FILTER_BANDS[0],
        band_6: Tuple[float] = DEFAULT_FILTER_BANDS[1],
        band_7: Tuple[float] = DEFAULT_FILTER_BANDS[2],
        band_8: Tuple[float] = DEFAULT_FILTER_BANDS[3],
        band_9: Tuple[float] = DEFAULT_FILTER_BANDS[4],
        band_10: Tuple[float] = DEFAULT_FILTER_BANDS[0],
        band_11: Tuple[float] = DEFAULT_FILTER_BANDS[1],
        band_12: Tuple[float] = DEFAULT_FILTER_BANDS[2],
        band_13: Tuple[float] = DEFAULT_FILTER_BANDS[3],
        band_14: Tuple[float] = DEFAULT_FILTER_BANDS[4],
    ) -> "FilterBank":
        """
        Create a filter bank with the given frequency bands.

        Args:
            band_0 (Tuple[float], optional): The frequency band of the zeroth filter. Defaults to DEFAULT_FILTER_BANDS[0].
            band_1 (Tuple[float], optional): The frequency band of the first filter. Defaults to DEFAULT_FILTER_BANDS[1].
            band_2 (Tuple[float], optional): The frequency band of the second filter. Defaults to DEFAULT_FILTER_BANDS[2].
            band_3 (Tuple[float], optional): The frequency band of the third filter. Defaults to DEFAULT_FILTER_BANDS[3].
            band_4 (Tuple[float], optional): The frequency band of the fourth filter. Defaults to DEFAULT_FILTER_BANDS[4].
            band_5 (Tuple[float], optional): The frequency band of the fifth filter. Defaults to DEFAULT_FILTER_BANDS[0].
            band_6 (Tuple[float], optional): The frequency band of the sixth filter. Defaults to DEFAULT_FILTER_BANDS[1].
            band_7 (Tuple[float], optional): The frequency band of the seventh filter. Defaults to DEFAULT_FILTER_BANDS[2].
            band_8 (Tuple[float], optional): The frequency band of the eighth filter. Defaults to DEFAULT_FILTER_BANDS[3].
            band_9 (Tuple[float], optional): The frequency band of the ninth filter. Defaults to DEFAULT_FILTER_BANDS[4].
            band_10 (Tuple[float], optional): The frequency band of the tenth filter. Defaults to DEFAULT_FILTER_BANDS[0].
            band_11 (Tuple[float], optional): The frequency band of the eleventh filter. Defaults to DEFAULT_FILTER_BANDS[1].
            band_12 (Tuple[float], optional): The frequency band of the twelfth filter. Defaults to DEFAULT_FILTER_BANDS[2].
            band_13 (Tuple[float], optional): The frequency band of the thirteenth filter. Defaults to DEFAULT_FILTER_BANDS[3].
            band_14 (Tuple[float], optional): The frequency band of the fourteenth filter. Defaults to DEFAULT_FILTER_BANDS[4].

        Returns:
            FilterBank: the filter bank with the given frequency bands.
        """

        return FilterBank(
            shape=(3, 15),
            filter_0=BandPassFilter.from_specification(*band_0),
            filter_1=BandPassFilter.from_specification(*band_1),
            filter_2=BandPassFilter.from_specification(*band_2),
            filter_3=BandPassFilter.from_specification(*band_3),
            filter_4=BandPassFilter.from_specification(*band_4),
            filter_5=BandPassFilter.from_specification(*band_5),
            filter_6=BandPassFilter.from_specification(*band_6),
            filter_7=BandPassFilter.from_specification(*band_7),
            filter_8=BandPassFilter.from_specification(*band_8),
            filter_9=BandPassFilter.from_specification(*band_9),
            filter_10=BandPassFilter.from_specification(*band_10),
            filter_11=BandPassFilter.from_specification(*band_11),
            filter_12=BandPassFilter.from_specification(*band_12),
            filter_13=BandPassFilter.from_specification(*band_13),
            filter_14=BandPassFilter.from_specification(*band_14),
        )

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

        # -- Filter
        data_out = []

        # iterate over batch
        for signal in input_data:
            channel_out = []
            for __filter, __ch in zip(self.filter_list, self.channel_mapping):
                out = __filter(signal.T[__ch])
                channel_out.append(out)

            data_out.append(channel_out)

        # convert into numpy
        data_out = np.asarray(data_out, dtype=object)
        data_out = data_out.transpose(0, 2, 1)  # BxTxC

        return data_out, {}, {}
