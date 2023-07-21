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
        return np.asarray(output, dtype=object)

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
        return sig_out

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
        B_b_list: Union[List[int], int] = 6,
        B_wf_list: Union[List[int], int] = 8,
        B_af_list: Union[List[int], int] = 9,
        a1_list: Optional[Union[List[int], int]] = None,
        a2_list: Optional[Union[List[int], int]] = None,
    ) -> None:
        """Object Constructor

        Args:
            shape (Optional[Union[Tuple, int]], optional): The number of input and output channels. Defaults to (3,15).
            B_b_list (Union[List[int], int], optional):  Maximum number of right bit shifts in the feedback loops of the filters. Defaults to [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6].
            B_wf_list (Union[List[int], int], optional): Maximum number of left bit shifts in the input for avoiding dead zones of the filters. Defaults to [8,8,8,8,8,8,8,8,8,8,8,8,8,8,8].
            B_af_list (Union[List[int], int], optional): Maximum number of right bit shifts in computing the AR part multiplication of the filters. Defaults to [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9].
            a1_list (Union[List[int], int], optional): a1 tap parameters of each filter. Defaults to [-64700,-64458,-64330,-64138,-63884,-63566,-63185,-62743,-62238,-61672,-61045,-60357,-59611,-58805,-57941].
            a2_list (Union[List[int], int], optional): a2 tap parameters of each filter (repeats if int). Defaults to [31935,31754,31754,31754,31754,31754,31754,31754,31754,31754,31754,31754,31754,31754,31754].
        """

        if shape[1] // shape[0] != shape[1] / shape[0]:
            raise ValueError(
                f"The number of output channels should be a multiple of the number of input channels."
            )

        super().__init__(shape=shape, spiking_input=False, spiking_output=False)

        if a1_list is None:
            a1_list = [
                64700,
                64458,
                64330,
                64138,
                63884,
                63566,
                63185,
                62743,
                62238,
                61672,
                61045,
                60357,
                59611,
                58805,
                57941,
            ]

        if a2_list is None:
            a2_list = [
                31935,
                31754,
                31754,
                31754,
                31754,
                31754,
                31754,
                31754,
                31754,
                31754,
                31754,
                31754,
                31754,
                31754,
                31754,
            ]

        def __make_list(val: Union[List[int], int]) -> List[int]:
            if isinstance(val, int):
                return [val] * shape[1]
            else:
                return val

        self.B_b_list = SimulationParameter(__make_list(B_b_list), shape=self.size_out)
        """Bits needed for scaling b0 values of each filter"""
        self.B_wf_list = SimulationParameter(
            __make_list(B_wf_list), shape=self.size_out
        )
        """Bits needed for fractional part of the filter output of each filter"""
        self.B_af_list = SimulationParameter(
            __make_list(B_af_list), shape=self.size_out
        )
        """Bits needed for encoding the fractional parts of taps of each filter"""
        self.a1_list = SimulationParameter(__make_list(a1_list), shape=self.size_out)
        """a1 tap parameters of each filter"""
        self.a2_list = SimulationParameter(__make_list(a2_list), shape=self.size_out)
        """a2 tap parameters of each filter"""

        self.filter_list = []
        for B_b, B_wf, B_af, a1, a2 in zip(
            self.B_b_list, self.B_wf_list, self.B_af_list, self.a1_list, self.a2_list
        ):
            self.filter_list.append(
                BandPassFilter(B_b=B_b, B_wf=B_wf, B_af=B_af, a1=a1, a2=a2)
            )

        if shape[1] != len(self.filter_list):
            raise ValueError(
                f"The output size should be {len(self.filter_list)} to compute filtered output!"
            )

        self.channel_mapping = np.sort([i % self.size_in for i in range(self.size_out)])
        """Mapping from IMU channels to filter channels. [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2] by default"""

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
