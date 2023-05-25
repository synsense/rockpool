"""
Hardware butterworth filter implementation for the Xylo IMU.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from rockpool.devices.xylo.imu.preprocessing.utils import type_check
from rockpool.nn.modules.module import Module

__all__ = ["FilterBank", "BandPassFilter"]


@dataclass(eq=False, repr=False)
class BandPassFilter:
    """
    Class containing the parameters of the filter in state-space representation
    This is the block-diagram structure proposed for implementation.
    """

    a1: int
    """Integer representation of a1 tap"""

    a2: int = 31754
    """Integer representation of a2 tap"""

    B_worst_case: int = 5
    """Number of additional bits devoted to storing filter taps such that no over- and under-flow can happen"""

    B_in: int = 16
    """Number of input bits that can be processed with the block diagram"""

    B_b: int = 6
    """Bits needed for scaling b0"""

    B_a: int = 17
    """Total number of bits devoted to storing filter a-taps"""

    B_af: int = 9
    """Bits needed for encoding the fractional parts of taps"""

    B_wf: int = 8
    """Bits needed for fractional part of the filter output"""

    B_w: Optional[int] = None
    """Total number of bits devoted to storing the values computed by the AR-filter. It should be equal to `B_in + B_worst_case + B_wf`"""

    B_out: Optional[int] = None
    """Total number of bits needed for storing the values computed by the WHOLE filter."""

    b: list = field(default_factory=lambda: [1, 0, -1])
    """Special case for normalized Butterworth filters"""

    scale_out: int = 0.9898
    """Surplus scaling due to `b` normalization surplus scaling due to `b` normalization. It is always in the range [0.5, 1.0]"""

    def __post_init__(self) -> None:
        """
        Fill `None` values with the correct values and check the validity of the parameters.
        """
        if self.B_w is None:
            self.B_w = self.B_in + self.B_worst_case + self.B_wf
        elif self.B_w != self.B_in + self.B_worst_case + self.B_wf:
            raise ValueError("`B_w` should be equal to `B_in + B_worst_case + B_wf`")

        if self.B_out is None:
            self.B_out = self.B_in + self.B_worst_case
        elif self.B_out != self.B_in + self.B_worst_case:
            raise ValueError("`B_out` should be equal to `B_in + B_worst_case`")

        if self.scale_out < 0.5 or self.scale_out > 1.0:
            raise ValueError(
                f"output surplus scale should be in the range [0.5, 1.0]. Got {self.scale_out}."
            )

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

        if np.max(np.abs(signal)) >= 2 ** (self.B_in - 1):
            raise ValueError(
                f"The input signal values can be in the range [-2^{self.B_in-1}, +2^{self.B_in-1}]!"
            )

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

        if np.max(np.abs(output)) >= 2 ** (self.B_w - 1):
            raise ValueError(
                f"output signal is beyond the valid output range of AR branch [-2^{self.B_w-1}, +2^{self.B_w-1}]!"
            )

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

        sig_out = self.b[0] * signal
        sig_out[2:] = sig_out[2:] + self.b[2] * signal[:-2]

        # apply the last B_wf bitshift to get rid of additional scaling needed to avoid dead-zone in the AR part
        sig_out = sig_out >> self.B_wf

        # check the validity of the computed output
        if np.max(np.abs(sig_out)) >= 2 ** (self.B_out - 1):
            raise OverflowError(
                f"overflow or underflow: computed filter output is beyond the valid range [-2^{self.B_out-1}, +2^{self.B_out-1}]!"
            )

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


class FilterBank(Module):
    """
    This class builds the block-diagram version of the filters, which is exactly as it is done in FPGA.

    NOTE: Here we have considered a collection of `candidate` band-pass filters that have the potential to be chosen and implemented by the algorithm team.
    Here we make sure that all those filters work properly.
    """

    def __init__(self, shape: Optional[Union[Tuple, int]] = (3, 48)) -> None:
        """Object Constructor

        Args:
            shape (Optional[Union[Tuple, int]], optional): The number of input and output channels. Defaults to (3,9).
        """
        self.bd_list = [
            BandPassFilter(B_worst_case=9, a1=-64700, a2=31935, scale_out=0.8139),
            BandPassFilter(a1=-64458),
            BandPassFilter(a1=-64330),
            BandPassFilter(a1=-64138),
            BandPassFilter(a1=-63884),
            BandPassFilter(a1=-63566),
            BandPassFilter(a1=-63185),
            BandPassFilter(a1=-62743),
            BandPassFilter(a1=-62238),
            BandPassFilter(a1=-61672),
            BandPassFilter(a1=-61045),
            BandPassFilter(a1=-60357),
            BandPassFilter(a1=-59611),
            BandPassFilter(a1=-58805),
            BandPassFilter(a1=-57941),
            BandPassFilter(a1=-57020),
        ]
        if shape[1] != shape[0] * len(self.bd_list):
            raise ValueError(
                f"The output size should be {shape[0]*len(self.bd_list)} to compute filtered output! Each filter will be applied to one channel."
            )
        super().__init__(shape=shape, spiking_input=False, spiking_output=False)

    @property
    def numF(self) -> int:
        """Number of filters in the collection"""
        return len(self.bd_list)

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
        __B, __T, __C = input_data.shape

        if __C != self.size_in:
            raise ValueError(f"The input data should have {self.size_in} channels!")

        input_data = np.array(input_data, dtype=np.int64).astype(object)

        # -- Filter
        data_out = []

        # iterate over batch
        for signal in input_data:
            # iterate over channels
            channel_out = []
            for single_channel in signal.T:
                for __filter in self.bd_list:
                    # apply the filter to the input signal
                    out = __filter(single_channel)
                    channel_out.append(out)

            data_out.append(channel_out)

        # convert into numpy
        data_out = np.asarray(data_out, dtype=object)
        data_out = data_out.transpose(0, 2, 1)  # BxTxC

        return data_out, {}, {}
