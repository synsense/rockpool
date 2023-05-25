"""
Hardware butterworth filter implementation for the Xylo IMU.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from rockpool.devices.xylo.imu.preprocessing.utils import type_check
from rockpool.nn.modules.module import Module

__all__ = ["ChipButterworth", "BlockDiagram"]


@dataclass(eq=False, repr=False)
class BlockDiagram:
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


class ChipButterworth(Module):
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
            BlockDiagram(B_worst_case=9, a1=-64700, a2=31935, scale_out=0.8139),
            BlockDiagram(a1=-64458),
            BlockDiagram(a1=-64330),
            BlockDiagram(a1=-64138),
            BlockDiagram(a1=-63884),
            BlockDiagram(a1=-63566),
            BlockDiagram(a1=-63185),
            BlockDiagram(a1=-62743),
            BlockDiagram(a1=-62238),
            BlockDiagram(a1=-61672),
            BlockDiagram(a1=-61045),
            BlockDiagram(a1=-60357),
            BlockDiagram(a1=-59611),
            BlockDiagram(a1=-58805),
            BlockDiagram(a1=-57941),
            BlockDiagram(a1=-57020),
        ]
        if shape[1] != shape[0] * len(self.bd_list):
            raise ValueError(
                f"The output size should be {shape[0]*len(self.bd_list)} to compute filtered output! Each filter will be applied to one channel."
            )

    @property
    def numF(self) -> int:
        """Number of filters in the collection"""
        return len(self.bd_list)

    @type_check
    def _filter_AR(self, bd: BlockDiagram, sig_in: np.ndarray):
        """
        This function computes the AR part of the filter in the block-diagram with the given parameters.

        Args:
            bd (BlockDiagram): block diagram representation of the filter.
            sig_in (np.ndarray): the quantized input signal in python-object integer format.

        Raises:
            OverflowError: if any overflow happens during the filter computation.

        Returns:
            np.ndarray: the output signal of the AR filter.
        """

        # check that the input is within the valid range of block-diagram

        if np.max(np.abs(sig_in)) >= 2 ** (bd.B_in - 1):
            raise ValueError(
                f"The input signal values can be in the range [-2^{bd.B_in-1}, +2^{bd.B_in-1}]!"
            )

        output = []

        # w[n], w[n-1], w[n-2]
        w = [0, 0, 0]

        for sig in sig_in:
            # computation after the clock
            # w_new = sig * 2**bd.B_wf + np.floor( (-bd.a2*w[2] - bd.a1 * w[1])/ 2**bd.B_af )
            # w_new = np.floor(w_new / 2**bd.B_b)

            w_new = (sig << bd.B_wf) + ((-bd.a2 * w[2] - bd.a1 * w[1]) >> bd.B_af)
            w_new = w_new >> bd.B_b

            w[0] = w_new

            # register shift at the rising edge of the clock
            w[1], w[2] = w[0], w[1]

            output.append(w[0])

            # check the overflow: here we have the integer version

        if np.max(np.abs(output)) >= 2 ** (bd.B_w - 1):
            raise ValueError(
                f"output signal is beyond the valid output range of AR branch [-2^{bd.B_w-1}, +2^{bd.B_w-1}]!"
            )

        # convert into numpy
        return np.asarray(output, dtype=object)

    @type_check
    def _filter_MA(self, bd: BlockDiagram, sig_in: np.ndarray):
        """
        This function computes the MA part of the filter in the block-diagram representation.

        Args:
            bd (BlockDiagram): block diagram representation of the filter.
            sig_in (np.ndarray): input signal (in this case output of AR part) of datatype `pyton.object`.

        Raises:
            OverflowError: if any overflow happens during the filter computation.

        Returns:
            np.ndarray: quantized filtered output signal.
        """

        # check dimension
        if sig_in.ndim > 1:
            raise ValueError("input signal should be 1-dim.")

        sig_out = bd.b[0] * sig_in
        sig_out[2:] = sig_out[2:] + bd.b[2] * sig_in[:-2]

        # apply the last B_wf bitshift to get rid of additional scaling needed to avoid dead-zone in the AR part
        sig_out = sig_out >> bd.B_wf

        # check the validity of the computed output
        if np.max(np.abs(sig_out)) >= 2 ** (bd.B_out - 1):
            raise OverflowError(
                f"overflow or underflow: computed filter output is beyond the valid range [-2^{bd.B_out-1}, +2^{bd.B_out-1}]!"
            )

        return sig_out

    @type_check
    def _filter(self, bd: BlockDiagram, sig_in: np.ndarray):
        """
        This filter combines the filtering done in the AR and MA part of the block-diagram representation.

        Args:
            bd (BlockDiagram): block diagram representation of the filter.
            sig_in (np.ndarray): quantized input signal of python.object integer type.

        Raises:
            OverflowError: if any overflow happens during the filter computation.

        Returns:
            np.nadarray: quantized filtered output signal.
        """
        w = self._filter_AR(bd, sig_in)
        out = self._filter_MA(bd, w)

        return out

    @type_check
    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """
        Compute the output of all filters for an input signal.

        Args:
            input_data (np.ndarray): the quantized input signal of datatype python.object integer. (BxTxC)

        Returns:
            Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
                np.ndarray: the filtered output signal of all filters (BxTxC)
                dict: empty record dictionary.
                dict: empty state dictionary.
        """

        output = []

        for __filter in self.bd_list:
            # apply the filter to the input signal
            sig_out = self._filter(__filter, input_data)
            output.append(sig_out)

        # convert into numpy
        output = np.asarray(output, dtype=object)

        return output, {}, {}
