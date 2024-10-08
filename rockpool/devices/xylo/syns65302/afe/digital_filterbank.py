"""
This module implements the digital filterbank in XyloAudio 3 chip.
This is the first version of Xylo chip in which the analog filters have been replaced with the digital ones.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial, wraps
from logging import info

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from rockpool.devices.xylo.syns65302.afe.params import NUM_FILTERS
from rockpool.nn.modules.module import Module
from rockpool.parameters import ParameterBase

P_int = Union[int, ParameterBase]
P_float = Union[float, ParameterBase]
P_array = Union[np.array, ParameterBase]


# list of modules exported
__all__ = ["ChipButterworth"]


# a simple decorator to make sure that the input to the filters has type `int` or `np.int64`
# this is needed to avoid any over- and under-flow in filter implementation.


def type_check(func):
    """
    this function is a type-check decorator for make sure that all the input data are of type `int` or `np.int64`.
    This assures that the hardware and software will behave the same for the register sizes we have in mind.

    Args:
        func (Callable): the function to be decorated.
    """

    valid_types = [int, np.int64]

    # function for checking the type
    def verify(input):
        if isinstance(input, list):
            if len(input) == 0:
                return
            for el in input:
                type_check(el)

        if isinstance(input, np.ndarray):
            if (input.dtype not in valid_types) or (
                type(input.ravel()[0]) not in valid_types
            ):
                raise ValueError(
                    f"The elements of the following variable are not of type {valid_types}. This may cause mismatch between hardware and python implementation.\n"
                    + f"issue with the following variable:\n{input}\n"
                    + f"To solve this issue make sure that all the arrays have `dtype in {valid_types}`."
                )

        return

    # original function implementation
    @wraps(func)
    def inner_func(*args, **kwargs):
        # verification phase
        for arg in args:
            verify(arg)

        for key in kwargs:
            verify(kwargs[key])

        # main functionality
        return func(*args, **kwargs)

    # return an instance of the inner function
    return inner_func


# class for capturing over-flow and under-flow in computation
class OverflowError(Exception):
    pass


# class containing the parameters of the filter in state-space representation
# This is the block-diagram structure proposed for implementation.
#
# for further details see the proposed filter structure in the report:
# https://paper.dropbox.com/doc/Feasibility-study-for-AFE-with-digital-intensive-approach--BoJoECnIUJvHVe~Htanu~Ee6Ag-b07tQKnwpfDFYrZ5E8seQ


@dataclass
class BlockDiagram:
    B_worst_case: int
    """ number of additional bits devoted to storing filter taps such that no over- and under-flow can happen"""

    B_b: int
    """ bits needed for scaling b0"""

    B_af: int
    """ bits needed for encoding the fractional parts of taps"""

    a1: int
    """ integer representation of a1 tap"""

    a2: int
    """ integer representation of a2 tap"""

    scale_out: int
    """ surplus scaling due to `b` normalizationsurplus scaling due to `b` normalization. It is always in the range [0.5, 1.0]"""

    B_in: int = 14
    """ number of input bits that can be processed with the block diagram"""

    B_a: int = 16
    """ total number of bits devoted to storing filter a-taps"""

    B_wf: int = 8
    """ bits needed for fractional part of the filter output"""

    b: tuple = (1, 0, -1)
    """ [1, 0 , -1] : special case for normalized Butterworth filters"""

    B_w: Optional[int] = None
    """ total number of bits devoted to storing the values computed by the AR-filter. It should be equal to `B_in + B_worst_case + B_wf`"""

    B_out: Optional[int] = None
    """ total number of bits needed for storing the values computed by the WHOLE filter."""

    def __post_init__(self) -> None:
        if self.B_w is None:
            self.B_w = self.B_in + self.B_worst_case + self.B_wf
        if self.B_out is None:
            self.B_out = self.B_in + self.B_worst_case + self.B_wf


class ChipButterworth(Module):
    """
    Implement a simulation module for a digital Butterworth filterbank.
    """

    def __init__(
        self,
        select_filters: Optional[Tuple[int]] = None,
    ):
        """
        This class builds the block-diagram version of the filters, which is exactly as it is done in FPGA.
        The proposed filters are candidates that may be chosen for preprocessing of the audio data.

        Args:
            select_filters (Optional[Tuple[int]], optional): The indices of the filters to be used in the filter bank. Defaults to None: use all filters.
                i.e. select_filters = (0,2,4,8,15) will use Filter 0, Filter 2, Filter 4, Filter 8, and Filter 15.

        """
        if select_filters is None:
            select_filters = tuple(range(NUM_FILTERS))
        if self.validate_filter_selection(select_filters):
            self.__select_filters = select_filters
        else:
            raise ValueError("Invalid filter selection.")

        shape = (1, len(self.__select_filters))
        super().__init__(shape=shape, spiking_input=False, spiking_output=False)

        # Create block diagram for each filter

        # Filter 0
        bd_filter_0 = BlockDiagram(
            B_worst_case=7,
            B_b=8,
            B_af=6,
            a1=-32694,
            a2=16313,
            scale_out=0.5573,
        )

        # Filter 1
        bd_filter_1 = BlockDiagram(
            B_worst_case=6,
            B_b=8,
            B_af=6,
            a1=-32663,
            a2=16284,
            scale_out=0.7810,
        )

        # Filter 2
        bd_filter_2 = BlockDiagram(
            B_worst_case=6,
            B_b=7,
            B_af=7,
            a1=-32617,
            a2=16244,
            scale_out=0.5470,
        )

        # Filter 3
        bd_filter_3 = BlockDiagram(
            B_worst_case=5,
            B_b=7,
            B_af=7,
            a1=-32551,
            a2=16188,
            scale_out=0.7660,
        )

        # Filter 4
        bd_filter_4 = BlockDiagram(
            B_worst_case=5,
            B_b=6,
            B_af=8,
            a1=-32453,
            a2=16110,
            scale_out=0.5359,
        )

        # Filter 5
        bd_filter_5 = BlockDiagram(
            B_worst_case=4,
            B_b=6,
            B_af=8,
            a1=-32305,
            a2=16000,
            scale_out=0.7492,
        )

        # Filter 6
        bd_filter_6 = BlockDiagram(
            B_worst_case=4,
            B_b=5,
            B_af=9,
            a1=-32077,
            a2=15848,
            scale_out=0.5230,
        )

        # Filter 7
        bd_filter_7 = BlockDiagram(
            B_worst_case=3,
            B_b=5,
            B_af=9,
            a1=-31718,
            a2=15638,
            scale_out=0.7288,
        )

        # Filter 8
        bd_filter_8 = BlockDiagram(
            B_worst_case=3,
            B_b=4,
            B_af=10,
            a1=-31139,
            a2=15347,
            scale_out=0.5065,
        )

        # Filter 9
        bd_filter_9 = BlockDiagram(
            B_worst_case=2,
            B_b=4,
            B_af=10,
            a1=-30185,
            a2=14947,
            scale_out=0.7018,
        )

        # Filter 10
        bd_filter_10 = BlockDiagram(
            B_worst_case=2,
            B_b=4,
            B_af=10,
            a1=-28582,
            a2=14402,
            scale_out=0.9679,
        )

        # Filter 11
        bd_filter_11 = BlockDiagram(
            B_worst_case=2,
            B_b=3,
            B_af=11,
            a1=-25862,
            a2=13666,
            scale_out=0.6635,
        )

        # Filter 12
        bd_filter_12 = BlockDiagram(
            B_worst_case=2,
            B_b=3,
            B_af=11,
            a1=-21262,
            a2=12687,
            scale_out=0.9026,
        )

        # Filter 13
        bd_filter_13 = BlockDiagram(
            B_worst_case=2,
            B_b=2,
            B_af=13,
            a1=-27375,
            a2=22803,
            scale_out=0.6082,
        )

        # Filter 14
        bd_filter_14 = BlockDiagram(
            B_worst_case=2,
            B_b=2,
            B_af=13,
            a1=-4180,
            a2=19488,
            scale_out=0.8105,
        )

        # Filter 15
        bd_filter_15 = BlockDiagram(
            B_worst_case=2,
            B_b=1,
            B_af=14,
            a1=25566,
            a2=15280,
            scale_out=0.5337,
        )

        # list of block-diagram representations corresponding to the filters

        __full_list = [
            bd_filter_0,
            bd_filter_1,
            bd_filter_2,
            bd_filter_3,
            bd_filter_4,
            bd_filter_5,
            bd_filter_6,
            bd_filter_7,
            bd_filter_8,
            bd_filter_9,
            bd_filter_10,
            bd_filter_11,
            bd_filter_12,
            bd_filter_13,
            bd_filter_14,
            bd_filter_15,
        ]

        self.bd_list = [__full_list[i] for i in self.__select_filters]

    @staticmethod
    def validate_filter_selection(select_filters: Tuple[int]) -> bool:
        """
        Validate the filter selection, check if the range and the values are valid.

        Args:
            select_filters (Tuple[int]): The indices of the filters to be used in the filter bank.

        Raises:
            TypeError: select_filters must be of type tuple.
            TypeError: Filter index is not an integer.
            ValueError: Filter index is out of range. Valid indices are between 0 and 15.
            ValueError: All filter indices in select_filters must be unique.

        Returns:
            bool: True if the filter selection is valid.
        """
        if not isinstance(select_filters, tuple):
            raise TypeError("select_filters must be of type tuple.")

        # Check if all elements are integers and are within the allowed range
        for filter_index in select_filters:
            if not isinstance(filter_index, int):
                raise TypeError(f"Filter index {filter_index} is not an integer.")
            if filter_index < 0 or filter_index > 15:
                raise ValueError(
                    f"Filter index {filter_index} is out of range. Valid indices are between 0 and 15."
                )

        if len(select_filters) != len(set(select_filters)):
            raise ValueError("All filter indices in select_filters must be unique.")

        return True

    @type_check
    def _filter_AR(self, bd: BlockDiagram, sig_in: np.ndarray) -> np.ndarray:
        """
        This function computes the AR (auto-regressive) part of the filter in the block-diagram with the given parameters.

        Args:
            bd (BlockDiagram): block diagram representation of the filter.
            sig_in (np.ndarray): the input signal.

        Raises:
            OverflowError: if any overflow or underflow happens during the filter computation.

        Returns:
            np.ndarray: the output signal of the AR filter.
        """

        # check that the input is within the valid range of block-diagram

        if np.max(np.abs(sig_in)) >= 2 ** (bd.B_in - 1):
            raise OverflowError(
                f"The input signal values can be in the range [-2^{bd.B_in-1}, +2^{bd.B_in-1}]!"
            )

        output = []

        # w[n], w[n-1], w[n-2]
        w = [0, 0, 0]

        for sig in sig_in:
            # NOTE: Here we assume that AR part uses the latched/gated version of the filter at its output.
            # we have used the same convention for jax version.
            output.append(w[0])

            # computation after the clock
            w_new = (sig << bd.B_wf) + ((-bd.a2 * w[2] - bd.a1 * w[1]) >> bd.B_af)
            w_new = w_new >> bd.B_b

            w[0] = w_new

            # register shift at the rising edge of the clock
            w[1], w[2] = w[0], w[1]

        # convert into numpy array
        output = np.asarray(output, dtype=np.int64)

        # check the overflow: here we have the integer version
        if np.max(np.abs(output)) >= 2 ** (bd.B_w - 1):
            raise OverflowError(
                f"output signal is beyond the valid output range of AR branch [-2^{bd.B_w-1}, +2^{bd.B_w-1}]!"
            )

        return output

    @type_check
    def _filter_MA(self, bd: BlockDiagram, sig_in: np.ndarray) -> np.ndarray:
        """
        This function computes the MA (moving-average) part of the filter in the block-diagram representation.

        Args:
            bd (BlockDiagram): block diagram representation of the filter.
            sig_in (np.ndarray): integer-valued input signal.

        Raises:
            OverflowError: if any overflow happens during the filter computation.

        Returns:
            np.ndarray: the output of the MA part of the filter.
        """

        # check dimension
        if sig_in.ndim > 1:
            raise ValueError("input signal should be 1-dim.")

        sig_out = bd.b[0] * sig_in
        sig_out[2:] = sig_out[2:] + bd.b[2] * sig_in[:-2]

        # check the validity of the computed output
        if np.max(np.abs(sig_out)) >= 2 ** (bd.B_out - 1):
            raise OverflowError(
                f"overflow or underflow: computed filter output is beyond the valid range [-2^{bd.B_out-1}, +2^{bd.B_out-1}]!"
            )

        return sig_out

    @type_check
    def _filter(self, bd: BlockDiagram, sig_in: np.ndarray) -> np.ndarray:
        """
        This filter combines the filtering done in the AR and MA part of the block-diagram representation.

        Args:
            bd (BlockDiagram): block diagram representation of the filter.
            sig_in (np.ndarray): integer-valued input signal.

        Raises:
            OverflowError: if any overflow happens during the filter computation.

        Returns:
            np.nadarray: the output of the whole filter.
        """

        # AR branch
        w = self._filter_AR(bd, sig_in)

        # followed by MA branch
        out = self._filter_MA(bd, w)

        return out

    def _filter_iter(self, args_tuple):
        """
        this is the the same as `_filter` function with the difference that it maps on a single argument which contains a tuple of other arguments.
        This is used then in the multi-processing version of the filter needed for further speedup.
        """
        return self._filter(*args_tuple)

    @type_check
    def evolve(
        self,
        sig_in: np.ndarray,
        record: bool = False,
        num_workers: int = 4,
        scale_out: bool = False,
        python_version: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        This function computes the output of all filters for an input signal.

        Args:
            sig_in (np.ndarray): integer-valued input signal.
            record (bool, optional): record the state of the filter (AR part). Defaults to False.
            num_workers (int): number of independent processes (or threads) used to compute the filters faster. Defaults to 4 (4 workers).
            scale_out (bool, optional): add the surplus scaling due to `b` normalization.
            NOTE: this is due to the fact that after integer-quantization, one may still need to scale
            the filter output by some value in the range [0.5, 1.0] to obtain the final output. Defaults to False.
            python_version (bool, optional): force computing the filters with python version. Defaults to False: use jax if available.
        """
        sig_in, _ = self._auto_batch(sig_in)
        Nb, Nt, Nc = sig_in.shape

        # - Make sure input is 1D
        if Nb > 1 or Nc > 1:
            raise ValueError("the input signal should be 1-dim.")
        sig_in = sig_in[0, :, 0]

        if scale_out:
            scale_out_list = np.asarray([bd.scale_out for bd in self.bd_list])
        else:
            scale_out_list = np.ones(self.size_out)

        # -- Revert and repeat the input signal in the beginning to avoid boundary effects
        l = np.shape(sig_in)[0]
        __input_rev = np.flip(sig_in, axis=0)
        sig_in = np.concatenate((__input_rev, sig_in), axis=0)

        # check if jax version is available
        if JAX_Filter and not python_version:
            # ===========================================================================
            #                            Jax Version
            # ===========================================================================

            Bwf_list = np.asarray([bd.B_wf for bd in self.bd_list])
            Bb_list = np.asarray([bd.B_b for bd in self.bd_list])
            Baf_list = np.asarray([bd.B_af for bd in self.bd_list])
            a_list = np.asarray([[bd.a1, bd.a2] for bd in self.bd_list])
            b_list = np.asarray([bd.b for bd in self.bd_list])

            sig_out, recording = jax_filter(
                sig_in=sig_in,
                Bwf_list=Bwf_list,
                Bb_list=Bb_list,
                Baf_list=Baf_list,
                a_list=a_list,
                b_list=b_list,
                scale_out_list=scale_out_list,
                record=record,
            )

        else:
            # ===========================================================================
            #                            Python Version
            # ===========================================================================

            # to avoid issue with multiprocessing when num_workers = 1 => switch to single-cpu version

            # create an iterator of the arguments
            args_iter = ((bd, sig_in) for bd in self.bd_list)

            if num_workers > 1:
                # ===========================
                #     Multi-CPU Version
                # ===========================
                # use the multi-processing version
                # NOTE: this is unstable sometimes: processes start but they do not return any output

                # create an executor
                with ProcessPoolExecutor(max_workers=num_workers) as PPE:
                    # obtain the results
                    results = PPE.map(self._filter_iter, args_iter)

                # convert the results into numpy array: a transpose is needed to be compatible with rockpool T x C format where time is the first index.
                sig_out = np.asarray([result for result in results], dtype=np.int64).T

            else:
                # ===========================
                #     Single-CPU Version
                # ===========================
                sig_out = []

                for args_tuple in args_iter:
                    output = self._filter_iter(args_tuple=args_tuple)
                    sig_out.append(output)

                # convert the results into numpy array: a transpose is needed to be compatible with rockpool T x C format where time is the first index.
                sig_out = np.asarray(sig_out, dtype=np.int64).T

            # add the surplus scaling factor
            if scale_out:
                sig_out = np.einsum("ij, j -> ij", sig_out, scale_out_list)

            # in the python version state is empty: for performance reasons
            recording = {}

        # Trim the part of the signal coresponding to __input_rev (which was added to avoid boundary effects)
        sig_out = sig_out[l:, :]

        # Trim recordings
        recording = {k: v[l:, :] for k, v in recording.items()}

        return sig_out, self.state(), recording

    # add the call version for further convenience
    def __call__(self, *args, **kwargs):
        """
        This function simply calls the `evolve()` function and is added for further convenience.
        """
        return self.evolve(*args, **kwargs)

    # utility functions
    def _info(self) -> str:
        string = "*" * 60
        for filt_num in range(self.size_out):
            bd = self.bd_list[filt_num]
            string += (
                f"filter {filt_num}:\n"
                + f"B_worst_case (worst case amplitude)= {bd.B_worst_case}\n"
                + f"B_b = {bd.B_b}\n"
                + f"B_in = {bd.B_in}\n"
                + f"B_a = {bd.B_a}\n"
                + f"B_w = {bd.B_w}\n"
                + f"B_out = {bd.B_out}\n"
                + f"B_wf = {bd.B_wf}\n"
                + f"B_af = {bd.B_af}\n"
                + f"output surplus scale = {bd.scale_out: 0.4f}\n\n"
            )

        return string

    def register_config_XA3(self) -> dict:
        register_config = {}

        if self.size_out > 16:
            raise ValueError(
                f"This filterbank specifes {self.size_out} filters; only 16 are supported."
            )

        # - Get list of Bb values
        Bb = np.array([bd.B_b for bd in self.bd_list], int)

        # - Check Bb values
        if np.any(Bb > 15):
            raise ValueError(f"`Bb` must be 0..15. Found {np.max(Bb)}.")
        Bb = np.clip(Bb, 0, 15)

        # - Encode Bb register values
        register_config["bpf_bb_reg0"] = np.sum(
            [b << 4 * n for n, b in enumerate(Bb[:8])]
        )
        register_config["bpf_bb_reg1"] = np.sum(
            [b << 4 * n for n, b in enumerate(Bb[8:])]
        )

        # - Get list of Bwf values
        Bwf = np.array([bd.B_wf for bd in self.bd_list], int)
        Bwf.resize(16)

        # - Check Bwf values
        if np.any(Bwf > 15):
            raise ValueError(f"`Bwf` must be 0..15. Found {np.max(Bwf)}.")
        Bwf = np.clip(Bwf, 0, 15)

        # - Encode Bwf register values
        register_config["bpf_bwf_reg0"] = np.sum(
            [b << 4 * n for n, b in enumerate(Bwf[:8])]
        )
        register_config["bpf_bwf_reg1"] = np.sum(
            [b << 4 * n for n, b in enumerate(Bwf[8:])]
        )

        # - Get list of Baf values
        Baf = np.array([bd.B_af for bd in self.bd_list], int)
        Baf.resize(16)

        # - Check Baf values
        if np.any(Baf > 15):
            raise ValueError(f"`Baf` must be 0..15. Found {np.max(Baf)}.")
        Baf = np.clip(Baf, 0, 15)

        # - Encode Baf values
        register_config["bpf_baf_reg0"] = np.sum(
            [b << 4 * n for n, b in enumerate(Baf[:8])]
        )
        register_config["bpf_baf_reg1"] = np.sum(
            [b << 4 * n for n, b in enumerate(Baf[8:])]
        )

        # - Get list of A1 values
        A1 = [bd.a1 for bd in self.bd_list]

        # - Convert to 2s complement positive representation
        A1 = [
            int.from_bytes(int.to_bytes(a, 2, "big", signed=True), "big", signed=False)
            for a in A1
        ]

        A1 = np.array(A1, int)
        A1.resize(16)

        # - Check A1 values
        if np.any(A1 > 2**16):
            raise ValueError(
                f"`A1` values must fit in 16 bits. Found 2s complement of {np.max(A1)}."
            )

        # - Encode A1 values
        A1 = np.reshape(A1, (-1, 2))
        for n, A1s in enumerate(A1):
            reg_name = f"bpf_a1_reg{n}"
            reg_value = np.sum([a << 16 * n for n, a in enumerate(A1s)])
            register_config[reg_name] = reg_value

        # - Get list of A2 values
        A2 = [bd.a2 for bd in self.bd_list]

        # - Convert to 2s complement positive representation
        A2 = [
            int.from_bytes(int.to_bytes(a, 2, "big", signed=True), "big", signed=False)
            for a in A2
        ]

        A2 = np.array(A2, int)
        A2.resize(16)

        # - Check A2 values
        if np.any(A2 > 2**16):
            raise ValueError(
                f"`A2` values must fit in 16 bits. Found 2s complement of {np.max(A2)}."
            )

        # - Encode A2 values
        A2 = np.reshape(A2, (-1, 2))
        for n, A2s in enumerate(A2):
            reg_name = f"bpf_a2_reg{n}"
            reg_value = np.sum([a << 16 * n for n, a in enumerate(A2s)])
            register_config[reg_name] = reg_value

        return register_config


# implement the jax version as well
try:
    import jax
    import jax.numpy as jnp

    # only jax.float32 version implemented as jax.int32 will not work in the filters due to their large number of bits.
    def jax_filter(
        sig_in: np.ndarray,
        Bwf_list: np.ndarray,
        Bb_list: np.ndarray,
        Baf_list: np.ndarray,
        a_list: np.ndarray,
        b_list: np.ndarray,
        scale_out_list: np.ndarray,
        record: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """this function implements the filters in jax with float32 format.

        NOTE: To have the exact chip version, one needs to implement the filters in integer format.
        But with jax int32 there might be high chance for over-flow and under-flow. So we decided to use only float32 version.

        The problem with float32, in contrast, is that there might be a slight difference between software and hardware version.

        Args:
            sig_in (np.ndarray): quantized input signal.
            Bwf_list (np.ndarray): an array containing the list of B_wf (bitshifts used to avoid dead zone).
            Bb_list (np.ndarary): an array containing the list of Bb values used for proper scaling and quantizing the filter coefficients.
            Baf_list (np.ndarray): an array containing the bistshift the feedback part of the AR part of the filter.
            a_list (np.ndarray): an array containing the a-params of the filter (AR part).
            b_list (np.ndarray): an array containing the b-params of the filter (MA part).
            scale_out_list (np.ndarray): an array containing surplus scaling factors at the filter outputs.
            NOTE: This is due to bit-wise quantization, since all scaling is implemented as division or multiplication by 2. And scale_out is the left-over scaling.
            record (bool, optional): record the states of the filter (here the taps of AR part of the filter). Defaults to False.

        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]: output containing the filtered signal in all filters + states.
        """
        # number of filters
        num_filter = len(a_list)

        # convert the parameters into jax.numpy
        sig_in = jnp.asarray(sig_in, dtype=jnp.float32)
        Bwf_list = jnp.asarray(Bwf_list, dtype=jnp.float32)
        Bb_list = jnp.asarray(Bb_list, dtype=jnp.float32)
        Baf_list = jnp.asarray(Baf_list, dtype=jnp.float32)
        a_list = jnp.asarray(a_list, dtype=jnp.float32)
        b_list = jnp.asarray(b_list, dtype=jnp.float32)
        scale_out_list = jnp.asarray(scale_out_list, dtype=jnp.float32)

        # number of states in AR part of the filter: 2 due to order 1 filters + 1
        # assuming that we latch the state at the output of AR filter -> then feed to AR
        NUM_STATE_AR = 2 + 1

        @partial(jax.jit, static_argnums=(6,))
        def _compile_filter(
            sig_in: jnp.ndarray,
            Bwf_list: np.ndarray,
            Bb_list: jnp.ndarray,
            a_list: jnp.ndarray,
            b_list: jnp.ndarray,
            scale_out_list: jnp.ndarray,
            record: bool = False,
        ):
            # initial state: all the registers of the AR filters
            init_state = jnp.zeros((num_filter, NUM_STATE_AR), dtype=jnp.float32)

            def forward(state_in, input):
                # compute the feedback part in AR part of the filters
                ar_feedback = (
                    -jnp.sum(state_in[:, 0:-1] * a_list, axis=1) / 2**Baf_list
                )

                # combine scaled input (to avoid dead zone) with feedback coming from AR part
                merged_input = input * 2**Bwf_list + ar_feedback

                # find the next state after bitshift by Bb -> the next state is going to be loaded in the next clock
                next_state = merged_input / 2**Bb_list

                # compute the next state
                state_out = jnp.zeros_like(state_in)

                # first shift right when the clock comes -> then replace the new value into the state
                state_out = state_out.at[:, 1:].set(state_in[:, 0:-1])
                state_out = state_out.at[:, 0].set(next_state)

                # compute the output based on the gated version of the state (for more signal stability)
                # so the past state is used in computation
                # NOTE: we have not done any truncation by B_wf because of the bits added for avoiding dead-zone
                # this was due to the fact that, low-freq filters get badly truncated.
                sig_out = jnp.sum(b_list * state_in, axis=1)

                # apply the final surplus scaling
                sig_out = sig_out * scale_out_list

                # if states needed to be recorded -> consider states as part of the output
                output = (sig_out, state_out) if record else (sig_out,)

                return state_out, output

            # apply the forward to compute
            final_state, output = jax.lax.scan(forward, init_state, sig_in)

            return final_state, output

        final_state, output = _compile_filter(
            sig_in=sig_in,
            Bwf_list=Bwf_list,
            Bb_list=Bb_list,
            a_list=a_list,
            b_list=b_list,
            scale_out_list=scale_out_list,
            record=record,
        )

        if record:
            sig_out, state = output

            sig_out = np.asarray(sig_out, dtype=np.int64)
            recording = {"filter_AR_state": np.asarray(state, dtype=np.int64)}

        else:
            (sig_out,) = output

            sig_out = np.asarray(sig_out, dtype=np.int64)
            recording = {}

        return sig_out, recording

    # set the  flag for jax version
    JAX_Filter = True

    info("Jax version was found. Filterbank will be computed using jax speedup.\n")

except ModuleNotFoundError as e:
    info(
        "No jax module was found for filter implementation. Filterbank will use python version (multi-processing version).\n"
        + str(e)
    )

    # set flag for jax
    JAX_Filter = False
