"""
PDM module employs a delta sigma modulator to convert the analog audio signal into a PDM bit stream where the relative frequency of 1-vs-0 depends on the amplitude of the signal.
This file contains the implementation of a custom deltasigma module implementation which simulates the microphone's implementation in hardware.
NOTE that the implementation is not 1-to-1 with the hardware implementation and is only used for simulation purposes.
"""

from logging import info
from typing import Any, Tuple

import numpy as np
import scipy.signal as sp

from rockpool.devices.xylo.syns65302.afe.params import (
    AUDIO_CUTOFF_FREQUENCY,
    DELTA_SIGMA_ORDER,
    PDM_SAMPLING_RATE,
)

__all__ = ["DeltaSigma"]


class DeltaSigma:
    def __init__(
        self,
        amplitude: float = 1.0,
        bandwidth: float = AUDIO_CUTOFF_FREQUENCY,
        order: int = DELTA_SIGMA_ORDER,
        fs: float = PDM_SAMPLING_RATE,
    ):
        """this class implements a simple deltasigma modulation module.
        NOTE: this class is going to replace the `deltasigma` library which is not supported anymore.

        Args:
            amplitude (float, optional): maximum amplitude of the input signal. Defaults to 1.0 to obtain +1, -1 as the output.
            bandwidth (float, optional): target bandwidth of the input signal. Defaults to AUDIO_CUTOFF_FREQUENCY = 20K in XyloAudio 3.
            order (int, optional): order of deltasigma module. Defaults to DELTA_SIGMA_ORDER = 4 in XyloAudio 3.
            fs (float, optional): sampling rate of deltasigma module. Defaults to PDM_SAMPLING_RATE = 1.6 M in XyloAudio 3.
        """
        self.amplitude = amplitude
        self.bandwidth = bandwidth

        if fs <= 2 * bandwidth:
            raise ValueError(
                "sampling rate of deltasigma module should be much larger than the bandwidth of the signal!"
            )

        self.fs = fs
        self.order = order

        # * build a low-pass filter with the given order
        # butterworth filter
        filter_order = self.order
        filter_cutoff = self.bandwidth
        filter_type = "lowpass"

        b, a = sp.butter(
            N=filter_order,
            Wn=2 * np.pi * filter_cutoff,
            btype=filter_type,
            analog=True,
            output="ba",
        )

        # reverse the coefficients to convert them into recursive feedback format
        b = b[::-1]
        a = a[::-1]

        # dimension of the state
        self.dim_state = len(a) - 1

        # * convert this into block-diagram format
        # compute the state space representation
        A = np.diag(np.ones(self.dim_state - 1), -1)
        A_Q = -a[:-1]

        B = np.zeros(self.dim_state)
        B[0] = b[0]

        C = np.zeros(self.dim_state)
        C[-1] = 1

        # * apply state normalization for better numerical stability
        # NOTE: this normalization is needed since for a signal at frequency f0, its first and second derivative are scaled by f0 and f0^2 and so on.
        # So without proper normalization the simulation may be numerically unstable especially when the input signal has high frequency.
        self.norm_factor = self.fs
        N = np.diag(1 / self.norm_factor ** np.arange(self.dim_state - 1, -1, step=-1))

        self.A = N @ A @ np.linalg.inv(N)
        self.A_Q = N @ A_Q
        self.B = N @ B
        self.C = np.linalg.inv(N) @ C

    def evolve(
        self,
        sig_in: np.ndarray,
        sample_rate: float = None,
        python_version: bool = False,
        record: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """this module processes the input signal and produces its corresponding sigmadelta modulation via simulating the corresponding ODE.

        Args:
            sig_in (np.ndarray): input signal.
            sample_rate (float): sample rate of the input signal. Defaults to None.
            python_version (bool, optional): ttthis flag forces deltasigma computation to be done in Python without any Jax speedup. Defaults to False.
            record (bool, optional): record the states of the filter. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: a tuple containing arrays corresponding to deltasigma +1, -1 signal, pre-quantization signal,
                                                                    possibly interpolated high-rate signal used for simulation, and states of the filter.

        """

        # * validate the amplitude
        if np.max(np.abs(sig_in)) > self.amplitude:
            raise ValueError(
                "the amplitude of the input signal is larger than the target amplitude for sigmadelta module! This may results in wrong modulation or divergence of the module!"
            )

        # * convert the sample rate of the signal if needed
        if sample_rate is None:
            sample_rate = self.fs

        # signal needs resampling?
        if sample_rate != self.fs:
            time_in = np.arange(len(sig_in)) / sample_rate

            duration = (len(sig_in) - 1) / sample_rate
            time_target = np.arange(0, duration, step=1 / self.fs)
            sig_in_resampled = np.interp(time_target, time_in, sig_in)

            # replace the original signal
            sig_in = sig_in_resampled

        # check if Jax is available for speedup and if it is permitted
        if JAX_DeltaSigma and not python_version:
            sig_out_Q, sig_out, state_list = jax_deltasigma_evolve(
                sig_in=sig_in,
                fs=self.fs,
                A=self.A,
                A_Q=self.A_Q,
                B=self.B,
                C=self.C,
                amplitude=self.amplitude,
                record=record,
            )

            return sig_out_Q, sig_out, sig_in, state_list

        # otherwise: proceed with the python version
        state_list = []

        # states for the simulation
        state = np.zeros(self.dim_state)

        sig_out = []
        sig_out_Q = []

        sigmadelta_out = 0
        sigmadelta_out_Q = 0

        for sig in sig_in:
            # differential of the state
            d_state = (
                np.einsum("ij, j -> i", self.A, state)
                + self.B * sig
                + self.A_Q * sigmadelta_out_Q
            )

            # update the state
            state += d_state * 1 / self.fs

            if record:
                state_list.append(np.copy(state))

            # compute the output
            sigmadelta_out = np.sum(self.C * state)
            sigmadelta_out_Q = self.amplitude * (
                2 * np.heaviside(sigmadelta_out, 0) - 1
            )

            sig_out.append(sigmadelta_out)
            sig_out_Q.append(sigmadelta_out_Q)

        sig_out = np.asarray(sig_out)
        sig_out_Q = np.asarray(sig_out_Q)
        state_list = np.asarray(state_list)

        return sig_out_Q, sig_out, sig_in, state_list

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """this is the same as `evolve` function."""
        return self.evolve(*args, **kwargs)

    def recover(self, bin_in: np.ndarray) -> np.ndarray:
        """this module takes the binary encoded signal and recovers the original signal via low-pass filtering and decimation.

        NOTE: here we are applying a simple low-pass filter to recover the signal.
        This is used only for sanity check and in practice more involved filters may be applied to obtain a better recovery.

        Args:
            bin_in (np.ndarray): binary input signal obtained from deltasigma modulation.

        Returns:
            np.ndarray: array containing the recovered signal from binary deltasigma output.
        """

        # * build a low-pass filter with the given order
        filter_order = self.order
        filter_cutoff = self.bandwidth
        filter_type = "lowpass"

        b, a = sp.butter(
            N=filter_order,
            Wn=filter_cutoff,
            btype=filter_type,
            analog=False,
            output="ba",
            fs=self.fs,
        )

        # * filter the binary signal and decimate it
        sig_rec = sp.lfilter(b, a, bin_in)

        oversampling = int(self.fs / self.bandwidth)
        sig_rec_dec = sig_rec[::oversampling]

        # return the high-res and low-res (decimated) version of the recovered signal
        return sig_rec, sig_rec_dec

    def validate(self, sig_in: np.ndarray, bin_in: np.ndarray) -> bool:
        """
        this module investigates if the generated deltasigma encoding is valid or not.

        NOTE: this is used as a sanity check since in practice the simulation of deltasigma in block-diagram format may diverge!

        Args:
            sig_in (np.ndarray): input signal.
            bin_in (np.ndarray): binary bitstream obtain from deltasigma modulation of the input signal.

        Returns:
            bool: returns True if the deltasigma modulation is not diverged.
        """

        # compute the mean values of the signam
        mean_sig_pos = np.mean(sig_in + np.abs(sig_in)) / 2
        mean_sig_neg = np.mean(np.abs(sig_in) - sig_in) / 2

        mean_sig_sum = mean_sig_pos + mean_sig_neg

        mean_sig_neg /= mean_sig_sum
        mean_sig_pos /= mean_sig_sum

        mean_bin_pos = np.mean(bin_in + np.abs(bin_in)) / 2
        mean_bin_neg = np.mean(np.abs(bin_in) + bin_in) / 2

        mean_bin_sum = mean_bin_pos + mean_bin_neg

        mean_bin_pos /= mean_bin_sum
        mean_bin_neg /= mean_bin_sum

        # measure the relative error
        rel_err = np.max(
            [abs(mean_sig_pos - mean_bin_pos), abs(mean_sig_neg - mean_bin_neg)]
        )

        # threshold for the relative error
        EPS = 0.2

        return rel_err < EPS


# - Jax implementation for further speedup of deltasigma modulation module
try:
    import jax
    import jax.numpy as jnp

    # only jax.float32 version implemented as jax.int32 will not work in the filters due to their large number of bits.
    def jax_deltasigma_evolve(
        sig_in: np.ndarray,
        A: np.ndarray,
        A_Q: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        amplitude: np.ndarray,
        fs: float,
        record: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """this module implements the jax version of deltasigma modulator for further speedup.

        Args:
            sig_in (np.ndarray): input signal
            A (np.ndarray): A matrix in block-diagram version.
            A_Q (np.ndarray): A_Q vector in block-diagram version.
            B (np.ndarray): B vector in block-diagramm version.
            C (np.ndarray): C vector in block-diagram version.
            amplitude (np.ndarray): amplitude of the antipodal output of deltasigma modulator.
            record (bool, optional): record the inner states of deltasigma. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: a tuple containing the deltasigma output after quantizatio and before quenatization, and the inner states of the deltasigma module.
        """
        # dimension of the state
        state_dim = A.shape[0]

        state_init = jnp.zeros(state_dim + 1, dtype=jnp.float32)
        A = jnp.asarray(A, dtype=jnp.float32)
        A_Q = jnp.asarray(A_Q, dtype=jnp.float32)
        B = jnp.asarray(B, dtype=jnp.float32)
        C = jnp.asarray(C, dtype=jnp.float32)

        # define the forward function of the dynamics
        def forward(state_in, input):
            # decompse the state and the quantized output
            state, sig_out_Q = state_in[:-1], state_in[-1]

            # compute diffrential of the state
            d_state = A @ state + B * input + A_Q * sig_out_Q

            # update the state
            state += d_state * 1 / fs

            # produce the quantized output
            sig_out = state[-1]
            sig_out_Q = amplitude * (2 * jnp.heaviside(sig_out, 0) - 1)

            # build the output state and output
            state_out = jnp.zeros_like(state_in)
            state_out = state_out.at[:-1].set(state)
            state_out = state_out.at[-1].set(sig_out_Q)

            output = state_out if record else state_out[-2:]

            return state_out, output

        # apply the forward dynamics to compute the deltasigma output
        final_state, output = jax.lax.scan(forward, state_init, sig_in)

        # convert into numpy format for return
        sig_out = np.asarray(output[:, -2], dtype=np.float64)
        sig_out_Q = np.asarray(output[:, -1], dtype=np.float64)

        state = (
            np.asarray(output[:, :-1], dtype=np.float64)
            if record
            else np.asarray([], dtype=np.float64)
        )

        return sig_out_Q, sig_out, state

    # set the  flag for jax version
    JAX_DeltaSigma = True

    info(
        "Jax version was found. DeltaSigma module will be computed using jax speedup.\n"
    )

except ModuleNotFoundError as e:
    info(
        "No jax module was found for DeltaSigma implementation. DeltaSigma module will use python version!\n"
        + str(e)
    )

    # set flag for jax
    JAX_DeltaSigma = False


# - In debug mode deactivate accelerated version
__DEBUG_MODE__ = False

if __DEBUG_MODE__:
    JAX_DeltaSigma = False


if JAX_DeltaSigma:
    # Jax version is active: use jax since it is slightly faster than CPP if all dependencies are ok!
    # apply simple embedding in Python
    info(
        f"JAX_DeltaSigma: {JAX_DeltaSigma}: Using Jax-JIT version of DeltaSigma modulation."
    )

else:
    # use the Python version
    info(f"No Jax version: Using Python native for DeltaSigma modulation.")
