""" 
This module implements the divisive normalization (DN) algorithm to balance the spike rate.

NOTE: In the previous version XyloAudio 2, DN module was after the spike generation because spikes were produced asynchronously within the analog part (analog filter + leaky IF spike generator).
In the current version XyloAudio 3 since the output of the filters (here digital filters) is directly available, one does not need to do (i) spike generation followed by (ii) DN on the generated spikes.
Instead, we merge these two so that DN is applied directly to the filter output to normalize its power, which yields a better performance than DN applied to spikes.

NOTE: There is of course an option to NOT apply DN where in that case, spikes are produced using ordinary IAF with given/fixed thresholds rather than adaptive ones computed and used in DN.

For further details on how DN is implemented and how its parameters should be selected, we refer to the following repo and documentation:
https://spinystellate.office.synsense.ai/research/auditoryprocessing/synchronous-divisive-normalization

Important Remarks:

(i) don't forget to activate the jax flag again otherwise DN will always use python vresion.

(ii) In the current version of DN, we have a scaling of threshold which is a power of two. This implies that the threshold of DN thus the spike rate moves in steps of power of 2:
we replaced this by the difference of two bit-shifts so that we can cover a more flexible range of spike rates.

(iii) the surplus scaling of the filters is not added in the filterbank. This simply means that they should be added in thresholds we use for DN.
"""

# required packages
import numpy as np
from rockpool.devices.xylo.syns65302.afe.params import NUM_FILTERS
from rockpool.devices.xylo.syns65302.afe.digital_filterbank import type_check

from functools import partial

from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter

from logging import info

from typing import Union, Tuple, List, Dict
from rockpool.typehints import P_ndarray, P_float

# exported modules
__all__ = ["DivisiveNormalization", "jax_spike_gen", "fjax_spike_gen", "py_spike_gen"]


class DivisiveNormalization(Module):
    """
    This class implements a divisive normalisation module for XyloAudio 3.
    """

    def __init__(
        self,
        fs: float,
        shape: Tuple[Tuple[int], int] = NUM_FILTERS,
        enable_DN_channel: bool = True,
        spike_rate_scale_bitshift1: int = 6,
        spike_rate_scale_bitshift2: int = 0,
        low_pass_bitshift: int = 12,
        EPS_vec: Union[int, np.ndarray] = 1,
        fixed_threshold_vec: Union[int, np.ndarray] = 2 ** (14 - 1 + 8 + 6),
    ):
        """
        Initialise a divisive normalisation module

        Args:
            fs (float): sampling frequency of the input audio in Hz (e.g., 48.8 or 50K).
            shape (int): number of channels (here filters) in the divisive normalization module. Defaults to NUM_FILTERS (16 in XyloAudio 3).
            enable_DN_channel (bool): if True, divisive normalization is applied to the channel. Defaults to True.
            spike_rate_scale_bitshift1 ( int ): how much the spike rate should be scaled compared with the sampling rate of the input audio. Defaults to 6.
            spike_rate_scale_bitshift2 ( int ): how much the spike rate should be scaled compared with the sampling rate of the input audio. Defaults to 0.
            NOTE #1: A bitshift of size 0 results in a spike rate around the sampling rate of the audio, thus very large.
            A bitshift of size `b1=6, b2=0` yields a spike rate of fs/(2^b1 - 2^b2) where fs is the sampling rate of the input audio.
            A default value of 6 bits and 1 bits yields an average of 1 (slightly larger than 1) spike per 2^6 - 1 (=63) clock periods. With a clock rate of around 50K -> around 800 ~ 1K spikes/sec per channel.
            NOTE #2: We could use a single bit-shift value but this may cause some gaps in spike rate. To avoid that we decided to use two bit-shifts [b1, b2]. The thresholds that we use for spike
            generation will be `M(t)<<b1 - M(t)<<b2` where by a proper choice of b1 and b2 we can adjust the spike rate more smoothly.
            NOTE #3: Each channel can have its own `b1` and `b2`. For that we need to pass a vector as the input.

            low_pass_bitshift (int): number of bitshifts used in filter implementation. A bitshift of size `b` implies an averaging window of `2^b` clock periods. Defaults to 12.
            NOTE: The default value of 12, implies an averaging window of size 4096 clock periods. For an audio of clock rate 50K, this yields an averaging window of size 80 ms.

            EPS_vec (Union[int, np.ndarray]): lower bound on spike generation threshold. Defaults to 1 when not set.
            NOTE: Using this parameter we can control the noise level in the sense that if average power in a channel is less than EPS, the spike rate of that channel is somehow diminished
            during spike generation. We refer to the documentation file (see the top part) for further explanation.

            fixed_threshold_vec (Union[int, np.ndarray]): vector of threshold vectors of size `num_channels`. These threshold vectors are used only when the fixed
            threshold mode is active (mode = 0). Defaults to 2**27 (for a spike rate of around 1K for an input sinusoid signal quantized to 14 bits).
            NOTE: How to set the value of threshold for a target spike rate?
            In the current implementation, input audio to filters has 14 bits which is further left-bit-shifted by 8 bits to improve numerical precision, thus, 22 bits.
            This implies that the output signal may have a maximum amplitude of at most `2^21 - 1 ~ 2^22`, for example, when fed by a sinusoid signal
            within the passband of the filter.
            For a target rate of around 1K. e.g., 1 spike every 50 clock period for an audio of sampling rate 50K, then we need to choose a threshold as large as
            `50 x 2^22 ~ 2^27`.
        """
        super().__init__(shape=shape, spiking_input=False, spiking_output=True)

        # how much spike rate should be reduced compared with the sampling rate of the audio
        # bitshift 1
        self.spike_rate_scale_bitshift1 = SimulationParameter(
            np.broadcast_to(spike_rate_scale_bitshift1, self.size_out),
            shape=self.size_out,
            cast_fn=lambda x: np.array(x, dtype=np.int64),
        )
        """ ndarray[int64]: bitshift 1 per channel ``(N,)`` """

        # bitshift 2
        self.spike_rate_scale_bitshift2 = SimulationParameter(
            np.broadcast_to(spike_rate_scale_bitshift2, self.size_out),
            shape=self.size_out,
            cast_fn=lambda x: np.array(x, dtype=np.int64),
        )
        """ ndarray[int64]: bitshift 2 per channel ``(N,)`` """

        # make sure that in all the channels b1 is larger than b2
        if np.any(self.spike_rate_scale_bitshift1 <= self.spike_rate_scale_bitshift2):
            raise ValueError(
                """
                For smooth spike rate variation, we use `M(t)<<b1 - M(t)<<b2` as the spike generation threshold. This requires than b1 in all channels is larger than b2 so that the threshold is always positive.
                """
            )

        # minimum value for spike generation threshold during the DN mode
        self.EPS_vec: P_ndarray = SimulationParameter(
            np.broadcast_to(EPS_vec, self.size_out),
            shape=self.size_out,
            cast_fn=lambda x: np.array(x, dtype=np.int64),
        )
        """ ndarray[int64]: minimum value for spike generation threshold during the DN mode ``(N,)`` """

        # spike generation thresholds when DN mode is inactive
        self.fixed_threshold_vec: P_ndarray = SimulationParameter(
            np.broadcast_to(fixed_threshold_vec, self.size_out),
            shape=self.size_out,
            cast_fn=lambda x: np.array(x, dtype=np.int64),
        )
        """ ndarray[int64]: spike generation thresholds when DN mode is inactive ``(N,)`` """

        self.low_pass_bitshift: P_ndarray = SimulationParameter(
            np.broadcast_to(low_pass_bitshift, self.size_out),
            shape=self.size_out,
            cast_fn=lambda x: np.array(x, dtype=np.int64),
        )
        """ ndarray[int64]: number of bit-shifts used in low-pass filtering ``(N,)`` """

        self.fs: P_float = SimulationParameter(float(fs))
        """ float: Sampling frequency of the module in Hz """

        self.enable_DN_channel = SimulationParameter(
            shape=self.size_out,
            init_func=lambda s: np.broadcast_to(
                np.array(enable_DN_channel, dtype=bool), s
            ),
        )
        """ ndarray[bool]: Enable divisive normalisation threshold for each channel ``(N,)`` """

        self.joint_normalization = SimulationParameter(True)
        """ bool: If ``True``, normalization occurs jointly across all channels. If ``False``, normalization occurs separately for each channel. """

    @type_check
    def evolve(
        self,
        sig_in: np.ndarray,
        record: bool = False,
        *args,
        **kwargs,
    ):
        """This module takes the input `B x T x C` signal `sig_in` and applies divisive normalization across each channel.

        Args:
            sig_in (np.ndarray): input signal of dimension `B x T x num_channels`.
            record (bool): record the state. Defaults to False.
        """
        sig_in, _ = self._auto_batch(sig_in)
        Nb, T, num_channels = sig_in.shape

        if self.size_out != num_channels:
            raise ValueError(
                f"number of channels ({num_channels}) in the input signal differs\n"
                + f"from the numbers of channels ({self.size_out} in DN module!"
            )

        if Nb > 1:
            raise ValueError("Only batch size of 1 is supported.")

        sig_in = sig_in[0, :, :]

        # -- Revert and repeat the input signal in the beginning to avoid boundary effects
        l = np.shape(sig_in)[0]
        __input_rev = np.flip(sig_in, axis=0)
        sig_in = np.concatenate((__input_rev, sig_in), axis=0)

        # check if jax is available
        if JAX_SPIKE_GEN:
            ## use jax version:
            # we have int32 and float32 version:
            #       - former matches the python version exactly but may have over- and under-flow issues due to int32 limitation in jax.
            #       - latter may have deviation from python version, e.g., spike times may be shifted slightly, which would be fine for applications in XyloAudio 3.

            # check if int32 version is ok

            # maximum value of signal in various channels
            max_sig_in = np.max(np.abs(sig_in), axis=0)

            # maximum value of low-pass filter in various channels
            max_low_pass_value = np.max(max_sig_in * (2**self.low_pass_bitshift))
            max_low_pass_channel = np.argmax(max_sig_in * (2**self.low_pass_bitshift))

            # choose a suitable version of jax
            JAX_MAX_BITS = 32
            if max_low_pass_value < 2 ** (JAX_MAX_BITS - 1):
                jax_spike_gen_func = fjax_spike_gen
            else:
                jax_spike_gen_func = fjax_spike_gen
                info(
                    "Jax float32 was chosen for spike generation.\n\n"
                    + f"NOTE #1: Since Jax has only 32 bit for integer simulation, it could not simulate the current divisive normalization setting. More specifically, the amplitude of the low-pass filter can go up to {max_low_pass_value} in channel {max_low_pass_channel}. This cannot be simulated with 32-bit integer format.\n\n"
                    + "NOTE #2: Jax float32 should be sufficiently good for XyloAudio 3 simulation but it has the issues that the generated spikes may have some jitter compared with the exact integer version obtained from the XyloAudio 3 chip. However, the average rate of spikes even over very small time intervals should be the same as exact integer version. This jitter simply implies that spikes should not be compared with MSE distance, for example, because jitter may yield a very large distance, whereas the spikes are indeed quite similar.\n"
                    + "Other metrics such as Wasserstein metric should be good in this case."
                )

            # convert the parameters to the format of jax
            spikes, recording = jax_spike_gen_func(
                sig_in=sig_in,
                mode_vec=self.enable_DN_channel.astype(jnp.int64),
                spike_rate_scale_bitshift1=self.spike_rate_scale_bitshift1,
                spike_rate_scale_bitshift2=self.spike_rate_scale_bitshift2,
                low_pass_bitshift=self.low_pass_bitshift,
                EPS_vec=self.EPS_vec,
                fixed_threshold_vec=self.fixed_threshold_vec,
                joint_normalization=self.joint_normalization,
                record=record,
            )
        else:
            ## use python version
            spikes, recording = py_spike_gen(
                sig_in=sig_in,
                mode_vec=self.enable_DN_channel.astype(np.int64),
                spike_rate_scale_bitshift1=self.spike_rate_scale_bitshift1,
                spike_rate_scale_bitshift2=self.spike_rate_scale_bitshift2,
                low_pass_bitshift=self.low_pass_bitshift,
                EPS_vec=self.EPS_vec,
                fixed_threshold_vec=self.fixed_threshold_vec,
                joint_normalization=self.joint_normalization,
                record=record,
            )

        # Trim the part of the signal coresponding to __input_rev (which was added to avoid boundary effects)
        spikes = spikes[l:, :]

        # Trim recordings
        for k, v in recording.items():
            recording[k] = v[l:, :] if "state" in k else v

        return spikes, self.state(), recording

    # some uitility functions
    def _compute_spike_scale_bitshift(self, spike_rate: float):
        """this function computes the closest number of bitshifts `b` needed to adjust the rate scaling parameter
        `p=1/2**b` to have a given spike rate in the presence of divisive normalization.

        Args:
            spike_rate (float): number of spikes/sec expected in the presence of divisive normalization.
        """
        p_val = spike_rate / self.fs

        rate_downscale_val = 1 / p_val
        rate_downscale_val = rate_downscale_val if rate_downscale_val > 1.0 else 1.0

        # number of bitshifts needed to implement this
        spike_scale_bitshift = round(np.log2(rate_downscale_val))

        return spike_scale_bitshift

    def _info(self):
        string = (
            "Synchronous divisive normalization module\n"
            + "-" * 41
            + "\n"
            + f"number of channels: {self.size_out}\n"
            + f"prefixed threshold values for channels for non-DN mode are:\n {self.fixed_threshold_vec}\n"
            + f"rate scaling value (parameter p in `p fs E(t) / (M(t) V EPS)`): 1/{2**self.spike_rate_scale_bitshift1 - 2**self.spike_rate_scale_bitshift2}\n"
            + f"NOTE: this means that the spike rate will be around 1/{2**self.spike_rate_scale_bitshift1 - 2**self.spike_rate_scale_bitshift2} x fs spikes/sec\n"
            + f"number of bitshifts used to apply rate scaling: \n{self.spike_rate_scale_bitshift1}\n{self.spike_rate_scale_bitshift2}"
            + f"\nLow-pass averaging filter:\n"
            + f"number of bitshifts used for averaging: {self.low_pass_bitshift}\n"
            + f"number of time samples used for averaging: {2**self.low_pass_bitshift}\n"
            + f"averaging window duration: {2**self.low_pass_bitshift/self.fs} sec"
        )

        return string

    def register_config_XA3(self) -> dict:
        register_config = {}

        # - Check size of DN block
        if self.size_out > 16:
            raise ValueError(
                f"This divisive normalisation block specifes {self.size_out} channels; only 16 are supported."
            )

        # - Check threshold values
        IAF_th = self.fixed_threshold_vec
        if np.any(IAF_th > 2**42):
            raise ValueError(f"`IAF_th` must be 0..2**42. Found {np.max(IAF_th)}.")
        IAF_th = np.clip(IAF_th, 0, 2**42 - 1)

        # - Encode IAF_th values
        for n, this_IAF_th in enumerate(IAF_th):
            register_config[f"iaf_thr{n}_l"] = this_IAF_th & 0xFFFF_FFFF
            register_config[f"iaf_thr{n}_h"] = this_IAF_th >> 32

        # - Check EPS values
        if np.any(self.EPS_vec > 2**16 - 1):
            raise ValueError(f"`EPS` must be 0..2**16. Found {np.max(self.EPS_vec)}.")

        # - Encode EPS values
        EPS_vec = self.EPS_vec
        EPS_vec.resize(16)
        EPS_vec = np.reshape(EPS_vec, (-1, 2))
        for n, this_eps in enumerate(EPS_vec):
            register_config[f"dn_eps_reg{n}"] = np.sum(
                [eps << 16 * n for n, eps in enumerate(this_eps)]
            )

        # - Encode B values (low pass bitshift)
        B_vec = self.low_pass_bitshift
        B_vec.resize(16)
        B_vec = np.reshape(B_vec, (-1, 2))
        for n, this_B in enumerate(B_vec):
            register_config[f"dn_b_reg{n}"] = np.sum(
                [b << 16 * n for n, b in enumerate(this_B)]
            )

        # - Encode K1 values (spike rate scale bitshift 1)
        K1_vec = self.spike_rate_scale_bitshift1
        K1_vec.resize(16)
        K1_vec = np.reshape(K1_vec, (-1, 2))
        for n, this_k1 in enumerate(K1_vec):
            register_config[f"dn_k1_reg{n}"] = np.sum(
                [k1 << 16 * n for n, k1 in enumerate(this_k1)]
            )

        # - Encode K2 values (spike rate scale bitshift 2)
        K2_vec = self.spike_rate_scale_bitshift2
        K2_vec.resize(16)
        K2_vec = np.reshape(K2_vec, (-1, 2))
        for n, this_k2 in enumerate(K2_vec):
            register_config[f"dn_k2_reg{n}"] = np.sum(
                [k2 << 16 * n for n, k2 in enumerate(this_k2)]
            )

        return register_config


# jit version of divisive normalization and spike generation
try:
    import jax
    import jax.numpy as jnp

    # NOTE: this function implements DN using the integer version of jax, which is unfortunately still 32-bit.
    # This is in general fine when the input comes from the filterbank filters.
    # The only problem is that in the filterbank we have this initial bitshift to avoid deadzone which makes the input 14 + 8 = 22 bits.
    # So there is only 10 bits left to avoid any over- and under-flow.
    # It seems that for large values of filter bitshift, where more averaging and signal accumulation is done in low-pass filter, 32 bits may not be enough.
    def jax_spike_gen(
        sig_in: np.ndarray,
        mode_vec: np.ndarray,
        spike_rate_scale_bitshift1: np.ndarray,
        spike_rate_scale_bitshift2: np.ndarray,
        low_pass_bitshift: np.ndarray,
        EPS_vec: np.ndarray,
        fixed_threshold_vec: np.ndarray,
        joint_normalization: bool = False,
        record: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """this module processes the input signal in various channels and converts the into spikes via joint divisive normalization.

        Args:
            sig_in (np.ndarray): input signal of dimension `T x C` where C is the number of channels.
            mode_vec (np.ndarray): C-dim array of {0,1} showing the mode of spikes generation at each channel.
                0 => use the fixed thresholds
                1 => use the thresholds obtained via divisive normalization module (its low-pass filter followed by scaling to be more precise).
            spike_rate_scale_bitshift1 (np.ndarray): C-dim array specifying how much bitshift should be applied to scale the normalized spike rate down to a target value.
            spike_rate_scale_bitshift2 (np.ndarray): C-dim array specifying how much bitshift should be applied to scale the normalized spike rate down to a target value.
                NOTE: without any bitshift the spike rate will be around sampling rate of the input signal in each channel, so quite large.
                With `b1, b2` bit scaling, the spike rate goes down by a factor 2^b1 - 2^b2.
            low_pass_bitshift (np.ndarray): C-dim array containing the bitshifts used in low-pass filter for average power estimation and threshold computation in DN module.
                NOTE: a bitshift of size `b` amount to an averaging window of `2^b` samples or a physical averaging time of `2^b/fs` where fs is the sampling rate of input audio.
            EPS_vec (np.ndarray): C-dim array containing lower bound on the threshold in DN module.
            fixed_threshold_vec (np.ndarray): C-dimm array containing the fixed thresholds used for DN.
                NOTE: these thresholds are used only when spike generation `mode` in mode_vec is set to 0.
            joint_normalization (bool): if normalization is going to be applied separately or jointly across channels. Defaults to True.
            record (bool, optional) if the inner states need to be recorded. Defaults to False.

        Returns:
            Tuple[np.ndarray, Dict[np.ndarray]]: generated spikes and a dictionary containing the states during the simulation.
        """
        T, num_channels = sig_in.shape

        # change the parameters:
        #   (i)     convert them into int
        #   (ii)    make them time x channel so that they fit in jax compilation model

        sig_in = jnp.asarray(sig_in, dtype=jnp.int64)

        mode_vec = jnp.asarray(mode_vec, dtype=jnp.int64)
        EPS_vec = jnp.asarray(EPS_vec, dtype=jnp.int64)
        fixed_threshold_vec = jnp.asarray(fixed_threshold_vec, dtype=jnp.int64)
        low_pass_bitshift = jnp.asarray(low_pass_bitshift, dtype=jnp.int64)
        spike_rate_scale_bitshift1 = jnp.asarray(
            spike_rate_scale_bitshift1, dtype=jnp.int64
        )
        spike_rate_scale_bitshift2 = jnp.asarray(
            spike_rate_scale_bitshift2, dtype=jnp.int64
        )

        # jit compiled function
        # we have two compiled version: (i) when states are not needed, (ii) when states also need to be returned
        @partial(jax.jit, static_argnums=(1,))
        def _compiled_spike_gen(sig_in: jnp.ndarray, record: bool = False):
            # jax state-evolution map for generating the spikes
            def forward(
                state_in_tuple: jnp.ndarray, data_in_tuple: List[jnp.ndarray]
            ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
                """state evolution equation for spike generation in divisive normalization module

                Args:
                    state_in_tuple (List[jnp.ndarray]): a list of arrays containing the states [high-res-filter, low-res-filter, IAF-state]
                    data_in (List[jnp.ndarray]): input array containing the input [rectified signal, threshold values, EPS values, threshold mode]

                Returns:
                    List[jnp.ndarray]: next state and output spikes.
                """
                # decompose the states
                state_high_res_filter, state_low_res_filter, state_IAF = state_in_tuple

                # decompose the input
                # sig_in_rect, fixed_threshold_vec, eps_vec, mode_vec = data_in_tuple
                (sig_in_rect,) = data_in_tuple

                # update the states
                state_low_res_filter = state_high_res_filter >> low_pass_bitshift
                state_high_res_filter = (
                    state_high_res_filter - state_low_res_filter + sig_in_rect
                )

                ### set the threshold for DN module
                adpative_thresholds = state_low_res_filter

                # put a lower bound on the thresholds using EPS
                adpative_thresholds = (
                    jnp.abs(adpative_thresholds - EPS_vec)
                    + adpative_thresholds
                    + EPS_vec
                ) >> 1

                # scale the thresholds based on the target spike rate scaling parameters
                # NOTE: here, as we explained, we use 2 bitshifts to obtains a smoother threshold and spike rate scaling
                adpative_thresholds = (
                    adpative_thresholds << spike_rate_scale_bitshift1
                ) - (adpative_thresholds << spike_rate_scale_bitshift2)

                # decide on separate or joint scaling of the spike rate across channels
                if joint_normalization:
                    adpative_thresholds = jnp.max(adpative_thresholds) * jnp.ones_like(
                        adpative_thresholds
                    )

                # decide on the final threshold: adaptive via DN or fixed thresholds
                applied_thresholds = (
                    1 - mode_vec
                ) * fixed_threshold_vec + mode_vec * adpative_thresholds

                # update the state of IAF neuron and produce spikes (accoridng to Moorley model)
                state_IAF += sig_in_rect
                spikes = jnp.where(state_IAF >= applied_thresholds, 1, 0)

                state_IAF -= spikes * applied_thresholds

                # copy the states
                state_out_tuple = (
                    state_high_res_filter,
                    state_low_res_filter,
                    state_IAF,
                )

                # copy the output
                # NOTE: here we have the option to include all the states we are interested in the output
                if record:
                    # return all the simulation states
                    data_out_tuple = (
                        spikes,
                        state_IAF,
                        applied_thresholds,
                        state_high_res_filter,
                        state_low_res_filter,
                    )
                else:
                    # return only spikes
                    data_out_tuple = (spikes,)

                return state_out_tuple, data_out_tuple

            # use forward state update to simulate DN and spike generation
            initial_state = (
                jnp.zeros(num_channels, dtype=jnp.int64),
                jnp.zeros(num_channels, dtype=jnp.int64),
                jnp.zeros(num_channels, dtype=jnp.int64),
            )
            sig_in_rect = jnp.abs(sig_in)

            # sig_in, threshold, eps, mode
            # data_in = (sig_in_rect, fixed_threshold_vec, EPS_vec, mode_vec)
            data_in_tuple = (sig_in_rect,)

            if record:
                final_state_tuple, (
                    spikes,
                    state_IAF,
                    adpative_thresholds,
                    state_high_res_filter,
                    state_low_res_filter,
                ) = jax.lax.scan(forward, initial_state, data_in_tuple)
                return spikes, (
                    state_IAF,
                    adpative_thresholds,
                    state_high_res_filter,
                    state_low_res_filter,
                )
            else:
                final_state_tuple, (spikes,) = jax.lax.scan(
                    forward, initial_state, data_in_tuple
                )
                return spikes

        if record:
            spikes, (
                state_IAF,
                applied_thresholds,
                state_high_res_filter,
                state_low_res_filter,
            ) = _compiled_spike_gen(sig_in=sig_in, record=record)

            recording = {
                "state_IAF": state_IAF,
                "state_high_res_filter": state_high_res_filter,
                "state_low_res_filter": state_low_res_filter,
                "spike_gen_thresholds": applied_thresholds,
                "threshold-mode": mode_vec,
                "fixed_thresholds": fixed_threshold_vec,
            }

            # convert to python version
            spikes = np.asarray(spikes)
            for key in recording.keys():
                recording[key] = np.asarray(recording[key])

        else:
            spikes = _compiled_spike_gen(
                sig_in=sig_in,
                record=record,
            )

            spikes = np.asarray(spikes)

            recording = {}

        return spikes, recording

    # NOTE: this function implements the floating point 32-bit version of spike generation. It may have slight imprecision but should work well for all usecases in XyloAudio 3.
    def fjax_spike_gen(
        sig_in: np.ndarray,
        mode_vec: np.ndarray,
        spike_rate_scale_bitshift1: np.ndarray,
        spike_rate_scale_bitshift2: np.ndarray,
        low_pass_bitshift: np.ndarray,
        EPS_vec: np.ndarray,
        fixed_threshold_vec: np.ndarray,
        joint_normalization: bool = False,
        record: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """this module processes the input signal in various channels and converts the into spikes via joint divisive normalization.

        Args:
            sig_in (np.ndarray): input signal of dimension `T x C` where C is the number of channels.
            mode_vec (np.ndarray): C-dim array of {0,1} showing the mode of spikes generation at each channel.
                0 => use the fixed thresholds
                1 => use the thresholds obtained via divisive normalization module (its low-pass filter followed by scaling to be more precise).
            spike_rate_scale_bitshift1 (np.ndarray): C-dim array specifying how much bitshift should be applied to scale the normalized spike rate down to a target value.
            spike_rate_scale_bitshift2 (np.ndarray): C-dim array specifying how much bitshift should be applied to scale the normalized spike rate down to a target value.
                NOTE: without any bitshift the spike rate will be around sampling rate of the input signal in each channel, so quite large.
                With `b` bit scaling, the spike rate goes down by a factor 2^b.
            low_pass_bitshift (np.ndarray): C-dim array containing the bitshifts used in low-pass filter for average power estimation and threshold computation in DN module.
                NOTE: a bitshift of size `b` amount to an averaging window of `2^b` samples or a physical averaging time of `2^b/fs` where fs is the sampling rate of input audio.
            EPS_vec (np.ndarray): C-dim array containing lower bound on the threshold in DN module.
            fixed_threshold_vec (np.ndarray): C-dimm array containing the fixed thresholds used for DN.
                NOTE: these thresholds are used only when spike generation `mode` in mode_vec is set to 0.
            joint_normalization (bool): separate (per-channel) or joint scaling of spike rates. Defaults to True.
                NOTE: in DN, we have the option to normalize each channel spike rate separately or all spike channels jointly. In separate case, the relative rate ratio between
                various channel is not preserved in general.
            record (bool, optional) if the inner states need to be recorded. Defaults to False.

        Returns:
            Tuple[np.ndarray, Dict[np.ndarray]]: generated spikes and a dictionary containing the states during the simulation.
        """
        T, num_channels = sig_in.shape

        # change the parameters:
        #   (i)     convert them into int
        #   (ii)    make them time x channel so that they fit in jax compilation model

        sig_in = jnp.asarray(sig_in, dtype=jnp.float32)
        # mode_vec = jnp.asarray(np.ones((T,1)) * mode_vec.reshape(1,-1), dtype=jnp.int64)
        # EPS_vec = jnp.asarray(np.ones((T,1)) * (EPS_vec.reshape(1,-1)), dtype=jnp.int64)
        # fixed_threshold_vec = jnp.asarray(np.ones((T,1)) * fixed_threshold_vec.reshape(1,-1), dtype=jnp.int64)
        mode_vec = jnp.asarray(mode_vec, dtype=jnp.float32)
        EPS_vec = jnp.asarray(EPS_vec, dtype=jnp.float32)
        fixed_threshold_vec = jnp.asarray(fixed_threshold_vec, dtype=jnp.float32)

        # jit compiled function
        # we have two compiled version: (i) when states are not needed, (ii) when states also need to be returned
        @partial(jax.jit, static_argnums=(1,))
        def _compiled_spike_gen(sig_in: jnp.ndarray, record: bool = False):
            # jax state-evolution map for generating the spikes
            def forward(
                state_in_tuple: jnp.ndarray, data_in_tuple: List[jnp.ndarray]
            ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
                """state evolution equation for spike generation in divisive normalization module

                Args:
                    state_in_tuple (List[jnp.ndarray]): a list of arrays containing the states [high-res-filter, low-res-filter, IAF-state]
                    data_in (List[jnp.ndarray]): input array containing the input [rectified signal, threshold values, EPS values, threshold mode]

                Returns:
                    List[jnp.ndarray]: next state and output spikes.
                """
                # decompose the states
                state_high_res_filter, state_low_res_filter, state_IAF = state_in_tuple

                # decompose the input
                (sig_in_rect,) = data_in_tuple

                # update the states: truncate
                state_low_res_filter = jnp.fix(
                    state_high_res_filter / (2**low_pass_bitshift)
                )
                state_high_res_filter = (
                    state_high_res_filter - state_low_res_filter + sig_in_rect
                )

                ### set the threshold for DN module
                adpative_thresholds = state_low_res_filter

                # put a lower bound on the thresholds using EPS
                adpative_threshold_vec = jnp.round(
                    (
                        jnp.abs(adpative_thresholds - EPS_vec)
                        + adpative_thresholds
                        + EPS_vec
                    )
                    / 2.0
                )

                # scale the thresholds based on the target spike rate scaling parameter
                adpative_threshold_vec = adpative_threshold_vec * (
                    2**spike_rate_scale_bitshift1 - 2**spike_rate_scale_bitshift2
                )

                # decide on separate of joint spike normalization across all channels
                if joint_normalization:
                    adpative_threshold_vec = jnp.max(
                        adpative_threshold_vec
                    ) * jnp.ones_like(adpative_threshold_vec)

                # decide on the final threshold: adaptive via DN or fixed thresholds
                applied_thresholds = (
                    1 - mode_vec
                ) * fixed_threshold_vec + mode_vec * adpative_threshold_vec

                # update the state of IAF neuron and produce spikes (accoridng to Moorley model)
                state_IAF += sig_in_rect
                spikes = jnp.where(state_IAF >= applied_thresholds, 1, 0)

                state_IAF -= spikes * applied_thresholds

                # copy the states
                state_out_tuple = (
                    state_high_res_filter,
                    state_low_res_filter,
                    state_IAF,
                )

                # copy the output
                # NOTE: here we have the option to include all the states we are interested in the output
                if record:
                    # return all the simulation states
                    data_out_tuple = (
                        spikes,
                        state_IAF,
                        applied_thresholds,
                        state_high_res_filter,
                        state_low_res_filter,
                    )
                else:
                    # return only spikes
                    data_out_tuple = (spikes,)

                return state_out_tuple, data_out_tuple

            # use forward state update to simulate DN and spike generation
            initial_state = (
                jnp.zeros(num_channels, dtype=jnp.float32),
                jnp.zeros(num_channels, dtype=jnp.float32),
                jnp.zeros(num_channels, dtype=jnp.float32),
            )
            sig_in_rect = jnp.abs(sig_in)

            # sig_in, threshold, eps, mode
            # data_in = (sig_in_rect, fixed_threshold_vec, EPS_vec, mode_vec)
            data_in_tuple = (sig_in_rect,)

            if record:
                final_state_tuple, (
                    spikes,
                    state_IAF,
                    adpative_thresholds,
                    state_high_res_filter,
                    state_low_res_filter,
                ) = jax.lax.scan(forward, initial_state, data_in_tuple)
                return spikes, (
                    state_IAF,
                    adpative_thresholds,
                    state_high_res_filter,
                    state_low_res_filter,
                )
            else:
                final_state_tuple, (spikes,) = jax.lax.scan(
                    forward, initial_state, data_in_tuple
                )
                return spikes

        if record:
            spikes, (
                state_IAF,
                applied_thresholds,
                state_high_res_filter,
                state_low_res_filter,
            ) = _compiled_spike_gen(sig_in=sig_in, record=record)

            recording = {
                "state_IAF": state_IAF,
                "state_high_res_filter": state_high_res_filter,
                "state_low_res_filter": state_low_res_filter,
                "spike_gen_thresholds": applied_thresholds,
                "threshold-mode": mode_vec,
                "fixed_thresholds": fixed_threshold_vec,
            }

            # convert to python version
            spikes = np.asarray(spikes, dtype=np.int64)
            for key in recording.keys():
                recording[key] = np.asarray(recording[key], dtype=np.int64)

        else:
            spikes = _compiled_spike_gen(
                sig_in=sig_in,
                record=record,
            )

            spikes = np.asarray(spikes, dtype=np.int64)

            recording = {}

        return spikes, recording

    # set flag for jax version
    JAX_SPIKE_GEN = True


except ModuleNotFoundError as e:
    info(
        f"jax module was not found! Using python version for DN and spike generation.\n{e}\n",
    )
    JAX_SPIKE_GEN = False


# we always implement the python version
def py_spike_gen(
    sig_in: np.ndarray,
    mode_vec: np.ndarray,
    spike_rate_scale_bitshift1: np.ndarray,
    spike_rate_scale_bitshift2: np.ndarray,
    low_pass_bitshift: np.ndarray,
    EPS_vec: np.ndarray,
    fixed_threshold_vec: np.ndarray,
    joint_normalization: bool = False,
    record: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """this module processes the input signal in various channels and converts the into spikes via joint divisive normalization.

    Args:
        sig_in (np.ndarray): input signal of dimension `T x C` where C is the number of channels.
        mode_vec (np.ndarray): `T x C` 0-1 matrix showing the mode of spikes generation at each channel at each instant:
            0 => use the fixed thresholds
            1 => use the thresholds obtained via divisive normalization module (its low-pass filter to be more precise).
        spike_rate_scale_bitshift1 (np.ndarray): C-dim array specifying how much bitshift should be applied to scale the normalized spike rate down to a target value.
        spike_rate_scale_bitshift2 (np.ndarray): C-dim array specifying how much bitshift should be applied to scale the normalized spike rate down to a target value.
            NOTE: without any bitshift the spike rate will be around sampling rate of the input signal in each channel, so quite large.
            With `b` bit scaling, the spike rate goes down by a factor 2^b.
        low_pass_bitshift (np.ndarray): C-dim array containing the bitshifts used in low-pass filter for average power estimation and threshold computation in DN module.
            NOTE: a bitshift of size `b` amount to an averaging window of `2^b` samples or a physical averaging time of `2^b/fs` where fs is the sampling rate of input audio.
        EPS_vec (np.ndarray): C-dim array containing lower bound on the threshold in DN module.
        fixed_threshold_vec (np.ndarray): C-dimm array containing the fixed thresholds used for DN.
            NOTE: these thresholds are used only when spike generation `mode` in mode_vec is set to 0.
        joint_normalization (bool): if normalization is going to be applied separately (per-channel) or jointly across channels. Defaults to True.
        record (bool, optional) if the inner states need to be recorded. Defaults to False.

    Returns:
        Tuple[np.ndarray, Dict[np.ndarray]]: generated spikes and a dictionary containing the states during the simulation.
    """
    # check if there is a single channel
    if sig_in.ndim == 1:
        sig_in = sig_in.reshape(1, -1)

    T, num_channels = sig_in.shape

    # input signal after full-wave rectification
    sig_in_rect = np.abs(sig_in)

    state_IAF = np.zeros(num_channels, dtype=np.int64)
    state_high_res_filter = np.zeros(num_channels, dtype=np.int64)
    state_low_res_filter = np.zeros(num_channels, dtype=np.int64)
    adaptive_threshold_vec = np.zeros(num_channels, dtype=np.int64)

    spikes = np.zeros(num_channels, dtype=np.int64)

    spikes_list = []

    if record:
        state_IAF_list = []
        state_high_res_filter_list = []
        state_low_res_filter_list = []
        applied_thresholds = []

    for sig_rect_sample in sig_in_rect:
        # NOTE: in the following operations, ORDER is very important otherwise the computed results will be different
        # the spikes are produced according to Morely (rather than Moore) state machine model.

        # low-resolution filter state
        state_low_res_filter = state_high_res_filter >> low_pass_bitshift.astype(
            np.int64
        )

        # update the state of high-res filter for next clock
        state_high_res_filter = (
            state_high_res_filter
            - (state_high_res_filter >> low_pass_bitshift.astype(np.int64))
            + sig_rect_sample
        )

        # compute the thresholds that need to be applied
        adaptive_threshold_vec[:] = state_low_res_filter[:]
        adaptive_threshold_vec[adaptive_threshold_vec < EPS_vec] = EPS_vec[
            adaptive_threshold_vec < EPS_vec
        ]
        adaptive_threshold_vec = (
            adaptive_threshold_vec << spike_rate_scale_bitshift1.astype(np.int64)
        ) - (adaptive_threshold_vec << spike_rate_scale_bitshift2.astype(np.int64))

        # decide on separate (per-channel) or joint spike rate normalization across channels
        if joint_normalization:
            adaptive_threshold_vec = np.max(adaptive_threshold_vec) * np.ones_like(
                adaptive_threshold_vec
            )

        # choose the threshold value based on the operation `mode`
        #       mode = 0 -> fixed thresholds are used in spike generations.
        #       mode = 1 -> adaptive thresholds computed by divisive normalization are used in spike generation.
        current_thresholds = (
            mode_vec * adaptive_threshold_vec + (1 - mode_vec) * fixed_threshold_vec
        )

        # state of the IAF -> spike -> state of IAF
        state_IAF[:] = state_IAF[:] + sig_rect_sample

        spikes[:] = (state_IAF >= current_thresholds).astype(np.int8).astype(object)

        state_IAF[:] -= spikes * current_thresholds

        # store the states and spikes: save by copy
        spikes_list.append(spikes.copy())

        if record:
            state_IAF_list.append(state_IAF.copy())
            state_high_res_filter_list.append(state_high_res_filter.copy())
            state_low_res_filter_list.append(state_low_res_filter.copy())
            applied_thresholds.append(current_thresholds.copy())

    # convert into numpy
    # squeeze in case there is only a single channel
    spikes_list = np.asarray(spikes_list).squeeze()

    if record:
        state_IAF_list = np.asarray(state_IAF_list, dtype=np.int64).squeeze()
        state_high_res_filter_list = np.asarray(
            state_high_res_filter_list, dtype=np.int64
        ).squeeze()
        state_low_res_filter_list = np.asarray(
            state_low_res_filter_list, dtype=np.int64
        ).squeeze()
        applied_thresholds = np.asarray(applied_thresholds, dtype=np.int64).squeeze()

        recording = {
            "state_IAF": state_IAF_list,
            "state_high_res_filter": state_high_res_filter_list,
            "state_low_res_filter": state_low_res_filter_list,
            "spike_gen_thresholds": applied_thresholds,
            "threshold-mode": mode_vec,
            "fixed_thresholds": fixed_threshold_vec,
        }
    else:
        recording = {}

    return spikes_list, recording
