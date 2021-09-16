# -----------------------------------------------------------
# simulates divisive normalization for a system with input spikes
# coming in possibly several channels
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
# finalized and tested on 13.09.2021
# -----------------------------------------------------------


from rockpool.timeseries import TSEvent
from rockpool.nn.modules.module import Module
from rockpool.typehints import P_int, P_float, P_ndarray
from rockpool.parameters import State, SimulationParameter

import numpy as np

import warnings

import imp
import pathlib as pl

basedir = pl.Path(imp.find_module("rockpool")[1]) / "devices" / "xylo"


def build_lfsr(filename) -> np.ndarray:
    """
    This function reads the LFSR code in a binary format from the file "filename"
    It return a Numpy array containing the integer values of LFSR code.
    """
    # read the LFSR state and build the pseudo-random code
    with open(filename) as f:
        lines = f.readlines()

    # for some reason state 'all-zero' is included in the file but
    # LFSR state cannot be all-zero because then the next state also would be
    # all-zero
    code_lfsr = np.zeros(len(lines), dtype="int")

    for i in range(len(lines) - 1):
        code_lfsr[i] = int(lines[i], 2)

    # remove the first element corresponding to the all-zero state
    code_lfsr = code_lfsr[:-1] + 1

    return code_lfsr


class DivisiveNormalisation(Module):
    """
    A digital divisive normalization block


    """

    def __init__(
        self,
        shape: tuple = 1,
        *args,
        bits_counter: int = 10,
        E_frame_counter: np.ndarray = None,
        IAF_counter: np.ndarray = None,
        bits_lowpass: int = 16,
        bits_shift_lowpass: int = 5,
        M_lowpass_state: int = None,
        dt: P_float = 0.1e-3,
        frame_dt: float = 50e-3,
        bits_lfsr: int = 10,
        code_lfsr: np.ndarray = None,
        p_local: int = 12,
        **kwargs,
    ):
        """

        Args:
            shape (tuple): The number of channels ``N`` for this Module
            bits_counter (int): Bit-width of frame counter E. Defualt: ``10``
            E_frame_counter (np.ndarray): Initialisation for frame counter E ``(N,)``. Default: ``None``, initialise to zero.
            IAF_counter (np.ndarray): Initialisation for IAF state ``(N,)``. Defualt: ``None``, initialise to zero.
            bits_lowpass (int): Number of bits used by the low-pass filter ``M``. Default: ``16``
            bits_shift_lowpass (int): Dash bitshift averaging parameter to use in low-pass filter. Default: ``5``
            M_lowpass_state (np.ndarray): Initialisation for low-pass state ``(N,)``. Default: ``None``, initialise to zero.
            dt (float): Global clock step in seconds. Default: 0.1ms
            frame_dt (float): Frame clock step in seconds. Default: 50ms
            bits_lfsr (int): Bit-width of LFSR. Default: ``10``
            code_lfsr (np.ndarray): LFSR sequence to use. **Must be supplied**
            p_local (int): Factor to multiply spike rate E. Factor ``p`` is given by ``p_local / 2``.
        """

        # intialize the Module superclass
        super().__init__(
            shape, spiking_input=True, spiking_output=True, *args, **kwargs
        )

        # initialize the value of the counter or set to zero if not specified
        self.E_frame_counter: P_ndarray = State(
            E_frame_counter,
            shape=(self.size_in),
            init_func=lambda s: np.zeros(s, "int"),
        )
        """ np.ndarray: Spike rate per frame ``(N,)`` """

        self.bits_counter: P_int = SimulationParameter(bits_counter)
        """ int: Number of bits of spike rate counter E """

        # initialize the state of IAF_counter
        self.IAF_counter: P_ndarray = State(
            IAF_counter, shape=(self.size_in), init_func=lambda s: np.zeros(s, "int")
        )
        """ np.ndarray: IAF state ``(N,)`` """

        # set the parameters of the low-pass filter (implemented by bit-shifts)
        self.bits_lowpass: P_int = SimulationParameter(bits_lowpass)
        """ int: Number of bits in low-pass filter state M """

        self.bits_shift_lowpass: P_int = SimulationParameter(bits_shift_lowpass)
        """ int: Dash decay parameter for low-pass filter M """

        # at the moment we implement averaging filter such that its value is
        # always a positive integer. We can do much better by using some
        # fixed-point implementation by using some extra bits as decimals
        # this is left for future
        self.M_lowpass_state: P_ndarray = State(
            M_lowpass_state,
            shape=(self.size_in),
            init_func=lambda s: np.zeros(s, "int"),
        )
        """ np.ndarray: Low-pass filter state M ``(N,)`` """

        ##
        # set the global clock frequency
        self.dt: P_float = SimulationParameter(dt)
        """ float: Global clock step in seconds """

        # set the period of the frame
        self.frame_dt: P_float = SimulationParameter(frame_dt)
        """ float: Frame clock step in seconds """

        # set the number of bits and also code for LFSR
        self.bits_lfsr: P_int = SimulationParameter(bits_lfsr)
        """ int: Number of LFSR bits """

        if code_lfsr is None:
            # - Load default LFSR sequence
            code_lfsr = build_lfsr(basedir / "lfsr_data.txt")

        if code_lfsr.size != 2 ** self.bits_lfsr - 1:
            raise ValueError(
                f"Length of LFSR is not compatible with its number of bits. Expected {2 ** self.bits_lfsr - 1} entries, found {code_lfsr.size}."
            )

        self.code_lfsr: P_ndarray = SimulationParameter(
            np.copy(code_lfsr).reshape((-1,))
        )
        """ np.ndarray: LFSR sequence to use for pRNG """

        # set the ratio between the rate of local and global clocks
        # note that because of return-to-zero pulses, the spike rate increases
        # by only p_local/2 -> set p_local to be an even number
        self.p_local: P_int = int((1 + p_local) / 2) * 2
        """ int: Factor to scale up internal spike generation."""

        if self.p_local != p_local:
            warnings.warn(f"`p_local` = {p_local} was rounded to an even integer!")

    def evolve(
        self, input_spike: np.ndarray, record: bool = False
    ) -> (np.ndarray, dict):
        """
        This class simulates divisive normalization for an input spike signal
        with possibly several channels.
        The output of the simulation is another spike signal with normalized rates.
        """
        # check the dimensionality first
        if input_spike.shape[1] != self.size_in:
            raise ValueError(
                f"Input size {input_spike.shape} did not match number of channels {self.size_in}"
            )

        # - Convert input spikes with duration 'dt' to frames of duration 'frame_dt'-> counter output
        # - output is counter output E(t) of duration 'frame_dt'
        # - input : (N, self.size_in) -> N is units of 'dt'
        # - E: (n_frame, self.size_in) -> units of 'frame_dt'
        ts_input = TSEvent.from_raster(input_spike, dt=self.dt)
        E = ts_input.raster(dt=self.frame_dt, add_events=True)

        num_frames = E.shape[0]
        num_input_steps = input_spike.shape[0]

        # add the effect of initial values in E_frame_counter
        E[0, :] += self.E_frame_counter.astype(int)

        # clip the counter to take the limited number of bits into account
        E = np.clip(E, 0, 2 ** self.bits_counter)

        # register the value of E_frame_counter
        self.E_frame_counter = E[-1, :]

        # Perform low-pass filter on E(t)-> M(t)
        # M(t) = s * E(t) + (1-s) M(t-1)
        #  with s=1/2**bits_shift_lowpass
        # M: (n_frame, self.size_in) -> units of 'frame_dt'

        M = np.zeros((num_frames + 1, self.size_in), dtype="int")

        # load the initialization of the filter
        M[0, :] = self.M_lowpass_state

        # we have implemented the improved version of filer to avoid truncation
        # this requires some additional bits for M
        for t in range(num_frames):
            # current implementation with truncation issue
            # M[t+1, :] = (E[t, :] >> int(self.bits_shift_lowpass) + M[t, :] - M[t, :] >> int(self.bits_shift_lowpass))

            # the improved version we have adopted here
            M[t + 1, :] = (
                E[t, :] + (M[t, :] << self.bits_shift_lowpass) - M[t, :]
            ) >> self.bits_shift_lowpass

        M = M[1:, :]

        # take the limited number of counter bits into account
        # we should make sure that teh controller does not allow count-back to zero
        # and keeps the value of the counter at its maximum
        M = np.clip(M, 0, 2 ** self.bits_lowpass - 1)

        # use the value of E(t) at each frame t to produce a pseudo-random
        # Poisson spike train using LFSR
        # as the value of LFSR varies with global clock rate f_s, we have 'frame_dt*f_s'
        # samples in each frame
        # the timing of the output in in units of 'dt'

        # Number of global clock cycles within a frame period
        cycles_per_frame = int(np.ceil(self.frame_dt / self.dt))

        # maximum number of spikes at the output
        # max_output_spikes = int(cycles_per_frame * E.shape[0] * self.p_local / 2)

        # initialize the matrix of output spikes in all channels
        # output_spike = np.zeros((max_output_spikes, E.shape[1]))

        # compute the specific slice of LFSR code used for comparison among different frames
        # find the total number of comparisons between all E(t) and LFSR output
        # this is given by the total number of spikes
        lfsr_ticks_needed = cycles_per_frame * num_frames

        # number of LFSR periods needed
        num_lfsr_period = int(np.ceil(lfsr_ticks_needed / self.code_lfsr.size))

        code_lfsr_frame = np.tile(self.code_lfsr, num_lfsr_period)[
            :lfsr_ticks_needed
        ].reshape(E.shape[0], -1)

        # initialize the IAF_state for further inspection
        if record:
            IAF_state_saved = [[]] * self.size_in

        # length of expanded frame after expansion of spikes and zero-pad
        len_frame_local = int(cycles_per_frame * self.p_local / 2)

        output_spike_times = [[]] * self.size_in
        output_spike_channels = [[]] * self.size_in

        # perform operation per channel
        for ch in range(self.size_in):
            # for each channel
            E_ch = np.copy(E[:, ch])
            M_ch = np.copy(M[:, ch])

            # repeat Each E(t) 'max_spike_in_frame' times
            E_ch = E_ch.reshape(num_frames, 1)  # column-vec
            E_ch_rep = np.tile(E_ch, (1, cycles_per_frame))

            # Spike train generated by SG: each row contains spikes generated in a specific frame
            # units of time are 'dt'
            S_sg = np.int_(E_ch_rep >= code_lfsr_frame)

            # multiply the frequency of spikes by a factor p_local/2
            # (i) unwrap all frames in time (column vec)
            # (ii) repeat along column to simulate the effect of local clock
            # (iii) zero-pad each pulse to take the return-to-zero pulse shape into account
            # (iv) wrap again to have the expanded frame in a row

            # (i)-(ii)
            # repeat the spikes in each frame by a factor p_local/2
            # note that due to return-to-zero pulse, each spike needs to be zero padded at the output
            # each row in the following array contains a pulse produced by SG expanded by p_local
            # by the local clock generator
            S_local_before_pad = np.tile(
                S_sg.reshape(S_sg.size, 1), (1, int(self.p_local / 2))
            )

            # (iii)
            # now zero-pad each row containing a pulse modulated by local clock
            # although the results are 'uint', we use 'int' to avoid unsigned difference issue
            # S_local = np.zeros(
            #     (S_local_before_pad.shape[0], 2 * S_local_before_pad.shape[1]),
            #     dtype="int",
            # )
            # S_local[:, : S_local_before_pad.shape[1]] = S_local_before_pad

            # (iv)
            # now reshape S_local so that pulses corresponding to a specific frame are in the same row
            S_local = S_local_before_pad.reshape(num_frames, len_frame_local)

            # apply IAF with threshold M(t) at each frame t (each row of S_local)
            # due to surplus from frame t-> t+1, we need to do this frame by frame
            # output_spike_ch = np.zeros(S_local.shape, dtype="int")

            # initialize the state of the corresponding counter
            # res_from_previous_frame = self.IAF_counter[ch]

            for t in range(num_frames):
                # find the largest integer less than the floating-value threshold M(t)
                # this way of thresholding works because the IAF is implemented by counter
                # so, we need to set the value of threshold to be an integer value
                # some care is needed when M(t)<1 because then IAF fires everytime a
                # spike comes from the local generator
                # we solve this by simply adding the threshold by 1
                threshold = np.ceil(M_ch[t]).astype("int") + 1

                # compute the cumulative number of firings starting from residual and take mode
                IAF_state = (
                    np.concatenate(([self.IAF_counter[ch]], np.cumsum(S_local[t, :])))
                    % threshold
                )

                # IAF_state = (
                #     self.IAF_counter[ch] + np.cumsum(S_local[t, :])
                # ) % threshold

                # save if needed
                if record:
                    IAF_state_saved[ch].append(np.copy(IAF_state[1:]))

                # to find the firing times, we need to find those times for which IAF_state[t]-IAF_state[t+1]<0
                # +1 is needed because of the delay we added
                firing_times_in_frame = np.argwhere(
                    (IAF_state[1:] - IAF_state[0:-1]) < 0
                ).reshape(-1)

                # register these firing times
                # output_spike_ch[t, firing_times_in_frame] = 1
                output_spike_times[ch].append(
                    firing_times_in_frame * (self.dt / (self.p_local / 2))
                    + t * self.frame_dt
                )

                # Save the IAF state for the next frame
                self.IAF_counter[ch] = IAF_state[-1]
                # res_from_previous_frame = output_spike_ch[t, -1]

            # register the state of the IAF counter
            # self.IAF_counter[ch] = res_from_previous_frame

            # unwrap the spikes and copy it in the output_spike for the channel
            # since we are not worried about modification: no need for copy -> ravel
            # output_spike[:, ch] = output_spike_ch.ravel()

            if record:
                IAF_state_saved[ch] = np.concatenate(IAF_state_saved[ch])

            # - Build a channels list for the spikes for this channel
            output_spike_times[ch] = np.concatenate(output_spike_times[ch])
            output_spike_channels[ch] = ch * np.ones(len(output_spike_times[ch]))

        # - Sort times and channels
        output_spike_times = np.sort(np.concatenate(output_spike_times))

        # - Build output spike raster
        output_spike = TSEvent(
            np.concatenate(output_spike_times),
            np.concatenate(output_spike_channels),
            t_start=0.0,
            t_stop=num_frames * self.frame_dt,
        ).raster(self.dt, add_events=True)

        # - Generate state record dictionary
        record_dict = (
            {
                "E_frame_counter": self.E_frame_counter,
                "IAF_counter": self.IAF_counter,
                "IAF_state_saved": np.array(IAF_state_saved),
            }
            if record
            else {}
        )

        state_dict = {
            "E_frame_counter": self.E_frame_counter,
            "IAF_counter": self.IAF_counter,
            "M_lowpass_state": M,
        }

        return output_spike, {}, record_dict

    @staticmethod
    def corr_metric(spikes: np.ndarray):
        # this function compute the matrix of correlation factors
        # for the input spikes in "spikes"

        # correlation matrix
        corr_mat = (spikes.T @ spikes) / spikes.shape[0]

        # compute the average
        mean_vec = np.mean(spikes, axis=0)

        # compute the covariance matrix
        cov_mat = corr_mat - np.outer(mean_vec, mean_vec)

        # extract the diagonal elements of covariance matrix as variances
        var_vec = np.diag(cov_mat)
        var_vec.reshape(var_vec.size, 1)

        # normalize the rows and columns of covariance matrix
        # with the variances obtained so far

        # normalization matrix
        norm_mat = np.outer(np.sqrt(1 / var_vec), np.sqrt(1 / var_vec))

        # correlation factor matrix obtained by normalization
        corr_factor_mat = norm_mat * cov_mat

        return corr_factor_mat, cov_mat, mean_vec
