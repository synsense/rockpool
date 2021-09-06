from rockpool.nn.modules.module import Module

from rockpool import TSEvent

from rockpool.typehints import P_int, P_float

from rockpool.parameters import Parameter, State, SimulationParameter

import numpy as np

from tqdm.autonotebook import tqdm, trange


class DivisiveNormalisation(Module):
    def __init__(
        self,
        shape: tuple,
        n_p: int = 4,
        n_s: int = 5,
        n_e: int = 1,
        bits_counter: int = 10,
        bits_lowpass: int = 16,
        frame_dt: float = 0.1e-3,
        fs: float = 44e3,
    ):
        super().__init__(shape, spiking_input=True, spiking_output=True)

        self.counter: P_int = State(np.zeros(self.size_in, "uint"))
        self.fs: P_float = SimulationParameter(fs)
        self.n_p: P_int = SimulationParameter(n_p)
        self.n_s: P_int = SimulationParameter(n_s)
        self.n_e: P_int = SimulationParameter(n_e)
        self.bits_counter: P_int = SimulationParameter(bits_counter)
        self.bits_lowpass: P_int = SimulationParameter(bits_lowpass)
        self.frame_dt: P_float = SimulationParameter(frame_dt)

    def evolve(
        self, input: np.ndarray, record: bool = False
    ) -> (np.ndarray, dict, dict):
        # - Convert input spikes to frames -> E(t)
        # - input : (T, Nin) -> T is units of `dt`
        # - E: (T_frames, Nin) -> units of `frame_dt`
        ts_input = TSEvent.from_raster(input, dt=1 / self.fs)
        E = ts_input.raster(dt=self.frame_dt, add_events=True)
        E = np.clip(E, 0, 2 ** self.bits_counter).astype("uint")

        # - Perform low-pass filter -> M(t)
        # - M: (T_frame, Nin) -> units of `frame_dt`
        #  M(t) = s * E(t) + (1-s) M(t-1)

        M = np.zeros((E.shape[0] + 1, E.shape[1])).astype("uint")
        M[0, :] = self.counter
        for t in trange(E.shape[0]):
            # M[t+1, :] = (E[t, :] >> int(self.n_s) + M[t, :] - M[t, :] >> int(self.n_s))
            M[t + 1, :] = (E[t, :] + (M[t, :] << int(self.n_s)) - M[t, :]) >> int(
                self.n_s
            )

        M = M[1:, :]

        # counter = np.zeros((E.shape[0] + 1, self.size_out), "uint")
        # counter[0, :] = self.counter

        # M = E << int(self.n_s) + counter[:-1] >> int(self.n_s)
        M = np.clip(M, 0, 2 ** self.bits_lowpass).astype("uint")

        # - Generate Poisson spike train -> S(t)
        # - S: (T_frame * 2**self.bits_counter, Nin)
        # noise = np.random.uniform(
        #     size=(M.shape[0] * 2 ** self.bits_counter, self.size_out)
        # )
        # *** PROBLEM: Noise should be independent for each channel ***
        E_wide = np.repeat(E, 2 ** self.bits_counter, axis=0)
        noise = np.tile(
            np.random.uniform(size=(E_wide.shape[0], 1)), (1, self.size_out),
        )

        S = E_wide > noise

        # - Multiply poisson spike train frequency by p
        S_p = np.repeat(S, 2 ** self.n_p, axis=0)

        # - IAF: Threshold by M(t) to produce output spike train
        M_wide = np.repeat(M, 2 ** self.bits_counter * 2 ** self.n_p, axis=0).astype(
            "uint"
        )
        IAF_state = np.cumsum(S_p, axis=0).astype("uint")

        output = np.zeros(M_wide.shape)
        for t in trange(M_wide.shape[0]):
            # - Find channels that spike this time step
            spikes_t = IAF_state[t, :] > M_wide[t, :]

            # - Reset IAF channels that spike
            IAF_state[(t + 1) :, spikes_t] -= M_wide[t, spikes_t]

            # - Store spikes in output
            output[t, :] = spikes_t

        # - Generate state record dictionary
        record_dict = (
            {
                "E": E,
                "S": S,
                "M": M,
                "IAF_state": IAF_state,
                "E_wide": E_wide,
                "M_wide": M_wide,
            }
            if record
            else {}
        )

        self.counter = E[-1, :]

        return output, self.state(), record_dict
