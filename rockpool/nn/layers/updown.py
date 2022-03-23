"""
Feedforward layer that converts each analogue input channel to one spiking up and one down channel

Runs in batch mode like UpDownTorch to save memory, but does not use pytorch. UpDownTorch seems to be slower...
"""

from typing import Optional, Union, Tuple
import json

import numpy as np

# - Local imports
from rockpool.timeseries import TSContinuous, TSEvent
from rockpool.utilities.property_arrays import ArrayLike
from rockpool.nn.layers.layer import Layer
from rockpool.nn.modules.timed_module import astimedmodule


# - Default maximum numbers of time steps for a single evolution batch
MAX_NUM_TIMESTEPS_DEFAULT = 5000

__all__ = ["FFUpDown"]

## - FFUpDown - Class: Define a spiking feedforward layer to convert analogue inputs to up and down channels
class FFUpDownV1(Layer):
    """
    Define a spiking feedforward layer to convert analogue inputs to up and down channels

    Feedforward layer that converts each analogue input channel to one spiking up and one down channel
            Runs in batch mode like `.UpDownTorch` to save memory, but does not use pytorch. `.UpDownTorch` seems to be slower...
    """

    ## - Constructor
    def __init__(
        self,
        weights: Union[int, np.ndarray, Tuple[int, int]],
        repeat_output: int = 1,
        dt: float = 0.001,
        tau_decay: Optional[Union[ArrayLike, float]] = None,
        noise_std: float = 0.0,
        thr_up: Union[ArrayLike, float] = 0.001,
        thr_down: Union[ArrayLike, float] = 0.001,
        name: str = "unnamed",
        max_num_timesteps: int = MAX_NUM_TIMESTEPS_DEFAULT,
        multiplex_spikes: bool = True,
    ):
        """
        Construct a spiking feedforward layer to convert analogue inputs to up and down channels.

        This layer is exceptional in that :py:attr:`.state` has the same size as :py:attr:`.size_in`, not :py:attr:`.size`. It corresponds to the input, inferred from the output spikes by inverting the up-/down-algorithm.

        :param np.array weights:                MxN weight matrix. Unlike other `.Layer` classes, the only important thing about weights its shape.
                                                The first dimension determines the number of input channels (self.size_in). The second dimension corresponds to size and has to be n*2*size_in, n up and n down channels for each input).
                                                If n>1 the up-/and down-spikes are distributed over multiple channels. The values of the weight matrix do not have any effect.
                                                It is also possible to pass only an integer, which will correspond to size_in.
                                                Size is then set to 2*size_in, i.e. n=1. Alternatively a tuple of two values, corresponding to size_in and n can be passed.
        :param int repeat_output:               Repeat each output spike x times.
        :param float dt:                        Time-step. Default: 0.1 ms
        :param Optional[ArrayLike] tau_decay:   The states that track the input signal for threshold comparison decay with this time constant, unless it is ``None``. Default: ``None``, do not decay tracking states.
        :param float noise_std:                 Noise std. dev. per second. Default: ``0.``, no noise
        :param ArrayLike thr_up:                Thresholds for creating up-spikes. Default: ``0.001``
        :param ArrayLike thr_down:              Thresholds for creating down-spikes. Default: ``0.001``
        :param str name:                        Name for the layer. Default: ``'unnamed'``
        :param int max_num_timesteps:           Maximum number of timesteps during single evolution batch. Longer evolution periods will automatically split in smaller batches. Default: ``MAX_NUM_TIMESTEPS_DEFAULT``
        :param bool multiplex_spikes:           If ``True``, allows a channel to emit multiple spikes per time, according to how much the corresponding threshold is exceeded. Default: ``True``, emit multiple spikes per time step
        """

        if np.size(weights) == 1:
            size_in = weights
            size = 2 * size_in

            # - Over how many output channels are the up-/down-spikes from each input distributed
            self._multi_channels = 1

        elif np.size(weights) == 2:
            # - Tuple determining shape
            (size_in, self._multi_channels) = weights
            size = 2 * self._multi_channels * size_in
        else:
            (size_in, size) = np.shape(weights)
            assert (
                size % (2 * size_in) == 0
            ), "Layer `{}`: size (here {}) must be a multiple of 2*size_in (here {}).".format(
                name, size, size_in
            )
            # - On how many output channels is the are the up-/down-spikes from each input distributed
            self._multi_channels = size / (2 * size_in)

        # - Make sure self._multi_channels is an integer
        self._multi_channels = int(self._multi_channels)

        # - Call super constructor
        super().__init__(
            weights=np.zeros((size_in, size)), dt=dt, noise_std=noise_std, name=name
        )

        # - Store layer parameters
        self.thr_up = thr_up
        self.thr_down = thr_down
        self.tau_decay = tau_decay
        self.max_num_timesteps = max_num_timesteps
        self.repeat_output = repeat_output
        self.multiplex_spikes = multiplex_spikes
        self.analog_value = np.zeros(size_in)
        self.initialized = False

        self.reset_all()

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        Evolve the states of this layer given an input

        :param Optional[TSContinuous] tsSpkInput:   Input signal
        :param Optional[float] duration:            Simulation/Evolution time
        :param Optional[int] num_timesteps:         Number of evolution time steps
        :param bool verbose:                        Currently no effect, just for conformity

        :return TSEvent:    Output spike series
        """

        # - Prepare time base
        __, inp, num_timesteps = self._prepare_input(ts_input, duration, num_timesteps)

        if self.noise_std > 0:
            # - Add noise to input
            inp += np.random.randn(*inp.shape) * self.noise_std

        # - Make sure that layer is able to represent input faithfully
        # mfInputDiff = np.diff(inp, axis=0)
        # if (
        #     ((mfInputDiff + 2 * np.abs(inp[1:]) * (1 - self._vfDecayFactor)) > self.thr_up).any()
        #     or ((mfInputDiff - 2 * np.abs(-inp[1:]) * (1 - self._vfDecayFactor)) < - self.thr_down).any()
        # ):
        #     print(
        #         "Layer `{}`: With the current settings it may not be possible".format(self.name)
        #         + " to represent the input faithfully. Consider increasing dt"
        #         + " or decreasing vtThrUp and vtTrhDown."
        #     )

        # - Matrix for collecting output spike raster
        mnOutputSpikes = np.zeros((num_timesteps, 2 * self.size_in))

        # # - Record states for debugging
        # record = np.zeros((num_timesteps, self.size_in))

        # - Iterate over batches and run evolution
        idx_curr = 0
        for matr_input_curr, num_ts_curr in self._batch_data(
            inp, num_timesteps, self.max_num_timesteps
        ):
            # (
            #     mnOutputSpikes[idx_curr : idx_curr+num_ts_curr],
            #     record[idx_curr : idx_curr+num_ts_curr]
            # ) = self._single_batch_evolution(
            mnOutputSpikes[
                idx_curr : idx_curr + num_ts_curr
            ] = self._single_batch_evolution(matr_input_curr, num_ts_curr, verbose)
            idx_curr += num_ts_curr

        ## -- Distribute output spikes over output channels by assigning to each channel
        ##    an interval of length self._multi_channels.
        # - Set each event to the first element of its corresponding interval
        vnTSSpike, spike_ids = np.nonzero(mnOutputSpikes)
        if self.multiplex_spikes:
            # - How many times is each spike repeated
            vnSpikeCounts = mnOutputSpikes[vnTSSpike, spike_ids].astype(int)
            vnTSSpike = vnTSSpike.repeat(vnSpikeCounts)
            spike_ids = spike_ids.repeat(vnSpikeCounts)
        if self.repeat_output > 1:
            # - Repeat output spikes
            spike_ids = spike_ids.repeat(self.repeat_output)
            vnTSSpike = vnTSSpike.repeat(self.repeat_output)
        if self._multi_channels > 1:
            # - Add a repeating series of (0,1,2,..,self._multi_channels) to distribute the
            #   events over the interval
            spike_ids *= self._multi_channels
            vnDistribute = np.tile(
                np.arange(self._multi_channels),
                int(np.ceil(spike_ids.size / self._multi_channels)),
            )[: spike_ids.size]
            spike_ids += vnDistribute

        # self.tsRecord = TSContinuous(self.dt * (np.arange(num_timesteps) + self._timestep), record)

        # - Start and stop times for output time series
        t_start = self._timestep * self.dt
        t_stop = (self._timestep + num_timesteps) * self.dt

        # - Output time series with events at middle of time bins
        spike_times = (vnTSSpike + self._timestep + 0.5) * self.dt
        event_out = TSEvent(
            times=np.clip(
                spike_times, t_start, t_stop
            ),  # Clip due to possible numerical errors,
            channels=spike_ids,
            num_channels=2 * self.size_in * self._multi_channels,
            name="Spikes from analogue",
            t_start=t_start,
            t_stop=t_stop,
        )

        # - Update time
        self._timestep += num_timesteps

        return event_out

    # @profile
    def _batch_data(
        self, inp: np.ndarray, num_timesteps: int, max_num_timesteps: int = None
    ) -> (np.ndarray, int):
        """_batch_data: Generator that returns the data in batches"""
        # - Handle None for max_num_timesteps
        max_num_timesteps = (
            num_timesteps if max_num_timesteps is None else max_num_timesteps
        )
        n_start = 0
        while n_start < num_timesteps:
            # - Endpoint of current batch
            n_end = min(n_start + max_num_timesteps, num_timesteps)
            # - Data for current batch
            matr_input_curr = inp[n_start:n_end]
            yield matr_input_curr, n_end - n_start
            # - Update n_start
            n_start = n_end

    # @profile
    def _single_batch_evolution(
        self, inp: np.ndarray, num_timesteps: int, verbose: bool = False
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input for a single batch

        :param np.ndarray inp:      Input
        :param int num_timesteps:   Number of evolution time steps
        :param bool verbose:        Currently no effect, just for conformity

        :return TSEvent:            Output spike series

        """

        # - Prepare local variables
        thr_up = self.thr_up
        thr_down = self.thr_down
        vfDecayFactor = self._vfDecayFactor

        # - Arrays for collecting spikes
        spike_raster = np.zeros((num_timesteps, 2 * self.size_in))

        # record = np.zeros((num_timesteps, self.size_in))

        # - Initialize state for comparing values: If self.state exists, assume input continues from
        #   previous evolution. Otherwise start with initial input data
        if not self.initialized:
            state = inp[0]
            self.initialized = True
        else:
            state = self.analog_value.copy()

        for iCurrentTS in range(num_timesteps):
            # - Decay mechanism
            state *= vfDecayFactor

            # record[iCurrentTS] = state.copy()

            if self.multiplex_spikes:
                # - By how many times are the upper thresholds exceeded for each input
                vnUp = np.clip(
                    np.floor((inp[iCurrentTS] - state) / thr_up).astype(int), 0, None
                )
                # - By how many times are the lower thresholds exceeded for each input
                vnDown = np.clip(
                    np.floor((state - inp[iCurrentTS]) / thr_down).astype(int), 0, None
                )
            else:
                # - Inputs where upper threshold is passed
                vnUp = inp[iCurrentTS] > state + thr_up
                # - Inputs where lower threshold is passed
                vnDown = inp[iCurrentTS] < state - thr_down
            # - Update state
            state += thr_up * vnUp
            state -= thr_down * vnDown
            # - Append spikes to array
            spike_raster[iCurrentTS, ::2] = vnUp
            spike_raster[iCurrentTS, 1::2] = vnDown

        # - Store state for future evolutions
        self.analog_value = state.copy()

        return spike_raster  # , record

    # def reset_state(self):
    #    """ Resets the state """
    #    # - Store None as state to indicate that future evolutions do not continue from previous input
    #    self.state = None

    @property
    def output_type(self):
        """Returns the output type class"""
        return TSEvent

    # @property
    # def state(self):
    #    """ Returns the state """
    #    return self._state

    # @state.setter
    ## Note that state here is of size self.size_in and not self.size
    # def state(self, new_state):
    #    if new_state is None:
    #        self._state = None
    #    else:
    #        self._state = self._expand_to_size(new_state, self.size_in, "state")

    @property
    def thr_up(self):
        """Returns the spiking threshold of the ``up`` channel"""
        return self._vfThrUp

    @thr_up.setter
    def thr_up(self, vfNewThr):
        assert (np.array(vfNewThr) >= 0).all(), "thr_up must not be negative."

        self._vfThrUp = self._expand_to_size(
            vfNewThr, self.size_in, "thr_up", allow_none=False
        )

    @property
    def thr_down(self):
        """Returns the spiking threshold of the ``down`` channel"""
        return self._vfThrDown

    @thr_down.setter
    def thr_down(self, vfNewThr):
        assert (np.array(vfNewThr) >= 0).all(), "thr_down must not be negative."
        self._vfThrDown = self._expand_to_size(
            vfNewThr, self.size_in, "thr_down", allow_none=False
        )

    @property
    def tau_decay(self):
        """Returns the decay time constant"""
        tau = np.repeat(None, self.size_in)
        # - Treat decay factors of 1 as not decaying (i.e. set them None)
        vbDecay = self._vfDecayFactor != 1
        tau[vbDecay] = self.dt / (1 - self._vfDecayFactor[vbDecay])
        return tau

    @tau_decay.setter
    def tau_decay(self, new_tau):
        new_tau = self._expand_to_size(
            new_tau, self.size_in, "tau_decay", allow_none=True
        )
        # - Find entries which are not None, indicating decay
        vbDecay = np.array([tTau is not None for tTau in new_tau])
        # - Check for too small entries
        assert (
            new_tau[vbDecay] >= self.dt
        ).all(), "Layer `{}`: Entries of tau_decay must be greater or equal to dt ({}).".format(
            self.name, self.dt
        )
        self._vfDecayFactor = np.ones(
            self.size_in
        )  # No decay corresponds to decay factor 1
        self._vfDecayFactor[vbDecay] = 1 - self.dt / new_tau[vbDecay]

    def to_dict(self):
        """
        Convert parameters of `self` to a dict if they are relevant for reconstructing an identical layer.
        """

        config = {}
        config["name"] = self.name
        config["weights"] = self.weights.tolist()
        config["dt"] = self.dt if type(self.dt) is float else self.dt.tolist()
        config["noise_std"] = self.noise_std
        config["repeat_output"] = self.repeat_output
        config["max_num_timesteps"] = self.max_num_timesteps
        config["multiplex_spikes"] = self.multiplex_spikes
        config["tau_decay"] = (
            self.tau_decay if type(self.tau_decay) is float else self.tau_decay.tolist()
        )
        config["thr_up"] = (
            self.thr_up if type(self.thr_up) is float else self.thr_up.tolist()
        )
        config["thr_down"] = (
            self.vfVhrDown if type(self.thr_down) is float else self.thr_down.tolist()
        )
        config["class_name"] = "FFUpDown"

        return config

    def save(self, config: dict, filename: str):
        """Saves the model configuration under a given filename
        :param dict config:     Configuration to save
        :param str filename:    Path the configuration should be saved to"""

        with open(filename, "w") as f:
            json.dump(config, f)

    @staticmethod
    def load_from_dict(config: dict):
        """Returns new object of FFUpDown using the passed configuration

        :param dict config:     Configuration to use
        """

        return FFUpDown(
            weights=config["weights"],
            noise_std=config["noise_std"],
            repeat_output=config["repeat_output"],
            max_num_timesteps=config["max_num_timesteps"],
            multiplex_spikes=config["multiplex_spikes"],
            dt=config["dt"],
            tau_decay=config["tau_decay"],
            thr_up=config["thr_up"],
            thr_down=config["thr_down"],
            name=config["name"],
        )

    @staticmethod
    def load_from_file(filename):
        """Loads and returns a new object of FFUpDown from filename
        :param str filename:    Path to the saved configuration
        """
        with open(filename, "r") as f:
            config = json.load(f)

        return FFUpDown(
            weights=config["weights"],
            noise_std=config["noise_std"],
            repeat_output=config["repeat_output"],
            max_num_timesteps=config["max_num_timesteps"],
            multiplex_spikes=config["multiplex_spikes"],
            dt=config["dt"],
            tau_decay=config["tau_decay"],
            thr_up=config["thr_up"],
            thr_down=config["thr_down"],
            name=config["name"],
        )


@astimedmodule(
    parameters=[],
    simulation_parameters=[
        "weights",
        "repeat_output",
        "dt",
        "tau_decay",
        "noise_std",
        "thr_up",
        "thr_down",
        "name",
        "max_num_timesteps",
        "multiplex_spikes",
    ],
    states=["analog_value"],
)
class FFUpDown(FFUpDownV1):
    pass
