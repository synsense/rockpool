"""
updown.py - Feedforward layer that converts each analogue input channel to one spiking up and one down channel
            Runs in batch mode like FFUpDownTorch to save memory, but does not use pytorch. FFUpDownTorch seems
            to be slower..
"""

import numpy as np
from typing import Optional, Union, Tuple, List
import json

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    bUseTqdm = False
else:
    bUseTqdm = True

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]
# - Default maximum numbers of time steps for a single evolution batch
nDefaultMaxNumTimeSteps = 5000

# - Local imports
from ...timeseries import TSContinuous, TSEvent
from ..layer import Layer

__all__ = ["FFUpDown"]

## - FFUpDown - Class: Define a spiking feedforward layer to convert analogue inputs to up and down channels
class FFUpDown(Layer):
    """
    FFUpDown - Class: Define a spiking feedforward layer to convert analogue inputs to up and down channels
    """

    ## - Constructor
    def __init__(
        self,
        weights: Union[int, np.ndarray],
        nRepeatOutput: int = 1,
        dt: float = 0.001,
        vtTauDecay: Union[ArrayLike, float, None] = None,
        noise_std: float = 0,
        vfThrUp: Union[ArrayLike, float] = 0.001,
        vfThrDown: Union[ArrayLike, float] = 0.001,
        name: str = "unnamed",
        nMaxNumTimeSteps: int = nDefaultMaxNumTimeSteps,
        bMultiplexSpikes: bool = True,
    ):
        """
        FFUpDownBatch - Construct a spiking feedforward layer to convert analogue inputs to up and down channels
        This layer is exceptional in that self.state has the same size as self.size_in, not self.size.
        It corresponds to the input, inferred from the output spikes by inverting the up-/down-algorithm.

        :param weights:         np.array MxN weight matrix.
            Unlike other Layer classes, only important thing about weights its shape. The first
            dimension determines the number of input channels (self.size_in). The second
            dimension corresponds to size and has to be n*2*size_in, n up and n down
            channels for each input). If n>1 the up-/and down-spikes are distributed over
            multiple channels. The values of the weight matrix do not have any effect.
            It is also possible to pass only an integer, which will correspond to size_in.
            size is then set to 2*size_in, i.e. n=1. Alternatively a tuple of two values,
            corresponding to size_in and n can be passed.
        :param dt:         float Time-step. Default: 0.1 ms
        :param vtTauDecay:  array-like  States that tracks input signal for threshold comparison
                                        decay with this time constant unless it is None

        :param noise_std:   float Noise std. dev. per second. Default: 0

        :param vfThrUp:     array-like Thresholds for creating up-spikes
        :param vfThrDown:   array-like Thresholds for creating down-spikes

        :param name:     str Name for the layer. Default: 'unnamed'

        :nMaxNumTimeSteps:  int   Maximum number of timesteps during single evolution batch. Longer
                                  evolution periods will automatically split in smaller batches.

        :bMultiplexSpikes:  bool  Allow a channel to emit multiple spikes per time, according to
                                  how much the corresponding threshold is exceeded
        """

        if np.size(weights) == 1:
            size_in = weights
            size = 2 * size_in * nRepeatOutput
            # - On how many output channels is the are the up-/down-spikes from each input distributed
            self._nMultiChannel = 1
        elif np.size(weights) == 2:
            # - Tuple determining shape
            (size_in, self._nMultiChannel) = weights
            size = 2 * self._nMultiChannel * size_in * nRepeatOutput
        else:
            (size_in, size) = np.shape(weights)
            assert (
                size % (2 * size_in) == 0
            ), "Layer `{}`: size (here {}) must be a multiple of 2*size_in (here {}).".format(
                name, size, size_in
            )
            # - On how many output channels is the are the up-/down-spikes from each input distributed
            self._nMultiChannel = size / (2 * size_in)
            size *= nRepeatOutput
        # - Make sure self._nMultiChannel is an integer
        self._nMultiChannel = int(self._nMultiChannel)

        # - Call super constructor
        super().__init__(
            weights=np.zeros((size_in, size)),
            dt=dt,
            noise_std=noise_std,
            name=name,
        )

        # - Store layer parameters
        self.vfThrUp = vfThrUp
        self.vfThrDown = vfThrDown
        self.vtTauDecay = vtTauDecay
        self.nMaxNumTimeSteps = nMaxNumTimeSteps
        self.nRepeatOutput = nRepeatOutput
        self.bMultiplexSpikes = bMultiplexSpikes

        self.reset_all()

    # @profile
    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSContinuous  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        # - Prepare time base
        __, mfInput, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        if self.noise_std > 0:
            # - Add noise to input
            mfInput += np.random.randn(*mfInput.shape) * self.noise_std

        # - Make sure that layer is able to represent input faithfully
        # mfInputDiff = np.diff(mfInput, axis=0)
        # if (
        #     ((mfInputDiff + 2 * np.abs(mfInput[1:]) * (1 - self._vfDecayFactor)) > self.vfThrUp).any()
        #     or ((mfInputDiff - 2 * np.abs(-mfInput[1:]) * (1 - self._vfDecayFactor)) < - self.vfThrDown).any()
        # ):
        #     print(
        #         "Layer `{}`: With the current settings it may not be possible".format(self.name)
        #         + " to represent the input faithfully. Consider increasing dt"
        #         + " or decreasing vtThrUp and vtTrhDown."
        #     )

        # - Matrix for collecting output spike raster
        mnOutputSpikes = np.zeros((num_timesteps, 2*self.size_in))

        # # - Record states for debugging
        # mfRecord = np.zeros((num_timesteps, self.size_in))

        # - Iterate over batches and run evolution
        iCurrentIndex = 0
        for mfCurrentInput, nCurrNumTS in self._batch_data(
                mfInput, num_timesteps, self.nMaxNumTimeSteps
            ):
            # (
            #     mnOutputSpikes[iCurrentIndex : iCurrentIndex+nCurrNumTS],
            #     mfRecord[iCurrentIndex : iCurrentIndex+nCurrNumTS]
            # ) = self._single_batch_evolution(
            mnOutputSpikes[iCurrentIndex : iCurrentIndex+nCurrNumTS] = self._single_batch_evolution(
                mfCurrentInput,
                nCurrNumTS,
                verbose,
            )
            iCurrentIndex += nCurrNumTS

        ## -- Distribute output spikes over output channels by assigning to each channel
        ##    an interval of length self._nMultiChannel.
        # - Set each event to the first element of its corresponding interval
        vnTSSpike, vnSpikeIDs = np.nonzero(mnOutputSpikes)
        if self.bMultiplexSpikes:
            # - How many times is each spike repeated
            vnSpikeCounts = mnOutputSpikes[vnTSSpike, vnSpikeIDs].astype(int)
            vnTSSpike = vnTSSpike.repeat(vnSpikeCounts)
            vnSpikeIDs = vnSpikeIDs.repeat(vnSpikeCounts)
        if self.nRepeatOutput > 1:
            # - Repeat output spikes
            vnSpikeIDs = vnSpikeIDs.repeat(self.nRepeatOutput)
            vnTSSpike = vnTSSpike.repeat(self.nRepeatOutput)
        if self._nMultiChannel > 1:
            # - Add a repeating series of (0,1,2,..,self._nMultiChannel) to distribute the
            #   events over the interval
            vnSpikeIDs *= self._nMultiChannel
            vnDistribute = np.tile(
                np.arange(self._nMultiChannel),
                int(np.ceil(vnSpikeIDs.size / self._nMultiChannel))
            )[:vnSpikeIDs.size]
            vnSpikeIDs += vnDistribute

        # self.tsRecord = TSContinuous(self.dt * (np.arange(num_timesteps) + self._timestep), mfRecord)

        # - Start and stop times for output time series
        t_start = self._timestep * self.dt
        t_stop = (self._timestep + num_timesteps) * self.dt

        # - Output time series
        vtSpikeTimes = (vnTSSpike + 1 + self._timestep) * self.dt
        tseOut = TSEvent(
            times=np.clip(vtSpikeTimes, t_start, t_stop),  # Clip due to possible numerical errors,
            channels=vnSpikeIDs,
            num_channels=2 * self.size_in * self._nMultiChannel,
            name="Spikes from analogue",
            t_start=t_start,
            t_stop=t_stop,
        )

        # - Update time
        self._timestep += num_timesteps

        return tseOut

    # @profile
    def _batch_data(
        self, mfInput: np.ndarray, num_timesteps: int, nMaxNumTimeSteps: int = None,
    ) -> (np.ndarray, int):
        """_batch_data: Generator that returns the data in batches"""
        # - Handle None for nMaxNumTimeSteps
        nMaxNumTimeSteps = num_timesteps if nMaxNumTimeSteps is None else nMaxNumTimeSteps
        nStart = 0
        while nStart < num_timesteps:
            # - Endpoint of current batch
            nEnd = min(nStart + nMaxNumTimeSteps, num_timesteps)
            # - Data for current batch
            mfCurrentInput = mfInput[nStart:nEnd]
            yield mfCurrentInput, nEnd-nStart
            # - Update nStart
            nStart = nEnd

    # @profile
    def _single_batch_evolution(
        self,
        mfInput: np.ndarray,
        num_timesteps: int,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input for a single batch

        :param mfInput:     np.ndarray   Input
        :param num_timesteps:   int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        # - Prepare local variables
        vfThrUp = self.vfThrUp
        vfThrDown = self.vfThrDown
        vfDecayFactor = self._vfDecayFactor

        # - Arrays for collecting spikes
        mnSpikeRaster = np.zeros((num_timesteps, 2*self.size_in))

        # mfRecord = np.zeros((num_timesteps, self.size_in))

        # - Initialize state for comparing values: If self.state exists, assume input continues from
        #   previous evolution. Otherwise start with initial input data
        state = mfInput[0] if self._state is None else self._state.copy()

        for iCurrentTS in range(num_timesteps):
            # - Decay mechanism
            state *= vfDecayFactor
        
            # mfRecord[iCurrentTS] = state.copy()
        
            if self.bMultiplexSpikes:
                # - By how many times are the upper thresholds exceeded for each input
                vnUp = np.clip(np.floor((mfInput[iCurrentTS]-state) / vfThrUp).astype(int), 0, None)
                # - By how many times are the lower thresholds exceeded for each input
                vnDown = np.clip(np.floor((state-mfInput[iCurrentTS]) / vfThrDown).astype(int), 0, None)
            else:
                # - Inputs where upper threshold is passed
                vnUp = mfInput[iCurrentTS] > state + vfThrUp
                # - Inputs where lower threshold is passed
                vnDown = mfInput[iCurrentTS] < state - vfThrDown
            # - Update state
            state += vfThrUp * vnUp
            state -= vfThrDown * vnDown
            # - Append spikes to array
            mnSpikeRaster[iCurrentTS, ::2] = vnUp
            mnSpikeRaster[iCurrentTS, 1::2] = vnDown

        # - Store state for future evolutions
        self._state = state.copy()

        return mnSpikeRaster #, mfRecord

    def reset_state(self):
        # - Store None as state to indicate that future evolutions do not continue from previous input
        self.state = None

    @property
    def output_type(self):
        return TSEvent

    @property
    def state(self):
        return self._state

    @state.setter
    # Note that state here is of size self.size_in and not self.size
    def state(self, vNewState):
        if vNewState is None:
            self._state = None
        else:
            self._state = self._expand_to_size(vNewState, self.size_in, "state")

    @property
    def vfThrUp(self):
        return self._vfThrUp

    @vfThrUp.setter
    def vfThrUp(self, vfNewThr):
        assert (np.array(vfNewThr) >= 0).all(), "vfThrUp must not be negative."

        self._vfThrUp = self._expand_to_size(
            vfNewThr, self.size_in, "vfThrUp", bAllowNone=False
        )

    @property
    def vfThrDown(self):
        return self._vfThrDown

    @vfThrDown.setter
    def vfThrDown(self, vfNewThr):
        assert (np.array(vfNewThr) >= 0).all(), "vfThrDown must not be negative."
        self._vfThrDown = self._expand_to_size(
            vfNewThr, self.size_in, "vfThrDown", bAllowNone=False
        )

    @property
    def vtTauDecay(self):
        vtTau = np.repeat(None, self.size_in)
        # - Treat decay factors of 1 as not decaying (i.e. set them None)
        vbDecay = self._vfDecayFactor != 1
        vtTau[vbDecay] = self.dt / (1 - self._vfDecayFactor[vbDecay])
        return vtTau

    @vtTauDecay.setter
    def vtTauDecay(self, vtNewTau):
        vtNewTau = self._expand_to_size(vtNewTau, self.size_in, "vtTauDecay", bAllowNone=True)
        # - Find entries which are not None, indicating decay
        vbDecay = np.array([tTau is not None for tTau in vtNewTau])
        # - Check for too small entries
        assert (vtNewTau[vbDecay] >= self.dt).all(), (
            "Layer `{}`: Entries of vtTauDecay must be greater or equal to dt ({}).".format(self.name, self.dt)
        )
        self._vfDecayFactor = np.ones(self.size_in)  # No decay corresponds to decay factor 1
        self._vfDecayFactor[vbDecay] = 1 - self.dt / vtNewTau[vbDecay]


    def to_dict(self):

        config = {}
        config['name'] = self.name
        config['weights'] = self.weights.tolist()
        config['dt'] = self.dt if type(self.dt) is float else self.dt.tolist()
        config['noise_std'] = self.noise_std
        config['nRepeatOutput'] = self.nRepeatOutput
        config['nMaxNumTimeSteps'] = self.nMaxNumTimeSteps
        config['bMultiplexSpikes'] = self.bMultiplexSpikes
        config['vtTauDecay'] = self.vtTauDecay if type(self.vtTauDecay) is float else self.vtTauDecay.tolist()
        config['vfThrUp'] = self.vfThrUp if type(self.vfThrUp) is float else self.vfThrUp.tolist()
        config['vfThrDown'] = self.vfVhrDown if type(self.vfThrDown) is float else self.vfThrDown.tolist()
        config["ClassName"] = "FFUpDown"

        return config

    def save(self, config, filename):
        with open(filename, "w") as f:
            json.dump(config, f)

    @staticmethod
    def load_from_dict(config):

        return FFUpDown(
            weights=config["weights"],
            noise_std=config['noise_std'],
            nRepeatOutput=config['nRepeatOutput'],
            nMaxNumTimeSteps=config['nMaxNumTimeSteps'],
            bMultiplexSpikes=config['bMultiplexSpikes'],
            dt=config['dt'],
            vtTauDecay=config['vtTauDecay'],
            vfThrUp=config['vfThrUp'],
            vfThrDown=config['vfThrDown'],
            name=config['name'],
        )

    @staticmethod
    def load_from_file(filename):
        with open(filename, "r") as f:
            config = json.load(f)

        return FFUpDown(
            weights=config["weights"],
            noise_std=config['noise_std'],
            nRepeatOutput=config['nRepeatOutput'],
            nMaxNumTimeSteps=config['nMaxNumTimeSteps'],
            bMultiplexSpikes=config['bMultiplexSpikes'],
            dt=config['dt'],
            vtTauDecay=config['vtTauDecay'],
            vfThrUp=config['vfThrUp'],
            vfThrDown=config['vfThrDown'],
            name=config['name'],
        )
