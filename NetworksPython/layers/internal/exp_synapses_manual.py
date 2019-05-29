###
# exp_synapses_manual.py - Class implementing a spike-to-current layer with exponential synapses
###


# - Imports
from typing import Union
import numpy as np
from scipy.signal import fftconvolve
import torch

from ...timeseries import TSContinuous, TSEvent
from ..layer import Layer

from typing import Optional, Union, Tuple, List

from warnings import warn

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Configure exports
__all__ = ["FFExpSyn"]

# - Absolute tolerance, e.g. for comparing float values
tol_abs = 1e-9


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


## - FFExpSyn - Class: define an exponential synapse layer (spiking input)
class FFExpSyn(Layer):
    """ FFExpSyn - Class: define an exponential synapse layer (spiking input)
    """

    ## - Constructor
    def __init__(
        self,
        weights: Union[np.ndarray, int] = None,
        vfBias: np.ndarray = 0,
        dt: float = 0.0001,
        noise_std: float = 0,
        tTauSyn: float = 0.005,
        name: str = "unnamed",
        bAddEvents: bool = True,
    ):
        """
        FFExpSyn - Construct an exponential synapse layer (spiking input)

        :param weights:             np.array MxN weight matrix
                                int Size of layer -> creates one-to-one conversion layer
        :param dt:             float Time step for state evolution
        :param noise_std:       float Std. dev. of noise added to this layer. Default: 0

        :param tTauSyn:         float Output synaptic time constants. Default: 5ms
        :param eqSynapses:      Brian2.Equations set of synapse equations for receiver. Default: exponential
        :param strIntegrator:   str Integrator to use for simulation. Default: 'exact'

        :param name:         str Name for the layer. Default: 'unnamed'

        :bAddEvents:            bool     If during evolution multiple input events arrive during one
                                         time step for a channel, count their actual number instead of
                                         just counting them as one.
        """

        # - Provide default weight matrix for one-to-one conversion
        if isinstance(weights, int):
            weights = np.identity(weights, "float")

        # - Check dt
        if dt is None:
            dt = tTauSyn / 10

        # - Call super constructor
        super().__init__(
            weights=weights, dt=dt, noise_std=np.asarray(noise_std), name=name
        )

        # - Parameters
        self.tTauSyn = tTauSyn
        self.vfBias = vfBias
        self.bAddEvents = bAddEvents

        # - set time and state to 0
        self.reset_all()

        # - Objects for training
        self._mfXTX = None
        self._mfXTY = None

    def _prepare_input(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> (np.ndarray, int):
        """
        _prepare_input - Sample input and return as raster.

        :param ts_input:      TimeSeries TxM or Tx1 Input signals for this layer
        :param duration:    float Duration of the desired evolution, in seconds
        :param num_timesteps int Number of evolution time steps

        :return:
            mnInput:          ndarray Raster containing spike info
            num_timesteps:    ndarray Number of evlution time steps
        """
        if num_timesteps is None:
            # - Determine num_timesteps
            if duration is None:
                # - Determine duration
                assert (
                    ts_input is not None
                ), "Layer {}: One of `num_timesteps`, `ts_input` or `duration` must be supplied".format(
                    self.name
                )

                if ts_input.periodic:
                    # - Use duration of periodic TimeSeries, if possible
                    duration = ts_input.duration

                else:
                    # - Evolve until the end of the input TImeSeries
                    duration = ts_input.t_stop - self.t + self.dt
                    assert duration > 0, (
                        "Layer {}: Cannot determine an appropriate evolution duration.".format(
                            self.name
                        )
                        + "`ts_input` finishes before the current "
                        "evolution time."
                    )
            # - Discretize duration wrt self.dt
            num_timesteps = int(np.floor((duration + tol_abs) / self.dt))
        else:
            assert isinstance(
                num_timesteps, int
            ), "Layer `{}`: num_timesteps must be of type int.".format(self.name)

        if ts_input is not None:
            # Extract spike data from the input variable
            mnSpikeRaster = ts_input.raster(
                dt=self.dt,
                t_start=self.t,
                num_timesteps=num_timesteps,
                channels=np.arange(self.size_in),
                add_events=self.bAddEvents,
            ).astype(float)

        else:
            mnSpikeRaster = np.zeros((num_timesteps, self.size_in))

        return mnSpikeRaster, num_timesteps

    ### --- State evolution

    def evolve(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSContinuous:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSEvent  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSContinuous  output spike series

        """

        # - Prepare weighted input signal
        mnInputRaster, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )
        mfWeightedInput = mnInputRaster @ self.weights

        # - Time base
        vtTimeBase = (np.arange(num_timesteps + 1) + self._timestep) * self.dt

        if self.noise_std > 0:
            # - Add a noise trace
            # - Noise correction is slightly different than in other layers
            mfNoise = (
                np.random.randn(*mfWeightedInput.shape)
                * self.noise_std
                * np.sqrt(2 * self.dt / self.tTauSyn)
            )
            mfNoise[0, :] = 0  # Make sure that noise trace starts with 0
            mfWeightedInput += mfNoise

        # Add current state to input
        mfWeightedInput[0, :] += self._vStateNoBias.copy() * np.exp(
            -self.dt / self.tTauSyn
        )

        # - Define exponential kernel
        vfKernel = np.exp(-np.arange(num_timesteps + 1) * self.dt / self.tTauSyn)
        # - Make sure spikes only have effect on next time step
        vfKernel = np.r_[0, vfKernel]

        # - Apply kernel to spike trains
        mfFiltered = np.zeros((num_timesteps + 1, self.size))
        for channel, vEvents in enumerate(mfWeightedInput.T):
            vConv = fftconvolve(vEvents, vfKernel, "full")
            vConvShort = vConv[: vtTimeBase.size]
            mfFiltered[:, channel] = vConvShort

        # - Update time and state
        self._timestep += num_timesteps
        self._vStateNoBias = mfFiltered[-1]

        # - Output time series with output data and bias
        return TSContinuous(
            vtTimeBase, mfFiltered + self.vfBias, name="Receiver current"
        )

    def evolve_train(
        self,
        tsTarget: TSContinuous,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        fRegularize: float = 0,
        fLearningRate: float = 0.01,
        verbose: bool = False,
    ) -> TSContinuous:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsTarget:        TSContinuous  Target time series
        :param tsSpkInput:      TSEvent  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param fRegularize:     float    Regularization parameter
        :param fLearningRate:   flaot    Factor determining scale of weight increments at each step
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSContinuous  output spike series

        """

        # - Prepare input signal
        num_timesteps = int(np.round(tsTarget.duration / self.dt))
        mnInputRaster, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Time base
        vtTimeBase = (np.arange(num_timesteps + 1) + self._timestep) * self.dt

        # - Define exponential kernel
        vfKernel = np.exp(-(np.arange(num_timesteps) * self.dt) / self.tTauSyn)
        # - Make sure spikes only have effect on next time step
        vfKernel = np.r_[0, vfKernel]

        # Empty input array with additional dimension for training biases
        mfInput = np.zeros((np.size(vtTimeBase), self.size_in + 1))
        mfInput[:, -1] = 1

        # - Apply kernel to spike trains and add filtered trains to input array
        for channel, vEvents in enumerate(mnInputRaster.T):
            mfInput[:, channel] = fftconvolve(vEvents, vfKernel, "full")[
                : vtTimeBase.size
            ]

        # - Evolution:
        mfWeighted = mfInput[:, :-1] @ self.weights
        mfOut = mfWeighted + self.vfBias

        # - Update time and state
        self._timestep += num_timesteps

        ## -- Training
        # - Prepare target data
        mfTarget = tsTarget(vtTimeBase)

        # - Make sure no nan is in target, as this causes learning to fail
        assert not np.isnan(
            mfTarget
        ).any(), "Layer `{}`: nan values have been found in mfTarget (where: {})".format(
            self.name, np.where(np.isnan(mfTarget))
        )

        # - Check target dimensions
        if mfTarget.ndim == 1 and self.size == 1:
            mfTarget = mfTarget.reshape(-1, 1)

        assert (
            mfTarget.shape[-1] == self.size
        ), "Layer `{}`: Target dimensions ({}) does not match layer size ({})".format(
            self.name, mfTarget.shape[-1], self.size
        )

        # - Weight update
        # mfUpdate = mfInput.T @ (mfTarget - mfOut)
        # print(np.linalg.norm(mfTarget-mfOut))
        # # Normalize learning rate by number of inputs
        # fLearningRate /= (self.size_in * mfInput.shape[0] * vfG)
        # self.weights += fLearningRate * (mfUpdate[:-1]) - fRegularize * self.weights
        # self.vfBias += fLearningRate * (mfUpdate[-1]) - fRegularize * self.vfBias

        mfXTX = mfInput.T @ mfInput
        mfXTY = mfInput.T @ mfTarget
        mfNewWeights = np.linalg.solve(
            mfXTX + fRegularize * np.eye(mfInput.shape[1]), mfXTY
        )
        print(np.linalg.norm(mfTarget - mfOut))
        self.weights = (self.weights + fLearningRate * mfNewWeights[:-1]) / (
            1.0 + fLearningRate
        )
        self.vfBias = (self.vfBias + fLearningRate * mfNewWeights[-1]) / (
            1.0 + fLearningRate
        )

        # - Output time series with output data and bias
        return TSContinuous(vtTimeBase, mfOut, name="Receiver current")

    def train_rr(
        self,
        tsTarget: TSContinuous,
        ts_input: TSEvent = None,
        fRegularize: float = 0,
        bFirst: bool = True,
        bFinal: bool = False,
        bStoreState: bool = True,
        bTrainBiases: bool = True,
    ):

        """
        train_rr - Train self with ridge regression over one of possibly
                   many batches. Use Kahan summation to reduce rounding
                   errors when adding data to existing matrices from
                   previous batches.
        :param tsTarget:        TimeSeries - target for current batch
        :param ts_input:         TimeSeries - input to self for current batch
        :fRegularize:           float - regularization for ridge regression
        :bFirst:                bool - True if current batch is the first in training
        :bFinal:                bool - True if current batch is the last in training
        :bStoreState:           bool - Include last state from previous training and store state from this
                                       traning. This has the same effect as if data from both trainings
                                       were presented at once.
        :param bTrainBiases:    bool - If True, train biases as if they were weights
                                       Otherwise present biases will be ignored in
                                       training and not be changed.
        """

        # - Discrete time steps for evaluating input and target time series
        num_timesteps = int(np.round(tsTarget.duration / self.dt))
        vtTimeBase = self._gen_time_trace(tsTarget.t_start, num_timesteps)

        if not bFinal:
            # - Discard last sample to avoid counting time points twice
            vtTimeBase = vtTimeBase[:-1]

        # - Make sure vtTimeBase does not exceed tsTarget
        vtTimeBase = vtTimeBase[vtTimeBase <= tsTarget.t_stop]

        # - Prepare target data
        mfTarget = tsTarget(vtTimeBase)

        # - Make sure no nan is in target, as this causes learning to fail
        assert not np.isnan(
            mfTarget
        ).any(), "Layer `{}`: nan values have been found in mfTarget (where: {})".format(
            self.name, np.where(np.isnan(mfTarget))
        )

        # - Check target dimensions
        if mfTarget.ndim == 1 and self.size == 1:
            mfTarget = mfTarget.reshape(-1, 1)

        assert (
            mfTarget.shape[-1] == self.size
        ), "Layer `{}`: Target dimensions ({}) does not match layer size ({})".format(
            self.name, mfTarget.shape[-1], self.size
        )

        # - Prepare input data
        nInputSize = self.size_in + int(bTrainBiases)
        # Empty input array with additional dimension for training biases
        mfInput = np.zeros((np.size(vtTimeBase), nInputSize))
        if bTrainBiases:
            mfInput[:, -1] = 1

        # - Generate spike trains from ts_input
        if ts_input is None:
            # - Assume zero input
            print(
                "Layer `{}`: No ts_input defined, assuming input to be 0.".format(
                    self.name
                )
            )

        else:
            # - Get data within given time range
            vtEventTimes, vnEventChannels = ts_input(
                t_start=vtTimeBase[0], t_stop=vtTimeBase[-1]
            )

            # - Make sure that input channels do not exceed layer input dimensions
            try:
                assert (
                    np.amax(vnEventChannels) <= self.size_in - 1
                ), "Layer `{}`: Number of input channels exceeds layer input dimensions.".format(
                    self.name
                )
            except ValueError as e:
                # - No events in input data
                if vnEventChannels.size == 0:
                    print(
                        "Layer `{}`: No input spikes for training.".format(self.name)
                    )
                else:
                    raise e

            # Extract spike data from the input
            mnSpikeRaster = (
                ts_input.raster(
                    dt=self.dt,
                    t_start=vtTimeBase[0],
                    num_timesteps=vtTimeBase.size,
                    channels=np.arange(self.size_in),
                    add_events=self.bAddEvents,
                )
            ).astype(float)

            if bStoreState and not bFirst:
                try:
                    # - Include last state from previous batch
                    mnSpikeRaster[0, :] += self._vTrainingState
                except AttributeError:
                    pass

            # - Define exponential kernel
            vfKernel = np.exp(
                -(np.arange(vtTimeBase.size - 1) * self.dt) / self.tTauSyn
            )
            # - Make sure spikes only have effect on next time step
            vfKernel = np.r_[0, vfKernel]

            # - Apply kernel to spike trains and add filtered trains to input array
            for channel, vEvents in enumerate(mnSpikeRaster.T):
                mfInput[:, channel] = fftconvolve(vEvents, vfKernel, "full")[
                    : vtTimeBase.size
                ]

        # - For first batch, initialize summands
        if bFirst:
            # Matrices to be updated for each batch
            self._mfXTY = np.zeros((nInputSize, self.size))  # mfInput.T (dot) mfTarget
            self._mfXTX = np.zeros((nInputSize, nInputSize))  # mfInput.T (dot) mfInput
            # Corresponding Kahan compensations
            self.mfKahanCompXTY = np.zeros_like(self._mfXTY)
            self.mfKahanCompXTX = np.zeros_like(self._mfXTX)

        # - New data to be added, including compensation from last batch
        #   (Matrix summation always runs over time)
        mfUpdXTY = mfInput.T @ mfTarget - self.mfKahanCompXTY
        mfUpdXTX = mfInput.T @ mfInput - self.mfKahanCompXTX

        if not bFinal:
            # - Update matrices with new data
            mfNewXTY = self._mfXTY + mfUpdXTY
            mfNewXTX = self._mfXTX + mfUpdXTX
            # - Calculate rounding error for compensation in next batch
            self.mfKahanCompXTY = (mfNewXTY - self._mfXTY) - mfUpdXTY
            self.mfKahanCompXTX = (mfNewXTX - self._mfXTX) - mfUpdXTX
            # - Store updated matrices
            self._mfXTY = mfNewXTY
            self._mfXTX = mfNewXTX

            if bStoreState:
                # - Store last state for next batch
                if bTrainBiases:
                    self._vTrainingState = mfInput[-1, :-1].copy()
                else:
                    self._vTrainingState = mfInput[-1, :].copy()

        else:
            # - In final step do not calculate rounding error but update matrices directly
            self._mfXTY += mfUpdXTY
            self._mfXTX += mfUpdXTX

            # - Weight and bias update by ridge regression
            mfSolution = np.linalg.solve(
                self._mfXTX + fRegularize * np.eye(nInputSize), self._mfXTY
            )
            if bTrainBiases:
                self.weights = mfSolution[:-1, :]
                self.vfBias = mfSolution[-1, :]
            else:
                self.weights = mfSolution

            # - Remove dat stored during this trainig
            self._mfXTY = None
            self._mfXTX = None
            self.mfKahanCompXTY = None
            self.mfKahanCompXTX = None
            self._vTrainingState = None

    def train_logreg(
        self,
        tsTarget: TSContinuous,
        ts_input: TSEvent = None,
        fLearningRate: float = 0,
        fRegularize: float = 0,
        nBatchSize: Optional[int] = None,
        nEpochs: int = 1,
        bStoreState: bool = True,
        verbose: bool = False,
    ):
        """
        train_logreg - Train self with logistic regression over one of possibly many batches.
                       Note that this training method assumes that a sigmoid funciton is applied
                       to the layer output, which is not the case in self.evolve.
        :param tsTarget:    TimeSeries - target for current batch
        :param ts_input:     TimeSeries - input to self for current batch
        :fLearningRate:     flaot - Factor determining scale of weight increments at each step
        :fRegularize:       float - regularization parameter
        :nBatchSize:        int - Number of samples per batch. If None, train with all samples at once
        :nEpochs:           int - How many times is training repeated
        :bStoreState:       bool - Include last state from previous training and store state from this
                                   traning. This has the same effect as if data from both trainings
                                   were presented at once.
        :verbose:          bool - Print output about training progress
        """

        # - Discrete time steps for evaluating input and target time series
        num_timesteps = int(np.round(tsTarget.duration / self.dt))
        vtTimeBase = self._gen_time_trace(tsTarget.t_start, num_timesteps)

        # - Discard last sample to avoid counting time points twice
        vtTimeBase = vtTimeBase[:-1]

        # - Make sure vtTimeBase does not exceed tsTarget
        vtTimeBase = vtTimeBase[vtTimeBase <= tsTarget.t_stop]

        # - Prepare target data
        mfTarget = tsTarget(vtTimeBase)

        # - Make sure no nan is in target, as this causes learning to fail
        assert not np.isnan(
            mfTarget
        ).any(), "Layer `{}`: nan values have been found in mfTarget (where: {})".format(
            self.name, np.where(np.isnan(mfTarget))
        )

        # - Check target dimensions
        if mfTarget.ndim == 1 and self.size == 1:
            mfTarget = mfTarget.reshape(-1, 1)

        assert (
            mfTarget.shape[-1] == self.size
        ), "Layer `{}`: Target dimensions ({}) does not match layer size ({})".format(
            self.name, mfTarget.shape[-1], self.size
        )

        # - Prepare input data

        # Empty input array with additional dimension for training biases
        mfInput = np.zeros((np.size(vtTimeBase), self.size_in + 1))
        mfInput[:, -1] = 1

        # - Generate spike trains from ts_input
        if ts_input is None:
            # - Assume zero input
            print(
                "Layer `{}`: No ts_input defined, assuming input to be 0.".format(
                    self.name
                )
            )

        else:
            # - Get data within given time range
            vtEventTimes, vnEventChannels = ts_input(
                t_start=vtTimeBase[0], t_stop=vtTimeBase[-1]
            )

            # - Make sure that input channels do not exceed layer input dimensions
            try:
                assert (
                    np.amax(vnEventChannels) <= self.size_in - 1
                ), "Layer `{}`: Number of input channels exceeds layer input dimensions.".format(
                    self.name
                )
            except ValueError as e:
                # - No events in input data
                if vnEventChannels.size == 0:
                    print(
                        "Layer `{}`: No input spikes for training.".format(self.name)
                    )
                else:
                    raise e

            # Extract spike data from the input
            mnSpikeRaster = (
                ts_input.raster(
                    dt=self.dt,
                    t_start=vtTimeBase[0],
                    num_timesteps=vtTimeBase.size,
                    channels=np.arange(self.size_in),
                    add_events=self.bAddEvents,
                )
            ).astype(float)

            if bStoreState:
                try:
                    # - Include last state from previous batch
                    mnSpikeRaster[0, :] += self._vTrainingState
                except AttributeError:
                    pass

            # - Define exponential kernel
            vfKernel = np.exp(
                -(np.arange(vtTimeBase.size - 1) * self.dt) / self.tTauSyn
            )

            # - Apply kernel to spike trains and add filtered trains to input array
            for channel, vEvents in enumerate(mnSpikeRaster.T):
                mfInput[:, channel] = fftconvolve(vEvents, vfKernel, "full")[
                    : vtTimeBase.size
                ]

        # - Prepare batches for training
        if nBatchSize is None:
            nNumBatches = 1
            nBatchSize = num_timesteps
        else:
            nNumBatches = int(np.ceil(num_timesteps / float(nBatchSize)))

        viSampleOrder = np.arange(
            num_timesteps
        )  # Indices to choose samples - shuffle for random order

        # - Iterate over epochs
        for iEpoch in range(nEpochs):
            # - Iterate over batches and optimize
            for iBatch in range(nNumBatches):
                viSampleIndices = viSampleOrder[
                    iBatch * nBatchSize : (iBatch + 1) * nBatchSize
                ]
                # - Gradients
                mfGradients = self._gradients(
                    mfInput[viSampleIndices], mfTarget[viSampleIndices], fRegularize
                )
                self.weights = self.weights - fLearningRate * mfGradients[:-1, :]
                self.vfBias = self.vfBias - fLearningRate * mfGradients[-1, :]
            if verbose:
                print(
                    "Layer `{}`: Training epoch {} of {}".format(
                        self.name, iEpoch + 1, nEpochs
                    ),
                    end="\r",
                )
            # - Shuffle samples
            np.random.shuffle(viSampleOrder)

        if verbose:
            print("Layer `{}`: Finished trainig.              ".format(self.name))

        if bStoreState:
            # - Store last state for next batch
            self._vTrainingState = mfInput[-1, :-1].copy()

    def _gradients(self, mfInput, mfTarget, fRegularize):
        # - Output with current weights
        mfLinear = mfInput[:, :-1] @ self.weights + self.vfBias
        mfOutput = sigmoid(mfLinear)
        # - Gradients for weights
        nNumSamples = mfInput.shape[0]
        mfError = mfOutput - mfTarget
        mfGradients = (mfInput.T @ mfError) / float(nNumSamples)
        # - Regularization of weights
        if fRegularize > 0:
            mfGradients[:-1, :] += fRegularize / float(self.size_in) * self.weights

        return mfGradients

    ### --- Properties

    @property
    def input_type(self):
        return TSEvent

    @property
    def tTauSyn(self):
        return self._tTauSyn

    @tTauSyn.setter
    def tTauSyn(self, tNewTau):
        assert tNewTau > 0, "Layer `{}`: tTauSyn must be greater than 0.".format(
            self.name
        )
        self._tTauSyn = tNewTau

    @property
    def vfBias(self):
        return self._vfBias

    @vfBias.setter
    def vfBias(self, vfNewBias):
        self._vfBias = self._expand_to_net_size(vfNewBias, "vfBias", bAllowNone=False)

    @property
    def state(self):
        return self._vStateNoBias + self._vfBias

    @state.setter
    def state(self, vNewState):
        vNewState = np.asarray(self._expand_to_net_size(vNewState, "state"))
        self._vStateNoBias = vNewState - self._vfBias

    @property
    def mfXTX(self):
        return self._mfXTX

    @property
    def mfXTY(self):
        return self._mfXTY
