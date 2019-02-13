###
# exp_synapses_manual.py - Class implementing a spike-to-current layer with exponential synapses
###


# - Imports
from typing import Union
import numpy as np
from scipy.signal import fftconvolve

from ...timeseries import TSContinuous, TSEvent
from ..layer import Layer

from typing import Optional, Union, Tuple, List

import torch
from warnings import warn

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Configure exports
__all__ = ["FFExpSyn"]


# - Absolute tolerance, e.g. for comparing float values
fTolAbs = 1e-9



def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def sigmoid_torch(z):
    return 1. / (1. + torch.exp(-z))

## - FFExpSyn - Class: define an exponential synapse layer (spiking input)
class FFExpSyn(Layer):
    """ FFExpSyn - Class: define an exponential synapse layer (spiking input)
    """

    ## - Constructor
    def __init__(
        self,
        mfW: Union[np.ndarray, int] = None,
        vfBias: np.ndarray = 0,
        tDt: float = 0.0001,
        fNoiseStd: float = 0,
        tTauSyn: float = 0.005,
        strName: str = "unnamed",
        bAddEvents: bool = True,
    ):
        """
        FFExpSyn - Construct an exponential synapse layer (spiking input)

        :param mfW:             np.array MxN weight matrix
                                int Size of layer -> creates one-to-one conversion layer
        :param tDt:             float Time step for state evolution
        :param fNoiseStd:       float Std. dev. of noise added to this layer. Default: 0

        :param tTauSyn:         float Output synaptic time constants. Default: 5ms
        :param eqSynapses:      Brian2.Equations set of synapse equations for receiver. Default: exponential
        :param strIntegrator:   str Integrator to use for simulation. Default: 'exact'

        :param strName:         str Name for the layer. Default: 'unnamed'

        :bAddEvents:            bool     If during evolution multiple input events arrive during one
                                         time step for a channel, count their actual number instead of
                                         just counting them as one.
        """

        # - Provide default weight matrix for one-to-one conversion
        if isinstance(mfW, int):
            mfW = np.identity(mfW, "float")

        # - Check tDt
        if tDt is None:
            tDt = tTauSyn / 10

        # - Call super constructor
        super().__init__(
            mfW=mfW,
            tDt=tDt,
            fNoiseStd=np.asarray(fNoiseStd),
            strName=strName,
        )

        # - Parameters
        self.tTauSyn = tTauSyn
        self.vfBias = vfBias
        self.bAddEvents = bAddEvents

        # - set time and state to 0
        self.reset_all()

    def _prepare_input(
        self,
        tsInput: Optional[TSEvent] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
    ) -> (np.ndarray, int):
        """
        _prepare_input - Sample input and return as raster.

        :param tsInput:      TimeSeries TxM or Tx1 Input signals for this layer
        :param tDuration:    float Duration of the desired evolution, in seconds
        :param nNumTimeSteps int Number of evolution time steps

        :return:
            mnInput:          ndarray Raster containing spike info
            nNumTimeSteps:    ndarray Number of evlution time steps
        """
        if nNumTimeSteps is None:
            # - Determine nNumTimeSteps
            if tDuration is None:
                # - Determine tDuration
                assert (
                    tsInput is not None
                ), "Layer {}: One of `nNumTimeSteps`, `tsInput` or `tDuration` must be supplied".format(
                    self.strName
                )

                if tsInput.bPeriodic:
                    # - Use duration of periodic TimeSeries, if possible
                    tDuration = tsInput.tDuration

                else:
                    # - Evolve until the end of the input TImeSeries
                    tDuration = tsInput.tStop - self.t + self.tDt
                    assert tDuration > 0, (
                        "Layer {}: Cannot determine an appropriate evolution duration.".format(
                            self.strName
                        )
                        + "`tsInput` finishes before the current "
                        "evolution time."
                    )
            # - Discretize tDuration wrt self.tDt
            nNumTimeSteps = int(np.floor((tDuration + fTolAbs) / self.tDt))
        else:
            assert isinstance(
                nNumTimeSteps, int
            ), "Layer `{}`: nNumTimeSteps must be of type int.".format(self.strName)

        if tsInput is not None:
            # Extract spike data from the input variable
            mnSpikeRaster = tsInput.raster(
                tDt=self.tDt,
                tStart=self.t,
                nNumTimeSteps=nNumTimeSteps,
                vnSelectChannels=np.arange(self.nSizeIn),
                bSamples=False,
                bAddEvents=self.bAddEvents,
            )[2].astype(float)

        else:
            mnSpikeRaster = np.zeros((nNumTimeSteps, self.nSizeIn))

        return mnSpikeRaster, nNumTimeSteps

    ### --- State evolution

    def evolve(
        self,
        tsInput: Optional[TSEvent] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
        bVerbose: bool = False,
    ) -> TSContinuous:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSEvent  Input spike trian
        :param tDuration:       float    Simulation/Evolution time
        :param nNumTimeSteps    int      Number of evolution time steps
        :param bVerbose:        bool     Currently no effect, just for conformity
        :return:            TSContinuous  output spike series

        """

        # - Prepare weighted input signal
        mnInputRaster, nNumTimeSteps = self._prepare_input(tsInput, tDuration, nNumTimeSteps)
        mfWeightedInput = mnInputRaster @ self.mfW

        # - Time base
        vtTimeBase = (np.arange(nNumTimeSteps + 1) + self._nTimeStep) * self.tDt

        if self.fNoiseStd > 0:
            # - Add a noise trace
            # - Noise correction is slightly different than in other layers
            mfNoise = (
                np.random.randn(*mfWeightedInput.shape)
                * self.fNoiseStd
                * np.sqrt(2 * self.tDt / self.tTauSyn)
            )
            mfNoise[0, :] = 0  # Make sure that noise trace starts with 0
            mfWeightedInput += mfNoise

        # Add current state to input
        mfWeightedInput[0, :] += self._vStateNoBias.copy() * np.exp(-self.tDt / self.tTauSyn)

        # - Define exponential kernel
        vfKernel = np.exp(-np.arange(nNumTimeSteps + 1) * self.tDt / self.tTauSyn)
        # - Make sure spikes only have effect on next time step
        vfKernel = np.r_[0, vfKernel]

        # - Apply kernel to spike trains
        mfFiltered = np.zeros((nNumTimeSteps+1, self.nSize))
        for channel, vEvents in enumerate(mfWeightedInput.T):
            vConv = fftconvolve(vEvents, vfKernel, "full")
            vConvShort = vConv[: vtTimeBase.size]
            mfFiltered[:, channel] = vConvShort

        # - Update time and state
        self._nTimeStep += nNumTimeSteps
        self._vStateNoBias = mfFiltered[-1]

        # - Output time series with output data and bias
        return TSContinuous(
            vtTimeBase, mfFiltered + self.vfBias, strName="Receiver current"
        )

    def evolve_train(
        self,
        tsTarget: TSContinuous,
        tsInput: Optional[TSEvent] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
        fRegularize: float = 0,
        fLearningRate: float = 0.01,
        bVerbose: bool = False,
    ) -> TSContinuous:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsTarget:        TSContinuous  Target time series
        :param tsSpkInput:      TSEvent  Input spike trian
        :param tDuration:       float    Simulation/Evolution time
        :param nNumTimeSteps    int      Number of evolution time steps
        :param fRegularize:     float    Regularization parameter
        :param fLearningRate:   flaot    Factor determining scale of weight increments at each step
        :param bVerbose:        bool     Currently no effect, just for conformity
        :return:            TSContinuous  output spike series

        """

        # - Prepare input signal
        nNumTimeSteps = int(np.round(tsTarget.tDuration / self.tDt))
        mnInputRaster, nNumTimeSteps = self._prepare_input(tsInput, tDuration, nNumTimeSteps)

        # - Time base
        vtTimeBase = (np.arange(nNumTimeSteps + 1) + self._nTimeStep) * self.tDt

        # - Define exponential kernel
        vfKernel = np.exp(
            - (np.arange(vtTimeBase.size-1) * self.tDt) / self.tTauSyn
        )

        # Empty input array with additional dimension for training biases
        mfInput = np.zeros((np.size(vtTimeBase), self.nSizeIn + 1))
        mfInput[:, -1] = 1

        # - Apply kernel to spike trains and add filtered trains to input array
        for channel, vEvents in enumerate(mnInputRaster.T):
            mfInput[:, channel] = fftconvolve(vEvents, vfKernel, "full")[
                : vtTimeBase.size
            ]

        # - Evolution:
        mfFiltered = mfInput[:, :-1] @ self.mfW
        mfOut = mfFiltered + self.vfBias

        # - Update time and state
        self._nTimeStep += nNumTimeSteps

        ## -- Training
        # - Prepare target data
        mfTarget = tsTarget(vtTimeBase)

        # - Make sure no nan is in target, as this causes learning to fail
        assert not np.isnan(
            mfTarget
        ).any(), "Layer `{}`: nan values have been found in mfTarget (where: {})".format(
            self.strName, np.where(np.isnan(mfTarget))
        )

        # - Check target dimensions
        if mfTarget.ndim == 1 and self.nSize == 1:
            mfTarget = mfTarget.reshape(-1, 1)

        assert (
            mfTarget.shape[-1] == self.nSize
        ), "Layer `{}`: Target dimensions ({}) does not match layer size ({})".format(
            self.strName, mfTarget.shape[-1], self.nSize
        )
        # - Weight update
        mfUpdate = mfInput.T @ (mfTarget - mfOut)
        self.mfW += fLearningRate * (mfUpdate[:-1] - fRegularize * self.mfW)
        self.vfBias += fLearningRate * (mfUpdate[-1] - fRegularize * self.vfBias)

        # - Output time series with output data and bias
        return TSContinuous(
            vtTimeBase, mfFiltered + self.vfBias, strName="Receiver current"
        )


    def train_rr(
        self,
        tsTarget: TSContinuous,
        tsInput: TSEvent = None,
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
        :param tsInput:         TimeSeries - input to self for current batch
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
        nNumTimeSteps = int(np.round(tsTarget.tDuration / self.tDt))
        vtTimeBase = self._gen_time_trace(tsTarget.tStart, nNumTimeSteps)

        if not bFinal:
            # - Discard last sample to avoid counting time points twice
            vtTimeBase = vtTimeBase[:-1]

        # - Make sure vtTimeBase does not exceed tsTarget
        vtTimeBase = vtTimeBase[vtTimeBase <= tsTarget.tStop]

        # - Prepare target data
        mfTarget = tsTarget(vtTimeBase)

        # - Make sure no nan is in target, as this causes learning to fail
        assert not np.isnan(
            mfTarget
        ).any(), "Layer `{}`: nan values have been found in mfTarget (where: {})".format(
            self.strName, np.where(np.isnan(mfTarget))
        )

        # - Check target dimensions
        if mfTarget.ndim == 1 and self.nSize == 1:
            mfTarget = mfTarget.reshape(-1, 1)

        assert (
            mfTarget.shape[-1] == self.nSize
        ), "Layer `{}`: Target dimensions ({}) does not match layer size ({})".format(
            self.strName, mfTarget.shape[-1], self.nSize
        )

        # - Prepare input data
        nInputSize = self.nSizeIn + int(bTrainBiases)
        # Empty input array with additional dimension for training biases
        mfInput = np.zeros((np.size(vtTimeBase), nInputSize))
        if bTrainBiases:
            mfInput[:, -1] = 1

        # - Generate spike trains from tsInput
        if tsInput is None:
            # - Assume zero input
            print(
                "Layer `{}`: No tsInput defined, assuming input to be 0.".format(
                    self.strName
                )
            )

        else:
            # - Get data within given time range
            vtEventTimes, vnEventChannels, __ = tsInput.find(
                [vtTimeBase[0], vtTimeBase[-1]]
            )

            # - Make sure that input channels do not exceed layer input dimensions
            try:
                assert (
                    np.amax(vnEventChannels) <= self.nSizeIn - 1
                ), "Layer `{}`: Number of input channels exceeds layer input dimensions.".format(
                    self.strName
                )
            except ValueError as e:
                # - No events in input data
                if vnEventChannels.size == 0:
                    print(
                        "Layer `{}`: No input spikes for training.".format(self.strName)
                    )
                else:
                    raise e

            # Extract spike data from the input
            mnSpikeRaster = (
                tsInput.raster(
                    tDt=self.tDt,
                    tStart=vtTimeBase[0],
                    nNumTimeSteps=vtTimeBase.size,
                    vnSelectChannels=np.arange(self.nSizeIn),
                    bSamples=False,
                    bAddEvents=self.bAddEvents,
                )[2]
            ).astype(float)

            if bStoreState and not bFirst:
                try:
                    # - Include last state from previous batch
                    mnSpikeRaster[0, :] += self._vTrainingState
                except AttributeError:
                    pass

            # - Define exponential kernel
            vfKernel = np.exp(
                - (np.arange(vtTimeBase.size-1) * self.tDt) / self.tTauSyn
            )

            # - Apply kernel to spike trains and add filtered trains to input array
            for channel, vEvents in enumerate(mnSpikeRaster.T):
                mfInput[:, channel] = fftconvolve(vEvents, vfKernel, "full")[
                    : vtTimeBase.size
                ]

        # - For first batch, initialize summands
        if bFirst:
            # Matrices to be updated for each batch
            self.mfXTY = np.zeros(
                (nInputSize, self.nSize)
            )  # mfInput.T (dot) mfTarget
            self.mfXTX = np.zeros(
                (nInputSize, nInputSize)
            )  # mfInput.T (dot) mfInput
            # Corresponding Kahan compensations
            self.mfKahanCompXTY = np.zeros_like(self.mfXTY)
            self.mfKahanCompXTX = np.zeros_like(self.mfXTX)

        # - New data to be added, including compensation from last batch
        #   (Matrix summation always runs over time)
        mfUpdXTY = mfInput.T @ mfTarget - self.mfKahanCompXTY
        mfUpdXTX = mfInput.T @ mfInput - self.mfKahanCompXTX

        if not bFinal:
            # - Update matrices with new data
            mfNewXTY = self.mfXTY + mfUpdXTY
            mfNewXTX = self.mfXTX + mfUpdXTX
            # - Calculate rounding error for compensation in next batch
            self.mfKahanCompXTY = (mfNewXTY - self.mfXTY) - mfUpdXTY
            self.mfKahanCompXTX = (mfNewXTX - self.mfXTX) - mfUpdXTX
            # - Store updated matrices
            self.mfXTY = mfNewXTY
            self.mfXTX = mfNewXTX
            
            if bStoreState:
                # - Store last state for next batch
                if bTrainBiases:
                    self._vTrainingState = mfInput[-1, :-1].copy()
                else:
                    self._vTrainingState = mfInput[-1, :].copy()

        else:
            # - In final step do not calculate rounding error but update matrices directly
            self.mfXTY += mfUpdXTY
            self.mfXTX += mfUpdXTX

            # - Weight and bias update by ridge regression
            mfSolution = np.linalg.solve(
                self.mfXTX + fRegularize * np.eye(nInputSize), self.mfXTY
            )
            if bTrainBiases:
                self.mfW = mfSolution[:-1, :]
                self.vfBias = mfSolution[-1, :]
            else:
                self.mfW = mfSolution

            # - Remove dat stored during this trainig
            self.mfXTY = None
            self.mfXTX = None
            self.mfKahanCompXTY = None
            self.mfKahanCompXTX = None
            self._vTrainingState = None


    def train_logreg(
        self,
        tsTarget: TSContinuous,
        tsInput: TSEvent = None,
        fLearningRate: float = 0,
        fRegularize: float = 0,
        nBatchSize: Optional[int] = None,
        nEpochs: int = 1,
        bStoreState: bool = True,
        bVerbose: bool = False,
    ):

        """
        train_logreg - Train self with logistic regression over one of possibly many batches.
                       Note that this training method assumes that a sigmoid funciton is applied
                       to the layer output, which is not the case in self.evolve.
        :param tsTarget:    TimeSeries - target for current batch
        :param tsInput:     TimeSeries - input to self for current batch
        :fLearningRate:     flaot - Factor determining scale of weight increments at each step
        :fRegularize:       float - regularization parameter
        :nBatchSize:        int - Number of samples per batch. If None, train with all samples at once
        :nEpochs:           int - How many times is training repeated
        :bStoreState:       bool - Include last state from previous training and store state from this 
                                   traning. This has the same effect as if data from both trainings
                                   were presented at once.
        :bVerbose:          bool - Print output about training progress
        """

        # - Discrete time steps for evaluating input and target time series
        nNumTimeSteps = int(np.round(tsTarget.tDuration / self.tDt))
        vtTimeBase = self._gen_time_trace(tsTarget.tStart, nNumTimeSteps)

        # - Discard last sample to avoid counting time points twice
        vtTimeBase = vtTimeBase[:-1]

        # - Make sure vtTimeBase does not exceed tsTarget
        vtTimeBase = vtTimeBase[vtTimeBase <= tsTarget.tStop]

        # - Prepare target data
        mfTarget = tsTarget(vtTimeBase)

        # - Make sure no nan is in target, as this causes learning to fail
        assert not np.isnan(
            mfTarget
        ).any(), "Layer `{}`: nan values have been found in mfTarget (where: {})".format(
            self.strName, np.where(np.isnan(mfTarget))
        )

        # - Check target dimensions
        if mfTarget.ndim == 1 and self.nSize == 1:
            mfTarget = mfTarget.reshape(-1, 1)

        assert (
            mfTarget.shape[-1] == self.nSize
        ), "Layer `{}`: Target dimensions ({}) does not match layer size ({})".format(
            self.strName, mfTarget.shape[-1], self.nSize
        )

        # - Prepare input data

        # Empty input array with additional dimension for training biases
        mfInput = np.zeros((np.size(vtTimeBase), self.nSizeIn + 1))
        mfInput[:, -1] = 1

        # - Generate spike trains from tsInput
        if tsInput is None:
            # - Assume zero input
            print(
                "Layer `{}`: No tsInput defined, assuming input to be 0.".format(
                    self.strName
                )
            )

        else:
            # - Get data within given time range
            vtEventTimes, vnEventChannels, __ = tsInput.find(
                [vtTimeBase[0], vtTimeBase[-1]]
            )

            # - Make sure that input channels do not exceed layer input dimensions
            try:
                assert (
                    np.amax(vnEventChannels) <= self.nSizeIn - 1
                ), "Layer `{}`: Number of input channels exceeds layer input dimensions.".format(
                    self.strName
                )
            except ValueError as e:
                # - No events in input data
                if vnEventChannels.size == 0:
                    print(
                        "Layer `{}`: No input spikes for training.".format(self.strName)
                    )
                else:
                    raise e

            # Extract spike data from the input
            mnSpikeRaster = (
                tsInput.raster(
                    tDt=self.tDt,
                    tStart=vtTimeBase[0],
                    nNumTimeSteps=vtTimeBase.size,
                    vnSelectChannels=np.arange(self.nSizeIn),
                    bSamples=False,
                    bAddEvents=self.bAddEvents,
                )[2]
            ).astype(float)

            if bStoreState:
                try:
                    # - Include last state from previous batch
                    mnSpikeRaster[0, :] += self._vTrainingState
                except AttributeError:
                    pass

            # - Define exponential kernel
            vfKernel = np.exp(
                - (np.arange(vtTimeBase.size-1) * self.tDt) / self.tTauSyn
            )

            # - Apply kernel to spike trains and add filtered trains to input array
            for channel, vEvents in enumerate(mnSpikeRaster.T):
                mfInput[:, channel] = fftconvolve(vEvents, vfKernel, "full")[
                    : vtTimeBase.size
                ]

        # - Prepare batches for training
        if nBatchSize is None:
            nNumBatches = 1
            nBatchSize = nNumTimeSteps
        else:
            nNumBatches = int(np.ceil(nNumTimeSteps / float(nBatchSize)))
        
        viSampleOrder = np.arange(nNumTimeSteps)  # Indices to choose samples - shuffle for random order

        # - Iterate over epochs
        for iEpoch in range(nEpochs):
            # - Iterate over batches and optimize
            for iBatch in range(nNumBatches):
                viSampleIndices = viSampleOrder[iBatch * nBatchSize: (iBatch+1) * nBatchSize]
                # - Gradients
                mfGradients = self._gradients(
                    mfInput[viSampleIndices], mfTarget[viSampleIndices], fRegularize
                )
                self.mfW = self.mfW - fLearningRate * mfGradients[: -1, :]
                self.vfBias = self.vfBias - fLearningRate * mfGradients[-1, :]
            if bVerbose:
                print("Layer `{}`: Training epoch {} of {}".format(self.strName, iEpoch+1, nEpochs), end="\r")
            # - Shuffle samples
            np.random.shuffle(viSampleOrder)
        
        if bVerbose:
            print("Layer `{}`: Finished trainig.              ".format(self.strName))
        
        if bStoreState:    
            # - Store last state for next batch
            self._vTrainingState = mfInput[-1, :-1].copy()


    def _gradients(self, mfInput, mfTarget, fRegularize):
        # - Output with current weights
        mfLinear = mfInput[:, : -1] @ self.mfW + self.vfBias
        mfOutput = sigmoid(mfLinear)
        # - Gradients for weights
        nNumSamples = mfInput.shape[0]
        mfError = mfOutput - mfTarget
        mfGradients = (mfInput.T @ mfError) / float(nNumSamples)
        # - Regularization of weights
        if fRegularize > 0:
            mfGradients[: -1, :] += fRegularize / float(self.nSizeIn) * self.mfW
        
        return mfGradients

    def train_logreg_torch(
        self,
        tsTarget: TSContinuous,
        tsInput: TSEvent = None,
        fLearningRate: float = 0,
        fRegularize: float = 0,
        nBatchSize: Optional[int] = None,
        nEpochs: int = 1,
        bStoreState: bool = True,
        bVerbose: bool = False,
    ):

        """
        train_logreg - Train self with logistic regression over one of possibly many batches.
                       Note that this training method assumes that a sigmoid funciton is applied
                       to the layer output, which is not the case in self.evolve.
                       Use pytorch as backend
        :param tsTarget:    TimeSeries - target for current batch
        :param tsInput:     TimeSeries - input to self for current batch
        :fLearningRate:     flaot - Factor determining scale of weight increments at each step
        :fRegularize:       float - regularization parameter
        :nBatchSize:        int - Number of samples per batch. If None, train with all samples at once
        :nEpochs:           int - How many times is training repeated
        :bStoreState:       bool - Include last state from previous training and store state from this 
                                   traning. This has the same effect as if data from both trainings
                                   were presented at once.
        :bVerbose:          bool - Print output about training progress
        """

        if not torch.cuda.is_available():
            warn("Layer `{}`: CUDA not available. Will use cpu".format(self.strName))
            self.train_logreg(
                tsTarget, tsInput, fLearningRate, fRegularize, nBatchSize, nEpochs, bStoreState, bVerbose
            )
            return

        # - Discrete time steps for evaluating input and target time series
        nNumTimeSteps = int(np.round(tsTarget.tDuration / self.tDt))
        vtTimeBase = self._gen_time_trace(tsTarget.tStart, nNumTimeSteps)

        # - Discard last sample to avoid counting time points twice
        vtTimeBase = vtTimeBase[:-1]

        # - Make sure vtTimeBase does not exceed tsTarget
        vtTimeBase = vtTimeBase[vtTimeBase <= tsTarget.tStop]

        # - Prepare target data
        mfTarget = tsTarget(vtTimeBase)

        # - Make sure no nan is in target, as this causes learning to fail
        assert not np.isnan(
            mfTarget
        ).any(), "Layer `{}`: nan values have been found in mfTarget (where: {})".format(
            self.strName, np.where(np.isnan(mfTarget))
        )

        # - Check target dimensions
        if mfTarget.ndim == 1 and self.nSize == 1:
            mfTarget = mfTarget.reshape(-1, 1)

        assert (
            mfTarget.shape[-1] == self.nSize
        ), "Layer `{}`: Target dimensions ({}) does not match layer size ({})".format(
            self.strName, mfTarget.shape[-1], self.nSize
        )

        # - Prepare input data

        # Empty input array with additional dimension for training biases
        mfInput = np.zeros((np.size(vtTimeBase), self.nSizeIn + 1))
        mfInput[:, -1] = 1

        # - Generate spike trains from tsInput
        if tsInput is None:
            # - Assume zero input
            print(
                "Layer `{}`: No tsInput defined, assuming input to be 0.".format(
                    self.strName
                )
            )

        else:
            # - Get data within given time range
            vtEventTimes, vnEventChannels, __ = tsInput.find(
                [vtTimeBase[0], vtTimeBase[-1]]
            )

            # - Make sure that input channels do not exceed layer input dimensions
            try:
                assert (
                    np.amax(vnEventChannels) <= self.nSizeIn - 1
                ), "Layer `{}`: Number of input channels exceeds layer input dimensions.".format(
                    self.strName
                )
            except ValueError as e:
                # - No events in input data
                if vnEventChannels.size == 0:
                    print(
                        "Layer `{}`: No input spikes for training.".format(self.strName)
                    )
                else:
                    raise e

            # Extract spike data from the input
            mnSpikeRaster = (
                tsInput.raster(
                    tDt=self.tDt,
                    tStart=vtTimeBase[0],
                    nNumTimeSteps=vtTimeBase.size,
                    vnSelectChannels=np.arange(self.nSizeIn),
                    bSamples=False,
                    bAddEvents=self.bAddEvents,
                )[2]
            ).astype(float)

            if bStoreState:
                try:
                    # - Include last state from previous batch
                    mnSpikeRaster[0, :] += self._vTrainingState
                except AttributeError:
                    pass

            # - Define exponential kernel
            vfKernel = np.exp(
                - (np.arange(vtTimeBase.size-1) * self.tDt) / self.tTauSyn
            )

            # - Apply kernel to spike trains and add filtered trains to input array
            for channel, vEvents in enumerate(mnSpikeRaster.T):
                mfInput[:, channel] = fftconvolve(vEvents, vfKernel, "full")[
                    : vtTimeBase.size
                ]

        # - Move data to cuda
        ctTarget = torch.from_numpy(mfTarget).float().to("cuda")
        ctInput = torch.from_numpy(mfInput).float().to("cuda")
        ctWeights = torch.from_numpy(self.mfW).float().to("cuda")
        ctBiases = torch.from_numpy(self.vfBias).float().to("cuda")

        # - Prepare batches for training
        if nBatchSize is None:
            nNumBatches = 1
            nBatchSize = nNumTimeSteps
        else:
            nNumBatches = int(np.ceil(nNumTimeSteps / float(nBatchSize)))
        
        ctSampleOrder = torch.arange(nNumTimeSteps)  # Indices to choose samples - shuffle for random order

        # - Iterate over epochs
        for iEpoch in range(nEpochs):
            # - Iterate over batches and optimize
            for iBatch in range(nNumBatches):
                ctSampleIndices = ctSampleOrder[iBatch * nBatchSize: (iBatch+1) * nBatchSize]
                # - Gradients
                ctGradients = self._gradients_torch(
                    ctWeights, ctBiases, ctInput[ctSampleIndices], ctTarget[ctSampleIndices], fRegularize
                )
                ctWeights -= fLearningRate * ctGradients[: -1, :]
                ctBiases -= fLearningRate * ctGradients[-1, :]
            if bVerbose:
                print("Layer `{}`: Training epoch {} of {}".format(self.strName, iEpoch+1, nEpochs), end="\r")
            # - Shuffle samples
            ctSampleOrder = torch.randperm(nNumTimeSteps)
        
        if bVerbose:
            print("Layer `{}`: Finished trainig.              ".format(self.strName))
        
        if bStoreState:    
            # - Store last state for next batch
            self._vTrainingState = ctInput[-1, :-1].cpu().numpy()


    def _gradients_torch(self, ctWeights, ctBiases, ctInput, ctTarget, fRegularize):
        # - Output with current weights
        ctLinear = torch.mm(ctInput[:, : -1], ctWeights) + ctBiases
        ctOutput = sigmoid_torch(ctLinear)
        # - Gradients for weights
        nNumSamples = ctInput.size()[0]
        ctError = ctOutput - ctTarget
        ctGradients = torch.mm(ctInput.t(), ctError) / float(nNumSamples)
        # - Regularization of weights
        if fRegularize > 0:
            ctGradients[: -1, :] += fRegularize / float(self.nSizeIn) * ctWeights
        
        return ctGradients


    ### --- Properties

    @property
    def cInput(self):
        return TSEvent

    @property
    def tTauSyn(self):
        return self._tTauSyn

    @tTauSyn.setter
    def tTauSyn(self, tNewTau):
        assert tNewTau > 0, "Layer `{}`: tTauSyn must be greater than 0.".format(self.strName)
        self._tTauSyn = tNewTau
    
    @property
    def vfBias(self):
        return self._vfBias
    
    @vfBias.setter
    def vfBias(self, vfNewBias):
        self._vfBias = self._expand_to_net_size(vfNewBias, "vfBias", bAllowNone=False)

    @property
    def vState(self):
        return self._vStateNoBias + self._vfBias

    @vState.setter
    def vState(self, vNewState):
        vNewState = (
            np.asarray(self._expand_to_net_size(vNewState, "vState"))
        )
        self._vStateNoBias = vNewState - self._vfBias