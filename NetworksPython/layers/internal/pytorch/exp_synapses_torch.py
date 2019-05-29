###
# exp_synapses_torch.py - Like exp_synapses_manual.py but with torch for computations.
###


# - Imports
import json
from warnings import warn
from typing import Union, Optional
import numpy as np
from scipy.signal import fftconvolve
import torch

from ....timeseries import TSContinuous, TSEvent
from ..exp_synapses_manual import FFExpSyn
from ...layer import ArrayLike, RefProperty


# - Configure exports
__all__ = ["FFExpSyn"]

# - Absolute tolerance, e.g. for comparing float values
tol_abs = 1e-9
# - Default maximum numbers of time steps for a single evolution batch
nDefaultMaxNumTimeSteps = 5000


def sigmoid(z):
    return 1.0 / (1.0 + torch.exp(-z))


## - FFExpSynTorch - Class: define an exponential synapse layer (spiking input, pytorch as backend)
class FFExpSynTorch(FFExpSyn):
    """ FFExpSynTorch - Class: define an exponential synapse layer (spiking input, pytorch as backend)
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
        nMaxNumTimeSteps: int = nDefaultMaxNumTimeSteps,
    ):
        """
        FFExpSynTorch - Construct an exponential synapse layer (spiking input, pytorch as backend)

        :param weights:             np.array MxN weight matrix
                                int Size of layer -> creates one-to-one conversion layer
        :param dt:             float Time step for state evolution
        :param noise_std:       float Std. dev. of noise added to this layer. Default: 0

        :param tTauSyn:         float Output synaptic time constants. Default: 5ms
        :param eqSynapses:      Brian2.Equations set of synapse equations for receiver. Default: exponential
        :param strIntegrator:   str Integrator to use for simulation. Default: 'exact'

        :param name:         str Name for the layer. Default: 'unnamed'

        :bAddEvents:            bool  If during evolution multiple input events arrive during one
                                      time step for a channel, count their actual number instead of
                                      just counting them as one.

        :nMaxNumTimeSteps:      int   Maximum number of timesteps during single evolution batch. Longer
                                      evolution periods will automatically split in smaller batches.
        """

        # - Set device to cuda if available and determine how tensors should be instantiated
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.tensors = torch.cuda
        else:
            self.device = torch.device("cpu")
            print(
                "Layer `{}`: Using CPU because CUDA is not available.".format(name)
            )
            self.tensors = torch


        # - Bypass property setter to avoid unnecessary convolution kernel update
        assert (
            type(nMaxNumTimeSteps) == int and nMaxNumTimeSteps > 0.0
        ), "Layer `{}`: nMaxNumTimeSteps ({nStep}) must be an integer greater than 0.".format(
            name, nStep=nMaxNumTimeSteps
        )
        self._nMaxNumTimeSteps = nMaxNumTimeSteps

        # - Call super constructor
        super().__init__(
            weights=weights,
            vfBias=vfBias,
            dt=dt,
            noise_std=noise_std,
            tTauSyn=tTauSyn,
            name=name,
            bAddEvents=bAddEvents,
        )

    ### --- State evolution

    # @profile
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

        # - Prepare input signal
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

        with torch.no_grad():
            # - Tensor for collecting output spike raster
            mfOutput = torch.FloatTensor(num_timesteps + 1, self.size).fill_(0)

            # - Iterate over batches and run evolution
            iCurrentIndex = 1
            for mfCurrentInput, nCurrNumTS in self._batch_data(
                mfWeightedInput, num_timesteps, self.nMaxNumTimeSteps
            ):
                mfOutput[
                    iCurrentIndex : iCurrentIndex + nCurrNumTS
                ] = self._single_batch_evolution(
                    mfCurrentInput,  # torch.from_numpy(mfCurrentInput).float().to(self.device),
                    nCurrNumTS,
                    verbose,
                )
                iCurrentIndex += nCurrNumTS

        # - Output time series with output data and bias
        return TSContinuous(
            vtTimeBase,
            (mfOutput + self._vfBias.cpu()).numpy(),
            name="Filtered spikes",
        )

    # @profile
    def _batch_data(
        self, mfInput: np.ndarray, num_timesteps: int, nMaxNumTimeSteps: int = None
    ) -> (np.ndarray, int):
        """_batch_data: Generator that returns the data in batches"""
        # - Handle None for nMaxNumTimeSteps
        nMaxNumTimeSteps = (
            num_timesteps if nMaxNumTimeSteps is None else nMaxNumTimeSteps
        )
        nStart = 0
        while nStart < num_timesteps:
            # - Endpoint of current batch
            nEnd = min(nStart + nMaxNumTimeSteps, num_timesteps)
            # - Data for current batch
            mfCurrentInput = (
                torch.from_numpy(mfInput[nStart:nEnd]).float().to(self.device)
            )
            yield mfCurrentInput, nEnd - nStart
            # - Update nStart
            nStart = nEnd

    # @profile
    def _single_batch_evolution(
        self, mfWeightedInput: np.ndarray, num_timesteps: int, verbose: bool = False
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input for a single batch

        :param mfWeightedInput: np.ndarray   Weighted input
        :param num_timesteps:   int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        with torch.no_grad():

            # Add current state to input
            mfWeightedInput[0, :] += self._vStateNoBias.clone() * np.exp(
                -self.dt / self.tTauSyn
            )

            # - Reshape input for convolution
            mfWeightedInput = mfWeightedInput.t().reshape(1, self.size, -1)

            # - Filter synaptic currents
            mfFiltered = (
                self._convSynapses(mfWeightedInput)[0].detach().t()[:num_timesteps]
            )

            # - Store current state and update internal time
            self._vStateNoBias = mfFiltered[-1].clone()
            self._timestep += num_timesteps

        return mfFiltered

    def train_rr(
        self,
        tsTarget: TSContinuous,
        ts_input: TSEvent = None,
        fRegularize: float = 0,
        bFirst: bool = True,
        bFinal: bool = False,
        bStoreState: bool = True,
        bTrainBiases: bool = True,
        bIntermediateResults: bool = False
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
        :param bIntermediateResults: bool - If True, calculates the intermediate weights not in the final batch
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

        with torch.no_grad():
            # - Move target data to GPU
            mfTarget = torch.from_numpy(mfTarget).float().to(self.device)

            # - Prepare input data

            # Empty input array with additional dimension for training biases
            nInputSize = self.size_in + int(bTrainBiases)
            mfInput = self.tensors.FloatTensor(vtTimeBase.size, nInputSize).fill_(0)
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

            with torch.no_grad():
                # Extract spike data from the input variable and bring to GPU
                mnSpikeRaster = (
                    torch.from_numpy(
                        ts_input.raster(
                            dt=self.dt,
                            t_start=vtTimeBase[0],
                            num_timesteps=vtTimeBase.size,
                            channels=np.arange(self.size_in),
                            add_events=self.bAddEvents,
                        ).astype(float)
                    )
                    .float()
                    .to(self.device)
                )

                if bStoreState and not bFirst:
                    try:
                        # - Include last state from previous batch
                        mnSpikeRaster[0, :] += self._vTrainingState
                    except AttributeError:
                        pass

                # - Reshape input for convolution
                mnSpikeRaster = mnSpikeRaster.t().reshape(1, self.size_in, -1)

                # - Filter synaptic currents and store in input tensor
                if bTrainBiases:
                    mfInput[:, :-1] = (
                        self._convSynapsesTraining(mnSpikeRaster)[0]
                        .detach()
                        .t()[: vtTimeBase.size]
                    )
                else:
                    mfInput[:, :] = (
                        self._convSynapsesTraining(mnSpikeRaster)[0]
                            .detach()
                            .t()[: vtTimeBase.size]
                    )


        with torch.no_grad():
            # - For first batch, initialize summands
            if bFirst:
                # Matrices to be updated for each batch
                self._mfXTY = self.tensors.FloatTensor(nInputSize, self.size).fill_(0)
                self._mfXTX = self.tensors.FloatTensor(nInputSize, nInputSize).fill_(0)
                # Corresponding Kahan compensations
                self._mfKahanCompXTY = self._mfXTY.clone()
                self._mfKahanCompXTX = self._mfXTX.clone()

            # - New data to be added, including compensation from last batch
            #   (Matrix summation always runs over time)
            mfUpdXTY = torch.mm(mfInput.t(), mfTarget) - self._mfKahanCompXTY
            mfUpdXTX = torch.mm(mfInput.t(), mfInput) - self._mfKahanCompXTX

            if not bFinal:
                # - Update matrices with new data
                mfNewXTY = self._mfXTY + mfUpdXTY
                mfNewXTX = self._mfXTX + mfUpdXTX
                # - Calculate rounding error for compensation in next batch
                self._mfKahanCompXTY = (mfNewXTY - self._mfXTY) - mfUpdXTY
                self._mfKahanCompXTX = (mfNewXTX - self._mfXTX) - mfUpdXTX
                # - Store updated matrices
                self._mfXTY = mfNewXTY
                self._mfXTX = mfNewXTX

                if bStoreState:
                    # - Store last state for next batch
                    if bTrainBiases:
                        self._vTrainingState = mfInput[-1, :-1].clone()
                    else:
                        self._vTrainingState = mfInput[-1, :].clone()

                if bIntermediateResults:
                    mfA = self._mfXTX + fRegularize * torch.eye(self.size_in + 1).to(
                        self.device
                    )
                    mfSolution = torch.mm(mfA.inverse(), self._mfXTY).cpu().numpy()
                    if bTrainBiases:
                        self.weights = mfSolution[:-1, :]
                        self.vfBias = mfSolution[-1, :]
                    else:
                        self.weights = mfSolution
            else:
                # - In final step do not calculate rounding error but update matrices directly
                self._mfXTY += mfUpdXTY
                self._mfXTX += mfUpdXTX

                # - Weight and bias update by ridge regression
                if bTrainBiases:
                    mfA = self._mfXTX + fRegularize * torch.eye(self.size_in + 1).to(
                        self.device
                    )
                else:
                    mfA = self._mfXTX + fRegularize * torch.eye(self.size_in).to(
                        self.device
                    )

                mfSolution = torch.mm(mfA.inverse(), self._mfXTY).cpu().numpy()
                if bTrainBiases:
                    self.weights = mfSolution[:-1, :]
                    self.vfBias = mfSolution[-1, :]
                else:
                    self.weights = mfSolution

                # - Remove data stored during this trainig
                self._mfXTY = None
                self._mfXTX = None
                self._mfKahanCompXTY = None
                self._mfKahanCompXTX = None
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
                       Use pytorch as backend
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

        if not torch.cuda.is_available():
            warn("Layer `{}`: CUDA not available. Will use cpu".format(self.name))
            self.train_logreg(
                tsTarget,
                ts_input,
                fLearningRate,
                fRegularize,
                nBatchSize,
                nEpochs,
                bStoreState,
                verbose,
            )
            return

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

        # - Move data to cuda
        ctTarget = torch.from_numpy(mfTarget).float().to("cuda")
        ctInput = torch.from_numpy(mfInput).float().to("cuda")
        ctWeights = torch.from_numpy(self.weights).float().to("cuda")
        ctBiases = torch.from_numpy(self.vfBias).float().to("cuda")

        # - Prepare batches for training
        if nBatchSize is None:
            nNumBatches = 1
            nBatchSize = num_timesteps
        else:
            nNumBatches = int(np.ceil(num_timesteps / float(nBatchSize)))

        ctSampleOrder = torch.arange(
            num_timesteps
        )  # Indices to choose samples - shuffle for random order

        # - Iterate over epochs
        for iEpoch in range(nEpochs):
            # - Iterate over batches and optimize
            for iBatch in range(nNumBatches):
                ctSampleIndices = ctSampleOrder[
                    iBatch * nBatchSize : (iBatch + 1) * nBatchSize
                ]
                # - Gradients
                ctGradients = self._gradients(
                    ctWeights,
                    ctBiases,
                    ctInput[ctSampleIndices],
                    ctTarget[ctSampleIndices],
                    fRegularize,
                )
                ctWeights -= fLearningRate * ctGradients[:-1, :]
                ctBiases -= fLearningRate * ctGradients[-1, :]
            if verbose:
                print(
                    "Layer `{}`: Training epoch {} of {}".format(
                        self.name, iEpoch + 1, nEpochs
                    ),
                    end="\r",
                )
            # - Shuffle samples
            ctSampleOrder = torch.randperm(num_timesteps)

        if verbose:
            print("Layer `{}`: Finished trainig.              ".format(self.name))

        if bStoreState:
            # - Store last state for next batch
            self._vTrainingState = ctInput[-1, :-1].cpu().numpy()

    def _gradients(self, ctWeights, ctBiases, ctInput, ctTarget, fRegularize):
        # - Output with current weights
        ctLinear = torch.mm(ctInput[:, :-1], ctWeights) + ctBiases
        ctOutput = sigmoid(ctLinear)
        # - Gradients for weights
        nNumSamples = ctInput.size()[0]
        ctError = ctOutput - ctTarget
        ctGradients = torch.mm(ctInput.t(), ctError) / float(nNumSamples)
        # - Regularization of weights
        if fRegularize > 0:
            ctGradients[:-1, :] += fRegularize / float(self.size_in) * ctWeights

        return ctGradients

    def _update_kernels(self):
        """Generate kernels for filtering input spikes during evolution and training"""
        nKernelSize = min(
            50
            * int(
                self._tTauSyn / self.dt
            ),  # - Values smaller than ca. 1e-21 are neglected
            self._nMaxNumTimeSteps
            + 1,  # Kernel does not need to be larger than batch duration
        )
        vtTimes = (
            torch.arange(nKernelSize).to(self.device).reshape(1, -1).float() * self.dt
        )

        # - Kernel for filtering recurrent spikes
        mfInputKernels = torch.exp(
            -vtTimes / self.tensors.FloatTensor(self.size, 1).fill_(self._tTauSyn)
        )
        # - Reverse on time axis and reshape to match convention of pytorch
        mfInputKernels = mfInputKernels.flip(1).reshape(self.size, 1, nKernelSize)
        # - Object for applying convolution
        self._convSynapses = torch.nn.Conv1d(
            self.size,
            self.size,
            nKernelSize,
            padding=nKernelSize - 1,
            groups=self.size,
            bias=False,
        ).to(self.device)
        self._convSynapses.weight.data = mfInputKernels

        # - Kernel for filtering recurrent spikes (uses unweighted input and therefore has different dimensions)
        mfInputKernelsTraining = self.tensors.FloatTensor(
            self.size_in, nKernelSize
        ).fill_(0)
        mfInputKernelsTraining[:, 1:] = torch.exp(
            -vtTimes[:, :-1]
            / self.tensors.FloatTensor(self.size_in, 1).fill_(self._tTauSyn)
        )
        # - Reverse on time axis and reshape to match convention of pytorch
        mfInputKernelsTraining = mfInputKernelsTraining.flip(1).reshape(
            self.size_in, 1, nKernelSize
        )
        # - Object for applying convolution
        self._convSynapsesTraining = torch.nn.Conv1d(
            self.size_in,
            self.size_in,
            nKernelSize,
            padding=nKernelSize - 1,
            groups=self.size_in,
            bias=False,
        ).to(self.device)
        self._convSynapsesTraining.weight.data = mfInputKernelsTraining

        print("Layer `{}`: Filter kernels have been updated.".format(self.name))

    ### --- Properties

    @property
    def tTauSyn(self):
        return self._tTauSyn

    @tTauSyn.setter
    def tTauSyn(self, tNewTau, bNoKernelUpdate=False):
        assert tNewTau > 0, "Layer `{}`: tTauSyn must be greater than 0.".format(
            self.name
        )
        self._tTauSyn = tNewTau
        self._update_kernels()

    @RefProperty
    def vfBias(self):
        return self._vfBias

    @vfBias.setter
    def vfBias(self, vfNewBias):
        vfNewBias = self._expand_to_net_size(vfNewBias, "vfBias", bAllowNone=False)
        self._vfBias = torch.from_numpy(vfNewBias).float().to(self.device)

    @RefProperty
    def mfXTX(self):
        return self._mfXTX

    @RefProperty
    def mfXTY(self):
        return self._mfXTY

    @property
    def state(self):
        warn(
            "Layer `{}`: Changing values of returned object by item assignment will not have effect on layer's state".format(
                self.name
            )
        )
        return (self._vStateNoBias + self._vfBias).cpu().numpy()

    @state.setter
    def state(self, vNewState):
        vNewState = np.asarray(self._expand_to_net_size(vNewState, "state"))
        self._vStateNoBias = (
            torch.from_numpy(vNewState).float().to(self.device) - self._vfBias
        )

    @property
    def nMaxNumTimeSteps(self):
        return self._nMaxNumTimeSteps

    @nMaxNumTimeSteps.setter
    def nMaxNumTimeSteps(self, nNewMax):
        assert (
            type(nNewMax) == int and nNewMax > 0.0
        ), "Layer `{}`: nMaxNumTimeSteps must be an integer greater than 0.".format(
            self.name
        )
        self._nMaxNumTimeSteps = nNewMax
        self._update_kernels()

    def to_dict(self):

        config = {}
        config["weights"] = self.weights.tolist()
        config["bias"] = self._vfBias if type(self._vfBias) is float else self._vfBias.tolist()
        config["dt"] = self.dt
        config["noise_std"] = self.noise_std
        config["tauS"] = self.tTauSyn if type(self.tTauSyn) is float else self.tTauSyn.tolist()
        config["name"] = self.name
        config["bAddEvents"] = self.bAddEvents
        config["nMaxNumTimeSteps"] = self.nMaxNumTimeSteps
        config["ClassName"] = "FFExpSynTorch"

        return config

    def save(self, config, filename):
        with open(filename, "w") as f:
            json.dump(config, f)

    @staticmethod
    def load_from_dict(config):
        return FFExpSynTorch(
            weights = config["weights"],
            vfBias = config["bias"],
            dt = config["dt"],
            noise_std = config["noise_std"],
            tTauSyn = config["tauS"],
            name = config["name"],
            bAddEvents = config["bAddEvents"],
            nMaxNumTimeSteps = config["nMaxNumTimeSteps"],
        )
