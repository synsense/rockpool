###
# exp_synapses_torch.py - Like exp_synapses_manual.py but with torch for computations.
###


# - Imports
from typing import Union
import numpy as np
from scipy.signal import fftconvolve
import torch

from ....timeseries import TSContinuous, TSEvent
from ..exp_synapses_manual import FFExpSyn

from typing import Optional, Union, Tuple, List

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Configure exports
__all__ = ["FFExpSyn"]

# - Absolute tolerance, e.g. for comparing float values
fTolAbs = 1e-9
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
        mfW: Union[np.ndarray, int] = None,
        vfBias: np.ndarray = 0,
        tDt: float = 0.0001,
        fNoiseStd: float = 0,
        tTauSyn: float = 0.005,
        strName: str = "unnamed",
        bAddEvents: bool = True,
        nMaxNumTimeSteps: int = nDefaultMaxNumTimeSteps,
    ):
        """
        FFExpSynTorch - Construct an exponential synapse layer (spiking input, pytorch as backend)

        :param mfW:             np.array MxN weight matrix
                                int Size of layer -> creates one-to-one conversion layer
        :param tDt:             float Time step for state evolution
        :param fNoiseStd:       float Std. dev. of noise added to this layer. Default: 0

        :param tTauSyn:         float Output synaptic time constants. Default: 5ms
        :param eqSynapses:      Brian2.Equations set of synapse equations for receiver. Default: exponential
        :param strIntegrator:   str Integrator to use for simulation. Default: 'exact'

        :param strName:         str Name for the layer. Default: 'unnamed'

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
                "Layer `{}`: Using CPU because CUDA is not available.".format(strName)
            )
            self.tensors = torch

        # - Bypass property setter to avoid unnecessary convolution kernel update
        assert (
            type(nMaxNumTimeSteps) == int and nMaxNumTimeSteps > 0.0
        ), "Layer `{}`: nMaxNumTimeSteps must be an integer greater than 0.".format(
            self.strName
        )
        self._nMaxNumTimeSteps = nMaxNumTimeSteps

        # - Call super constructor
        super().__init__(
            mfW=mfW,
            vfBias=vfBias,
            tDt=tDt,
            fNoiseStd=fNoiseStd,
            tTauSyn=tTauSyn,
            strName=strName,
            bAddEvents=bAddEvents,
        )

    ### --- State evolution

    # @profile
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

        # - Prepare input signal
        mnInputRaster, nNumTimeSteps = self._prepare_input(
            tsInput, tDuration, nNumTimeSteps
        )
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

        with torch.no_grad():
            # - Tensor for collecting output spike raster
            mfOutput = torch.FloatTensor(nNumTimeSteps + 1, self.nSize).fill_(0)

            # - Iterate over batches and run evolution
            iCurrentIndex = 1
            for mfCurrentInput, nCurrNumTS in self._batch_data(
                mfWeightedInput, nNumTimeSteps, self.nMaxNumTimeSteps
            ):
                mfOutput[
                    iCurrentIndex : iCurrentIndex + nCurrNumTS
                ] = self._single_batch_evolution(
                    mfCurrentInput,  # torch.from_numpy(mfCurrentInput).float().to(self.device),
                    nCurrNumTS,
                    bVerbose,
                )
                iCurrentIndex += nCurrNumTS

        # - Output time series with output data and bias
        return TSContinuous(
            vtTimeBase,
            (mfOutput + self._vfBias.cpu()).numpy(),
            strName="Filtered spikes",
        )

    # @profile
    def _batch_data(
        self, mfInput: np.ndarray, nNumTimeSteps: int, nMaxNumTimeSteps: int = None
    ) -> (np.ndarray, int):
        """_batch_data: Generator that returns the data in batches"""
        # - Handle None for nMaxNumTimeSteps
        nMaxNumTimeSteps = (
            nNumTimeSteps if nMaxNumTimeSteps is None else nMaxNumTimeSteps
        )
        nStart = 0
        while nStart < nNumTimeSteps:
            # - Endpoint of current batch
            nEnd = min(nStart + nMaxNumTimeSteps, nNumTimeSteps)
            # - Data for current batch
            mfCurrentInput = (
                torch.from_numpy(mfInput[nStart:nEnd]).float().to(self.device)
            )
            yield mfCurrentInput, nEnd - nStart
            # - Update nStart
            nStart = nEnd

    # @profile
    def _single_batch_evolution(
        self, mfWeightedInput: np.ndarray, nNumTimeSteps: int, bVerbose: bool = False
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input for a single batch

        :param mfWeightedInput: np.ndarray   Weighted input
        :param nNumTimeSteps:   int      Number of evolution time steps
        :param bVerbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        with torch.no_grad():

            # Add current state to input
            mfWeightedInput[0, :] += self._vStateNoBias.clone() * np.exp(
                -self.tDt / self.tTauSyn
            )

            # - Reshape input for convolution
            mfWeightedInput = mfWeightedInput.t().reshape(1, self.nSize, -1)

            # - Filter synaptic currents
            mfFiltered = (
                self._convSynapses(mfWeightedInput)[0].detach().t()[:nNumTimeSteps]
            )

            # - Store current state and update internal time
            self._vStateNoBias = mfFiltered[-1].clone()
            self._nTimeStep += nNumTimeSteps

        return mfFiltered

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

        with torch.no_grad():
            # - Move target data to GPU
            mfTarget = torch.from_numpy(mfTarget).float().to(self.device)

            # - Prepare input data

            # Empty input array with additional dimension for training biases
            nInputSize = self.nSizeIn + int(bTrainBiases)
            mfInput = self.tensors.FloatTensor(vtTimeBase.size, nInputSize).fill_(0)
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

            with torch.no_grad():
                # Extract spike data from the input variable and bring to GPU
                mnSpikeRaster = (
                    torch.from_numpy(
                        tsInput.raster(
                            tDt=self.tDt,
                            tStart=vtTimeBase[0],
                            nNumTimeSteps=vtTimeBase.size,
                            vnSelectChannels=np.arange(self.nSizeIn),
                            bSamples=False,
                            bAddEvents=self.bAddEvents,
                        )[2].astype(float)
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
                mnSpikeRaster = mnSpikeRaster.t().reshape(1, self.nSizeIn, -1)

                # - Filter synaptic currents and store in input tensor
                mfInput[:, :-1] = (
                    self._convSynapsesTraining(mnSpikeRaster)[0]
                    .detach()
                    .t()[: vtTimeBase.size]
                )

        with torch.no_grad():
            # - For first batch, initialize summands
            if bFirst:
                # Matrices to be updated for each batch
                self._mfXTY = self.tensors.FloatTensor(nInputSize, self.nSize).fill_(0)
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

            else:
                # - In final step do not calculate rounding error but update matrices directly
                self._mfXTY += mfUpdXTY
                self._mfXTX += mfUpdXTX

                # - Weight and bias update by ridge regression
                mfA = self._mfXTX + fRegularize * torch.eye(self.nSizeIn + 1).to(
                    self.device
                )
                mfSolution = torch.mm(mfA.inverse(), self._mfXTY).cpu().numpy()
                if bTrainBiases:
                    self.mfW = mfSolution[:-1, :]
                    self.vfBias = mfSolution[-1, :]
                else:
                    self.mfW = mfSolution

                # - Remove data stored during this trainig
                self._mfXTY = None
                self._mfXTX = None
                self._mfKahanCompXTY = None
                self._mfKahanCompXTX = None
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
                tsTarget,
                tsInput,
                fLearningRate,
                fRegularize,
                nBatchSize,
                nEpochs,
                bStoreState,
                bVerbose,
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
                -(np.arange(vtTimeBase.size - 1) * self.tDt) / self.tTauSyn
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

        ctSampleOrder = torch.arange(
            nNumTimeSteps
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
            if bVerbose:
                print(
                    "Layer `{}`: Training epoch {} of {}".format(
                        self.strName, iEpoch + 1, nEpochs
                    ),
                    end="\r",
                )
            # - Shuffle samples
            ctSampleOrder = torch.randperm(nNumTimeSteps)

        if bVerbose:
            print("Layer `{}`: Finished trainig.              ".format(self.strName))

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
            ctGradients[:-1, :] += fRegularize / float(self.nSizeIn) * ctWeights

        return ctGradients

    def _update_kernels(self):
        """Generate kernels for filtering input spikes during evolution and training"""
        nKernelSize = min(
            50
            * int(
                self._tTauSyn / self.tDt
            ),  # - Values smaller than ca. 1e-21 are neglected
            self._nMaxNumTimeSteps
            + 1,  # Kernel does not need to be larger than batch duration
        )
        vtTimes = (
            torch.arange(nKernelSize).to(self.device).reshape(1, -1).float() * self.tDt
        )

        # - Kernel for filtering recurrent spikes
        mfInputKernels = torch.exp(
            -vtTimes / self.tensors.FloatTensor(self.nSize, 1).fill_(self._tTauSyn)
        )
        # - Reverse on time axis and reshape to match convention of pytorch
        mfInputKernels = mfInputKernels.flip(1).reshape(self.nSize, 1, nKernelSize)
        # - Object for applying convolution
        self._convSynapses = torch.nn.Conv1d(
            self.nSize,
            self.nSize,
            nKernelSize,
            padding=nKernelSize - 1,
            groups=self.nSize,
            bias=False,
        ).to(self.device)
        self._convSynapses.weight.data = mfInputKernels

        # - Kernel for filtering recurrent spikes (uses unweighted input and therefore has different dimensions)
        mfInputKernelsTraining = self.tensors.FloatTensor(
            self.nSizeIn, nKernelSize
        ).fill_(0)
        mfInputKernelsTraining[:, 1:] = torch.exp(
            -vtTimes[:, :-1]
            / self.tensors.FloatTensor(self.nSizeIn, 1).fill_(self._tTauSyn)
        )
        # - Reverse on time axis and reshape to match convention of pytorch
        mfInputKernelsTraining = mfInputKernelsTraining.flip(1).reshape(
            self.nSizeIn, 1, nKernelSize
        )
        # - Object for applying convolution
        self._convSynapsesTraining = torch.nn.Conv1d(
            self.nSizeIn,
            self.nSizeIn,
            nKernelSize,
            padding=nKernelSize - 1,
            groups=self.nSizeIn,
            bias=False,
        ).to(self.device)
        self._convSynapsesTraining.weight.data = mfInputKernelsTraining

        print("Layer `{}`: Filter kernels have been updated.".format(self.strName))

    ### --- Properties

    @property
    def tTauSyn(self):
        return self._tTauSyn

    @tTauSyn.setter
    def tTauSyn(self, tNewTau, bNoKernelUpdate=False):
        assert tNewTau > 0, "Layer `{}`: tTauSyn must be greater than 0.".format(
            self.strName
        )
        self._tTauSyn = tNewTau
        self._update_kernels()

    @property
    def vfBias(self):
        return self._vfBias.cpu().numpy()

    @vfBias.setter
    def vfBias(self, vfNewBias):
        vfNewBias = self._expand_to_net_size(vfNewBias, "vfBias", bAllowNone=False)
        self._vfBias = torch.from_numpy(vfNewBias).float().to(self.device)

    @property
    def vState(self):
        return (self._vStateNoBias + self._vfBias).cpu().numpy()

    @vState.setter
    def vState(self, vNewState):
        vNewState = np.asarray(self._expand_to_net_size(vNewState, "vState"))
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
            self.strName
        )
        self._nMaxNumTimeSteps = nNewMax
        self._update_kernels()
