###
# exp_synapses_manual.py - Class implementing a spike-to-current layer with exponential synapses with pytorch as backend
###


# - Imports
from typing import Union
import numpy as np
from scipy.signal import fftconvolve
import torch

from ...timeseries import TSContinuous, TSEvent
from ..layer import Layer

from typing import Optional, Union, Tuple, List

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Configure exports
__all__ = ["FFExpSyn"]

# - Absolute tolerance, e.g. for comparing float values
fTolAbs = 1e-9
# - Default maximum numbers of time steps for a single evolution batch
nDefaultMaxNumTimeSteps = 400

## - FFExpSynTorch - Class: define an exponential synapse layer (spiking input, pytorch as backend)
class FFExpSynTorch(Layer):
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
        bAddEvents: bool = False,
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
            fNoiseStd=fNoiseStd,
            strName=strName,
        )

        # - Set device to cuda if available and determine how tensors should be instantiated
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.tensors = torch.cuda
        else:
            self.device = torch.device('cpu')
            print("Layer `{}`: Using CPU as CUDA is not available.".format(strName))
            self.tensors = torch

        # - Record layer parameters
        self.tTauSyn = tTauSyn
        self.vfBias = vfBias
        self.bAddEvents = bAddEvents

        # - Set time and state to 0
        self.reset_all()


    def _prepare_input(
        self,
        tsInput: Optional[TSEvent] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
    ) -> (np.ndarray, int):
        """
        _prepare_input - Sample input and apply weights. Also add noise.

        :param tsInput:      TimeSeries TxM or Tx1 Input signals for this layer
        :param tDuration:    float Duration of the desired evolution, in seconds
        :param nNumTimeSteps int Number of evolution time steps

        :return:
            mnWeightedInput:  ndarray Raster containing weighted spike info
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
            )[2]
            # Apply input weights
            mfWeightedInput = mnSpikeRaster @ self.mfW

        else:
            mfWeightedInput = np.zeros((nNumTimeSteps, self.nSize))

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

        return mfWeightedInput, nNumTimeSteps

    ### --- State evolution

    def evolve(
        self,
        tsInput: Optional[TSEvent] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
        bVerbose: bool = False,
        nMaxNumTimeSteps: int = nDefaultMaxNumTimeSteps,
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
        mfWeightedInput, nNumTimeSteps = self._prepare_input(tsInput, tDuration, nNumTimeSteps)
        # - Time base
        vtTimeBase = (np.arange(nNumTimeSteps + 1) + self._nTimeStep) * self.tDt

        # - Tensor for collecting output spike raster
        mfOutput = torch.FloatTensor(nNumTimeSteps+1, self.nSize).fill_(0)
        
        # - Iterate over batches and run evolution
        iCurrentIndex = 1
        for mfCurrentInput, nCurrNumTS in self._batch_data(
                mfWeightedInput, nNumTimeSteps, nMaxNumTimeSteps
            ):
            mfOutput[iCurrentIndex : iCurrentIndex+nCurrNumTS] = self._single_batch_evolution(
                torch.from_numpy(mfCurrentInput).float().to(self.device),
                nCurrNumTS,
                bVerbose,
            )
            iCurrentIndex += nCurrNumTS

        # - Output time series with output data and bias
        return TSContinuous(
            vtTimeBase, (mfOutput + self._vfBias.cpu()).numpy(), strName="Filtered spikes"
        )
    
    def _batch_data(
        self, mfInput: np.ndarray, nNumTimeSteps: int, nMaxNumTimeSteps: int = None,
    ) -> (np.ndarray, int):
        """_batch_data: Generator that returns the data in batches"""
        # - Handle None for nMaxNumTimeSteps
        nMaxNumTimeSteps = nNumTimeSteps if nMaxNumTimeSteps is None else nMaxNumTimeSteps
        nStart = 0
        while nStart < nNumTimeSteps:
            # - Endpoint of current batch
            nEnd = min(nStart + nMaxNumTimeSteps, nNumTimeSteps)
            # - Data for current batch
            mfCurrentInput = mfInput[nStart:nEnd]
            yield mfCurrentInput, nEnd-nStart
            # - Update nStart
            nStart = nEnd

    def _single_batch_evolution(
        self,
        mfWeightedInput: np.ndarray,
        nNumTimeSteps: int,
        bVerbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input for a single batch

        :param mfWeightedInput: np.ndarray   Weighted input
        :param nNumTimeSteps:   int      Number of evolution time steps
        :param bVerbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        # Add current state to input
        mfWeightedInput[0, :] += self._vStateNoBias.clone() * np.exp(-self.tDt / self.tTauSyn)

        # - Reshape input for convolution
        mfWeightedInput = mfWeightedInput.t().reshape(1,self.nSize,-1)
        
        # - Filter synaptic currents
        mfFiltered = self._convSynapses(mfWeightedInput)[0].detach().t()[:nNumTimeSteps]
        
        # - Store current state and update internal time
        self._vStateNoBias = mfFiltered[-1].clone()
        self._nTimeStep += nNumTimeSteps

        return mfFiltered

    def train_rr(
        self,
        tsTarget: TSContinuous,
        tsInput: TSEvent = None,
        fRegularize=0,
        bFirst=True,
        bFinal=False,
    ):

        """
        train_rr - Train self with ridge regression over one of possibly
                   many batches. Use Kahan summation to reduce rounding
                   errors when adding data to existing matrices from
                   previous batches.
        :param tsTarget:    TimeSeries - target for current batch
        :param tsInput:     TimeSeries - input to self for current batch
        :fRegularize:       float - regularization for ridge regression
        :bFirst:            bool - True if current batch is the first in training
        :bFinal:            bool - True if current batch is the last in training
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

        # # - Store some objects for debuging
        # self.tsTarget = tsTarget
        # self.tsInput = tsInput
        # self.mfTarget = mfTarget
        # self.vtTimeBase = vtTimeBase

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

        # - Move target data to GPU
        mfTarget = torch.from_numpy(mfTarget).float().to(self.device)

        # - Prepare input data

        # Empty input array with additional dimension for training biases
        mfInput = self.tensors.FloatTensor(vtTimeBase.size, self.nSizeIn + 1).fill_(0)
        mfInput[:, -1] = 1
        print(mfTarget.shape)
        print(mfInput.shape)

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

            # Extract spike data from the input variable and bring to GPU
            mnSpikeRaster = torch.from_numpy(
                tsInput.raster(
                    tDt=self.tDt,
                    tStart=vtTimeBase[0],
                    nNumTimeSteps=vtTimeBase.size,
                    vnSelectChannels=np.arange(self.nSizeIn),
                    bSamples=False,
                    bAddEvents=self.bAddEvents,
                )[2].astype(int)
            ).float().to(self.device)

            # - Reshape input for convolution
            mnSpikeRaster = mnSpikeRaster.t().reshape(1,self.nSizeIn,-1)
            
            # - Filter synaptic currents and store in input tensor
            mfInput[:, :-1] = self._convSynapsesTraining(mnSpikeRaster)[0].detach().t()[:vtTimeBase.size]

        # - For first batch, initialize summands
        if bFirst:
            # Matrices to be updated for each batch
            self._mfXTY = self.tensors.FloatTensor(
                self.nSizeIn + 1, self.nSize
            ).fill_(0)
            self._mfXTX = self.tensors.FloatTensor(
                self.nSizeIn + 1, self.nSizeIn + 1
            ).fill_(0)
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

        else:
            # - In final step do not calculate rounding error but update matrices directly
            self._mfXTY += mfUpdXTY
            self._mfXTX += mfUpdXTX

            # - Weight and bias update by ridge regression
            mfA = self._mfXTX + fRegularize * torch.eye(self.nSizeIn + 1).to(self.device)
            mfSolution = torch.mm(mfA.inverse(), self._mfXTY).cpu().numpy()
            self.mfW = mfSolution[:-1, :]
            self.vfBias = mfSolution[-1, :]

            # - Remove data stored during this trainig
            self._mfXTY = self._mfXTX = self._mfKahanCompXTY = self._mfKahanCompXTX = None

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
    
        # - Kernel for filtering recurrent spikes
        nKernelSize = 50 * int(tNewTau / self.tDt) # - Values smaller than ca. 1e-21 are neglected
        vtTimes = torch.arange(nKernelSize).to(self.device).reshape(1,-1).float() * self.tDt
        mfInputKernels = torch.exp(-vtTimes / self.tensors.FloatTensor(self.nSize, 1).fill_(tNewTau))
        # - Reverse on time axis and reshape to match convention of pytorch
        mfInputKernels = mfInputKernels.flip(1).reshape(self.nSize, 1, nKernelSize)
        # - Object for applying convolution
        self._convSynapses = torch.nn.Conv1d(
            self.nSize, self.nSize, nKernelSize, padding=nKernelSize-1, groups=self.nSize, bias=False
        ).to(self.device)
        self._convSynapses.weight.data = mfInputKernels

        # - Kernel for filtering recurrent spikes (uses unweighted input and therefore has different dimensions)
        mfInputKernelsTraining = torch.exp(-vtTimes / self.tensors.FloatTensor(self.nSizeIn, 1).fill_(tNewTau))
        # - Reverse on time axis and reshape to match convention of pytorch
        mfInputKernelsTraining = mfInputKernelsTraining.flip(1).reshape(self.nSizeIn, 1, nKernelSize)
        # - Object for applying convolution
        self._convSynapsesTraining = torch.nn.Conv1d(
            self.nSizeIn, self.nSizeIn, nKernelSize, padding=nKernelSize-1, groups=self.nSizeIn, bias=False
        ).to(self.device)
        self._convSynapsesTraining.weight.data = mfInputKernelsTraining
        
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
        vNewState = (
            np.asarray(self._expand_to_net_size(vNewState, "vState"))
        )
        self._vStateNoBias = torch.from_numpy(vNewState).float().to(self.device) - self._vfBias

    