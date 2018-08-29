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

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Configure exports
__all__ = ["FFExpSyn"]


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
            tDt=np.asarray(tDt),
            fNoiseStd=np.asarray(fNoiseStd),
            strName=strName,
        )

        # - Parameters
        self.tTauSyn = tTauSyn
        self.vfBias = self._correct_param_shape(vfBias)

        # - set time and state to 0
        self.reset_all()

    def _correct_param_shape(self, v) -> np.ndarray:
        """
        _correct_param_shape - Convert v to 1D-np.ndarray and verify
                              that dimensions match self.nSize
        :param v:   Float or array-like that is to be converted
        :return:    v as 1D-np.ndarray
        """
        v = np.array(v, dtype=float).flatten()
        assert v.shape in (
            (1,),
            (self.nSize,),
            (1, self.nSize),
            (self.nSize),
            1,
        ), "Numbers of elements in v must be 1 or match layer size"
        return v

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

        # - Prepare time base
        vtTimeBase, __, nNumTimeSteps = self._prepare_input(
            tsInput, tDuration, nNumTimeSteps
        )

        mSpikeTrains = np.zeros((vtTimeBase.size, self.nSize))

        # - Generate spike trains from tsInput
        if tsInput is None:
            # - Assume zero input
            mWeightedSpikeTrains = np.zeros((vtTimeBase.size, self.nSize))

        else:
            vtEventTimes, vnEventChannels, __ = tsInput.find(
                [vtTimeBase[0], vtTimeBase[-1]]
            )

            # - Make sure that input channels do not exceed layer input dimensions
            if vnEventChannels.size > 0:
                assert (
                    np.max(vnEventChannels) <= self.nSizeIn
                ), "Number of input channels exceeds layer input dimensions "

            # - Convert input events to spike trains
            mSpikeTrains = np.zeros((vtTimeBase.size, self.nSizeIn))
            #   Iterate over channel indices and create their spike trains
            for channel in range(self.nSizeIn):
                # Times with event in current channel
                vtEventTimesChannel = vtEventTimes[np.where(vnEventChannels == channel)]
                # Indices of vtTimeBase corresponding to these times
                viEventIndicesChannel = (
                    (vtEventTimesChannel - vtTimeBase[0]) / self.tDt
                ).astype(int)
                # Set spike trains for current channel
                mSpikeTrains[viEventIndicesChannel, channel] = 1

            # - Apply weights
            mWeightedSpikeTrains = mSpikeTrains @ self.mfW

        # Add current state
        mWeightedSpikeTrains[0, :] += self.vState

        # - Add a noise trace
        # - Noise correction is slightly different than in other layers
        mfNoise = (
            np.random.randn(*mWeightedSpikeTrains.shape)
            * self.fNoiseStd
            * np.sqrt(2 * self.tDt / self.tTauSyn)
        )
        mfNoise[0, :] = 0  # Make sure that noise trace starts with 0
        # mfNoise = np.zeros_like(mWeightedSpikeTrains)
        # mfNoise[0,:] = self.fNoiseStd

        mWeightedSpikeTrains += mfNoise

        # - Define exponential kernel
        vfKernel = np.exp(-np.arange(nNumTimeSteps + 1) * self.tDt / self.tTauSyn)
        # - Make sure spikes only have effect on next time step
        vfKernel = np.r_[0, vfKernel]

        # - Apply kernel to spike trains
        mfFiltered = np.zeros_like(mWeightedSpikeTrains)
        for channel, vEvents in enumerate(mWeightedSpikeTrains.T):
            vConv = fftconvolve(vEvents, vfKernel, "full")
            vConvShort = vConv[: vtTimeBase.size]
            mfFiltered[:, channel] = vConvShort

        # - Update time and state
        self._nTimeStep += nNumTimeSteps
        self.vState = mfFiltered[-1]

        # - Output time series with output data and bias
        return TSContinuous(
            vtTimeBase, mfFiltered + self.vfBias, strName="Receiver current"
        )

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

            # - Convert input events to spike trains
            mSpikeTrains = np.zeros((vtTimeBase.size, self.nSizeIn))
            #   Iterate over channel indices and create their spike trains
            for channel in range(self.nSizeIn):
                # Times with event in current channel
                vtEventTimesChannel = vtEventTimes[np.where(vnEventChannels == channel)]
                # Indices of vtTimeBase corresponding to these times
                viEventIndicesChannel = (
                    (vtEventTimesChannel - vtTimeBase[0]) / self.tDt
                ).astype(int)
                # Set spike trains for current channel
                mSpikeTrains[viEventIndicesChannel, channel] = 1

            # - Define exponential kernel
            vfKernel = np.exp(
                -np.arange(0, vtTimeBase[-1] - vtTimeBase[0], self.tDt) / self.tTauSyn
            )

            # - Apply kernel to spike trains and add filtered trains to input array
            for channel, vEvents in enumerate(mSpikeTrains.T):
                mfInput[:, channel] = fftconvolve(vEvents, vfKernel, "full")[
                    : vtTimeBase.size
                ]

        # - For first batch, initialize summands
        if bFirst:
            # Matrices to be updated for each batch
            self.mfXTY = np.zeros(
                (self.nSizeIn + 1, self.nSize)
            )  # mfInput.T (dot) mfTarget
            self.mfXTX = np.zeros(
                (self.nSizeIn + 1, self.nSizeIn + 1)
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

        else:
            # - In final step do not calculate rounding error but update matrices directly
            self.mfXTY += mfUpdXTY
            self.mfXTX += mfUpdXTX

            # - Weight and bias update by ridge regression
            mfSolution = np.linalg.solve(
                self.mfXTX + fRegularize * np.eye(self.nSizeIn + 1), self.mfXTY
            )
            self.mfW = mfSolution[:-1, :]
            self.vfBias = mfSolution[-1, :]

            # - Remove dat stored during this trainig
            self.mfXTY = self.mfXTX = self.mfKahanCompXTY = self.mfKahanCompXTX = None

    ### --- Properties

    @property
    def cInput(self):
        return TSEvent
