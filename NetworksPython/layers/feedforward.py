import numpy as np
import time
from abc import abstractmethod
from typing import Callable
from numba import njit

import TimeSeries as ts
from layers.layer import Layer

fTolerance = 1e-5

@njit
def fhReLU(vfX: np.ndarray) -> np.ndarray:
    """
    Activation function for rectified linear units.
    :param vfX:             ndarray with current neuron potentials
    :return:                np.clip(vfX, 0, None)
    """
    mfCopy = np.copy(vfX)
    mfCopy[np.where(vfX < 0)] = 0
    return mfCopy

@njit
def noisy(mX: np.ndarray, fStdDev: float) -> np.ndarray:
    """
    noisy - Add randomly distributed noise to each element of mX
    :param mX:  Array-like with values that noise is added to
    :param fStdDev: Float, the standard deviation of the noise to be added
    :return:        Array-like, mX with noise added
    """
    return fStdDev * np.random.randn(*mX.shape) + mX

def print_progress(iCurr: int, nTotal: int, tPassed: float):
    print('Progress: [{:6.1%}]    in {:6.1f} s. Remaining:   {:6.1f}'.format(
             iCurr/nTotal, tPassed, tPassed*(nTotal-iCurr)/max(0.1, iCurr)),
           end='\r')

def get_evolution_function(fhActivation: Callable[[np.ndarray], np.ndarray]):
    """
    get_evolution_function: Construct a compiled Euler solver for a given activation function

    :param fhActivation: Callable (x) -> f(x)
    :return: Compiled function evolve_Euler_complete(vState, nSize, mfW, mfInputStep, nNumSteps, vfBias, vtTau)
    """

    # - Compile an Euler solver for the desired activation function
    @njit
    def evolve_Euler_complete(vState: np.ndarray,
                              mfInput: np.ndarray,
                              mfW: np.ndarray,
                              vfGain: np.ndarray,
                              vfBias: np.ndarray,
                              vfAlpha: np.ndarray,
                              fNoiseStd) -> np.ndarray:

        # - Initialise storage of layer output
        nNumSteps = len(mfInput)
        mfWeightedInput = mfInput@mfW
        mfActivities = np.zeros_like(mfWeightedInput)

        # - Loop over time steps. The updated vState already corresponds to
        # subsequent time step. Therefore skip state update in final step
        # and only update activation.
        for nStep in range(nNumSteps-1):
            # - Store layer activity
            mfActivities[nStep, :] = fhActivation(vState + vfBias)
            # - Evolve layer state
            vDState = -vState + noisy(vfGain * mfWeightedInput[nStep, :], fNoiseStd)
            vState += vDState * vfAlpha
        mfActivities[-1, :] = fhActivation(vState + vfBias)

        return mfActivities

    # - Return the compiled function
    return evolve_Euler_complete


class PassThrough(Layer):
    """ Neuron states directly correspond to input, but can be delayed. """

    def __init__(self,
                 mfW: np.ndarray,
                 tDt: float = 1,
                 fNoiseStd: float = 0,
                 tDelay: float = 0,
                 strName: str = None):
        super().__init__(mfW=mfW, tDt=tDt, fNoiseStd=fNoiseStd, strName=strName)
        self._tDelay = tDelay
        self.reset_all()

        # Buffer already reset by super().__init__ which calls self.reset_all()
        # self.reset_buffer()

    def reset_buffer(self):
        if self.tDelay != 0:
            # - Make sure that self.tDelay is a multiple of self.tDt
            if (min(self.tDelay%self.tDt, self.tDt-self.tDelay%self.tDt) 
                > fTolerance * self.tDt):
                raise ValueError('tDelay must be a multiple of tDt')

            vtBuffer = np.arange(0, self.tDelay+self._tDt, self._tDt)
            self.tsBuffer = ts.TimeSeries(vtBuffer,
                                          np.zeros((len(vtBuffer), self.nSize)))
        else:
            self.tsBuffer = None

    def evolve(self, tsInput: np.ndarray, tDuration: float = None):
        """
        evolve - Evolve the state of this layer

        :param tsInput:     TimeSeries TxM or Tx1 input to this layer
        :param tDuration:   float Duration of evolution, in seconds

        :return: TimeSeries Output of this layer during evolution period
        """

        # - Discretize input time series
        vtTime, mfInput, tTrueDuration = self._prepare_input(tsInput, tDuration)

        # - Apply input weights and add noise
        mfProcessed = noisy(mfInput@self.mfW, self.fNoiseStd)

        if self.tsBuffer is not None:
            nBufferSteps = len(self.tsBuffer.vtTimeTrace)
            nInputSteps = len(vtTime)
            if nInputSteps >= nBufferSteps: # Input is as least as buffer
                # - Output buffer content, then first part of new input
                mSamplesOut = np.vstack((self.tsBuffer.mfSamples[:-1],
                                         mfProcessed[:-(nBufferSteps-1)]))
                # - Fill buffer with last part of new input
                self.tsBuffer.mfSamples = mfProcessed[-nBufferSteps:]
            else:  # Buffer is longer than input
                # - Output older part of buffer content
                # mSamplesOut = self.tsBuffer(vtTime-self.t)
                mSamplesOut = self.tsBuffer.mfSamples[:nInputSteps]
                # - Remove older part from buffer, move newer part to beginning and add new input
                self.tsBuffer.mfSamples = np.vstack((self.tsBuffer.mfSamples[nInputSteps-1:-1],
                                                     mfProcessed))
        else:
            mSamplesOut = noisy(mfInput@self.mfW, self.fNoiseStd)

        self._t += tTrueDuration

        return ts.TimeSeries(vtTime, mSamplesOut)

    def reset_state(self):
        super().reset_state()
        self.reset_buffer()

    def reset_all(self):
        super().reset_all()
        self.reset_buffer()

    @property
    def tDelay(self):
        return self._tDelay

    # @tDelay.setter
    # def tDelay(self, tNewDelay):
        # Some method to extend self.tsBuffer


class FFRateEuler(Layer):
    """ Feedforward layer consisting of rate-based neurons """

    def __init__(self,
                 mfW: np.ndarray,
                 tDt: float = 1,
                 strName: str = None,
                 fNoiseStd: float = 0,
                 fhActivation: Callable[[np.ndarray], np.ndarray] = fhReLU,
                 vtTau: np.ndarray = 10,
                 vfGain: np.ndarray = 1,
                 vfBias: np.ndarray = 0):
        super().__init__(mfW=mfW.astype(float), tDt=tDt, fNoiseStd=fNoiseStd, strName=strName)
        self.reset_all()
        try:
            self.vtTau, self.vfGain, self.vfBias = map(self.correct_param_shape, (vtTau, vfGain, vfBias))
        except AssertionError:
            raise AssertionError('Numbers of elements in vtTau, vfGain and vfBias'
                                 + ' must be 1 or match layer size.')
        self.vfAlpha = self._tDt/self.vtTau
        self.fhActivation = fhActivation

    def correct_param_shape(self, v) -> np.ndarray:
        """
        correct_param_shape - Convert v to 1D-np.ndarray and verify
                              that dimensions match self.nSize
        :param v:   Float or array-like that is to be converted
        :return:    v as 1D-np.ndarray
        """
        v = np.array(v, dtype=float).flatten()
        assert v.shape in ((1,), (self.nSize,), (1,self.nSize), (self.nSize), 1), (
            'Numbers of elements in v must be 1 or match layer size')
        return v

    def evolve(self, tsInput: ts.TimeSeries = None, tDuration: float = None) -> ts.TimeSeries:
        """
        evolve - Evolve the state of this layer

        :param tsInput:     TimeSeries TxM or Tx1 input to this layer
        :param tDuration:   float Duration of evolution, in seconds

        :return: TimeSeries Output of this layer during evolution period
        """

        vtTime, mfInput, tTrueDuration = self._prepare_input(tsInput, tDuration)

        mSamplesAct = self._evolveEuler(vState=self._vState,     #self._vState is automatically updated
                                        mfInput=mfInput,
                                        mfW=self.mfW,
                                        vfGain=self.vfGain,
                                        vfBias=self.vfBias,
                                        vfAlpha=self.vfAlpha,
                                        fNoiseStd=self.fNoiseStd/np.sqrt(self.tDt))

        # rtStart = time.time()
        # for i, vIn in enumerate(mSamplesIn):
        #     self.vState = self.potential(vIn)
        #     mSamplesOut[i] = self.vActivation
        #     print_progress(i, len(vtTime), time.time()-rtStart)
        # print('')

        # - Increment internal time representation
        self._t += tTrueDuration

        return ts.TimeSeries(vtTime, mSamplesAct)

    def train_rr(self,
                 tsTarget: TimeSeries,
                 tsInput: TimeSeries = None,
                 fRegularize=0,
                 bFirst = True,
                 bFinal = False):
        """
        train_rr - Train self with ridge regression over one of possibly
                   many batches. Use Kahan summation to reduce rounding
                   errors when adding data to existing matrices.
        :param tsTarget:    TimeSeries - target for current batch
        :param tsInput:     TimeSeries - input to self for current batch
        :fRegularize:       float - regularization for ridge regression
        :bFirst:            bool - True if current batch is the first in training
        :bFinal:            bool - True if current batch is the last in training
        """

        # - Discrete time steps for evaluating input and target time series
        vtTimeBase = self._gen_time_trace(tsTarget.tStart, tsTarget.tDuration)

        if not bFinal:
            # - Discard last sample to avoid counting time points twice
            vtTimeBase = vtTimeBase[:-1]
        
        # - Prepare target data, check dimensions
        mfTarget = tsTarget(vtTimeTrace)
        assert mfTarget.shape[-1] == self.nSize, 
            ('Target dimensions ({}) does not match layer size ({})'.format(
                mfTarget.shape[-1], self.nSize))
        
        # - Prepare input data
        if tsInput is not None:
            # Warn if intput time range does not cover whole target time range
            if not tsInput.contains(vtTimeBase) or tsInput.bPeriodic:
                print('WARNING: tsInput (t = {} to {}) does not cover '.format(
                      tsInput.tStart, tsInput.tStop)
                      +'full time range of tsTarget (t = {} to {})'.format(
                      tsTarget.tStart, tsTarget.tStop)
                      +'Assuming input to be 0 outside of defined range.')

            # - Sample input trace and check for correct dimensions
            mfInput = self._check_input_dims(tsInput(vtTimeBase))
            # - Treat "NaN" as zero inputs
            mfInput[np.where(np.isnan(mfInput))] = 0

        else:
            # - Assume zero input
            mfInput = np.zeros((np.size(vtTimeBase), self.nDimIn))
            print('No tsInput defined, assuming input to be 0.')
            
        # - For first batch, initialize summands
        if bFirst:
            # Matrices to be updated for each batch
            self.mfYXT = np.zeros((nDimOut, nNetSize))
            self.mfXXT = np.zeros((nNetSize, nNetSize))
            # Corresponding Kahan compensations 
            self.mfKahanCompYXT = np.zeros_like(mfYXT)
            self.mfKahanCompXXT = np.zeros_like(mfXXT)

        # - Actual computations
        self._computation_training(mfTarget, mfInput, bFinal)

    @njit
    def _computation_training(self,
                              mfTarget: np.ndarray,
                              mfInput: np.ndarray,
                              bFinal: bool):
        """
        _computation_training - Perform matrix updates for training and
                                in final batch also update weights
        :param mfTarget:    2D-Array - training target     
        :param mfInput:     2D-Array - training input to layer
        :param bFinal:      bool - True for final batch
        """

        # - New data to be added, including compensation from last batch
        mfUpdYXT = mfTarget.T@mfInput - mfKahanCompYXT
        mfUpdXXT = mfInput.T@mfInput - mfKahanCompXXT

        if not bFinal:
            # - Update matrices with new data
            mfNewYXT = self.mfYXT + mfUpdYXT
            mfNewXXT = self.mfXXT + mfUpdXXT
            # - Calculate rounding error for compensation in next batch
            self.mfKahanCompYXT = (mfNewYXT-self.mfYXT) - mfUpdYXT
            self.mfKahanCompXXT = (mfNewXXT-self.mfXXT) - mfUpdXXT
            # - Store updated matrices
            self.mfYXT = mfNewYXT
            self.mfXXT = mfNewXXT

        else:
            # - In final step do not calculate rounding error but update matrices directly
            self.mfYXT += mfUpdYXT
            self.mfXXT += mfUpdXXT 

            # - Weight update by ridge regression
            self.mfW = np.linalg.solve(self.mfXXT+fRegularize*np.eye(nNetSize),
                                       self.mfYXT.T).T

            # - Remove dat stored during this trainig
            self.mfYXT = self.mfXXT = self.mfKahanCompYXT = self.mfKahanCompXXT = None              


    @njit
    def potential(self, vInput: np.ndarray) -> np.ndarray:
        return (self._vfAlpha * noisy(vInput*self.vfGain + self.vfBias, self.fNoiseStd)
                + (1-self._vfAlpha)*self.vState)

    @property
    def vActivation(self):
        return self.fhActivation(self.vState)

    ### --- properties

    @property
    def vtTau(self):
        return self._vtTau

    @vtTau.setter
    def vtTau(self, vNewTau):
        vNewTau = self.correct_param_shape(vNewTau)
        if not (vNewTau >= self._tDt).all(): raise ValueError('All vtTau must be at least tDt.')
        self._vtTau = vNewTau
        self._vfAlpha = self._tDt/vNewTau

    @property
    def vfAlpha(self):
        return self._vfAlpha

    @vfAlpha.setter
    def vfAlpha(self, vNewAlpha):
        vNewAlpha = self.correct_param_shape(vNewAlpha)
        if not (vNewAlpha <= 1).all(): raise ValueError('All vfAlpha must be at most 1.')
        self._vfAlpha = vNewAlpha
        self._vtTau = self._tDt/vNewAlpha

    @property
    def vfBias(self):
        return self._vfBias

    @vfBias.setter
    def vfBias(self, vNewBias):
        self._vfBias = self.correct_param_shape(vNewBias)

    @property
    def vfGain(self):
        return self._vfGain

    @vfGain.setter
    def vfGain(self, vNewGain):
        self._vfGain = self.correct_param_shape(vNewGain)

    @property
    def fhActivation(self):
        return self._fhActivation

    @fhActivation.setter
    def fhActivation(self, f):
        self._fhActivation = f
        self._evolveEuler = get_evolution_function(f)

    @Layer.tDt.setter
    def tDt(self, tNewDt):
        if not (self.vtTau >= tNewDt).all(): raise ValueError('All vtTau must be at least tDt.')
        self._tDt = tNewDt
        self._vfAlpha = tNewDt/self._vtTau

