import numpy as np
from typing import Callable
from numba import njit

from TimeSeries import TimeSeries
from layers.layer import Layer
from layers import noisy, fhReLU

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

        # - Compute final activity
        mfActivities[-1, :] = fhActivation(vState + vfBias)

        return mfActivities

    # - Return the compiled function
    return evolve_Euler_complete

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

    def evolve(self, tsInput: TimeSeries = None, tDuration: float = None) -> TimeSeries:
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

        # - Increment internal time representation
        self._t += tTrueDuration

        return TimeSeries(vtTime, mSamplesAct)

    def train_rr(self,
                 tsTarget: TimeSeries,
                 tsInput: TimeSeries = None,
                 fRegularize=0,
                 bFirst = True,
                 bFinal = False):
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
        vtTimeBase = self._gen_time_trace(tsTarget.tStart, tsTarget.tDuration)

        if not bFinal:
            # - Discard last sample to avoid counting time points twice
            vtTimeBase = vtTimeBase[:-1]

        # - Prepare target data, check dimensions
        mfTarget = tsTarget(vtTimeBase)
        assert mfTarget.shape[-1] == self.nSize, \
            ('Target dimensions ({}) does not match layer size ({})'.format(
                mfTarget.shape[-1], self.nSize))

        # - Prepare input data

        # Empty input array with additional dimension for training biases
        mfInput = np.zeros((np.size(vtTimeTrace), self.nDimIn+1))
        mfInput[:,-1] = 1

        if tsInput is not None:
            # Warn if intput time range does not cover whole target time range
            if not tsInput.contains(vtTimeBase) or tsInput.bPeriodic:
                print('WARNING: tsInput (t = {} to {}) does not cover '.format(
                      tsInput.tStart, tsInput.tStop)
                      +'full time range of tsTarget (t = {} to {})'.format(
                      tsTarget.tStart, tsTarget.tStop)
                      +'Assuming input to be 0 outside of defined range.')

            # - Sample input trace and check for correct dimensions
            mfInput[:, :-1] = self._check_input_dims(tsInput(vtTimeBase))
            # - Treat "NaN" as zero inputs
            mfInput[np.where(np.isnan(mfInput))] = 0

        else:
            print('No tsInput defined, assuming input to be 0 and only training biases.')

        # - For first batch, initialize summands
        if bFirst:
            # Matrices to be updated for each batch
            self.mfXTY = np.zeros((self.nSize, self.nDimIn+1))  # mfInput.T (dot) mfTarget
            self.mfXTX = np.zeros((self.nSize, self.nSize))     # mfInput.T (dot) mfInput
            # Corresponding Kahan compensations
            self.mfKahanCompXTY = np.zeros_like(mfXTY)
            self.mfKahanCompXTX = np.zeros_like(mfXTX)


        # - Actual computations
        self._computation_training(mfTarget, mfInput, fRegularize, bFinal)

    @njit
    def _computation_training(self,
                              mfTarget: np.ndarray,
                              mfInput: np.ndarray,
                              fRegularize : float,
                              bFinal: bool):
        """
        _computation_training - Perform matrix updates for training and
                                in final batch also update weights
        :param mfTarget:    2D-Array - Training target
        :param mfInput:     2D-Array - Training input to layer
        :param fRegularize: float - Regularization parameter
        :param bFinal:      bool - True for final batch
        """

        # - New data to be added, including compensation from last batch
        #   (Matrix summation always runs over time)
        mfUpdXTY = mfTarget.T@mfInput - self.mfKahanCompXTY
        mfUpdXTX = mfInput.T@mfInput - self.mfKahanCompXTX

        if not bFinal:
            # - Update matrices with new data
            mfNewXTY = self.mfXTY + mfUpdXTY
            mfNewXTX = self.mfXTX + mfUpdXTX
            # - Calculate rounding error for compensation in next batch
            self.mfKahanCompXTY = (mfNewXTY-self.mfXTY) - mfUpdXTY
            self.mfKahanCompXTX = (mfNewXTX-self.mfXTX) - mfUpdXTX
            # - Store updated matrices
            self.mfXTY = mfNewXTY
            self.mfXTX = mfNewXTX

        else:
            # - In final step do not calculate rounding error but update matrices directly
            self.mfXTY += mfUpdXTY
            self.mfXTX += mfUpdXTX

            # - Weight and bias update by ridge regression
            mfSolution = np.linalg.solve(self.mfXTX+fRegularize*np.eye(self.nDimIn),
                                         self.mfXTY)
            self.mfW = mfSolution[:-1, :]
            self.vfBias = mfSolution[-1, :]


            # - Remove dat stored during this trainig
            self.mfXTY = self.mfXTX = self.mfKahanCompXTY = self.mfKahanCompXTX = None


    # @njit
    # def potential(self, vInput: np.ndarray) -> np.ndarray:
    #     return (self._vfAlpha * noisy(vInput@self.mfW*self.vfGain + self.vfBias, self.fNoiseStd)
    #             + (1-self._vfAlpha)*self.vState)

    def __repr__(self):
        return 'FFRateEuler layer object `{}`.\nnSize: {}, nDimIn: {}   '.format(
            self.strName, self.nSize, self.nDimIn)

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

