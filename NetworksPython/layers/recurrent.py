import numpy as np
from abc import ABC
from typing import Callable
from numba import njit

from .layer import Layer
from TimeSeries import TimeSeries



### --- Provide a default ReLu activation function

@njit
def fhReLu(vfX: np.ndarray) -> np.ndarray:
    vCopy = np.copy(vfX)
    vCopy[np.where(vfX < 0)] = 0
    return vCopy


def get_evolution_function(fhActivation: Callable[[np.ndarray], np.ndarray]):
    """
    get_evolution_function: Construct a compiled Euler solver for a given activation funciton

    :param fhActivation: Callable (x) -> f(x)
    :return: Compiled function evolve_Euler_complete(vState, nSize, mfW, mfInputStep, nNumSteps, vfBias, vtTau)
    """

    # - Compile an Euler solver for the desired activation function
    @njit
    def evolve_Euler_complete(vState: np.ndarray,
                              nSize: int,
                              mfW: np.ndarray,
                              mfInputStep: np.ndarray,
                              nNumSteps: int,
                              vfBias: np.ndarray,
                              vtTau: np.ndarray) -> np.ndarray:
        # - Initialise storage of network output
        mfActivity = np.zeros((nNumSteps, nSize))

        # - Loop over time steps
        for nStep in range(nNumSteps):
            # - Evolve network state
            vfThisAct = fhActivation(vState + vfBias)
            vDState = -vState + mfInputStep[nStep, :] + mfW @ vfThisAct
            vState += vDState * vtTau

            # - Store network state
            mfActivity[nStep, :] = vfThisAct

        return mfActivity

    # - Return the compiled function
    return evolve_Euler_complete


### --- Recurrent rate class, with a Euler integrator

class RecRateEuler(Layer):
    def __init__(self,
                 mfW: np.ndarray,
                 vfBias: np.ndarray = 0,
                 vtTau: np.ndarray = 1,
                 fhActivation: Callable[[np.ndarray], np.ndarray] = fhReLu,
                 tDt: float = None,
                 fNoiseStd: float = 0,
                 sName: str = None):
        """
        RecRate: Implement a recurrent layer with firing rate neurons

        :param mfW:
        :param vfBias:
        :param vtTau:
        :param fhActivation:
        :param tDt:
        :param fNoiseStd:
        :param sName:
        """

        # - Call super-class init
        super().__init__(mfW = mfW,
                         tDt = tDt,
                         sName = sName)

        # - Assign properties
        self.vfBias = vfBias
        self.vtTau = vtTau
        self.fhActivation = fhActivation
        self.fNoiseStd = fNoiseStd



    ### --- Properties

    @property
    def vfBias(self) -> np.ndarray:
        return self._vfBias

    @vfBias.setter
    def vfBias(self, vfNewBias: np.ndarray):
        if np.size(vfNewBias) == 1:
            # - Expand bias to array
            vfNewBias = np.repeat(vfNewBias, self.nSize)

        else:
            assert np.size(vfNewBias) == self.nSize, \
                '`vfNewBias` must be a scalar or have {} elements'.format(self.nSize)

        # - Assign biases
        self._vfBias = np.reshape(vfNewBias, self.nSize)


    ### --- State evolution methods

    def evolve(self,
               tsInput: TimeSeries = None,
               tDuration: float = None):

        # - Determine default duration
        if tDuration is None:
            assert tsInput is not None, \
                'One of `tsInput` or `tDuration` must be supplied'

            tDuration = tsInput.tDuration

        # - Discretise tsInput to the desired evolution time base
        vtTimeBase = self.t + np.arange(0, tDuration, self.tDt)
        nNumSteps = np.size(vtTimeBase)

        if tsInput is not None:
            mfInputStep = tsInput(vtTimeBase)
        else:
            mfInputStep = np.zeros((nNumSteps, self.nSize))

        # - Generate a noise trace
        mfNoiseStep = np.random.randn(nNumSteps, self.nSize) * self.fNoiseStd

        # - Call Euler method integrator
        mfActivity = self._evolveEuler(self.vState, self.nSize, self.mfW, mfInputStep + mfNoiseStep,
                                       nNumSteps, self.vfBias, self.vtTau)

        # - Construct a return TimeSeries
        return TimeSeries(vtTimeBase, mfActivity)


    def reset_state(self):
        super().reset_state()

    def reset_all(self):
        super().reset_all()

    @property
    def tDt(self):
        return super().tDt

    @tDt.setter
    def tDt(self, tNewDt):
        # - Check that the time step is reasonable
        tMinTau = np.min(self.vtTau)
        assert tNewDt <= tMinTau / 10, \
            '`tNewDt` must be <= {}'.format(tMinTau/10)

        # - Assign time step
        self._tDt = tNewDt

    @property
    def fhActivation(self):
        return self._fhActivation

    @fhActivation.setter
    def fhActivation(self, fhNewActivation):
        self._fhActivation = fhNewActivation

        # - Build a state evolution function
        self._evolveEuler = get_evolution_function(fhNewActivation)