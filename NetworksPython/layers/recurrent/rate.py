###
# rate.py - Classes for recurrent rate model layers
#
###

# - Imports
import numpy as np
from typing import Callable
from numba import njit

from ..layer import Layer
from TimeSeries import TimeSeries


# - Define imports
__all__ = ['RecRateEuler']


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

        :param mfW:             np.ndarray (NxN) matrix of recurrent weights
        :param vfBias:          np.ndarray (N) vector (or scalar) of bias currents
        :param vtTau:           np.ndarray (N) vector (or scalar) of neuron time constants
        :param fhActivation:    Callable (x) -> f(x) Activation function
        :param tDt:             float Time step for integration (Euler method)
        :param fNoiseStd:       float Std. Dev. of state noise injected at each time step
        :param sName:           str Name of this layer
        """

        # - Call super-class init
        super().__init__(mfW = mfW,
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
        self._vfBias = self._expand_to_net_size(vfNewBias, 'vfNewBias')

    @property
    def vtTau(self) -> np.ndarray:
        return self._vtTau

    @vtTau.setter
    def vtTau(self, vtNewTau: np.ndarray):
        self._vtTau = self._expand_to_net_size(vtNewTau, 'vtNewTau')

        # - Ensure tDt is reasonable for numerical accuracy
        self.tDt = np.min(self.vtTau) / 10


    ### --- State evolution method

    def evolve(self,
               tsInput: TimeSeries = None,
               tDuration: float = None):

        # - Determine default duration
        if tDuration is None:
            assert tsInput is not None, \
                'One of `tsInput` or `tDuration` must be supplied'

            tDuration = tsInput.tDuration

        # - Discretise tsInput to the desired evolution time base
        vtTimeBase = self._gen_time_trace(self.t, tDuration)
        nNumSteps = np.size(vtTimeBase)

        if tsInput is not None:
            # - Sample input trace
            mfInputStep = tsInput(vtTimeBase)

            # - Treat "NaN" as zero inputs
            mfInputStep[np.where(np.isnan(mfInputStep))] = 0

        else:
            # - Assume zero inputs
            mfInputStep = np.zeros((nNumSteps, self.nSize))

        # - Generate a noise trace
        mfNoiseStep = np.random.randn(nNumSteps, self.nSize) * self.fNoiseStd

        # - Call Euler method integrator
        #   Note: Bypass setter method for .vState
        mfActivity = self._evolveEuler(self._vState, self.nSize, self.mfW, mfInputStep + mfNoiseStep,
                                       nNumSteps, self.vfBias, self.vtTau)

        # - Increment internal time representation
        self.t += tDuration

        # - Construct a return TimeSeries
        return TimeSeries(vtTimeBase, mfActivity)


    ### --- Properties

    @Layer.tDt.setter
    def tDt(self, tNewDt: float):
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