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
    :return: Compiled function evolve_Euler_complete(vState, nSize, mfW, mfInputStep, tDt, nNumSteps, vfBias, vtTau)
    """

    # - Compile an Euler solver for the desired activation function
    @njit
    def evolve_Euler_complete(vState: np.ndarray,
                              nSize: int,
                              mfW: np.ndarray,
                              mfInputStep: np.ndarray,
                              nNumSteps: int,
                              tDt: float,
                              vfBias: np.ndarray,
                              vtTau: np.ndarray) -> np.ndarray:
        # - Initialise storage of network output
        mfActivity = np.zeros((nNumSteps + 1, nSize))

        # - Precompute tDt / vtTau
        vfLambda = tDt / vtTau

        # - Loop over time steps
        for nStep in range(nNumSteps):
            # - Evolve network state
            vfThisAct = fhActivation(vState + vfBias)
            vDState = -vState + mfInputStep[nStep, :] + mfW @ vfThisAct
            vState += vDState * vfLambda

            # - Store network state
            mfActivity[nStep, :] = vfThisAct

        # - Get final activation
        mfActivity[-1, :] = fhActivation(vState + vfBias)

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
                 strName: str = None):
        """
        RecRate: Implement a recurrent layer with firing rate neurons

        :param mfW:             np.ndarray (NxN) matrix of recurrent weights
        :param vfBias:          np.ndarray (N) vector (or scalar) of bias currents
        :param vtTau:           np.ndarray (N) vector (or scalar) of neuron time constants
        :param fhActivation:    Callable (x) -> f(x) Activation function
        :param tDt:             float Time step for integration (Euler method)
        :param fNoiseStd:       float Std. Dev. of state noise injected at each time step
        :param strName:           str Name of this layer
        """

        # - Call super-class init
        super().__init__(mfW = mfW,
                         strName = strName)

        # - Assign properties
        self.vfBias = vfBias
        self.vtTau = vtTau
        self.fhActivation = fhActivation
        self.fNoiseStd = fNoiseStd

        if tDt is not None:
            self.tDt = tDt

        # - Reset the internal state
        self.reset_all()


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
               tDuration: float = None) -> TimeSeries:
        """
        evolve - Evolve the state of this layer

        :param tsInput:     TimeSeries TxM or Tx1 input to this layer
        :param tDuration:   float Duration of evolution, in seconds

        :return: TimeSeries Output of this layer during evolution period
        """

        # - Discretise input, prepare time base
        vtTimeBase, mfInputStep, tDuration = self._prepare_input(tsInput, tDuration)

        # - Generate a noise trace
        mfNoiseStep = np.random.randn(np.size(vtTimeBase), self.nSize) * self.fNoiseStd * np.sqrt(self.tDt)

        # - Call Euler method integrator
        #   Note: Bypass setter method for .vState
        mfActivity = self._evolveEuler(self._vState, self.nSize, self.mfW, mfInputStep + mfNoiseStep,
                                       np.size(vtTimeBase)-1, self.tDt, self.vfBias, self.vtTau)

        # - Increment internal time representation
        self._t = vtTimeBase[-1]

        # - Construct a return TimeSeries
        return TimeSeries(vtTimeBase, mfActivity)


    ### --- Properties

    @Layer.tDt.setter
    def tDt(self, tNewDt: float):
        # - Check that the time step is reasonable
        tMinTau = np.min(self.vtTau)
        assert tNewDt <= tMinTau / 10, \
            '`tNewDt` must be <= {}'.format(tMinTau/10)

        # - Call super-class setter
        super(RecRateEuler, RecRateEuler).tDt.__set__(self, tNewDt)

    @property
    def fhActivation(self):
        return self._fhActivation

    @fhActivation.setter
    def fhActivation(self, fhNewActivation):
        self._fhActivation = fhNewActivation

        # - Build a state evolution function
        self._evolveEuler = get_evolution_function(fhNewActivation)