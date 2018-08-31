###
# rate.py - Classes for recurrent rate model layers
#
###

# - Imports
import numpy as np
from typing import Callable, Tuple, List
from numba import njit

from ..layer import Layer
from ...timeseries import TimeSeries

from typing import Optional, Union, Tuple, List

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Define imports
__all__ = ["RecRateEuler"]


### --- Provide a default ReLu activation function


# @njit
# def fhReLu(vfX: np.ndarray) -> np.ndarray:
#    vCopy = np.copy(vfX)
#    vCopy[np.where(vfX < 0)] = 0
#    return vCopy


# def get_evolution_function(fhActivation: Callable[[np.ndarray], np.ndarray]):
#    """
#    get_evolution_function: Construct a compiled Euler solver for a given activation funciton
#
#    :param fhActivation: Callable (x) -> f(x)
#    :return: Compiled function evolve_Euler_complete(vState, nSize, mfW, mfInputStep, tDt, nNumSteps, vfBias, vtTau)
#    """
#
#    # - Compile an Euler solver for the desired activation function
#    @njit
#    def evolve_Euler_complete(
#        vState: np.ndarray,
#        nSize: int,
#        mfW: np.ndarray,
#        mfInputStep: np.ndarray,
#        nNumSteps: int,
#        tDt: float,
#        vfBias: np.ndarray,
#        vtTau: np.ndarray,
#    ) -> np.ndarray:
#        # - Initialise storage of network output
#        mfActivity = np.zeros((nNumSteps + 1, nSize))
#
#        # - Precompute tDt / vtTau
#        vfLambda = tDt / vtTau
#
#        # - Loop over time steps
#        for nStep in range(nNumSteps):
#            # - Evolve network state
#            vfThisAct = fhActivation(vState + vfBias)
#            vDState = -vState + mfInputStep[nStep, :] + mfW @ vfThisAct
#            vState += vDState * vfLambda
#
#            # - Store network state
#            mfActivity[nStep, :] = vfThisAct
#
#        # - Get final activation
#        mfActivity[-1, :] = fhActivation(vState + vfBias)
#
#        return mfActivity
#
#    # - Return the compiled function
#    return evolve_Euler_complete


### --- Recurrent rate class, with a Euler integrator


# class RecRateEuler(Layer):
#    def __init__(
#        self,
#        mfW: np.ndarray,
#        vfBias: np.ndarray = 0,
#        vtTau: np.ndarray = 1,
#        fhActivation: Callable[[np.ndarray], np.ndarray] = fhReLu,
#        tDt: float = None,
#        fNoiseStd: float = 0,
#        strName: str = None,
#    ):
#        """
#        RecRate: Implement a recurrent layer with firing rate neurons
#
#        :param mfW:             np.ndarray (NxN) matrix of recurrent weights
#        :param vfBias:          np.ndarray (N) vector (or scalar) of bias currents
#        :param vtTau:           np.ndarray (N) vector (or scalar) of neuron time constants
#        :param fhActivation:    Callable (x) -> f(x) Activation function
#        :param tDt:             float Time step for integration (Euler method)
#        :param fNoiseStd:       float Std. Dev. of state noise injected at each time step
#        :param strName:           str Name of this layer
#        """
#
#        # - Call super-class init
#        super().__init__(mfW=mfW, strName=strName)
#
#        # - Assign properties
#        self.vfBias = vfBias
#        self.vtTau = vtTau
#        self.fhActivation = fhActivation
#        self.fNoiseStd = fNoiseStd
#
#        if tDt is not None:
#            self.tDt = tDt
#
#        # - Reset the internal state
#        self.reset_all()
#
#    ### --- Properties
#
#    @property
#    def vfBias(self) -> np.ndarray:
#        return self._vfBias
#
#    @vfBias.setter
#    def vfBias(self, vfNewBias: np.ndarray):
#        self._vfBias = self._expand_to_net_size(vfNewBias, "vfNewBias")
#
#    @property
#    def vtTau(self) -> np.ndarray:
#        return self._vtTau
#
#    @vtTau.setter
#    def vtTau(self, vtNewTau: np.ndarray):
#        self._vtTau = self._expand_to_net_size(vtNewTau, "vtNewTau")
#
#        # - Ensure tDt is reasonable for numerical accuracy
#        self.tDt = np.min(self.vtTau) / 10
#
#    ### --- State evolution method
#
#    def evolve(
#        self,
#        tsInput: Optional[TimeSeries] = None,
#        tDuration: Optional[float] = None,
#        nNumTimeSteps: Optional[int] = None,
#        bVerbose: bool = False,
#    ) -> TimeSeries:
#        """
#        evolve : Function to evolve the states of this layer given an input
#
#        :param tsSpkInput:      TimeSeries  Input spike trian
#        :param tDuration:       float    Simulation/Evolution time
#        :param nNumTimeSteps    int      Number of evolution time steps
#        :param bVerbose:        bool     Currently no effect, just for conformity
#        :return:            TimeSeries  output spike series
#
#        """
#
#        # - Prepare time base
#        vtTimeBase, mfInputStep, nNumTimeSteps = self._prepare_input(
#            tsInput, tDuration, nNumTimeSteps
#        )
#
#        # - Generate a noise trace
#        # Noise correction: Standard deviation after some time would be fNoiseStd * sqrt(0.5*tDt/vtTau)
#        mfNoiseStep = (
#            np.random.randn(np.size(vtTimeBase), self.nSize)
#            * self.fNoiseStd
#            * np.sqrt(2. * self._vtTau / self._tDt)
#        )
#
#        # - Call Euler method integrator
#        #   Note: Bypass setter method for .vState
#        mfActivity = self._evolveEuler(
#            self._vState,
#            self._nSize,
#            self._mfW,
#            mfInputStep + mfNoiseStep,
#            nNumTimeSteps,
#            self._tDt,
#            self._vfBias,
#            self._vtTau,
#        )
#
#        # - Increment internal time representation
#        self._nTimeStep += nNumTimeSteps
#
#        # - Construct a return TimeSeries
#        return TimeSeries(vtTimeBase, mfActivity)
#
#    def stream(
#        self, tDuration: float, tDt: float, bVerbose: bool = False
#    ) -> Tuple[float, List[float]]:
#        """
#        stream - Stream data through this layer
#        :param tDuration:   float Total duration for which to handle streaming
#        :param tDt:         float Streaming time step
#        :param bVerbose:    bool Display feedback
#
#        :yield: (t, vState)
#
#        :return: Final (t, vState)
#        """
#
#        # - Initialise simulation, determine how many tDt to evolve for
#        if bVerbose:
#            print("Layer: I'm preparing")
#        vtTimeTrace = np.arange(0, tDuration + tDt, tDt)
#        nNumSteps = np.size(vtTimeTrace) - 1
#        nEulerStepsPerDt = int(tDt / self._tDt)
#
#        # - Generate a noise trace
#        mfNoiseStep = (
#            np.random.randn(np.size(vtTimeBase), self.nSize)
#            * self.fNoiseStd
#            * np.sqrt(2. * self._vtTau / self._tDt)
#        )
#
#        if bVerbose:
#            print("Layer: Prepared")
#
#        # - Loop over tDt steps
#        for nStep in range(nNumSteps):
#            if bVerbose:
#                print("Layer: Yielding from internal state.")
#            if bVerbose:
#                print("Layer: step", nStep)
#            if bVerbose:
#                print("Layer: Waiting for input...")
#
#            # - Yield current activity, receive input for next time step
#            tupInput = (
#                yield self._t,
#                np.reshape(self._fhActivation(self._vState + self._vfBias), (1, -1)),
#            )
#
#            # - Set zero input if no input provided
#            if tupInput is None:
#                mfInput = np.zeros(nEulerStepsPerDt, self._nSizeIn)
#            else:
#                mfInput = np.repeat(
#                    np.atleast_2d(tupInput[1][0, :]), nEulerStepsPerDt, axis=0
#                )
#
#            if bVerbose:
#                print("Layer: Input was: ", tupInput)
#
#            # - Evolve layer
#            _ = self._evolveEuler(
#                vState=self._vState,  # self._vState is automatically updated
#                nSize=self._nSize,
#                mfW=self._mfW,
#                mfInputStep=mfInput + mfNoiseStep[nStep, :],
#                nNumSteps=nEulerStepsPerDt,
#                tDt=self._tDt,
#                vfBias=self._vfBias,
#                vtTau=self._vtTau,
#            )
#
#            # - Increment time
#            self._nTimeStep += nEulerStepsPerDt
#
#        # - Return final activity
#        return (
#            self.t,
#            np.reshape(self._fhActivation(self._vState + self._vfBias), (1, -1)),
#        )
#
#    ### --- Properties
#
#    @Layer.tDt.setter
#    def tDt(self, tNewDt: float):
#        # - Check that the time step is reasonable
#        tMinTau = np.min(self.vtTau)
#        assert tNewDt <= tMinTau / 10, "`tNewDt` must be <= {}".format(tMinTau / 10)
#
#        # - Call super-class setter
#        super(RecRateEuler, RecRateEuler).tDt.__set__(self, tNewDt)
#
#    @property
#    def fhActivation(self):
#        return self._fhActivation
#
#    @fhActivation.setter
#    def fhActivation(self, fhNewActivation):
#        self._fhActivation = fhNewActivation
#
#        # - Build a state evolution function
#        self._evolveEuler = get_evolution_function(fhNewActivation)
