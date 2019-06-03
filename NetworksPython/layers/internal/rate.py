##
# rate.py - Non-spiking rate-coded dynamical layers, with ReLu / LT neurons. Euler solvers
##

import numpy as np
from typing import Callable
from numba import njit

from ...timeseries import TSContinuous
from ..layer import Layer
from typing import Optional, Union, Tuple, List

from warnings import warn

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Relative tolerance for float comparions
fTolerance = 1e-5

### --- Configure exports

__all__ = ["FFRateEuler", "PassThrough", "RecRateEuler"]


### --- Helper functions


def isMultiple(a: float, b: float, fTolerance: float = fTolerance) -> bool:
    """
    isMultiple - Check whether a%b is 0 within some tolerance.
    :param a: float The number that may be multiple of b
    :param b: float The number a may be a multiple of
    :param fTolerance: float Relative tolerance
    :return bool: True if a is a multiple of b within some tolerance
    """
    fMinRemainder = min(a % b, b - a % b)
    return fMinRemainder < fTolerance * b


def print_progress(iCurr: int, nTotal: int, tPassed: float):
    print(
        "Progress: [{:6.1%}]    in {:6.1f} s. Remaining:   {:6.1f}".format(
            iCurr / nTotal, tPassed, tPassed * (nTotal - iCurr) / max(0.1, iCurr)
        ),
        end="\r",
    )


@njit
def fhReLu(vfX: np.ndarray) -> np.ndarray:
    vCopy = np.copy(vfX)
    vCopy[np.where(vfX < 0)] = 0
    return vCopy


@njit
def noisy(mX: np.ndarray, fStdDev: float) -> np.ndarray:
    """
    noisy - Add randomly distributed noise to each element of mX
    :param mX:  Array-like with values that noise is added to
    :param fStdDev: Float, the standard deviation of the noise to be added
    :return:        Array-like, mX with noise added
    """
    return fStdDev * np.random.randn(*mX.shape) + mX


### --- Functions used in connection with FFRateEuler class


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


def get_ff_evolution_function(fhActivation: Callable[[np.ndarray], np.ndarray]):
    """
    get_ff_evolution_function: Construct a compiled Euler solver for a given activation function

    :param fhActivation: Callable (x) -> f(x)
    :return: Compiled function evolve_Euler_complete(state, mfInput, weights, size, nNumSteps, vfGain, vfBias, vfAlpha, noise_std)
    """

    # - Compile an Euler solver for the desired activation function
    @njit
    def evolve_Euler_complete(
        state: np.ndarray,
        mfInput: np.ndarray,
        weights: np.ndarray,
        size: int,
        nNumSteps: int,
        vfGain: np.ndarray,
        vfBias: np.ndarray,
        vfAlpha: np.ndarray,
        noise_std,
    ) -> np.ndarray:

        # - Initialise storage of layer output
        mfWeightedInput = mfInput @ weights
        mfActivities = np.zeros((nNumSteps + 1, size))

        # - Loop over time steps. The updated state already corresponds to
        # subsequent time step. Therefore skip state update in final step
        # and only update activation.
        for nStep in range(nNumSteps):
            # - Store layer activity
            mfActivities[nStep, :] = fhActivation(state + vfBias)

            # - Evolve layer state
            vDState = -state + noisy(vfGain * mfWeightedInput[nStep, :], noise_std)
            state += vDState * vfAlpha

        # - Compute final activity
        mfActivities[-1, :] = fhActivation(state + vfBias)

        return mfActivities

    # - Return the compiled function
    return evolve_Euler_complete


def get_rec_evolution_function(fhActivation: Callable[[np.ndarray], np.ndarray]):
    """
   get_rec_evolution_function: Construct a compiled Euler solver for a given activation function

   :param fhActivation: Callable (x) -> f(x)
   :return: Compiled function evolve_Euler_complete(state, size, weights, mfInputStep, dt, nNumSteps, vfBias, vtTau)
   """

    # - Compile an Euler solver for the desired activation function
    @njit
    def evolve_Euler_complete(
        state: np.ndarray,
        size: int,
        weights: np.ndarray,
        mfInputStep: np.ndarray,
        nNumSteps: int,
        dt: float,
        vfBias: np.ndarray,
        vtTau: np.ndarray,
    ) -> np.ndarray:
        # - Initialise storage of network output
        mfActivity = np.zeros((nNumSteps + 1, size))

        # - Precompute dt / vtTau
        vfLambda = dt / vtTau

        # - Loop over time steps
        for nStep in range(nNumSteps):
            # - Evolve network state
            vfThisAct = fhActivation(state + vfBias)
            vDState = -state + mfInputStep[nStep, :] + vfThisAct @ weights
            state += vDState * vfLambda

            # - Store network state
            mfActivity[nStep, :] = vfThisAct

        # - Get final activation
        mfActivity[-1, :] = fhActivation(state + vfBias)

        return mfActivity

    # - Return the compiled function
    return evolve_Euler_complete


### --- FFRateEuler class


class FFRateEuler(Layer):
    """ Feedforward layer consisting of rate-based neurons """

    def __init__(
        self,
        weights: np.ndarray,
        dt: float = None,
        name: str = None,
        noise_std: float = 0.0,
        fhActivation: Callable[[np.ndarray], np.ndarray] = fhReLU,
        vtTau: Union[float, np.ndarray] = 10.0,
        vfGain: Union[float, np.ndarray] = 1.0,
        vfBias: Union[float, np.ndarray] = 0.0,
    ):
        """
        FFRateEuler - Implement a feed-forward non-spiking neuron layer, with an Euler method solver

        :param weights:             np.ndarray [MxN] Weight matrix
        :param dt:             float Time step for Euler solver, in seconds
        :param name:         string Name of this layer
        :param noise_std:       float Noise std. dev. per second
        :param fhActivation:    Callable a = f(x) Neuron activation function (Default: ReLU)
        :param vtTau:           np.ndarray [Nx1] Vector of neuron time constants
        :param vfGain:          np.ndarray [Nx1] Vector of gain factors
        :param vfBias:          np.ndarray [Nx1] Vector of bias currents
        """

        # - Make sure some required parameters are set
        assert weights is not None, "`weights` is required"

        assert vtTau is not None, "`vtTau` is required"

        assert vfBias is not None, "`vfBias` is required"

        assert vfGain is not None, "`vfGain` is required"

        # - Set a reasonable dt
        if dt is None:
            tMinTau = np.min(vtTau)
            dt = tMinTau / 10

        # - Call super-class initialiser
        super().__init__(
            weights=np.asarray(weights, float), dt=dt, noise_std=noise_std, name=name
        )

        # - Check all parameter shapes
        try:
            self.vtTau, self.vfGain, self.vfBias = map(
                self._correct_param_shape, (vtTau, vfGain, vfBias)
            )
        except AssertionError:
            raise AssertionError(
                "Numbers of elements in vtTau, vfGain and vfBias"
                + " must be 1 or match layer size."
            )

        # - Reset this layer state and set attributes
        self.reset_all()
        self.vfAlpha = self._dt / self.vtTau
        self.fhActivation = fhActivation

    def _correct_param_shape(self, v) -> np.ndarray:
        """
        _correct_param_shape - Convert v to 1D-np.ndarray and verify
                              that dimensions match self.size
        :param v:   Float or array-like that is to be converted
        :return:    v as 1D-np.ndarray
        """
        v = np.array(v, dtype=float).flatten()
        assert v.shape in (
            (1,),
            (self.size,),
            (1, self.size),
            (self.size),
            1,
        ), "Numbers of elements in v must be 1 or match layer size"
        return v

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSContinuous:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSContinuous  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSContinuous  output spike series

        """

        # - Prepare time base
        vtTimeBase, mfInput, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        mSamplesAct = self._evolveEuler(
            state=self._state,  # self._state is automatically updated
            mfInput=mfInput,
            weights=self._mfW,
            size=self._size,
            nNumSteps=num_timesteps,
            vfGain=self._vfGain,
            vfBias=self._vfBias,
            vfAlpha=self._vfAlpha,
            # Without correction, standard deviation after some time will be
            # self._noise_std * sqrt(self._vfAlpha/2)
            noise_std=self._noise_std * np.sqrt(2.0 / self._vfAlpha),
        )

        # - Increment internal time representation
        self._timestep += num_timesteps

        return TSContinuous(vtTimeBase, mSamplesAct)

    def stream(
        self, duration: float, dt: float, verbose: bool = False
    ) -> Tuple[float, List[float]]:
        """
        stream - Stream data through this layer
        :param duration:   float Total duration for which to handle streaming
        :param dt:         float Streaming time step
        :param verbose:    bool Display feedback

        :yield: (t, state)

        :return: Final (t, state)
        """

        # - Initialise simulation, determine how many dt to evolve for
        if verbose:
            print("Layer: I'm preparing")
        vtTimeTrace = np.arange(0, duration + dt, dt)
        nNumSteps = np.size(vtTimeTrace) - 1
        nEulerStepsPerDt = int(np.round(dt / self._dt))

        if verbose:
            print("Layer: Prepared")

        # - Loop over dt steps
        for nStep in range(nNumSteps):
            if verbose:
                print("Layer: Yielding from internal state.")
            if verbose:
                print("Layer: step", nStep)
            if verbose:
                print("Layer: Waiting for input...")

            # - Yield current output, receive input for next time step
            tupInput = (
                yield self._t,
                np.reshape(self._fhActivation(self._state + self._vfBias), (1, -1)),
            )

            # - Set zero input if no input provided
            if tupInput is None:
                mfInput = np.zeros(nEulerStepsPerDt, self._size_in)
            else:
                mfInput = np.repeat(
                    np.atleast_2d(tupInput[1][0, :]), nEulerStepsPerDt, axis=0
                )

            if verbose:
                print("Layer: Input was: ", tupInput)

            # - Evolve layer
            _ = self._evolveEuler(
                state=self._state,  # self._state is automatically updated
                mfInput=mfInput,
                weights=self._mfW,
                size=self._size,
                nNumSteps=nEulerStepsPerDt,
                vfGain=self._vfGain,
                vfBias=self._vfBias,
                vfAlpha=self._vfAlpha,
                # Without correction, standard deviation after some time will be
                # self._noise_std * sqrt(self._vfAlpha/2)
                noise_std=self._noise_std * np.sqrt(2.0 / self._vfAlpha),
            )

            # - Increment time
            self._timestep += nEulerStepsPerDt

        # - Return final state
        return (
            self.t,
            np.reshape(self._fhActivation(self._state + self._vfBias), (1, -1)),
        )

    def train_rr(
        self,
        tsTarget: TSContinuous,
        ts_input: TSContinuous,
        fRegularize=0,
        bFirst=True,
        bFinal=False,
    ):
        """
        train_rr - Train self with ridge regression over one of possibly
                   many batches. Use Kahan summation to reduce rounding
                   errors when adding data to existing matrices from
                   previous batches.
        :param tsTarget:    TSContinuous - target for current batch
        :param ts_input:     TSContinuous - input to self for current batch
        :fRegularize:       float - regularization for ridge regression
        :bFirst:            bool - True if current batch is the first in training
        :bFinal:            bool - True if current batch is the last in training
        """

        # - Discrete time steps for evaluating input and target time series
        num_timesteps = int(np.round(ts_input.duration / self.dt))
        vtTimeBase = self._gen_time_trace(ts_input.t_start, num_timesteps)

        if not bFinal:
            # - Discard last sample to avoid counting time points twice
            vtTimeBase = vtTimeBase[:-1]

        # - Make sure vtTimeBase does not exceed ts_input
        vtTimeBase = vtTimeBase[vtTimeBase <= ts_input.t_stop]

        # - Prepare target data
        mfTarget = tsTarget(vtTimeBase)

        # - Make sure no nan is in target, as this causes learning to fail
        assert not np.isnan(
            mfTarget
        ).any(), "nan values have been found in mfTarget (where: {})".format(
            np.where(np.isnan(mfTarget))
        )

        # - Check target dimensions
        if mfTarget.ndim == 1 and self.size == 1:
            mfTarget = mfTarget.reshape(-1, 1)

        assert (
            mfTarget.shape[-1] == self.size
        ), "Target dimensions ({}) does not match layer size ({})".format(
            mfTarget.shape[-1], self.size
        )

        # - Prepare input data

        # Empty input array with additional dimension for training biases
        mfInput = np.zeros((np.size(vtTimeBase), self.size_in + 1))
        mfInput[:, -1] = 1

        # Warn if input time range does not cover whole target time range
        if (
            not tsTarget.contains(vtTimeBase)
            and not ts_input.periodic
            and not tsTarget.periodic
        ):
            warn(
                "WARNING: ts_input (t = {} to {}) does not cover ".format(
                    ts_input.t_start, ts_input.t_stop
                )
                + "full time range of tsTarget (t = {} to {})\n".format(
                    tsTarget.t_start, tsTarget.t_stop
                )
                + "Assuming input to be 0 outside of defined range.\n"
                + "If you are training by batches, check that the target signal is also split by batch.\n"
            )

        # - Sample input trace and check for correct dimensions
        mfInput[:, :-1] = self._check_input_dims(ts_input(vtTimeBase))

        # - Treat "NaN" as zero inputs
        mfInput[np.where(np.isnan(mfInput))] = 0

        # - For first batch, initialize summands
        if bFirst:
            # Matrices to be updated for each batch
            self._mfXTY = np.zeros(
                (self.size_in + 1, self.size)
            )  # mfInput.T (dot) mfTarget
            self._mfXTX = np.zeros(
                (self.size_in + 1, self.size_in + 1)
            )  # mfInput.T (dot) mfInput

            # Corresponding Kahan compensations
            self._mfKahanCompXTY = np.zeros_like(self._mfXTY)
            self._mfKahanCompXTX = np.zeros_like(self._mfXTX)

        # - New data to be added, including compensation from last batch
        #   (Matrix summation always runs over time)
        mfUpdXTY = mfInput.T @ mfTarget - self._mfKahanCompXTY
        mfUpdXTX = mfInput.T @ mfInput - self._mfKahanCompXTX

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
            mfSolution = np.linalg.solve(
                self._mfXTX + fRegularize * np.eye(self.size_in + 1), self._mfXTY
            )

            self.weights = mfSolution[:-1, :]
            self.vfBias = mfSolution[-1, :]

            # - Remove data stored during this training
            self._mfXTY = (
                self._mfXTX
            ) = self._mfKahanCompXTY = self._mfKahanCompXTX = None

    # @njit
    # def potential(self, vInput: np.ndarray) -> np.ndarray:
    #     return (self._vfAlpha * noisy(vInput@self.weights*self.vfGain + self.vfBias, self.noise_std)
    #             + (1-self._vfAlpha)*self.state)

    def __repr__(self):
        return "FFRateEuler layer object `{}`.\nnSize: {}, size_in: {}   ".format(
            self.name, self.size, self.size_in
        )

    @property
    def vActivation(self):
        return self.fhActivation(self.state)

    ### --- properties

    @property
    def vtTau(self):
        return self._vtTau

    @vtTau.setter
    def vtTau(self, vNewTau):
        vNewTau = self._correct_param_shape(vNewTau)
        if not (vNewTau >= self._dt).all():
            raise ValueError("All vtTau must be at least dt.")
        self._vtTau = vNewTau
        self._vfAlpha = self._dt / vNewTau

    @property
    def vfAlpha(self):
        return self._vfAlpha

    @vfAlpha.setter
    def vfAlpha(self, vNewAlpha):
        vNewAlpha = self._correct_param_shape(vNewAlpha)
        if not (vNewAlpha <= 1).all():
            raise ValueError("All vfAlpha must be at most 1.")
        self._vfAlpha = vNewAlpha
        self._vtTau = self._dt / vNewAlpha

    @property
    def vfBias(self):
        return self._vfBias

    @vfBias.setter
    def vfBias(self, vNewBias):
        self._vfBias = self._correct_param_shape(vNewBias)

    @property
    def vfGain(self):
        return self._vfGain

    @vfGain.setter
    def vfGain(self, vNewGain):
        self._vfGain = self._correct_param_shape(vNewGain)

    @property
    def fhActivation(self):
        return self._fhActivation

    @fhActivation.setter
    def fhActivation(self, f):
        self._fhActivation = f
        self._evolveEuler = get_ff_evolution_function(f)

    @Layer.dt.setter
    def dt(self, tNewDt):
        if not (self.vtTau >= tNewDt).all():
            raise ValueError("All vtTau must be at least dt.")
        self._dt = tNewDt
        self._vfAlpha = tNewDt / self._vtTau


### --- PassThrough Class


class PassThrough(FFRateEuler):
    """ Neuron states directly correspond to input, but can be delayed. """

    def __init__(
        self,
        weights: np.ndarray,
        dt: float = 1.0,
        noise_std: float = 0.0,
        vfBias: Union[float, np.ndarray] = 0.0,
        tDelay: float = 0.0,
        name: str = None,
    ):
        """
        PassThrough - Implement a feed-forward layer that simply passes input (possibly delayed)

        :param weights:         np.ndarray [MxN] Weight matrix
        :param dt:         float Time step for Euler solver, in seconds
        :param noise_std:   float Noise std. dev. per second
        :param vfBias:      np.ndarray [Nx1] Vector of bias currents
        :param tDelay:      float Delay between input and output, in seconds
        :param name:     string Name of this layer
        """
        # - Set delay
        self._nDelaySteps = 0 if tDelay is None else int(np.round(tDelay / dt))

        # - Call super-class initialiser
        super().__init__(
            weights=np.asarray(weights, float),
            dt=dt,
            noise_std=noise_std,
            fhActivation=lambda x: x,
            vfBias=vfBias,
            name=name,
        )

        self.reset_all()

    def reset_buffer(self):
        if self.tDelay != 0:
            vtBuffer = np.arange(self._nDelaySteps + 1) * self._dt
            self.tsBuffer = TSContinuous(
                vtBuffer, np.zeros((len(vtBuffer), self.size))
            )
        else:
            self.tsBuffer = None

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSContinuous:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSContinuous  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSContinuous  output spike series

        """

        # - Prepare time base
        vtTimeBase, mfInput, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Apply input weights and add noise
        mfInProcessed = noisy(mfInput @ self.weights, self.noise_std)

        if self.tsBuffer is not None:
            # - Combined time trace for buffer and processed input
            nNumTimeStepsComb = num_timesteps + self._nDelaySteps
            vtTimeComb = self._gen_time_trace(self.t, nNumTimeStepsComb)

            # - Array for buffered and new data
            mfSamplesComb = np.zeros((vtTimeComb.size, self.size_in))
            nStepsIn = vtTimeBase.size

            # - Buffered data: last point of buffer data corresponds to self.t,
            #   which is also part of current input
            mfSamplesComb[:-nStepsIn] = self.tsBuffer.samples[:-1]

            # - Processed input data (weights and noise)
            mfSamplesComb[-nStepsIn:] = mfInProcessed

            # - Output data
            mfSamplesOut = mfSamplesComb[:nStepsIn]

            # - Update buffer with new data
            self.tsBuffer.samples = mfSamplesComb[nStepsIn - 1 :]

        else:
            # - Undelayed processed input
            mfSamplesOut = mfInProcessed

        # - Update state and time
        self.state = mfSamplesOut[-1]
        self._timestep += num_timesteps

        # - Return time series with output data and bias
        return TSContinuous(vtTimeBase, mfSamplesOut + self.vfBias)

    def __repr__(self):
        return "PassThrough layer object `{}`.\nnSize: {}, size_in: {}, tDelay: {}".format(
            self.name, self.size, self.size_in, self.tDelay
        )

    def print_buffer(self, **kwargs):
        if self.tsBuffer is not None:
            self.tsBuffer.print(**kwargs)
        else:
            print("This layer does not use a delay.")

    @property
    def mfBuffer(self):
        if self.tsBuffer is not None:
            return self.tsBuffer.samples
        else:
            print("This layer does not use a delay.")

    def reset_state(self):
        super().reset_state()
        self.reset_buffer()

    def reset_all(self):
        super().reset_all()
        self.reset_buffer()

    @property
    def tDelay(self):
        return self._nDelaySteps * self.dt

    @property
    def nDelaySteps(self):
        return self._nDelaySteps

    # @tDelay.setter
    # def tDelay(self, tNewDelay):
    # Some method to extend self.tsBuffer


class RecRateEuler(Layer):
    def __init__(
        self,
        weights: np.ndarray,
        vfBias: np.ndarray = 0.0,
        vtTau: np.ndarray = 1.0,
        fhActivation: Callable[[np.ndarray], np.ndarray] = fhReLu,
        dt: float = None,
        noise_std: float = 0.0,
        name: str = None,
    ):
        """
        RecRate: Implement a recurrent layer with firing rate neurons

        :param weights:             np.ndarray (NxN) matrix of recurrent weights
        :param vfBias:          np.ndarray (N) vector (or scalar) of bias currents
        :param vtTau:           np.ndarray (N) vector (or scalar) of neuron time constants
        :param fhActivation:    Callable (x) -> f(x) Activation function
        :param dt:             float Time step for integration (Euler method)
        :param noise_std:       float Std. Dev. of state noise injected at each time step
        :param name:           str Name of this layer
        """

        # - Call super-class init
        super().__init__(weights=np.asarray(weights, float), name=name)

        # - Check size and shape of `weights`
        assert len(weights.shape) == 2, "`weights` must be a matrix with 2 dimensions"
        assert weights.shape[0] == weights.shape[1], "`weights` must be a square matrix"

        # - Check arguments
        assert vtTau is not None, "`vtTau` may not be None"

        assert noise_std is not None, "`noise_std` may not be None"

        # - Assign properties
        self.vfBias = vfBias
        self.vtTau = vtTau
        self.fhActivation = fhActivation
        self.noise_std = noise_std

        if dt is not None:
            self.dt = dt

        # - Reset the internal state
        self.reset_all()

    ### --- Properties

    @property
    def vfBias(self) -> np.ndarray:
        return self._vfBias

    @vfBias.setter
    def vfBias(self, vfNewBias: np.ndarray):
        self._vfBias = self._expand_to_net_size(vfNewBias, "vfNewBias")

    @property
    def vtTau(self) -> np.ndarray:
        return self._vtTau

    @vtTau.setter
    def vtTau(self, vtNewTau: np.ndarray):
        self._vtTau = self._expand_to_net_size(vtNewTau, "vtNewTau")

        # - Ensure dt is reasonable for numerical accuracy
        self.dt = np.min(self.vtTau) / 10

    ### --- State evolution method

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSContinuous:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSContinuous  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:                TSContinuous  output spike series

        """

        # - Prepare time base
        vtTimeBase, mfInputStep, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Generate a noise trace
        # Noise correction: Standard deviation after some time would be noise_std * sqrt(0.5*dt/vtTau)
        mfNoiseStep = (
            np.random.randn(np.size(vtTimeBase), self.size)
            * self.noise_std
            * np.sqrt(2.0 * self._vtTau / self._dt)
        )

        # - Call Euler method integrator
        #   Note: Bypass setter method for .state
        mfActivity = self._evolveEuler(
            self._state,
            self._size,
            self._mfW,
            mfInputStep + mfNoiseStep,
            num_timesteps,
            self._dt,
            self._vfBias,
            self._vtTau,
        )

        # - Increment internal time representation
        self._timestep += num_timesteps

        # - Construct a return TimeSeries
        return TSContinuous(vtTimeBase, mfActivity)

    def stream(
        self, duration: float, dt: float, verbose: bool = False
    ) -> Tuple[float, List[float]]:
        """
        stream - Stream data through this layer
        :param duration:   float Total duration for which to handle streaming
        :param dt:         float Streaming time step
        :param verbose:    bool Display feedback

        :yield: (t, state)

        :return: Final (t, state)
        """

        # - Initialise simulation, determine how many dt to evolve for
        if verbose:
            print("Layer: I'm preparing")
        vtTimeTrace = np.arange(0, duration + dt, dt)
        nNumSteps = np.size(vtTimeTrace) - 1
        nEulerStepsPerDt = int(dt / self._dt)

        # - Generate a noise trace
        mfNoiseStep = (
            np.random.randn(np.size(vtTimeBase), self.size)
            * self.noise_std
            * np.sqrt(2.0 * self._vtTau / self._dt)
        )

        if verbose:
            print("Layer: Prepared")

        # - Loop over dt steps
        for nStep in range(nNumSteps):
            if verbose:
                print("Layer: Yielding from internal state.")
            if verbose:
                print("Layer: step", nStep)
            if verbose:
                print("Layer: Waiting for input...")

            # - Yield current activity, receive input for next time step
            tupInput = (
                yield self._t,
                np.reshape(self._fhActivation(self._state + self._vfBias), (1, -1)),
            )

            # - Set zero input if no input provided
            if tupInput is None:
                mfInput = np.zeros(nEulerStepsPerDt, self._size_in)
            else:
                mfInput = np.repeat(
                    np.atleast_2d(tupInput[1][0, :]), nEulerStepsPerDt, axis=0
                )

            if verbose:
                print("Layer: Input was: ", tupInput)

            # - Evolve layer
            _ = self._evolveEuler(
                state=self._state,  # self._state is automatically updated
                size=self._size,
                weights=self._mfW,
                mfInputStep=mfInput + mfNoiseStep[nStep, :],
                nNumSteps=nEulerStepsPerDt,
                dt=self._dt,
                vfBias=self._vfBias,
                vtTau=self._vtTau,
            )

            # - Increment time
            self._timestep += nEulerStepsPerDt

        # - Return final activity
        return (
            self.t,
            np.reshape(self._fhActivation(self._state + self._vfBias), (1, -1)),
        )

    ### --- Properties

    @Layer.dt.setter
    def dt(self, tNewDt: float):
        # - Check that the time step is reasonable
        tMinTau = np.min(self.vtTau)
        assert tNewDt <= tMinTau / 10, "`tNewDt` must be <= {}".format(tMinTau / 10)

        # - Call super-class setter
        super(RecRateEuler, RecRateEuler).dt.__set__(self, tNewDt)

    @property
    def fhActivation(self):
        return self._fhActivation

    @fhActivation.setter
    def fhActivation(self, fhNewActivation):
        self._fhActivation = fhNewActivation

        # - Build a state evolution function
        self._evolveEuler = get_rec_evolution_function(fhNewActivation)
