import numpy as np
from warnings import warn
from abc import ABC, abstractmethod
from functools import reduce
from typing import Optional, Union, List, Tuple
import torch

from ..timeseries import TimeSeries, TSContinuous, TSEvent

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Configure exports
__all__ = ["Layer"]


# - Absolute tolerance, e.g. for comparing float values
fTolAbs = 1e-9

### --- Convenience functions


def to_scalar(value, sClass: str = None):
    # - Check the value is a scalar
    assert np.size(value) == 1, "The value must be a scalar"

    if sClass is not None:
        return np.asscalar(np.array(value).astype(sClass))
    else:
        return np.asscalar(np.array(value))


### --- RefArray class


class RefArray(np.ndarray):
    """
    RefArray - np.ndarray subclass that is generated from an array-like or torch.Tensor
               and contains a reference to the original array-like or to a third object
               with same shape. Item assignment on a RefArray instance (i.e. refarray[i,j]
               = x) will also change this third object accordingly. Typically this object
               is some original container from which the array-like has been created.
               Therefore the objects in the RefArray are typically copies of those in the
               referenced object.
               This is useful for layers that contain torch tensors with properties
               returning a numpy array. Here, item assignment expected to modify also the
               original tensor object, which is not the case when using normal ndarrays.
    """

    def __new__(
        cls,
        arraylike: Union[ArrayLike, torch.Tensor],
        reference: Optional[Union[ArrayLike, torch.Tensor]] = None,
    ):
        """
        ___new__ - Customize instance creation. Necessary for custom subclasses of
                   np.ndarray. Create new object as view on existing ndarray or on a new
                   ndarray generated from an array-like object or tensor. Then add a
                   reference to a third object, with same shape. Typically the original
                   array is some form of copy of the referenced object. Alternatively a
                   reference to the original array-like or tensor can be added. In this
                   case the new instance is always a copy of the array-like and not a
                   reference.
        :param arraylike:  Array-like object or torch tensor to be copied.
        :param reference:  Indexable container with same dimensions as arraylike
                           If None, a reference to arraylike will be added.
        :return:
            obj  np.ndarray  Numpy array upon which new instance will be based
        """
        if reference is not None and tuple(np.shape(arraylike)) != tuple(
            np.shape(reference)
        ):
            raise TypeError(
                "Referenced object and array object need to have same shape"
            )
        # - Convert torch tensor to numpy array on cpu
        arraylikeNew = (
            arraylike.cpu().numpy()
            if isinstance(arraylike, torch.Tensor)
            else arraylike
        )
        if reference is None:
            # New class instance is a copy of arraylike (and never a view to original arraylike)
            obj = np.array(arraylikeNew).view(cls)
            # Store reference to original arraylike
            obj._reference = arraylike
        else:
            # New class instance is a copy of original array-like or a view, if arraylike is np.ndarray
            obj = np.asarray(arraylikeNew).view(cls)
            # - Add reference to third object
            obj._reference = reference
        return obj

    def __array_finalize(self, obj: np.ndarray):
        """
        __array_finalize - Method to be used for np.ndarray subclasses to include
                           additional elements in instance.
        :param obj:  np.ndarray upon which self is based
        """
        # - Store reference to third object as attribute of self
        self._reference = getattr(obj, "_reference", None)

    def __setitem__(self, position, value):
        """
        ___setitem___ - Update items of self and of self.reference in the same way.
        """
        super().__setitem__(position, value)
        if isinstance(self._reference, torch.Tensor):
            if not isinstance(value, torch.Tensor):
                # - Genrate tensor with new data
                value = torch.from_numpy(np.array(value))
            # - Match dtype and device with self.reference
            value = value.to(self._reference.dtype).to(self._reference.device)
        # - Update data in self.reference
        self._reference[position] = value

    def copy(self):
        """copy - Return np.ndarray as copy to get original __setitem__ method."""
        arrCopy = super().copy()
        return np.array(arrCopy)


class RefProperty(property):
    """
    RefProperty - The purpose of this class' is to provide a decorator @RefProperty
                  to be used instead of @property for objects that require that a copy
                  is returned instead of the original object. The returned object is
                  a RefArray with reference to the original object, allowing item
                  assignment to work.
    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        # - Change fget so that it returns a RefArray
        fget = self.fct_refarray(fget)
        super().__init__(fget=fget, fset=fset, fdel=fdel, doc=doc)

    def fct_refarray(self, fct):
        """
        fct_refarray - Return a function that does the same as fct but convert its return
                       value to a RefArray
        :param fct:  Callable  Function whose return value should be converted
        """

        def inner(owner):
            original = fct(owner)
            return RefArray(original)

        return inner


### --- Implements the Layer abstract class


class Layer(ABC):
    def __init__(
        self,
        mfW: np.ndarray,
        tDt: Optional[float] = 1,
        fNoiseStd: Optional[float] = 0,
        strName: Optional[str] = "unnamed",
    ):
        """
        Layer class - Implement an abstract layer of neurons (no implementation)

        :param mfW:         np.ndarray Weight matrix for this layer
        :param tDt:         float Time-step used for evolving this layer. Default: 1
        :param fNoiseStd:   float Std. Dev. of state noise when evolving this layer. Default: 0. Defined as the expected
                                    std. dev. after 1s of integration time
        :param strName:       str Name of this layer. Default: 'unnamed'
        """

        # - Assign properties
        if strName is None:
            self.strName = "unnamed"
        else:
            self.strName = strName

        try:
            # Try this before enforcing with Numpy atleast to account for custom classes for weights
            self._nSizeIn, self._nSize = mfW.shape
            self._mfW = mfW
        except Exception:
            mfW = np.atleast_2d(mfW)
            self._nSizeIn, self._nSize = mfW.shape
            self._mfW = mfW

        # - Check and assign tDt and fNoiseStd
        assert (
            np.size(tDt) == 1 and np.size(fNoiseStd) == 1
        ), "Layer `{}`: `tDt` and `fNoiseStd` must be scalars.".format(self.strName)

        # - Assign default noise
        if fNoiseStd is None:
            fNoiseStd = 0.0

        # - Check tDt
        assert tDt is not None, "`tDt` must be a numerical value"

        self._tDt = tDt
        self.fNoiseStd = fNoiseStd
        self._nTimeStep = 0

    ### --- Common methods

    def _determine_timesteps(
        self,
        tsInput: Optional[TimeSeries] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
    ) -> int:
        """
        _determine_timesteps - Determine over how many time steps to evolve with the given input

        :param tsInput:       TimeSeries  TxM or Tx1 Input signals for this layer
        :param tDuration:     float  Duration of the desired evolution, in seconds
        :param nNumTimeSteps: int  Number of evolution time steps

        :return nNumTimeSteps: int  Number of evolution time steps
        """

        if nNumTimeSteps is None:
            # - Determine nNumTimeSteps
            if tDuration is None:
                # - Determine tDuration
                assert (
                    tsInput is not None
                ), "Layer `{}`: One of `nNumTimeSteps`, `tsInput` or `tDuration` must be supplied".format(
                    self.strName
                )

                if tsInput.periodic:
                    # - Use duration of periodic TimeSeries, if possible
                    tDuration = tsInput.duration

                else:
                    # - Evolve until the end of the input TImeSeries
                    tDuration = tsInput.t_stop - self.t
                    assert tDuration > 0, (
                        "Layer `{}`: Cannot determine an appropriate evolution duration.".format(
                            self.strName
                        )
                        + " `tsInput` finishes before the current evolution time."
                    )
            nNumTimeSteps = int(np.floor((tDuration + fTolAbs) / self.tDt))
        else:
            assert (
                isinstance(nNumTimeSteps, int) and nNumTimeSteps >= 0
            ), "Layer `{}`: nNumTimeSteps must be a non-negative integer.".format(
                self.strName
            )

        return nNumTimeSteps

    def _prepare_input(
        self,
        tsInput: Optional[TSContinuous] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
    ) -> (np.ndarray, np.ndarray, float):
        """
        _prepare_input - Sample input, set up time base

        :param tsInput:       TimeSeries TxM or Tx1 Input signals for this layer
        :param tDuration:     float Duration of the desired evolution, in seconds
        :param nNumTimeSteps: int Number of evolution time steps

        :return: (vtTimeBase, mfInputStep, tDuration)
            vtTimeBase:     ndarray T1 Discretised time base for evolution
            mfInputStep:    ndarray (T1xN) Discretised input signal for layer
            nNumTimeSteps:  int Actual number of evolution time steps
        """

        nNumTimeSteps = self._determine_timesteps(tsInput, tDuration, nNumTimeSteps)

        # - Generate discrete time base
        vtTimeBase = self._gen_time_trace(self.t, nNumTimeSteps)

        if tsInput is not None:
            # - Make sure vtTimeBase matches tsInput
            if not isinstance(tsInput, TSEvent):
                if not tsInput.periodic:
                    # - If time base limits are very slightly beyond tsInput.t_start and tsInput.t_stop, match them
                    if (
                        tsInput.t_start - 1e-3 * self.tDt
                        <= vtTimeBase[0]
                        <= tsInput.t_start
                    ):
                        vtTimeBase[0] = tsInput.t_start
                    if (
                        tsInput.t_stop
                        <= vtTimeBase[-1]
                        <= tsInput.t_stop + 1e-3 * self.tDt
                    ):
                        vtTimeBase[-1] = tsInput.t_stop

                # - Warn if evolution period is not fully contained in tsInput
                if not (tsInput.contains(vtTimeBase) or tsInput.periodic):
                    warn(
                        "Layer `{}`: Evolution period (t = {} to {}) ".format(
                            self.strName, vtTimeBase[0], vtTimeBase[-1]
                        )
                        + "not fully contained in input signal (t = {} to {})".format(
                            tsInput.t_start, tsInput.t_stop
                        )
                    )

            # - Sample input trace and check for correct dimensions
            mfInputStep = self._check_input_dims(tsInput(vtTimeBase))

            # - Treat "NaN" as zero inputs
            mfInputStep[np.where(np.isnan(mfInputStep))] = 0

        else:
            # - Assume zero inputs
            mfInputStep = np.zeros((np.size(vtTimeBase), self.nSizeIn))

        return vtTimeBase, mfInputStep, nNumTimeSteps

    def _prepare_input_events(
        self,
        tsInput: Optional[TSEvent] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
    ) -> (np.ndarray, int):
        """
        _prepare_input_events - Sample input from TSEvent, set up time base

        :param tsInput:      TimeSeries TxM or Tx1 Input signals for this layer
        :param tDuration:    float Duration of the desired evolution, in seconds
        :param nNumTimeSteps int Number of evolution time steps

        :return:
            mnSpikeRaster:    ndarray Boolean or integer raster containing spike info
            nNumTimeSteps:    ndarray Number of evlution time steps
        """
        nNumTimeSteps = self._determine_timesteps(tsInput, tDuration, nNumTimeSteps)

        # - Extract spike timings and channels
        if tsInput is not None:
            # Extract spike data from the input variable
            mnSpikeRaster = tsInput.raster(
                dt=self.tDt,
                t_start=self.t,
                num_timesteps=nNumTimeSteps,
                channels=np.arange(self.nSizeIn),
                add_events=(self.bAddEvents if hasattr(self, "bAddEvents") else False),
            )[2]
            # - Make sure size is correct
            mnSpikeRaster = mnSpikeRaster[:nNumTimeSteps, :]

        else:
            mnSpikeRaster = np.zeros((nNumTimeSteps, self.nSizeIn))

        return mnSpikeRaster, nNumTimeSteps

    def _check_input_dims(self, mfInput: np.ndarray) -> np.ndarray:
        """
        Verify if dimension of input matches layer instance. If input
        dimension == 1, scale it up to self._nSizeIn by repeating signal.
            mfInput : np.ndarray with input data
            return : mfInput, possibly with dimensions repeated
        """
        # - Replicate `tsInput` if necessary
        if mfInput.ndim == 1 or (mfInput.ndim > 1 and mfInput.shape[1]) == 1:
            mfInput = np.repeat(mfInput.reshape((-1, 1)), self._nSizeIn, axis=1)
        else:
            # - Check dimensionality of input
            assert (
                mfInput.shape[1] == self._nSizeIn
            ), "Layer `{}`: Input dimensionality {} does not match layer input size {}.".format(
                self.strName, mfInput.shape[1], self._nSizeIn
            )

        # - Return possibly corrected input
        return mfInput

    def _gen_time_trace(self, tStart: float, nNumTimeSteps: int) -> np.ndarray:
        """
        Generate a time trace starting at tStart, of length nNumTimeSteps+1 with
        time step length self._tDt. Make sure it does not go beyond
        tStart+tDuration.

        :return vtTimeTrace, tDuration
        """
        # - Generate a trace
        vtTimeTrace = np.arange(nNumTimeSteps + 1) * self._tDt + tStart

        return vtTimeTrace

    def _expand_to_shape(
        self,
        oInput,
        tupShape: tuple,
        sVariableName: str = "input",
        bAllowNone: bool = True,
    ) -> np.ndarray:
        """
        _expand_to_shape: Replicate out a scalar to an array of shape tupShape

        :param oInput:          scalar or array-like (nSize)
        :param tupShape:        tuple of int Shape that input should be expanded to
        :param sVariableName:   str Name of the variable to include in error messages
        :param bAllowNone:      bool Allow None as argument for oInput
        :return:                np.ndarray (N) vector
        """
        if not bAllowNone:
            assert oInput is not None, "Layer `{}`: `{}` must not be None".format(
                self.strName, sVariableName
            )

        nTotalSize = reduce(lambda m, n: m * n, tupShape)

        if np.size(oInput) == 1:
            # - Expand input to full size
            oInput = np.repeat(oInput, nTotalSize)

        assert (
            np.size(oInput) == nTotalSize
        ), "Layer `{}`: `{}` must be a scalar or have {} elements".format(
            self.strName, sVariableName, nTotalSize
        )

        # - Return object of correct shape
        return np.reshape(oInput, tupShape)

    def _expand_to_size(
        self, oInput, nSize: int, sVariableName: str = "input", bAllowNone: bool = True
    ) -> np.ndarray:
        """
        _expand_to_size: Replicate out a scalar to nSize

        :param oInput:          scalar or array-like (nSize)
        :param nSize:           integer Size that input should be expanded to
        :param sVariableName:   str Name of the variable to include in error messages
        :param bAllowNone:      bool Allow None as argument for oInput
        :return:                np.ndarray (N) vector
        """
        return self._expand_to_shape(oInput, (nSize,), sVariableName, bAllowNone)

    def _expand_to_net_size(
        self, oInput, sVariableName: str = "input", bAllowNone: bool = True
    ) -> np.ndarray:
        """
        _expand_to_net_size: Replicate out a scalar to the size of the layer

        :param oInput:          scalar or array-like (N)
        :param sVariableName:   str Name of the variable to include in error messages
        :param bAllowNone:      bool Allow None as argument for oInput
        :return:                np.ndarray (N) vector
        """
        return self._expand_to_shape(oInput, (self.nSize,), sVariableName, bAllowNone)

    def _expand_to_weight_size(
        self, oInput, sVariableName: str = "input", bAllowNone: bool = True
    ) -> np.ndarray:
        """
        _expand_to_weight_size: Replicate out a scalar to the size of the layer's weights

        :param oInput:          scalar or array-like (NxN)
        :param sVariableName:   str Name of the variable to include in error messages
        :param bAllowNone:      bool Allow None as argument for oInput
        :return:                np.ndarray (NxN) vector
        """

        return self._expand_to_shape(
            oInput, (self.nSize, self.nSize), sVariableName, bAllowNone
        )

    ### --- String representations

    def __str__(self):
        return '{} object: "{}" [{} {} in -> {} {} out]'.format(
            self.__class__.__name__,
            self.strName,
            self.nSizeIn,
            self.cInput.__name__,
            self.nSize,
            self.cOutput.__name__,
        )

    def __repr__(self):
        return self.__str__()

    ### --- State evolution methods

    @abstractmethod
    def evolve(
        self,
        tsInput: Optional[TimeSeries] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
    ) -> TimeSeries:
        """
        evolve - Abstract method to evolve the state of this layer

        :param tsInput:     TimeSeries (TxM) External input trace to use when evolving the layer
        :param tDuration:   float Duration in seconds to evolve the layer
        :param nNumTimeSteps: int Number of time steps to evolve the layer
        :return:            TimeSeries (TxN) Output of this layer
        """
        pass

    # @abstractmethod
    # def stream(self,
    #            tDuration: float,
    #            tDt: float,
    #            bVerbose: bool = False,
    #           ) -> TimeSeries:
    #     """
    #     stream - Abstract method to evolve the state of this layer, in a streaming format
    #
    #     :param tDuration: float Total duration to be streamed
    #     :param tDt:       float Streaming time-step (multiple of layer.tDt)
    #
    #     :yield TimeSeries raw tuple representation on each time step
    #     """
    #     pass

    def reset_state(self):
        """
        reset_state - Reset the internal state of this layer. Sets state to zero

        :return: None
        """
        self.vState = np.zeros(self.nSize)

    def reset_time(self):
        """
        reset_time - Reset the internal clock
        :return:
        """
        self._nTimeStep = 0

    def randomize_state(self):
        """
        randomize_state - Randomise the internal state of this layer

        :return: None
        """
        # create random initial state with a gaussian distribution with mean
        # the values that were given and std the 20% of the absolute value
        self.vState = np.random.normal(self.vState, np.abs(self.vState)*0.02, size=(self.nSize,))

    def reset_all(self):
        self.reset_time()
        self.reset_state()

    #### --- Properties

    @property
    def cOutput(self):
        return TSContinuous

    @property
    def cInput(self):
        return TSContinuous

    @property
    def nSize(self) -> int:
        return self._nSize

    @property
    def nSizeIn(self) -> int:
        return self._nSizeIn

    @property
    def tDt(self) -> float:
        return self._tDt

    @tDt.setter
    def tDt(self, fNewDt: float):
        self._tDt = to_scalar(fNewDt)

    @property
    def mfW(self) -> np.ndarray:
        return self._mfW

    @mfW.setter
    def mfW(self, mfNewW: np.ndarray):
        assert mfNewW is not None, "Layer `{}`: mfW must not be None.".format(
            self.strName
        )

        # - Ensure weights are at least 2D
        try:
            assert mfNewW.ndim >= 2
        except AssertionError:
            warn(
                "Layer `{}`: `mfNewW must be at least of dimension 2".format(
                    self.strName
                )
            )
            mfNewW = np.atleast_2d(mfNewW)

        # - Check dimensionality of new weights
        assert (
            mfNewW.size == self.nSizeIn * self.nSize
        ), "Layer `{}`: `mfNewW` must be of shape {}".format(
            (self.strName, self.nSizeIn, self.nSize)
        )

        # - Save weights with appropriate size
        self._mfW = np.reshape(mfNewW, (self.nSizeIn, self.nSize))

    @property
    def vState(self):
        return self._vState

    @vState.setter
    def vState(self, vNewState):
        assert (
            np.size(vNewState) == self.nSize
        ), "Layer `{}`: `vNewState` must have {} elements".format(
            self.strName, self.nSize
        )

        self._vState = vNewState

    @property
    def fNoiseStd(self):
        return self._fNoiseStd

    @fNoiseStd.setter
    def fNoiseStd(self, fNewNoiseStd):
        self._fNoiseStd = to_scalar(fNewNoiseStd)

    @property
    def t(self):
        return self._nTimeStep * self.tDt

    @t.setter
    def t(self, new_t):
        self._nTimeStep = int(np.floor(new_t / self.tDt))

    # - Temporary, for maintaining compatibility with layers that still use _t
    @property
    def _t(self):
        return self._nTimeStep * self.tDt

    @_t.setter
    def _t(self, new_t):
        self._nTimeStep = int(np.floor(new_t / self.tDt))
