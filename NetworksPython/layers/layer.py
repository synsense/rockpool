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
tol_abs = 1e-9

### --- Convenience functions


def to_scalar(value, str_type: str = None):
    # - Check the value is a scalar
    assert np.size(value) == 1, "The value must be a scalar"

    if str_type is not None:
        return np.asscalar(np.array(value).astype(str_type))
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
        arraylike_new = (
            arraylike.cpu().numpy()
            if isinstance(arraylike, torch.Tensor)
            else arraylike
        )
        if reference is None:
            # New class instance is a copy of arraylike (and never a view to original arraylike)
            obj = np.array(arraylike_new).view(cls)
            # Store reference to original arraylike
            obj._reference = arraylike
        else:
            # New class instance is a copy of original array-like or a view, if arraylike is np.ndarray
            obj = np.asarray(arraylike_new).view(cls)
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
        array_copy = super().copy()
        return np.array(array_copy)


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
        weights: np.ndarray,
        dt: Optional[float] = 1,
        noise_std: Optional[float] = 0,
        name: Optional[str] = "unnamed",
    ):
        """
        Layer class - Implement an abstract layer of neurons (no implementation)

        :param weights:         np.ndarray Weight matrix for this layer
        :param dt:         float Time-step used for evolving this layer. Default: 1
        :param noise_std:   float Std. Dev. of state noise when evolving this layer. Default: 0. Defined as the expected
                                    std. dev. after 1s of integration time
        :param name:       str Name of this layer. Default: 'unnamed'
        """

        # - Assign properties
        if name is None:
            self.name = "unnamed"
        else:
            self.name = name

        try:
            # Try this before enforcing with Numpy atleast to account for custom classes for weights
            self._size_in, self._size = weights.shape
            self._mfW = weights
        except Exception:
            weights = np.atleast_2d(weights)
            self._size_in, self._size = weights.shape
            self._mfW = weights

        # - Check and assign dt and noise_std
        assert (
            np.size(dt) == 1 and np.size(noise_std) == 1
        ), "Layer `{}`: `dt` and `noise_std` must be scalars.".format(self.name)

        # - Assign default noise
        if noise_std is None:
            noise_std = 0.0

        # - Check dt
        assert dt is not None, "`dt` must be a numerical value"

        self._dt = dt
        self.noise_std = noise_std
        self._timestep = 0

    ### --- Common methods

    def _determine_timesteps(
        self,
        ts_input: Optional[TimeSeries] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> int:
        """
        _determine_timesteps - Determine over how many time steps to evolve with the given input

        :param ts_input:       TimeSeries  TxM or Tx1 Input signals for this layer
        :param duration:     float  Duration of the desired evolution, in seconds
        :param num_timesteps: int  Number of evolution time steps

        :return num_timesteps: int  Number of evolution time steps
        """

        if num_timesteps is None:
            # - Determine num_timesteps
            if duration is None:
                # - Determine duration
                assert (
                    ts_input is not None
                ), "Layer `{}`: One of `num_timesteps`, `ts_input` or `duration` must be supplied".format(
                    self.name
                )

                if ts_input.periodic:
                    # - Use duration of periodic TimeSeries, if possible
                    duration = ts_input.duration

                else:
                    # - Evolve until the end of the input TImeSeries
                    duration = ts_input.t_stop - self.t
                    assert duration > 0, (
                        "Layer `{}`: Cannot determine an appropriate evolution duration.".format(
                            self.name
                        )
                        + " `ts_input` finishes before the current evolution time."
                    )
            num_timesteps = int(np.floor((duration + tol_abs) / self.dt))
        else:
            assert (
                isinstance(num_timesteps, int) and num_timesteps >= 0
            ), "Layer `{}`: num_timesteps must be a non-negative integer.".format(
                self.name
            )

        return num_timesteps

    def _prepare_input(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> (np.ndarray, np.ndarray, float):
        """
        _prepare_input - Sample input, set up time base

        :param ts_input:       TimeSeries TxM or Tx1 Input signals for this layer
        :param duration:     float Duration of the desired evolution, in seconds
        :param num_timesteps: int Number of evolution time steps

        :return: (vtTimeBase, mfInputStep, duration)
            vtTimeBase:     ndarray T1 Discretised time base for evolution
            mfInputStep:    ndarray (T1xN) Discretised input signal for layer
            num_timesteps:  int Actual number of evolution time steps
        """

        num_timesteps = self._determine_timesteps(ts_input, duration, num_timesteps)

        # - Generate discrete time base
        vtTimeBase = self._gen_time_trace(self.t, num_timesteps)

        if ts_input is not None:
            # - Make sure vtTimeBase matches ts_input
            if not isinstance(ts_input, TSEvent):
                if not ts_input.periodic:
                    # - If time base limits are very slightly beyond ts_input.t_start and ts_input.t_stop, match them
                    if (
                        ts_input.t_start - 1e-3 * self.dt
                        <= vtTimeBase[0]
                        <= ts_input.t_start
                    ):
                        vtTimeBase[0] = ts_input.t_start
                    if (
                        ts_input.t_stop
                        <= vtTimeBase[-1]
                        <= ts_input.t_stop + 1e-3 * self.dt
                    ):
                        vtTimeBase[-1] = ts_input.t_stop

                # - Warn if evolution period is not fully contained in ts_input
                if not (ts_input.contains(vtTimeBase) or ts_input.periodic):
                    warn(
                        "Layer `{}`: Evolution period (t = {} to {}) ".format(
                            self.name, vtTimeBase[0], vtTimeBase[-1]
                        )
                        + "not fully contained in input signal (t = {} to {})".format(
                            ts_input.t_start, ts_input.t_stop
                        )
                    )

            # - Sample input trace and check for correct dimensions
            mfInputStep = self._check_input_dims(ts_input(vtTimeBase))

            # - Treat "NaN" as zero inputs
            mfInputStep[np.where(np.isnan(mfInputStep))] = 0

        else:
            # - Assume zero inputs
            mfInputStep = np.zeros((np.size(vtTimeBase), self.size_in))

        return vtTimeBase, mfInputStep, num_timesteps

    def _prepare_input_events(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> (np.ndarray, int):
        """
        _prepare_input_events - Sample input from TSEvent, set up time base

        :param ts_input:      TimeSeries TxM or Tx1 Input signals for this layer
        :param duration:    float Duration of the desired evolution, in seconds
        :param num_timesteps int Number of evolution time steps

        :return:
            mnSpikeRaster:    ndarray Boolean or integer raster containing spike info
            num_timesteps:    ndarray Number of evlution time steps
        """
        num_timesteps = self._determine_timesteps(ts_input, duration, num_timesteps)

        # - Extract spike timings and channels
        if ts_input is not None:
            # Extract spike data from the input variable
            mnSpikeRaster = ts_input.raster(
                dt=self.dt,
                t_start=self.t,
                num_timesteps=num_timesteps,
                channels=np.arange(self.size_in),
                add_events=(self.bAddEvents if hasattr(self, "bAddEvents") else False),
            )[2]
            # - Make sure size is correct
            mnSpikeRaster = mnSpikeRaster[:num_timesteps, :]

        else:
            mnSpikeRaster = np.zeros((num_timesteps, self.size_in))

        return mnSpikeRaster, num_timesteps

    def _check_input_dims(self, mfInput: np.ndarray) -> np.ndarray:
        """
        Verify if dimension of input matches layer instance. If input
        dimension == 1, scale it up to self._size_in by repeating signal.
            mfInput : np.ndarray with input data
            return : mfInput, possibly with dimensions repeated
        """
        # - Replicate `ts_input` if necessary
        if mfInput.ndim == 1 or (mfInput.ndim > 1 and mfInput.shape[1]) == 1:
            mfInput = np.repeat(mfInput.reshape((-1, 1)), self._size_in, axis=1)
        else:
            # - Check dimensionality of input
            assert (
                mfInput.shape[1] == self._size_in
            ), "Layer `{}`: Input dimensionality {} does not match layer input size {}.".format(
                self.name, mfInput.shape[1], self._size_in
            )

        # - Return possibly corrected input
        return mfInput

    def _gen_time_trace(self, tStart: float, num_timesteps: int) -> np.ndarray:
        """
        Generate a time trace starting at tStart, of length num_timesteps+1 with
        time step length self._dt. Make sure it does not go beyond
        tStart+duration.

        :return vtTimeTrace, duration
        """
        # - Generate a trace
        vtTimeTrace = np.arange(num_timesteps + 1) * self._dt + tStart

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

        :param oInput:          scalar or array-like (size)
        :param tupShape:        tuple of int Shape that input should be expanded to
        :param sVariableName:   str Name of the variable to include in error messages
        :param bAllowNone:      bool Allow None as argument for oInput
        :return:                np.ndarray (N) vector
        """
        if not bAllowNone:
            assert oInput is not None, "Layer `{}`: `{}` must not be None".format(
                self.name, sVariableName
            )

        nTotalSize = reduce(lambda m, n: m * n, tupShape)

        if np.size(oInput) == 1:
            # - Expand input to full size
            oInput = np.repeat(oInput, nTotalSize)

        assert (
            np.size(oInput) == nTotalSize
        ), "Layer `{}`: `{}` must be a scalar or have {} elements".format(
            self.name, sVariableName, nTotalSize
        )

        # - Return object of correct shape
        return np.reshape(oInput, tupShape)

    def _expand_to_size(
        self, oInput, size: int, sVariableName: str = "input", bAllowNone: bool = True
    ) -> np.ndarray:
        """
        _expand_to_size: Replicate out a scalar to size

        :param oInput:          scalar or array-like (size)
        :param size:           integer Size that input should be expanded to
        :param sVariableName:   str Name of the variable to include in error messages
        :param bAllowNone:      bool Allow None as argument for oInput
        :return:                np.ndarray (N) vector
        """
        return self._expand_to_shape(oInput, (size,), sVariableName, bAllowNone)

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
        return self._expand_to_shape(oInput, (self.size,), sVariableName, bAllowNone)

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
            oInput, (self.size, self.size), sVariableName, bAllowNone
        )

    ### --- String representations

    def __str__(self):
        return '{} object: "{}" [{} {} in -> {} {} out]'.format(
            self.__class__.__name__,
            self.name,
            self.size_in,
            self.input_type.__name__,
            self.size,
            self.output_type.__name__,
        )

    def __repr__(self):
        return self.__str__()

    ### --- State evolution methods

    @abstractmethod
    def evolve(
        self,
        ts_input: Optional[TimeSeries] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> TimeSeries:
        """
        evolve - Abstract method to evolve the state of this layer

        :param ts_input:     TimeSeries (TxM) External input trace to use when evolving the layer
        :param duration:   float Duration in seconds to evolve the layer
        :param num_timesteps: int Number of time steps to evolve the layer
        :return:            TimeSeries (TxN) Output of this layer
        """
        pass

    # @abstractmethod
    # def stream(self,
    #            duration: float,
    #            dt: float,
    #            verbose: bool = False,
    #           ) -> TimeSeries:
    #     """
    #     stream - Abstract method to evolve the state of this layer, in a streaming format
    #
    #     :param duration: float Total duration to be streamed
    #     :param dt:       float Streaming time-step (multiple of layer.dt)
    #
    #     :yield TimeSeries raw tuple representation on each time step
    #     """
    #     pass

    def reset_state(self):
        """
        reset_state - Reset the internal state of this layer. Sets state to zero

        :return: None
        """
        self.state = np.zeros(self.size)

    def reset_time(self):
        """
        reset_time - Reset the internal clock
        :return:
        """
        self._timestep = 0

    def randomize_state(self):
        """
        randomize_state - Randomise the internal state of this layer

        :return: None
        """
        # create random initial state with a gaussian distribution with mean
        # the values that were given and std the 20% of the absolute value
        self.state = np.random.normal(
            self.state, np.abs(self.state) * 0.02, size=(self.size,)
        )

    def reset_all(self):
        self.reset_time()
        self.reset_state()

    #### --- Properties

    @property
    def output_type(self):
        return TSContinuous

    @property
    def input_type(self):
        return TSContinuous

    @property
    def size(self) -> int:
        return self._size

    @property
    def size_in(self) -> int:
        return self._size_in

    @property
    def dt(self) -> float:
        return self._dt

    @dt.setter
    def dt(self, fNewDt: float):
        self._dt = to_scalar(fNewDt)

    @property
    def weights(self) -> np.ndarray:
        return self._mfW

    @weights.setter
    def weights(self, mfNewW: np.ndarray):
        assert mfNewW is not None, "Layer `{}`: weights must not be None.".format(
            self.name
        )

        # - Ensure weights are at least 2D
        try:
            assert mfNewW.ndim >= 2
        except AssertionError:
            warn(
                "Layer `{}`: `mfNewW must be at least of dimension 2".format(
                    self.name
                )
            )
            mfNewW = np.atleast_2d(mfNewW)

        # - Check dimensionality of new weights
        assert (
            mfNewW.size == self.size_in * self.size
        ), "Layer `{}`: `mfNewW` must be of shape {}".format(
            (self.name, self.size_in, self.size)
        )

        # - Save weights with appropriate size
        self._mfW = np.reshape(mfNewW, (self.size_in, self.size))

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, vNewState):
        assert (
            np.size(vNewState) == self.size
        ), "Layer `{}`: `vNewState` must have {} elements".format(
            self.name, self.size
        )

        self._state = vNewState

    @property
    def noise_std(self):
        return self._noise_std

    @noise_std.setter
    def noise_std(self, fNewNoiseStd):
        self._noise_std = to_scalar(fNewNoiseStd)

    @property
    def t(self):
        return self._timestep * self.dt

    @t.setter
    def t(self, new_t):
        self._timestep = int(np.floor(new_t / self.dt))

    # - Temporary, for maintaining compatibility with layers that still use _t
    @property
    def _t(self):
        return self._timestep * self.dt

    @_t.setter
    def _t(self, new_t):
        self._timestep = int(np.floor(new_t / self.dt))
