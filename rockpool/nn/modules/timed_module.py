"""
Contains the base classes for TimedModules in Rockpool. Also contains classes to adapt Module classes -> TimedModule classes, and to wrap Module objects as TimedModule objects.
"""

# - Rockpool imports
from rockpool.timeseries import TimeSeries, TSContinuous, TSEvent
from rockpool.nn.modules.module import Module, ModuleBase, PostInitMetaMixin
from rockpool.parameters import SimulationParameter, Parameter, State

# - Other imports
from typing import Optional, Union, Tuple, Dict, Any, List, Iterable, Callable
from warnings import warn
from abc import abstractmethod
import numpy as np
import functools
from decimal import Decimal
from collections import abc

# - Tolerance constants to use when comparing floats
tol_rel = 1e-5
tol_abs = 1e-6
decimal_base = 1e-7

# - Define some type aliases
RealValue = Union[float, Decimal, SimulationParameter, str]
Tree = Union[abc.Iterable, abc.MutableMapping]

__all__ = ["TimedModule", "TimedModuleWrapper", "astimedmodule"]


def is_multiple(
    a: RealValue,
    b: RealValue,
    tol_rel: RealValue = tol_rel,
    tol_abs: RealValue = tol_abs,
) -> bool:
    """
    Check whether a % b is 0 within some tolerance.

    :param float a:         The number that may be multiple of `b`
    :param float b:         The number `a` may be a multiple of
    :param float tol_rel:   Relative tolerance
    :param float tol_abs:   Absolute tolerance

    :return bool:   True if `a` is a multiple of `b` within some tolerance
    """
    # - Convert to decimals
    a = Decimal(str(a))
    b = Decimal(str(b))
    tol_rel = Decimal(str(tol_rel))
    tol_abs = Decimal(str(tol_abs))
    min_remainder = min(a % b, b - a % b)
    return min_remainder < tol_rel * b + tol_abs


def gcd(a: RealValue, b: RealValue) -> Decimal:
    """
    Return the greatest common divisor of two values

    :param float a: Value `a`
    :param float b: Value `b`

    :return int: Greatest common divisor of `a` and `b`
    """
    a = Decimal(str(a))
    b = Decimal(str(b))
    if b == 0:
        return a
    else:
        return gcd(b, a % b)


def lcm(a: RealValue, b: RealValue) -> Decimal:
    """
    Return the least common multiple of two values

    :param float a: Value a
    :param float b: Value b

    :return int: Least common integer multiple of `a` and `b`
    """
    # - Make sure that values used are sufficiently large
    # Transform to integer-values
    a_rnd = round(float(a) / decimal_base)
    b_rnd = round(float(b) / decimal_base)
    # - Make sure that a and b are not too small
    if (
        np.abs(a_rnd - float(a) / decimal_base) > tol_rel
        or np.abs(b_rnd - float(b) / decimal_base) > tol_rel
    ):
        raise ValueError(
            "network: Too small values to find lcm. Try changing 'decimal_base'"
        )
    a = Decimal(str(a_rnd))
    b = Decimal(str(b_rnd))
    return a / gcd(a, b) * b * Decimal(str(decimal_base))


def tree_map(func: Callable[[Any], Any], tree: Tree) -> Tree:
    """
    Map a function over a tree

    A ``Tree`` is a nested ``list``, ``dict`` or ``tuple`` object, containing other ``list``s, ``dict``s and ``tuple``s. Any other type is considered a leaf node. The supplied function will be applied independently to all leaf nodes in the tree, and the transformed ``Tree`` will be returned.

    Args:
        func (Callable): A function to apply to each node in the tree
        tree (Tree): A tree over which to iterate. `func` will be applied to every leaf node in `tree`

    Returns:
        Tree: A transformed tree, with `func` applied to every leaf node

    """
    if isinstance(tree, dict):  # if dict, apply to each key
        return {k: tree_map(func, v) for k, v in tree.items()}

    elif isinstance(tree, list):  # if list, apply to each element
        return [tree_map(func, elem) for elem in tree]

    elif isinstance(tree, tuple):  # if tuple, apply to each element
        return tuple([tree_map(func, elem) for elem in tree])

    else:
        #  - Apply function
        return func(tree)


def leaves(d: dict):
    """
    A generator that yields the leaf nodes in a nested dict

    `leaves` will perform a depth-first traversal of the nested ``dict`` `d`, and will yield each ``(key, value)`` tuple in turn

    Args:
        d (dict): The dict over which to traverse

    Yields:
        tuple: (key, value)
            key (str): The key of a given leaf
            value (Any): The value of a given leaf

    """
    for k, v in d.items():
        if isinstance(v, dict):
            yield from leaves(v)
        else:
            yield k, v


class TimedModule(ModuleBase, metaclass=PostInitMetaMixin):
    """
    The Rockpool base class for all :py:class:`.TimedModule` modules

    :py:class:`.TimedModule` provides functionality for :py:class:`.Module` s to understand time series data, and to conveniently evolve, handle and return time series data from modules.

    The :py:meth:`.evolve` method provided by :py:class:`.TimedModule` can accept :py:class:`.TimeSeries` objects natively as input, or can accept clocked / rasterised input data.

    See Also:
        :py:class:`.TimedModule` provides the useful methods :py:meth:`~.TimedModule._prepare_input` and :py:meth:`~.TimedModule._gen_timeseries` to help you in rasterising data for your own :py:class:`.TimedModule` subclasses.

        For more information on how to used the :py:class:`.TimedModule` API for Rockpool, see :ref:`/in-depth/api-high-level.ipynb`.
    """

    __in_TimedModule_init: bool = False
    """ A flag indicating that this ``TimedModule`` is currently being initialised """

    def __init__(
        self,
        dt: Union[float, SimulationParameter],
        spiking_input: bool = False,
        spiking_output: bool = False,
        add_events: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialise this :py:class:`.TimedModule` object

        When initialised, the :py:class:`.TimedModule` will have a :py:attr:`~.TimedModule.dt` attribute assigned, as well as initialising the internal module :py:attr:`~.TimedModule._timestep`, :py:attr:`~.TimedModule._parent_dt_factor` and :py:attr:`~.TimedModule._is_child`. The subclass :py:meth:`~.TimedModule.evolve` method will be wrapped to update the internal timestamp clock.

        Args:
            dt (float): The duration of a single time step for this module, in seconds
            spiking_input (bool): If ``True``, this module accepts :py:class:`.TSEvent` event time series objects as input. If ``False`` (default), this module accepts :py:class:`TSContinuous` continuous time series objects as input.
            spiking_output (bool): If ``True``, this module sends :py:class:`.TSEvent` event time series objects as output. If ``False`` (default), this module sends :py:class:`TSContinuous` continuous time series objects as output.
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        # - Initialise superclass
        super().__init__(
            spiking_input=spiking_input, spiking_output=spiking_output, *args, **kwargs
        )

        # - Assign dt
        self.dt: Union[float, SimulationParameter] = SimulationParameter(dt, "dt")
        """ float: The simulation and input rasterisation timestep for this `.TimedModule` """

        # - Initialise internal timestep
        self._timestep: int = 0
        """ The current time-step count in units of :py:attr:`.dt` """

        # - Initialise dt factor (1.0 by default)
        self._parent_dt_factor: float = 1.0
        """ The factor between the parent's :py:attr:`.dt` and this module's :py:attr:`.dt`. Given by ``self.dt / parent.dt`` """

        # - Initialise a flag indicating that this is a child module
        self._is_child: bool = TimedModule.__in_TimedModule_init
        """ Flag indicating that this is a child module """

        # - Record that we are currently initialising the module tree
        TimedModule.__in_TimedModule_init = True

        # - Wrap `evolve()` method to perform timestep updates
        self.__evolve = self.evolve
        self.evolve = self._evolve_wrapper

        # - Remember "add events" argument
        self._add_events = add_events

    def __post_init__(self) -> None:
        """
        Perform post-initialisation work for :py:class:`.TimedModule`

        Handles setting the :py:attr:`.dt` attribute for a :py:class:`.TimedModule` tree with all sub-modules. Manages the :py:attr:`._is_child` attribute.
        """
        # - Find least-common-multiple `dt` for base module
        if not self._is_child:
            self._set_dt()

        # - Restore in_init flag
        TimedModule.__in_TimedModule_init = self._is_child

    def _set_dt(self, max_factor: float = 100) -> None:
        """
        Set a time step size for the network which is the lowest common multiple of all sub-module's :py:attr:`.dt` s.

        :param float max_factor:    Factor by which the module :py:attr:`.dt` may exceed the largest sub-module :py:attr:`.dt` before an error is raised. Default: 100.

        :raises ValueError: If a sensible :py:attr:`.dt` cannot be found
        """
        if self.modules():
            ## -- Try to determine self.dt from layer time steps
            dt_list = [
                Decimal(str(dt)) for _, dt in leaves(self.attributes_named("dt"))
            ]

            # - Determine least common multiple
            t_lcm = dt_list[0]
            for dt in dt_list[1:]:
                try:
                    t_lcm = lcm(t_lcm, dt)
                except ValueError:
                    raise ValueError(
                        "Network: dt is too small for one or more layers. Try larger"
                        + " value or decrease `decimal_base`."
                    )

            if (
                # If result is way larger than largest dt, assume it hasn't worked
                t_lcm > max_factor * np.amax(dt_list)
                # Also make sure that t_lcm is indeed a multiple of all dt's
                or any(not is_multiple(t_lcm, dt) for dt in dt_list)
            ):
                raise ValueError(
                    "Network: Couldn't find a reasonable common time step "
                    + f"(layer dt's: {dt_list}, found: {t_lcm}"
                )

            # - Store base-level time step, for now as float for compatibility
            self.dt = float(t_lcm)

        # - Store number of layer time steps per global time step for each layer
        for _, mod in self.modules().items():
            if hasattr(mod, "dt"):
                mod._parent_dt_factor = float(round(self.dt / mod.dt))

    def _evolve_wrapper(
        self,
        ts_input=None,
        duration=None,
        num_timesteps=None,
        kwargs_timeseries=None,
        record: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[TimeSeries, Dict, Dict]:
        """
        Wrap a call to :py:meth:`.evolve` to update the internal time-steps count

        See :py:meth:`.evolve` for calling syntax.
        """
        # - Determine number of timesteps
        num_timesteps = self._determine_timesteps(ts_input, duration, num_timesteps)

        # - Call wrapped evolve
        ts_output, state_dict, record_dict = self.__evolve(
            ts_input,
            duration,
            num_timesteps,
            kwargs_timeseries,
            record,
            *args,
            **kwargs,
        )

        # - We could re-wrap outputs as TimeSeries here, if desired

        # - Update internal time
        self._timestep += num_timesteps

        return ts_output, state_dict, record_dict

    def _determine_timesteps(
        self,
        ts_input: Optional[TimeSeries] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> int:
        """
        Determine how many time steps to evolve with the given input specification

        :param Optional[TimeSeries] ts_input:   TxM or Tx1 time series of input signals for this layer
        :param Optional[float] duration:        Duration of the desired evolution, in seconds. If not provided, ``num_timesteps`` or the duration of ``ts_input`` will be used to determine evolution time
        :param Optional[int] num_timesteps:     Number of evolution time steps, in units of :py:attr:`.dt`. If not provided, ``duration`` or the duration of ``ts_input`` will be used to determine evolution time

        :return int:
            num_timesteps: Number of evolution time steps
        """

        if num_timesteps is None:
            # - Determine `num_timesteps`
            if duration is None:
                # - Determine duration
                if ts_input is None:
                    raise TypeError(
                        self.full_name
                        + "One of 'num_timesteps', 'ts_input' or 'duration' must be supplied."
                    )

                if ts_input.periodic:
                    # - Use duration of periodic TimeSeries, if possible
                    duration = ts_input.duration

                else:
                    # - Evolve until the end of the input TimeSeries
                    duration = ts_input.t_stop - self.t
                    if duration <= 0:
                        raise ValueError(
                            self.full_name
                            + "Cannot determine an appropriate evolution duration."
                            + " 'ts_input' finishes before the current evolution time."
                        )
            num_timesteps = int(np.floor((duration + tol_abs) / self.dt))
        else:
            if not isinstance(num_timesteps, int):
                raise TypeError(
                    self.full_name + "'num_timesteps' must be a non-negative integer."
                )
            elif num_timesteps < 0:
                raise ValueError(
                    self.full_name + "'num_timesteps' must be a non-negative integer."
                )

            # - Convert parent num_timestamps to self-compatible num-timestamps
            num_timesteps = int(np.ceil(num_timesteps * self._parent_dt_factor))

        return num_timesteps

    def _gen_time_trace(self, t_start: float, num_timesteps: int) -> np.ndarray:
        """
        Generate a time trace starting at ``t_start``, of length ``num_timesteps`` with time step :py:attr:`.dt`

        :param float t_start:       Start time, in seconds
        :param int num_timesteps:   Number of time steps to generate, in units of :py:attr:`.dt`

        :return ndarray: Generated time trace
        """
        # - Generate a trace
        time_trace = np.arange(num_timesteps) * self.dt + t_start

        return time_trace

    def _prepare_input(
        self,
        ts_input: Optional[TimeSeries] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Sample input, set up time base

        This function checks an input signal, and prepares a discretised time base according to the time step of the current module

        :param Optional[TimeSeries] ts_input:   :py:class:`.TimeSeries` of TxM or Tx1 Input signals for this layer
        :param Optional[float] duration:        Duration of the desired evolution, in seconds. If not provided, then either ``num_timesteps`` or the duration of ``ts_input`` will define the evolution time
        :param Optional[int] num_timesteps:     Integer number of evolution time steps, in units of :py:attr:`.dt`. If not provided, then ``duration`` or the duration of ``ts_input`` will define the evolution time

        :return (ndarray, ndarray, int): (time_base, input_steps, num_timesteps)
            time_base:      T1 Discretised time base for evolution
            input_raster    (T1xN) Discretised input signal for layer
            num_timesteps:  Actual number of evolution time steps, in units of :py:attr:`.dt`
        """
        if (ts_input is not None) and not isinstance(ts_input, self.input_type):
            raise TypeError(
                self.full_name
                + f": This TimedModule can only receive inputs of class `{self.input_type.__name__}`"
            )

        if self.spiking_input:
            return self._prepare_input_events(
                ts_input, duration, num_timesteps, add_events=self._add_events
            )
        else:
            return self._prepare_input_continuous(ts_input, duration, num_timesteps)

    def _prepare_input_continuous(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Sample input, set up time base

        This function checks an input signal, and prepares a discretised time base according to the time step of the current module

        :param Optional[TSContinuous] ts_input: :py:class:`.TSContinuous` of TxM or Tx1 Input signals for this layer
        :param Optional[float] duration:        Duration of the desired evolution, in seconds. If not provided, then either ``num_timesteps`` or the duration of ``ts_input`` will define the evolution time
        :param Optional[int] num_timesteps:     Integer number of evolution time steps, in units of :py:attr:`.dt`. If not provided, then ``duration`` or the duration of ``ts_input`` will define the evolution time

        :return (ndarray, ndarray, int): (time_base, input_raster, num_timesteps)
            time_base:      T1 Discretised time base for evolution
            input_raster:    (T1xN) Discretised input signal for layer
            num_timesteps:  Actual number of evolution time steps, in units of :py:attr:`.dt`
        """

        # - Work out how many time steps to take
        num_timesteps = self._determine_timesteps(ts_input, duration, num_timesteps)

        # - Generate discrete time base
        time_base = self._gen_time_trace(self.t, num_timesteps)

        if ts_input is not None:

            # - Make sure time series is of correct type
            if not isinstance(ts_input, TSContinuous):
                raise TypeError(
                    self.full_name
                    + ": 'ts_input' must be of type 'TSContinuous' or 'None'."
                )

            # - Warn if evolution period is not fully contained in ts_input
            if not (ts_input.contains(time_base)):
                warn(
                    self.full_name
                    + f": Evolution period (t = {time_base[0]} to {time_base[-1]}) "
                    + "is not fully contained in input signal "
                    + f"(t = {ts_input.t_start} to {ts_input.t_stop})."
                    + " You may need to use a 'periodic' time series."
                )

            # - Sample input trace
            input_raster = ts_input(time_base)

        else:
            # - Assume zero inputs
            input_raster = np.zeros((num_timesteps, self.size_in))

        return time_base, input_raster, num_timesteps

    def _prepare_input_events(
        self: "TimedModule",
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        add_events: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Sample input from a :py:class:`TSEvent` time series, set up evolution time base

        This function checks an input signal, and prepares a discretised time base according to the time step of the current module

        :param Optional[TSEvent] ts_input:  TimeSeries of TxM or Tx1 Input signals for this layer
        :param Optional[float] duration:    Duration of the desired evolution, in seconds. If not provided, then either ``num_timesteps`` or the duration of ``ts_input`` will determine evolution itme
        :param Optional[int] num_timesteps: Number of evolution time steps, in units of :py:attr:`.dt`. If not provided, then either ``duration`` or the duration of ``ts_input`` will determine evolution time

        :return (ndarray, ndarray, int):
            time_base:      T1X1 vector of time points -- time base for the rasterisation
            spike_raster:   Boolean or integer raster containing spike information. T1xM array
            num_timesteps:  Actual number of evolution time steps, in units of :py:attr:`.dt`
        """

        # - Work out how many time steps to take
        num_timesteps = self._determine_timesteps(ts_input, duration, num_timesteps)

        # - Generate discrete time base
        time_base = self._gen_time_trace(self.t, num_timesteps)

        # - Extract spike timings and channels
        if ts_input is not None:

            # - Make sure time series is of correct type
            if not isinstance(ts_input, TSEvent):
                raise TypeError(
                    self.full_name + "'ts_input' must be of type 'TSEvent' or 'None'."
                )

            # Extract spike data from the input variable
            spike_raster = ts_input.raster(
                dt=self.dt,
                t_start=self.t,
                num_timesteps=np.size(time_base),
                channels=np.arange(self.size_in),
                add_events=add_events,
            )

        else:
            spike_raster = np.zeros((np.size(time_base), self.size_in))

        return time_base, spike_raster, num_timesteps

    def _gen_timeseries(self, output: np.ndarray, **kwargs) -> TimeSeries:
        """
        Wrap a clocked / rasterised output array into a :py:class:`.TimeSeries` object

        Output :py:class:`.TimeSeries` will be of the appropriate subclass, and will be named nicely.

        Args:
            output (np.ndarray): The clocked or rasterised output data ``(T, N)``
            **kwargs: Additional keyword arguments to :py:class:`.TimeSeries`

        Returns:
            TimeSeries: The data in ``output`` wrapped into a :py:class:`.TimeSeries` object
        """
        if len(output.shape) > 2:
            output = output[0]

        if self.spiking_output:
            return self._gen_tsevent(output, **kwargs)
        else:
            return self._gen_tscontinuous(output, **kwargs)

    def _gen_tsevent(
        self,
        output: np.ndarray,
        dt: Optional[float] = None,
        t_start: Optional[float] = None,
        name: Optional[str] = None,
        periodic: bool = False,
        num_channels: Optional[int] = None,
        spikes_at_bin_start: bool = False,
    ) -> TSEvent:
        """
        Wrap a rasterised output array as a :py:class:`.TSEvent` object to present as output for this module

        Output :py:class:`.TSEvent` s will be named nicely, with correct start timesm durations, etc. Several attributes of the :py:class:`.TSEvent` object can be set as arguments here.

        Args:
            output (np.ndarray): A rasterised event array ``(T, N)``
            dt (Optional[float]): The time-step of the rasterised array ``output``. If not provided, the module :py:attr:`.dt` will be used
            t_start (Optional[float]): The start time of the output series, in seconds. If not provided, the module time before evolution will be used
            name (Optional[str]): The desired name of the :py:class:`.TSEvent` object. If not provided, the object will be named nicely according to the module name
            periodic (bool): Flag to indicate whether the returned :py:class:`.TSEvent` should be periodic. Default: ``False``, the :py:class:`.TSEvent` will not be periodic
            num_channels (Optional[int]): The desired number of total channels for the output :py:class:`.TSEvent` object. If not provided, the output size :py:attr:`.size_out` of the current module will be used
            spikes_at_bin_start (bool): If ``False`` (default), spike events will be considered to fall in the middle of the time bin they fall in. If ``True``, all spike events will be considered to occur at the start of the time bin they fall in.

        Returns:
            TSEvent: The wrapped output raster as a :py:class:`.TSEvent` object
        """
        # - Build a name for the time series
        if name is None:
            if self.name:
                name = f"Output events '{self.name}'"
            else:
                name = f"Output events"

        # - Create and return a new event time series
        return TSEvent.from_raster(
            raster=output,
            dt=self.dt if dt is None else dt,
            t_start=self.t if t_start is None else t_start,
            name=name,
            periodic=periodic,
            num_channels=self.size_out if num_channels is None else num_channels,
            spikes_at_bin_start=spikes_at_bin_start,
        )

    def _gen_tscontinuous(
        self,
        output: np.ndarray,
        dt: Optional[float] = None,
        t_start: Optional[float] = None,
        name: Optional[str] = None,
        periodic: bool = False,
        interp_kind: str = "previous",
    ) -> TSContinuous:
        """
        Wrap a rasterised output array as a :py:class:`TSContinuous` object to present as output for this module

        Output :py:class:`.TSContinuous` s will be named nicely, with correct start times, durations, etc. Several attributes of the :py:class:`.TSContinuous` object can be set as arguments here.

        Args:
            output (np.ndarray): A clocked time series data array ``(T, N)``
            dt (Optional[float]): The time-step of the clocked array ``output``. If not provided, the module :py:attr:`.dt` will be used
            t_start (Optional[float]): The start time of the output :py:class:`.TSContinuous` object, in seconds. If not provided, the module time before evolution will be used
            name (Optional[str]): The desired name of the :py:class:`.TSContinuous` object. If not provided, the object will be named nicely according to the module name
            periodic (bool): Flag to indicate whether the returned :py:class:`.TSContinuous` should be periodic. Default: ``False``, the :py:class:`.TSContinuous` will not be periodic
            interp_kind (str): The style of interpolation to apply to the returned :py:class:`.TSContinuous` object. Default: ``"previous"``

        Returns:
            TSContinuous: The wrapped output data as a `TSContinuous` object
        """
        # - Build a name for the time series
        if name is None:
            if self.name:
                name = f"Output samples '{self.name}'"
            else:
                name = f"Output samples"

        # - Create and return a new continuous time series
        return TSContinuous.from_clocked(
            samples=output,
            dt=self.dt if dt is None else dt,
            t_start=self.t if t_start is None else t_start,
            periodic=periodic,
            name=name,
            interp_kind=interp_kind,
        )

    @abstractmethod
    def evolve(
        self,
        ts_input: Union[TimeSeries, np.ndarray] = None,
        duration: float = None,
        num_timesteps: int = None,
        kwargs_timeseries: dict = None,
        record: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[TimeSeries, Dict, Dict]:
        """
        Evolve the state of this module over time

        Warnings:
            If you are seeing this message in documentation for a :py:class:`.TimedModule` subclass, then THIS CLASS HAS NOT PROVIDED DOCUMENTATION FOR ITS EVOLVE METHOD. PLEASE UPDATE THE DOCUMENTATION TO INCLUDE SPECIFIC DETAILS FOR THIS CLASS.

        You need to implement an :py:meth:`.evolve` method for each class which inherits from :py:class:`.TimedModule`.

        Here is an example :py:meth:`.evolve` method that rasterises a time series and uses the rasterised version for further processing. The output data is re-wrapped as a time series and returned.

        .. code-block:: python

            def evolve(...):
                # - Rasterise input and prepare input time steps
                time_base, input_raster, num_timesteps = self._prepare_input(
                    ts_input, duration, num_timesteps
                )

                # - Call sub-modules, do your evolution, etc.

                # - Return and wrap outputs if necessary
                return (
                    self._gen_timeseries(output, **kwargs_timeseries),
                    new_state,
                    record_dict,
                )

        Here is an example :py:meth:`.evolve` method that uses :py:class:`.TimeSeries` objects natively. Any rasterisation would be taken care of by submodules, if and when required.

        .. code-block:: python

            def evolve(...):
                new_state = {}
                record = {}

                x1, new_state1, record1 = self.submodule(input_ts)
                new_state.update({'submodule': new_state1})
                record.update({'submodule': record1})

                x2, new_state2, record2 = self.submodule2(x1)
                new_state.update({'submodule2': new_state2})
                record.update({'submodule2': record2})

                return x2, new_state, record

        You can of course use a mixture of these approaches.

        Args:
            ts_input (Union[TimeSeries, np.ndarray]): The input time series over which to evolve
            duration (float): The duration over which to evolve, in seconds
            num_timesteps (int): The number of time steps (in terms of the :py:attr:`.dt` attribute of this module) to evolve over
            kwargs_timeseries (Optional[dict]): Any additional arguments to pass when generating output time series
            record (bool): If ``True``, this module and sub-modules must record their state during evolution and return it in the ``record_state`` dict. If ``False`` (default), no recording is requested
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            tuple: (output_ts, new_state, record_state)
                output_ts :py:class:`.TimeSeries`: A time series containing the output time series produces by this module.
                new_state dict: A dictionary containing the updated state of this module and sub-modules, after evolution
                record_state dict: If the argument ``record`` is ``True``, ``record_state`` must contain a dictionary of the recorded states o this and all sub-modules during evolution. Otherwise it may be an empty dict.
        """
        raise NotImplementedError

        # - Rasterise input and prepare input time steps
        time_base, input_raster, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Return and wrap outputs if necessary
        return (
            self._gen_timeseries(output, **kwargs_timeseries),
            new_state,
            record_dict,
        )

    def __call__(self, *args, **kwargs) -> Tuple[TimeSeries, Dict, Dict]:
        """
        Evolve the state of this :py:class:`.TimedModule` over time

        Args:
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            tuple: (output_ts, new_state, record_state)
                output_ts (TimeSeries): The output time series produced by this module.
                new_state (dict): A dictionary containing the updated state of this and all sub-modules after evolution
                record_state (dict): If the argument ``record`` is ``True``, ``record_state`` must contain a dictionary of the recorded states o this and all sub-modules during evolution. Otherwise it may be an empty dict.
        """
        return self.evolve(*args, **kwargs)

    @property
    def input_type(self) -> type:
        """type: The :py:class:`.TimeSeries` class accepted by this module"""
        if self.spiking_input:
            return TSEvent
        else:
            return TSContinuous

    @property
    def output_type(self) -> type:
        """type: The :py:class:`.TimeSeries` class returned by this module"""
        if self.spiking_output:
            return TSEvent
        else:
            return TSContinuous

    @property
    def t(self) -> float:
        """float: The current evolution time of this layer, in seconds"""
        return self._timestep * self.dt

    @t.setter
    def t(self, new_t) -> None:
        self._timestep = int(np.floor(new_t / self.dt))

    def reset_time(self) -> None:
        """
        Reset the internal time of this module and all sub-modules to zero
        """
        # - Reset own time
        self._timestep = 0

        # - Get attribute registry
        __registered_attributes, __modules = self._get_attribute_registry()

        # - Reset submodule time
        for (k, m) in __modules.items():
            if hasattr(m, "reset_time"):
                m[0].reset_time()

    def reset_all(self) -> None:
        """
        Reset the internal state and time of this module and all sub-modules
        """
        self.reset_state()
        self.reset_time()


class TimedModuleWrapper(TimedModule):
    """
    Wrap a low-level Rockpool :py:class:`.Module` automatically into a :py:class:`.TimedModule` object

    Use this class to automatically convert a :py:class:`.Module` subclass, implementing the low-level API of Rockpool, into a :py:class:`.TimedModule` object that supports the high-level time series API directly.

    Notes:
        Only a single output argument may be returned from the wrapped :py:class:`.TimedModule`. However, multiple return arguments from the internal module can be handled through the ``output_num`` argument to :py:meth:`.__init__`.

        Recorded state from the wrapped module is not currently converted automatically into :py:class:`.TimeSeries` objects. Just keep that in mind.

    Examples:
        Constract a low-level module, wrap it into a :py:class:`.TimedModule`:

         >>> from rockpool.nn.modules import RateEuler, TimedModuleWrapper
         >>> mod = RateEuler(...)
         >>> tmod = TimedModuleWrapper(mod)

    See Also:
        If you want to convert a :py:class:`.Module` object, use this class.

        If you need to convert a Rockpool v1 :py:class:`~rockpool.nn.layers.Layer` subclass, use either the :py:class:`.LayerToTimedModule` or the `.astimedmodule` decorator.

        For more information, see :ref:`in-depth/api-high-level.ipynb`.
    """

    def __init__(
        self,
        module: Module,
        output_num: int = 0,
        dt: float = None,
        add_events: bool = True,
        *args,
        **kwargs,
    ):
        """
        Wrap a low-level module as a high-level :py:class:`.TimedModule`

        Args:
            module (:py:class:`.Module`): The module to wrap. Must inherit from :py:class:`.Module`.
            output_num (int): If the output of the evolution function for ``module`` returns multiple outputs, then here you should specify which of the outputs to wrap into a time series to return. :py:class:`.TimedModuleWrapper` only supports returning one output argument from :py:meth:`.evolve`.
            dt (float): The timestep to set for ``module``, if ``module.dt`` does not exist. Note that ``module.dt`` will not be overridden by this argument!
            add_events (bool): If ``True``, then multiple events per time bin will be summed when converting to a raster. If ``False``, only a single event will be retained per time bin. Default: ``True``, sum events in each time bin.
        """
        # - Check that we are wrapping a Module object
        if not isinstance(module, Module):
            raise TypeError(self.full_name + ": `module` must be a 'Module' object.")

        # - Warn that an extra `dt` is ignored
        if dt is not None and hasattr(module, "dt"):
            warn(
                "`dt` argument to `TimedModuleWrapper` is ignored if the module already has a `dt` attribute."
            )

        # - Assign a `dt`, if the submodule doesn't already have one
        if not hasattr(module, "dt"):
            if dt is None:
                raise KeyError(
                    self.full_name
                    + ": If 'module' has no `dt`, it must be passed as an argument."
                )

            module.dt = SimulationParameter(dt)

        # - Initialise superclass
        super().__init__(
            shape=(module.size_in, module.size_out),
            spiking_input=module.spiking_input,
            spiking_output=module.spiking_output,
            dt=module.dt,
            add_events=add_events,
            *args,
            **kwargs,
        )

        # - Keep a handle to the submodule
        self._module = module

        # - Remember which output to select
        self._output_num = output_num

    @property
    def module(self):
        """`Module`: The wrapped module"""
        return self._module

    def __repr__(self) -> str:
        """str: A representation of this module as a string"""
        return f"{super().__repr__()} with {self._module.full_name} as module"

    def evolve(
        self,
        ts_input: Optional[TimeSeries] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        kwargs_timeseries: Optional[Dict] = None,
        record: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[TimeSeries, Any, Any]:
        """
        Evolve the wrapped :py:class:`.Module`, handling :py:class:`.TimeSeries` input and output

        Args:
            ts_input (Optional[TimeSeries]): The input data for this evolution. If not provided, zero input will be used
            duration (Optional[float]): The duration over which to evolve this module, in seconds. If not provided, it will be inferred
            num_timesteps (Optional[int]): The number of time steps over which to evolve this module, in units of `dt`. If not provided, it will be inferred
            kwargs_timeseries (Optional[dict]): Additional keyword arguments to pass when generating the output time series
            record (bool): If ``True``, a dictionary containing a record of state during evolution for this and all submodules will be returned. If ``False`` (default), no record is requested.
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            tuple: (output_ts, new_state, record_dict)
                output_ts (`.TimeSeries`): The output of this module, wrapped as a `.TimeSeries`.
                new_state (dict): A dictionary containing teh updated state of this and all sub-modules after evolution
                record_dict (dict): If ``True``, a dictionary containing a record of state during evolution for this and all submodules will be returned. If ``False`` (default), no record is requested.
        """
        # - Rasterise input time series
        time_base, input_data, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Call evolution method of wrapped module
        output, state_dict, record_dict = self._module.evolve(input_data, record=record)

        # - Get the first output, if more than one is returned
        if isinstance(output, tuple):
            output = output[self._output_num]

        # - Convert output to TimeSeries
        if kwargs_timeseries is None:
            kwargs_timeseries = {}
        ts_out = self._gen_timeseries(output, **kwargs_timeseries)

        # - Use the optional `_wrap_recorded_state` method to convert the recorded state to TimeSeries objects
        if record:
            record_dict = self._module._wrap_recorded_state(record_dict, time_base[0])

        return ts_out, state_dict, record_dict

    def reset_state(self) -> None:
        self._module = self._module.reset_state()


class LayerToTimedModule(TimedModule):
    """
    An adapter class to wrap a Rockpool v1 :py:class:`.Layer` object, converting the object to support the :py:class:`.TimedModule` high-level Rockpool v2 API

    Use this class to automagically convert a Rockpool v1 :py:class:`.Layer` object into a Rockpool v2 :py:class:`.TimedModule` object. This class is used internally by the :py:func:`~.timed_module.astimedmodule` decorator to convert a v1 class into a v2 class.

    Examples:
        Construct a v1 :py:class:`.RecRateEulerV1` layer, and convert it to a v2 :py:class:`.TimedModule`:

        >>> from rockpool.nn.layers import RecRateEulerV1
        >>> from rockpool.nn.modules.timed_module import LayerToTimedModule
        >>> lyr = RecRateEuler(...)
        >>> tmod = LayerToTimedModule(lyr)
        >>> output_ts, new_state, record = tmod(input_ts)

    See Also:
        If you want to convert a :py:class:`.Module` object implementing the low-level v2 API to the high-level :py:class:`.TimedModule` v2 API, use the :py:class:`.TimedModuleWrapper` class.

        For more information, see :ref:`in-depth/api-high-level.ipynb`.
    """

    def __init__(
        self,
        layer: "Layer",
        parameters: Iterable[str] = None,
        states: Iterable[str] = None,
        simulation_parameters: Iterable[str] = None,
    ):
        """
        Wrap a v1 :py:class:`.Layer` object as a v2 :py:class:`.TimedModule` object

        Args:
            layer (:py:class:`.Layer`): The v1 layer object to wrap
            parameters (Iterable[str]): A list (or tuple) containing the names of all attributes of `layer` that should be registered as Rockpool :py:class:`.Parameter` s
            states (Iterable[str]): A list (or tuple) containing the names of all attributes of `layer` that should be registered as Rockpool :py:class:`.State` s
            simulation_parameters (Iterable[str]): A list (or tuple) containing the names of all attributes of `layer` that should be registered as Rockpool :py:class:`.SimulationParameter` s
        """
        from rockpool.nn.layers.layer import Layer

        if not isinstance(layer, Layer):
            raise TypeError("LayerToTimedModule can only wrap a Rockpool v1 Layer.")

        spiking_input = layer.input_type is TSEvent
        spiking_output = layer.output_type is TSEvent

        # - Record layer as submodule
        self._module: Layer = layer
        """ Layer: The wrapped layer object """

        super().__init__(
            shape=(layer.size_in, layer.size),
            dt=layer.dt,
            spiking_input=spiking_input,
            spiking_output=spiking_output,
        )

        self._name: str = layer.name
        """ str: The name of the wrapped layer """

        # - Record parameters
        if parameters is not None:
            for param in parameters:
                self._register_attribute(param, Parameter(getattr(self._module, param)))

        # - Record states
        if states is not None:
            for state in states:
                self._register_attribute(state, State(getattr(self._module, state)))

        # - Record simulation parameters
        if simulation_parameters is not None:
            for sim_param in simulation_parameters:
                self._register_attribute(
                    sim_param, SimulationParameter(getattr(self._module, sim_param))
                )

    def reset_time(self) -> None:
        """Reset the internal clock for this `TimedModule`"""
        super().reset_time()
        self._module.reset_time()

    def reset_state(self) -> None:
        """Reset the internal state for this `TimedModule`"""
        super().reset_state()
        self._module.reset_state()

    def evolve(
        self,
        ts_input: Optional[TimeSeries] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        kwargs_timeseries: Optional[dict] = None,
        record: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[TimeSeries, dict, dict]:
        """
        Evolve the wrapped layer, handle inputs and outputs

        Args:
            ts_input (Optional[`TimeSeries`]): The input time series to evolve over. If not provided, zero input will be used
            duration (Optional[float]): The duration of the evolution, in seconds. If not provided it will be inferred
            num_timesteps (Optional[int]): The duration of evolution in integer units of :py:attr:`dt`. If not provided it will be inferred
            kwargs_timeseries (Optional[dict]): Additional keyword arguments to pass when creating the return `TimeSeries` object
            record (bool): If ``True``, a dictionary of recorded state will be returned for the module. If ``False`` (default), no recorded state is requested.
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            tuple: ts_output, new_state, record_dict
                ts_output (`TimeSeries`): A time series containing the output of the module evolution
                new_state (dict): A dictionary containing the state of the module post evolution
                record_dict (dict): If ``record == True``, a dictionary of recorded state will be returned for the module. If ``record == False`` (default), no recorded state is requested
        """
        # - Call submodule layer to evolve
        ts_output = self._module.evolve(
            ts_input, duration, num_timesteps, *args, **kwargs
        )

        # - Return output, state and record dict
        return ts_output, self.state(), {}

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Set an attribute of the wrapped layer, if it has been registered

        Args:
            key (str): The name of the attribute to set
            value (Any): The value to set to the attribute
        """
        # - Set attribute in module, if registered
        if self._has_registered_attribute(key):
            if hasattr(self, "_module"):
                setattr(self._module, key, value)

                # - Ensure we get validated value from submodule
                value = getattr(self._module, key)

        # - Set value using superclass
        super().__setattr__(key, value)

    def __getattr__(self, key: str) -> Any:
        """
        Get an attribute of the wrapped layer, if it has been registered

        Args:
            key (str): The name of the attribute to get

        Returns:
            Any: The value of the attribute

        """
        if key == "_ModuleBase__registered_attributes" or key == "_ModuleBase__modules":
            raise AttributeError

        # - Get attribute from module if registered
        if self._has_registered_attribute(key):
            return getattr(self._module, key)
        else:
            raise AttributeError(
                f"Attribute {key} not found in TimedModule class {self.class_name} named {self.name}"
            )

    # def _get_attribute_family(self, type_name: str, family: str = None):
    #     # - Get matching attributes
    #     return super()._get_attribute_family(type_name, family)


def astimedmodule(
    v1_cls: type = None,
    parameters: Optional[Iterable[str]] = None,
    states: Optional[Iterable[str]] = None,
    simulation_parameters: Optional[Iterable[str]] = None,
) -> Union[type, Callable]:
    """
    Convert a Rockpool v1 class to a v2 class

    This decorator transparently converts a Rockpool v1 :py:class:`.Layer` subclass to a Rockpool v2 high-level API :py:class:`.TimedModule` subclass.

    You can specify the parameter, state and simulation parameter attributes of the v1 layer to expose via the v2 API.

    Evolution should just work™, and ideally you won't need to modify anything in the v1 code to use the class within the v2 API. Depending on the complexity of the v1 layer, this may or may not be the case.

    Examples:
        Specify a simple v1 layer, and convert it to a v2 :py:class:`.Module`:

        .. code-block:: python

            from rockpool.nn.layers import Layer
            from rockpool.nn.modules.timed_module import astimedmodule

            @astimedmodule(
                parameters = ['tau_mem', 'tau_syn', 'bias'],
                states = ['v_mem', 'i_syn'],
                simulation_parameters = ['noise_std']
            )
            class my_v1_layer(Layer):
                def __init__(...):
                    ...

                def evolve(...):
                    ...

    See Also
        For more information, see :ref:`in-depth/api-high-level.ipynb`.

    Args:
        v1_cls (type): A v1 :py:class:`.Layer` subclass to wrap
        parameters (Optional[Iterable[str]]): An iterable set of strings, specifying the names of attributes provided by ``v1_cls`` that should be automatically registered as Rockpool :py:class:`.Parameter` s.
        states (Optional[Iterable[str]]): An iterable set of strings, specifying the names of attributes provided by ``v1_cls`` that should be automatically registered as Rockpool :py:class:`.State` s.
        simulation_parameters (Optional[Iterable[str]]): An iterable set of strings, specifying the names of attributes provided by ``v1_cls`` that should be automatically registered as Rockpool :py:class:`.SimulationParameter` s.

    Returns:
        :py:class:`.LayerToTimedModule`: A wrapped class instantiator the will create a v2 high-level API object
    """
    # - Be lenient if any parameters are not lists/tuples
    if not isinstance(parameters, (tuple, list)) and parameters is not None:
        parameters = [parameters]

    if not isinstance(states, (tuple, list)) and states is not None:
        states = [states]

    if (
        not isinstance(simulation_parameters, (tuple, list))
        and simulation_parameters is not None
    ):
        simulation_parameters = [simulation_parameters]

    from rockpool.nn.layers import Layer

    def wrapper_function(v1_cls):
        if not issubclass(v1_cls, Layer):
            raise ValueError(
                "`@astimedmodule` may only be applied to Rockpool v1 `Layer` subclasses."
            )

        # - Define a wrapping class
        @functools.wraps(v1_cls, updated=())
        class wrapper(LayerToTimedModule):
            def __init__(self, *args, **kwargs):
                # - Instantiate layer
                layer = v1_cls(*args, **kwargs)

                # - Wrap layer
                super().__init__(layer, parameters, states, simulation_parameters)

        # - Update docstrings for wrapped class
        if v1_cls.__doc__ is None:
            # - Inherit docs from parent class
            wrapper.__doc__ = v1_cls.__base__.__doc__
        else:
            wrapper.__doc__ = v1_cls.__doc__

        wrapper.__init__.__doc__ = v1_cls.__init__.__doc__
        wrapper.evolve.__doc__ = v1_cls.evolve.__doc__

        # - Return the decorated class
        return wrapper

    if v1_cls is None:
        return wrapper_function
    else:
        return wrapper_function(v1_cls)
