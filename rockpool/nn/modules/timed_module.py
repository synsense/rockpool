from typing import Optional, Union, Tuple, Dict, Any, List, Iterable, Callable
from warnings import warn
from abc import abstractmethod

import numpy as np

from rockpool.timeseries import TimeSeries, TSContinuous, TSEvent

from rockpool.nn.modules.module import Module, ModuleBase, PostInitMetaMixin
from rockpool.parameters import SimulationParameter, Parameter, State

from rockpool.nn.layers.layer import Layer

import functools

from decimal import Decimal

tol_rel = 1e-5
tol_abs = 1e-6
decimal_base = 1e-7


RealValue = Union[float, Decimal, SimulationParameter, str]

from collections import abc

Tree = Union[abc.Iterable, abc.MutableMapping]


def is_multiple(
    a: RealValue,
    b: RealValue,
    tol_rel: RealValue = tol_rel,
    tol_abs: RealValue = tol_abs,
) -> bool:
    """
    Check whether a%b is 0 within some tolerance.

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
    for k, v in d.items():
        if isinstance(v, dict):
            yield from leaves(v)
        else:
            yield k, v


class TimedModule(ModuleBase, metaclass=PostInitMetaMixin):
    __in_TimedModule_init: bool = False

    def __init__(
        self,
        dt: float,
        spiking_input: bool = False,
        spiking_output: bool = False,
        *args,
        **kwargs,
    ):
        print("TimedModule.__init__ start")
        # - Initialise superclass
        super().__init__(
            spiking_input=spiking_input, spiking_output=spiking_output, *args, **kwargs
        )

        # - Assign dt
        self.dt: float = SimulationParameter(dt, "dt")

        # - Initialise internal timestep
        self._timestep: int = 0

        # - Initialise dt factor (1.0 by default)
        self._parent_dt_factor: float = 1.0

        # - Initialise a flag indicating that this is a child module
        self._is_child = TimedModule.__in_TimedModule_init
        TimedModule.__in_TimedModule_init = True

        # - Wrap `evolve()` method to perform timestep updates
        self.__evolve = self.evolve
        self.evolve = self._evolve_wrapper

    def __post_init__(self) -> None:
        # - Find least-common-multiple `dt` for base module
        if not self._is_child:
            self._set_dt()

        # - Restore in_init flag
        TimedModule.__in_TimedModule_init = self._is_child

    def _set_dt(self, max_factor: float = 100) -> None:
        """
        Set a time step size for the network which is the lcm of all layers' dt's.

        :param float max_factor:    Factor by which the network `.dt` may exceed the largest layer `.Layer.dt` before an error is raised

        :raises ValueError: If a sensible `.dt` cannot be found
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
        # - Determine number of timesteps
        num_timesteps = self._determine_timesteps(ts_input, duration, num_timesteps)

        # - Call wrapped evolve
        ts_output, state_dict, record_dict = self.__evolve(
            ts_input, duration, num_timesteps, kwargs_timeseries, record, *args, *kwargs
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

        :return int:                            num_timesteps: Number of evolution time steps
        """

        if num_timesteps is None:
            # - Determine ``num_timesteps``
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
                            + " 'ts_input' finishes before the current evolution time.",
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
        Generate a time trace starting at ``t_start``, of length ``num_timesteps`` with time step length :py:attr:`._dt`

        :param float t_start:       Start time, in seconds
        :param int num_timesteps:   Number of time steps to generate, in units of ``.dt``

        :return (ndarray): Generated time trace
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

        This function checks an input signal, and prepares a discretised time base according to the time step of the current layer

        :param Optional[TimeSeries] ts_input:   :py:class:`.TimeSeries` of TxM or Tx1 Input signals for this layer
        :param Optional[float] duration:        Duration of the desired evolution, in seconds. If not provided, then either ``num_timesteps`` or the duration of ``ts_input`` will define the evolution time
        :param Optional[int] num_timesteps:     Integer number of evolution time steps, in units of ``.dt``. If not provided, then ``duration`` or the duration of ``ts_input`` will define the evolution time

        :return (ndarray, ndarray, int): (time_base, input_steps, num_timesteps)
            time_base:      T1 Discretised time base for evolution
            input_raster    (T1xN) Discretised input signal for layer
            num_timesteps:  Actual number of evolution time steps, in units of ``.dt``
        """
        if (ts_input is not None) and not isinstance(ts_input, self.input_type):
            raise TypeError(
                self.full_name
                + f": This TimedModule can only receive inputs of class `{self.input_type.__name__}`"
            )

        if self.spiking_input:
            return self._prepare_input_events(ts_input, duration, num_timesteps)
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

        This function checks an input signal, and prepares a discretised time base according to the time step of the current layer

        :param Optional[TSContinuous] ts_input: :py:class:`.TSContinuous` of TxM or Tx1 Input signals for this layer
        :param Optional[float] duration:        Duration of the desired evolution, in seconds. If not provided, then either ``num_timesteps`` or the duration of ``ts_input`` will define the evolution time
        :param Optional[int] num_timesteps:     Integer number of evolution time steps, in units of ``.dt``. If not provided, then ``duration`` or the duration of ``ts_input`` will define the evolution time

        :return (ndarray, ndarray, int): (time_base, input_raster, num_timesteps)
            time_base:      T1 Discretised time base for evolution
            input_raster:    (T1xN) Discretised input signal for layer
            num_timesteps:  Actual number of evolution time steps, in units of ``.dt``
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
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Sample input from a :py:class:`TSEvent` time series, set up evolution time base

        This function checks an input signal, and prepares a discretised time base according to the time step of the current layer

        :param Optional[TSEvent] ts_input:  TimeSeries of TxM or Tx1 Input signals for this layer
        :param Optional[float] duration:    Duration of the desired evolution, in seconds. If not provided, then either ``num_timesteps`` or the duration of ``ts_input`` will determine evolution itme
        :param Optional[int] num_timesteps: Number of evolution time steps, in units of ``.dt``. If not provided, then either ``duration`` or the duration of ``ts_input`` will determine evolution time

        :return (ndarray, ndarray, int):
            time_base:      T1X1 vector of time points -- time base for the rasterisation
            spike_raster:   Boolean or integer raster containing spike information. T1xM array
            num_timesteps:  Actual number of evolution time steps, in units of ``.dt``
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
                # add_events=getattr(self.module, "add_events", False),
            )

        else:
            spike_raster = np.zeros((np.size(time_base), self.size_in))

        return time_base, spike_raster, num_timesteps

    def _gen_timeseries(self, output: np.ndarray, **kwargs) -> TimeSeries:
        if self.spiking_output:
            return self._gen_tsevent(output, **kwargs)
        else:
            return self._gen_tscontinuous(output, **kwargs)

    def _gen_tsevent(
        self,
        output: np.ndarray,
        dt=None,
        t_start=None,
        name=None,
        periodic=False,
        num_channels=None,
        spikes_at_bin_start=False,
    ) -> TSEvent:
        return TSEvent.from_raster(
            raster=output,
            dt=self.dt if dt is None else dt,
            t_start=self.t if t_start is None else t_start,
            name=f"Output events '{self.name}'" if name is None else name,
            periodic=periodic,
            num_channels=self.size_out if num_channels is None else num_channels,
            spikes_at_bin_start=spikes_at_bin_start,
        )

    def _gen_tscontinuous(
        self,
        output: np.ndarray,
        dt: Optional[float] = None,
        t_start: Optional[float] = None,
        periodic: bool = False,
        name: Optional[str] = None,
        interp_kind: str = "previous",
    ) -> TSContinuous:
        return TSContinuous.from_clocked(
            samples=output,
            dt=self.dt if dt is None else dt,
            t_start=self.t if t_start is None else t_start,
            periodic=periodic,
            name=f"Output samples '{self.name}'" if name is None else name,
            interp_kind=interp_kind,
        )

    @abstractmethod
    def evolve(
        self,
        ts_input=None,
        duration=None,
        num_timesteps=None,
        kwargs_timeseries=None,
        record: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[TimeSeries, Dict, Dict]:
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

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> Tuple[TimeSeries, Dict, Dict]:
        return self.evolve(*args, **kwargs)

    @property
    def input_type(self):
        if self.spiking_input:
            return TSEvent
        else:
            return TSContinuous

    @property
    def output_type(self):
        if self.spiking_output:
            return TSEvent
        else:
            return TSContinuous

    @property
    def t(self):
        """
        (float) The current evolution time of this layer
        """
        return self._timestep * self.dt

    @t.setter
    def t(self, new_t):
        self._timestep = int(np.floor(new_t / self.dt))

    def reset_time(self):
        # - Reset own time
        self._timestep = 0

        # - Reset submodule time
        for m in self.modules():
            m.reset_time()


class TimedModuleWrapper(TimedModule):
    def __init__(
        self,
        module: Module,
        output_num: int = 0,
        dt: float = None,
        *args,
        **kwargs,
    ):
        # - Check that we are wrapping a Module object
        if not isinstance(module, Module):
            raise TypeError(self.full_name + ": `module` must be a 'Module' object.")

        # - Warn that an extra `dt` is ignored
        if dt is not None and hasattr(self.module, "dt"):
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
            *args,
            **kwargs,
        )

        # - Keep a handle to the submodule
        self._module = module

        # - Remember which output to select
        self._output_num = output_num

    @property
    def module(self):
        return self._module

    def __repr__(self):
        return f"{super().__repr__()} with {self.module.full_name} as module"

    def evolve(
        self,
        ts_input: Union[TimeSeries, None] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        kwargs_timeseries: Optional[Dict] = None,
        record: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[TimeSeries, Any, Any]:
        # - Rasterise input time series
        time_base, input_data, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Call evolution method of wrapped module
        output, state_dict, record_dict = self.module.evolve(input_data, record=record)

        # - Get the first output, if more than one is returned
        if isinstance(output, tuple):
            output = output[self._output_num]

        # - Convert output to TimeSeries
        if kwargs_timeseries is None:
            kwargs_timeseries = {}
        ts_out = self._gen_timeseries(output, **kwargs_timeseries)

        # - We would need to convert record_dict elements here, if we are going to do it

        return ts_out, state_dict, record_dict


class LayerToTimedModule(TimedModule):
    def __init__(
        self,
        layer: Layer,
        parameters: Iterable = None,
        states: Iterable = None,
        simulation_parameters: Iterable = None,
    ):
        if not isinstance(layer, Layer):
            raise TypeError("LayerToTimedModule can only wrap a Rockpool v1 Layer.")

        spiking_input = layer.input_type is TSEvent
        spiking_output = layer.output_type is TSEvent

        super().__init__(
            shape=(layer.size_in, layer.size),
            dt=layer.dt,
            spiking_input=spiking_input,
            spiking_output=spiking_output,
        )

        # - Record layer as submodule
        self._module = layer
        self._name = layer.name

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
                    sim_param,
                    SimulationParameter(getattr(self._module, sim_param)),
                )

    def evolve(
        self,
        ts_input: TimeSeries = None,
        duration: float = None,
        num_timesteps: int = None,
        kwargs_timeseries: Any = None,
        record: bool = False,
        *args,
        **kwargs,
    ):
        # - Call submodule layer to evolve
        ts_output = self._module.evolve(ts_input, duration, num_timesteps)

        # - Return output, state and record dict
        return ts_output, {}, {}

    def __setattr__(self, key, value):
        # - Set value using superclass
        super().__setattr__(key, value)

        # - Set attribute in module, if registered
        if self._has_registered_attribute(key):
            if hasattr(self, "_module"):
                setattr(self._module, key, getattr(self, key))

    def __getattr__(self, key):
        if key is "_ModuleBase__registered_attributes" or key is "_ModuleBase__modules":
            raise AttributeError

        # - Get attribute from module if registered
        if self._has_registered_attribute(key):
            val = getattr(self._module, key)
            try:
                return np.array(val)
            except:
                return val
        else:
            raise AttributeError

    def _get_attribute_family(self, type_name: str, family: str = None):
        # - Get matching attributes
        attributes = super()._get_attribute_family(type_name, family)

        # - Convert types if possible
        def try_array(item):
            try:
                return np.array(item)
            except:
                return item

        return tree_map(try_array, attributes)


def astimedmodule(
    cls: type = None,
    parameters: Iterable = None,
    states: Iterable = None,
    simulation_parameters: Iterable = None,
):
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

    # - Define a wrapping function
    def wrap(cls: type) -> Callable[[Any], TimedModule]:
        @functools.wraps(cls)
        def __init__(*args, **kwargs):
            # - Instantiate layer
            layer = cls(*args, **kwargs)

            # - Wrap layer
            return LayerToTimedModule(layer, parameters, states, simulation_parameters)

        return __init__

    # - Apply or return the decorator
    if cls is None:
        return wrap
    else:
        return wrap(cls)
