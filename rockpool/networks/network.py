"""

This module encapsulates networks -- combinations of multiple `.Layer` objects, connected in a directed acyclic graph.

"""

###
# network.py - Code for encapsulating networks
###


### --- Imports
import json
from decimal import Decimal
from copy import deepcopy
from typing import Callable, Union, Tuple, List, Dict, Type, Optional, Any
from warnings import warn

import numpy as np

from ..timeseries import TimeSeries
from .. import layers

# - Try to import tqdm
try:
    from tqdm.autonotebook import tqdm

    use_tqdm = True

except ImportError:
    use_tqdm = False

RealValue = Union[float, Decimal, str]

# - Configure exports
__all__ = ["Network"]

# - Relative tolerance for float comparisons
tol_rel = 1e-5
decimal_base = 1e-7
tol_abs = 1e-10

### --- Helper functions


def digits_after_point(value):
    strval = str(value)
    # - Make sure that value is actually a number
    try:
        _ = float(value)
    except TypeError as e:
        raise e
    if "." in strval:
        # Contrains decimal point
        return len(strval) - strval.index(".") - 1
    elif "e-" in strval:
        # Scientific notation -> get exponent
        return int(strval.split("-")[1])
    else:
        return 0


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
    a_rnd = np.round(float(a) / decimal_base)
    b_rnd = np.round(float(b) / decimal_base)
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


### --- Network class


class Network:
    """
    Base class to manage networks (collections of `.Layer` objects)

    `.Network` objects allow you to encapsulate a stack of layers with various configurations. Using a `.Network` object allows you to connect layers in an acyclic graph, and verifies that adjacent layers have compatible sizes and signal classes.

    :Example of building a network:

    Specify the network sizes

    >>> input_size = 2
    >>> reservoir_size = 10
    >>> output_size = 1

    Generate layer weights

    >>> weights_in = np.random.rand(input_size, reservoir_size)
    >>> weights_rec = np.random.randn(reservoir_size, reservoir_size)
    >>> weights_out = np.random.rand(reservoir_size, output_size)

    Generate the layers

    >>> lyr_in = FFRateEuler(weights_in)
    >>> lyr_rec = RecRateEuler(weights_rec)
    >>> lyr_out = PassThrough(weights_out)

    Generate the `Network` object

    >>> net = Network([lyr_in, lyr_rec, lyr_out])

    .. seealso:: The tutorial :ref:`/tutorials/building_reservoir.ipynb` illustrates using a `.Network` object to encapsulate a reservoir network.

    """

    def __init__(self, *layers: List[layers.Layer], dt: Optional[float] = None):
        """
        Base class to encapsulate several `.Layer` objects and manage signal routing

        :param Iterable[Layer] layers:   Layers to be added to the network. They will be connected in series. The order in which they are received determines the order in which they are connected. First layer will receive external input
        :param Optional[float] dt: If not none, network time step is forced to this values. Layers that are added must have time step that is multiple of dt. If None, network will try to determine suitable dt each time a layer is added.
        """

        # - Network time
        self._timestep = 0

        # Maintain set of all layers
        self.layerset = set()

        if dt is not None:
            assert dt > 0, "Network: dt must be positive."
            # - Force dt
            self._dt = dt
            self._force_dt = True
        else:
            self._force_dt = False

        if layers:
            # - First layer receives external input
            self.input_layer = self.add_layer(layers[0], external_input=True)

            # - Keep track of most recently added layer
            recent_layer: layers.Layer = layers[0]

            # - Add and connect subsequent layers
            for lyr in layers[1:]:
                self.add_layer(lyr, input_layer=recent_layer)
                recent_layer = lyr

            # - Handle to last layer
            self.output_layer = recent_layer

        # - Set evolution order and time step if no layers have been connected
        if not hasattr(self, "evol_order"):
            self.evol_order: List[layers.Layer] = self._set_evolution_order()
        if not hasattr(self, "_dt"):
            self._dt = None

    def add_layer(
        self,
        lyr: layers.Layer,
        input_layer: layers.Layer = None,
        output_layer: layers.Layer = None,
        external_input: bool = False,
        verbose: bool = False,
    ) -> layers.Layer:
        """
        Add a new layer to the network

        Add `lyr` to `self` and to `.layerset`. Its attribute name is 'lyr'+lyr.name. Check whether layer with this name already exists (replace anyway). Connect `lyr` to `input_layer` and `output_layer`.

        :param Layer lyr:                       layer to be added to the network
        :param Optional[Layer] input_layer:     Layer to connect as an input layer to `lyr`
        :param Optional[Layer] output_layer:    Layer to connect as an output layer from `lyr`
        :param Optional[bool] external_input:   If `True`, this layer should receive external input. Default: `False`, `lyr` should not be connected to external input
        :param Optional[bool] verbose:          If `True`, print feedback about layer addition. Default: `False`, do not display feedback

        :return Layer:                          `lyr`, the connected layer
        """
        # - Check whether layer time matches network time
        assert np.isclose(lyr.t, self.t), (
            "Network: Layer time must match network time "
            + "(network: t={}, layer: t={})".format(self.t, lyr.t)
        )

        # - Check whether self already contains a layer with the same name as lyr.
        if hasattr(self, lyr.name):
            # - Check if layers are the same object.
            if getattr(self, lyr.name) is lyr:
                print(
                    "Network: Layer `{}` is already part of the network".format(
                        lyr.name
                    )
                )
                return lyr
            else:
                newname = lyr.name
                # - Find a new name for lyr.
                while hasattr(self, newname):
                    newname = self._new_name(newname)
                if verbose:
                    print(
                        "Network: A layer with name `{}` already exists.".format(
                            lyr.name
                        )
                        + "The new layer will be renamed to  `{}`.".format(newname)
                    )
                lyr.name = newname

        # - Add set of input layers and flag to determine if lyr receives external input
        lyr.pre_layer = None
        lyr.external_input = external_input

        # - Add lyr to the network
        setattr(self, lyr.name, lyr)
        if verbose:
            print("Network: Added layer `{}` to network\n".format(lyr.name))

        # - Update inventory of layers
        self.layerset.add(lyr)

        # - Update global dt
        self._set_dt()

        # - Connect in- and outputs
        if input_layer is not None:
            self.connect(input_layer, lyr)
        if output_layer is not None:
            self.connect(lyr, output_layer)

        # - Make sure evolution order is updated if it hasn't been before
        if input_layer is None and output_layer is None:
            self.evol_order = self._set_evolution_order()

        return lyr

    @staticmethod
    def _new_name(name: str) -> str:
        """
        Generate a new name by first checking whether the old name ends with '_i', with i an integer. If so, replace i by i+1, otherwise append '_0'

        :param str name:    Name to be modified

        :return str:        Modified name
        """

        # - Check wheter name already ends with '_...'
        splitted_name: List[str] = name.split("_")
        if len(splitted_name) > 1:
            try:
                # - If the part after the last '_' is an integer, raise it by 1
                i = int(splitted_name[-1])
                splitted_name[-1] = str(i + 1)
                newname = "_".join(splitted_name)
            except ValueError:
                newname = name + "_0"
        else:
            newname = name + "_0"

        return newname

    def remove_layer(self, del_layer: layers.Layer):
        """
        Remove a layer from the network by removing it from the layer inventory and make sure that no other layer receives input from it

        :param Layer del_layer: Layer to be removed from the network
        """

        # - Remove connections from del_layer to others
        post_layers = []
        for lyr in self.layerset:
            if del_layer is lyr.pre_layer:
                lyr.pre_layer = None
                post_layers.append(lyr)
        if post_layers:
            warn(
                f"Network: After removing layer {del_layer.name} the following layers "
                + "will not receive input anymore:\n"
                + "\n".join(repr(lyr) for lyr in post_layers)
            )

        # - Remove del_layer from the inventory and delete it
        self.layerset.remove(del_layer)

        # - Update global dt
        self._dt = self._set_dt()

        # - Reevaluate the layer evolution order
        self.evol_order: List[layers.Layer] = self._set_evolution_order()

    def connect(
        self, pre_layer: layers.Layer, post_layer: layers.Layer, verbose: bool = False
    ):
        """
        Connect two layers by defining one as the input layer of the other

        :param Layer pre_layer:         The source layer
        :param Layer post_layer:        The target layer
        :param Optional[bool] verbose:  If `True`, display feedback about the connection process. Default: `False`, do not display feedback

        :raises NetworkError: if layers do not have compatible output / input sizes, or incompatible time series classes
        """
        # - Make sure that layer dimensions match

        if pre_layer.size_out != post_layer.size_in:
            raise NetworkError(
                "Network: Dimensions of layers `{}` (size_out={}) and `{}`".format(
                    pre_layer.name, pre_layer.size_out, post_layer.name
                )
                + " (size_in={}) do not match".format(post_layer.size_in)
            )

        # - Check for compatible input / output
        if pre_layer.output_type != post_layer.input_type:
            raise NetworkError(
                "Network: Input / output classes of layer `{}` (output_type = {})".format(
                    pre_layer.name, pre_layer.output_type.__name__
                )
                + " and `{}` (input_type = {}) do not match".format(
                    post_layer.name, post_layer.input_type.__name__
                )
            )

        # - Add source layer to target's set of inputs
        post_layer.pre_layer = pre_layer

        # - Make sure that the network remains a directed acyclic graph
        #   and reevaluate evolution order
        try:
            self.evol_order = self._set_evolution_order()
            if verbose:
                print(
                    "Network: Layer `{}` now receives input from layer `{}` \n".format(
                        post_layer.name, pre_layer.name
                    )
                )
        except NetworkError as e:
            post_layer.pre_layer = None
            raise e

    def disconnect(
        self, pre_layer: layers.Layer, post_layer: layers.Layer, verbose: bool = False
    ):
        """
        Remove the connection between two layers by setting the input of the target layer to `None`

        :param Layer pre_layer:     The source layer
        :param Layer post_layer:    The target layer
        :param Optional[bool] verbose:  If `True`, display feedback about the connection process. Default: `False`, do not display feedback
        """

        # - Check whether layers are connected at all
        if post_layer.pre_layer is pre_layer:
            # - Remove the connection
            post_layer.pre_layer = None

            if verbose:
                print(
                    "Network: Layer {} no longer receives input from layer `{}`".format(
                        post_layer.name, pre_layer.name
                    )
                )

            # - Reevaluate evolution order
            try:
                self.evol_order = self._set_evolution_order()
            except NetworkError as e:
                raise e

        else:
            if verbose:
                print(
                    "Network: There is no connection from layer `{}` to layer `{}`".format(
                        pre_layer.name, post_layer.name
                    )
                )

    def _set_evolution_order(self) -> list:
        """
        Determine the order in which layers are evolved. Requires Network to be a directed acyclic graph, otherwise evolution has to happen timestep-wise instead of layer-wise
        """

        # - Function to find next evolution layer
        def find_next_layer(candidates: set) -> layers.Layer:
            while True:
                try:
                    candidate_lyr = candidates.pop()

                # If no candidate is left, raise an exception
                except KeyError:
                    raise NetworkError(
                        "Network: Cannot resolve evolution order of layers"
                    )
                    # Could implement timestep-wise evolution...
                else:
                    # - If there is a candidate and none of the remaining layers is its input layer, this will be the next
                    if not (candidate_lyr.pre_layer in remaining_lyrs):
                        return candidate_lyr

        # - Set of layers that are not in evolution order yet
        remaining_lyrs: set = self.layerset.copy()

        # - Loop through layers
        order = []
        while bool(remaining_lyrs):
            # - Find the next layer to be evolved
            next_lyr = find_next_layer(remaining_lyrs.copy())
            order.append(next_lyr)
            remaining_lyrs.remove(next_lyr)

        # - Return a list with the layers in their evolution order
        return order

    def _set_dt(self, max_factor: float = 100):
        """
        Set a time step size for the network which is the lcm of all layers' dt's.

        :param float max_factor:    Factor by which the network `.dt` may exceed the largest layer `.Layer.dt` before an error is raised

        :raises ValueError: If a sensible `.dt` cannot be found
        """
        if self._force_dt:
            # - Just make sure layer dt are multiples of self.dt
            for lyr in self.layerset:
                if not is_multiple(self.dt, lyr.dt):
                    raise ValueError(
                        f"Network: dt is set to {self.dt}, which is not a multiple of "
                        + f"layer `{lyr.name}`'s time step ({lyr.dt})."
                    )
        else:
            ## -- Try to determine self._dt from layer time steps
            # - Collect layer time steps, convert to Decimals for numerical stability
            dt_list = [Decimal(str(lyr.dt)) for lyr in self.layerset]

            # - If list is empty, there are no layers in the network
            if not dt_list:
                return None

            # - Determine lcm
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

            # - Store global time step, for now as float for compatibility
            self._dt = float(t_lcm)

        # - Store number of layer time steps per global time step for each layer
        for lyr in self.layerset:
            lyr._timesteps_per_network_dt = int(np.round(self._dt / lyr.dt))

    def _fix_duration(self, t: float) -> float:
        """
        Correct an evolution duration so that it is a multiple of `.dt`

        Due to rounding errors it can happen that a duration or end time `t` is slightly below its intened value, causing the layers to not evolve sufficiently. This method fixes the problem by increasing `t` if it is slightly below a multiple of `.dt` of any of the layers in the network.

        :param float t: Time to be fixed

        :return float:  Corrected duration
        """

        # - All dt
        v_dt = np.array([lyr.dt for lyr in self.evol_order])

        if ((np.abs(t % v_dt) > tol_abs) & (np.abs(t % v_dt) - v_dt < tol_abs)).any():
            return t + tol_abs
        else:
            return t

    def evolve(
        self,
        ts_input: Optional[TimeSeries] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Evolve the network by evolving each layer in turn

        Evolve each layer in the network according to self.evol_order. For layers with external_input==True their input is ts_input. If not but an input layer is defined, it will be the output of that, otherwise None. Return a dict with each layer's output.

        .. seealso:: :ref:`/basics/getting_started.ipynb` and the tutorial :ref:`/tutorials/building_reservoir.ipynb` show examples of using the `.evolve` method.

        :param Optional[TimeSeries] ts_input:   External input to the network. Default: `None`, no external input
        :param Optional[float] duration:        Duration over which network should be evolved. If not provided, then `num_timesteps` or the duration of `ts_input` will determine the evolution duration
        :param Optional[int] num_timesteps:     Number of evolution time steps, in units of `.dt`. If not provided, then `duration` of the duration of `ts_input` will determine evolution duration
        :param Optional[bool] verbose:         If `True`, display info about evolution state. Default: `True`, display feedback

        :return dict:                           Dictionary containing the output time series of each layer. Entries in the dictionary will be have keys taken from the names of each layer

        :raises AssertionError: If no duration can be determined
        """

        if num_timesteps is None:
            # - Determine num_timesteps
            if duration is None:
                # - Determine duration
                assert (
                    ts_input is not None
                ), "Network: One of `num_timesteps`, `ts_input` or `duration` must be supplied"

                if ts_input.periodic:
                    # - Use duration of periodic TimeSeries, if possible
                    duration: float = ts_input.duration

                else:
                    # - Evolve until the end of the input TimeSeries
                    duration: float = ts_input.t_stop - self.t
                    assert duration > 0, (
                        "Network: Cannot determine an appropriate evolution duration. "
                        + "`ts_input` finishes before the current evolution time."
                    )
            num_timesteps = int(np.floor(duration / self.dt))

        if ts_input is not None:
            # - Set external input name if not set already
            if ts_input.name is None:
                ts_input.name = "External input"
            # - Check if input contains information about trial timings
            try:
                trial_start_times: np.ndarray = ts_input.trial_start_times
            except AttributeError:
                try:
                    # Old variable name
                    trial_start_times: np.ndarray = ts_input.trial_start_times
                except AttributeError:
                    trial_start_times = None
        else:
            trial_start_times = None

        # - Dict to store external input and each layer's output time series
        signal_dict = {"external": ts_input}

        # - Make sure layers are in sync with netowrk
        self._check_sync(verbose=False)

        # - Iterate over evolution order and evolve layers
        for lyr in self.evol_order:

            # - Determine input for current layer
            if lyr.external_input:
                # - External input
                ts_current_input = ts_input
                str_in = "external input"

            elif lyr.pre_layer is not None:
                # - Output of current layer's input layer
                ts_current_input = signal_dict[lyr.pre_layer.name]
                str_in = lyr.pre_layer.name + "'s output"

            else:
                # - No input
                ts_current_input = None
                str_in = "nothing"

            if verbose:
                print(
                    "Network: Evolving layer `{}` with {} as input".format(
                        lyr.name, str_in
                    )
                )
            # - Evolve layer and store output in signal_dict
            signal_dict[lyr.name] = lyr.evolve(
                ts_input=ts_current_input,
                num_timesteps=int(num_timesteps * lyr._timesteps_per_network_dt),
                verbose=verbose,
            )

            # - Add information about trial timings if present
            if trial_start_times is not None:
                signal_dict[lyr.name].trial_start_times = trial_start_times.copy()

            # - Set name for response time series, if not already set
            if signal_dict[lyr.name].name is None:
                signal_dict[lyr.name].name = lyr.name

        # - Update network time
        self._timestep += num_timesteps

        # - Make sure layers are still in sync with netowrk
        self._check_sync(verbose=False)

        # - Return dict with layer outputs
        return signal_dict

    def train(
        self,
        training_fct: Callable[["Network", Dict[str, TimeSeries], bool, bool], Any],
        ts_input: Optional[TimeSeries] = None,
        duration: Optional[float] = None,
        batch_durs: Union[np.ndarray, float, None] = None,
        num_timesteps: int = None,
        nums_ts_batch: Union[np.ndarray, int, None] = None,
        verbose: bool = True,
        high_verbosity: bool = False,
    ):
        """
        Train the network batch-wise by evolving the layers and calling the training function

        .. seealso:: The tutorial :ref:`/tutorials/building_reservoir.ipynb` illustrates how to call `.train` and how to build a training function.

        :param Callable training_fct:           Function that is called after each evolution, taking the following arguments:
            - `net` (`Network`):  Network the network object to be trained.
            - `signals` (`Dict`): Dictionary containing all signals in the current evolution batch.
            - `is_first` (`bool`):   Is this the first batch?
            - `is_last` (`bool`):    Is this the final batch?

        :param Optional[TimeSeries] ts_input:           Time series containing external input to network
        :param Optional[float] duration:                Duration over which network should be evolved. If None, evolution is over the duration of ts_input
        :param Optional[ArrayLike[float]] batch_durs:   Array-like or float - Duration of one batch (can also pass array with several values)
        :param Optional[int] num_timesteps:              Total number of training time steps
        :param Optional[ArrayLike[int]] nums_ts_batch:  Array-like or int - Number of time steps per batch (or array of several values)
        :param Optional[bool] verbose:                  If `True`, print info about training progress. Default: `True`, display progress
        :param Optional[bool] high_verbosity:           If `True`, print info about layer evolution (only has effect if `verbose` is `True`) Default: `False`, dont' display extra feedback
        """

        if num_timesteps is None:
            # - Try to determine num_timesteps from duration
            if duration is None:
                # - Determine duration
                assert (
                    ts_input is not None
                ), "Network: One of `num_timesteps`, `ts_input` or `duration` must be supplied"

                if ts_input.periodic:
                    # - Use duration of periodic TimeSeries, if possible
                    duration = ts_input.duration

                else:
                    # - Evolve until the end of the input TimeSeries
                    duration = ts_input.t_stop - self.t
                    assert duration > 0, (
                        "Network: Cannot determine an appropriate evolution duration. "
                        + "`ts_input` finishes before the current evolution time."
                    )
            num_timesteps = int(np.floor(duration / self.dt))

        # - Number of time steps per batch
        if nums_ts_batch is None:
            if batch_durs is None:
                v_ts_batch = np.array([num_timesteps], dtype=int)
            elif np.size(batch_durs) == 1:
                # - Same value for all batches
                num_ts_single_batch = int(
                    np.floor(np.asscalar(np.asarray(batch_durs)) / self.dt)
                )
                num_batches = int(np.ceil(num_timesteps / num_ts_single_batch))
                v_ts_batch = np.repeat(num_ts_single_batch, num_batches)
                v_ts_batch[-1] = num_timesteps - np.sum(v_ts_batch[:-1])
            else:
                # - Individual batch durations
                # - Convert batch durations to time step numbers - Rounding down should
                #   not be too problematic as total training will always be num_timesteps
                v_ts_batch = np.floor(np.array(batch_durs) / self.dt).astype(int)
        else:
            if np.size(nums_ts_batch) == 1:
                # - Same value for all batches
                num_ts_single_batch = np.asscalar(np.asarray(nums_ts_batch, dtype=int))
                num_batches = int(np.ceil(num_timesteps / num_ts_single_batch))
                v_ts_batch = np.repeat(num_ts_single_batch, num_batches)
                v_ts_batch[-1] = num_timesteps - np.sum(v_ts_batch[:-1])
            else:
                # - Individual batch durations
                v_ts_batch = np.asarray(nums_ts_batch)

        # - Make sure time steps add up to num_timesteps
        diff_ts: int = num_timesteps - np.sum(v_ts_batch)
        if diff_ts > 0:
            v_ts_batch = np.r_[v_ts_batch, diff_ts]
        elif diff_ts < 0:
            # - Index of first element where cumulated number of time steps > num_timesteps
            first_idx_beyond: int = np.where(np.cumsum(v_ts_batch) > num_timesteps)[0][
                0
            ]
            v_ts_batch = v_ts_batch[: first_idx_beyond + 1]
            # - Correct last value

        ## -- Actual training starts here:

        # - Iterate over batches
        num_batches: int = np.size(v_ts_batch)

        def next_batch(batch_num, current_ts, num_batches):
            if high_verbosity or (verbose and not use_tqdm):
                print(
                    "Network: Training batch {} of {} from t = {:.3f} to {:.3f}.".format(
                        batch_num + 1,
                        num_batches,
                        self.t,
                        self.t + current_ts * self.dt,
                        end="",
                    ),
                    end="\r",
                )

            # - Evolve network
            signal_dict = self.evolve(
                ts_input=ts_input.clip(
                    self.t, self.t + current_ts * self.dt, include_stop=True
                ),
                num_timesteps=current_ts,
                verbose=high_verbosity,
            )

            # - Call the callback
            training_fct(
                self, signal_dict, batch_num == 0, batch_num == num_batches - 1
            )

        if verbose and use_tqdm:
            with tqdm(total=num_batches, desc="Network training") as pbar:
                for batch_num, current_ts in enumerate(v_ts_batch):
                    next_batch(batch_num, current_ts, num_batches)
                    pbar.update(1)
        else:
            for batch_num, current_ts in enumerate(v_ts_batch):
                next_batch(batch_num, current_ts, num_batches)

        if verbose:
            print(
                "Network: Training successful                                        \n"
            )

    def stream(
        self,
        ts_input: TimeSeries,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: Optional[bool] = False,
        step_callback: Optional[Callable] = None,
    ) -> dict:
        """
        Stream data through layers, evolving by single time steps

        :param TimeSeries ts_input:                 External input to the network
        :param Optional[float] duration:            Total duration to stream for. If not provided, use `num_timesteps` or the duration of `ts_input` to determine duration
        :param Optional[int] num_timesteps:         Number of time steps to stream for, in units of `.dt`. If not provided, using `duration` of the duration of `ts_input` to determine duration
        :param Optional[bool] verbose:              If `True`, display feedback during streaming. Default: `False`, do not display feedback
        :param Optional[Callable] step_callback:    Callback function that will be called on each time step. Has the signature Callable[[Network]]

        :return dict:       Collected output signals from each layer
        """

        # - Check that all layers implement the streaming interface
        assert all(
            [hasattr(lyr, "stream") for lyr in self.layerset]
        ), "Network: Not all layers implement the `stream` interface."

        # - Check that external input has the correct class
        assert isinstance(
            ts_input, self.input_layer.input_type
        ), "Network: External input must be of class {} for this network.".format(
            self.input_layer.input_type.__name__
        )

        # - Check that external input has the correct size
        assert (
            ts_input.num_channels == self.input_layer.size_in
        ), "Network: External input must have {} traces for this network.".format(
            self.input_layer.size_in
        )

        if num_timesteps is None:
            # - Try to determine time step number from duration
            assert (
                duration is not None
            ), "Network: Either `num_timesteps` or `duration` must be provided."
            num_timesteps = int(np.floor(duration / self.dt))

        # - Prepare time base
        timebase = np.arange(num_timesteps + 1) * self._dt + self.t
        duration = timebase[-1] - timebase[0]

        # - Prepare all layers
        self.l_streamers = [
            lyr.stream(duration, self.dt, verbose=verbose) for lyr in self.evol_order
        ]
        num_layers = np.size(self.evol_order)

        # - Prepare external input
        if ts_input is not None:
            l_input = [ts_input(t, t + self.dt) for t in timebase]
        else:
            l_input = [None] * num_timesteps

        # - Get initial state of all layers
        if verbose:
            print("Network: getting initial state")

        # - Determine input state size, obtain initial layer state
        input_state: Tuple = l_input[0]
        l_laststate = [input_state] + [
            deepcopy(lyr.send(None)) for lyr in self.l_streamers
        ]

        # - Initialise layer output variables with initial state, convert to lists
        l_layer_outputs = [
            tuple([np.reshape(x, (1, -1))] for x in state) for state in l_laststate[1:]
        ]

        # - Display some feedback
        if verbose:
            print("Network: got initial state:")
            print(l_layer_outputs)

        # - Streaming loop
        l_state = deepcopy(l_laststate)
        for step in range(num_timesteps):
            if verbose:
                print("Network: Start of step", step)

            # - Set up external input
            l_laststate[0] = l_input[step]

            # - Loop over layers, stream data in and out
            for layer_idx in range(num_layers):
                # - Display some feedback
                if verbose:
                    print("Network: Evolving layer {}".format(layer_idx))

                # - Try / Catch to handle end of streaming iteration
                try:
                    # - `send` input data for current layer
                    # - wait for the output state for the current layer
                    l_state[layer_idx + 1] = deepcopy(
                        self.l_streamers[layer_idx].send(l_laststate[layer_idx])
                    )

                except StopIteration as e:
                    # - StopIteration returns the final state
                    l_state[layer_idx + 1] = e.args[0]

            # - Collate layer outputs
            for layer_idx in range(num_layers):
                for tuple_idx in range(len(l_layer_outputs[layer_idx])):
                    l_layer_outputs[layer_idx][tuple_idx].append(
                        np.reshape(l_state[layer_idx + 1][tuple_idx], (1, -1))
                    )

            # - Save last state to use as input for next step
            l_laststate = deepcopy(l_state)

            # - Call callback function
            if step_callback is not None:
                step_callback(self)

        # - Build return dictionary
        signal_dict = {"external": ts_input.copy()}
        for layer_idx in range(num_layers):
            # - Concatenate time series
            lv_data = [
                np.stack(np.array(data, "float")) for data in l_layer_outputs[layer_idx]
            ]

            # - Filter out nans in time trace (always first data element)
            vb_use_samples: np.ndarray = ~np.isnan(lv_data[0]).flatten()
            tup_data = tuple(data[vb_use_samples, :] for data in lv_data)

            if verbose:
                print(tup_data[0])

            # - Build output dictionary (using appropriate output class)
            signal_dict[self.evol_order[layer_idx].name] = self.evol_order[
                layer_idx
            ].output_type(*tup_data)

            # - Set name for time series, if not already set
            if signal_dict[self.evol_order[layer_idx].name].name is None:
                signal_dict[self.evol_order[layer_idx].name].name = self.evol_order[
                    layer_idx
                ].name

        # - Increment time
        self._timestep += num_timesteps

        # - Return collated signals
        return signal_dict

    def _check_sync(self, verbose: bool = True) -> bool:
        """
        Check whether the time `t` of all layers matches `self.t`. If not, raise an exception

        :param Optional[bool] verbose:  If `True`, display feedback. Default: `True`, display feedback.

        :raises NetworkError: If layers are not in synch with global network time
        """
        in_sync = True
        if verbose:
            print("Network: Network time is {}. \n\t Layer times:".format(self.t))
            print(
                "\n".join(
                    ("\t\t {}: {}".format(lyr.name, lyr.t) for lyr in self.evol_order)
                )
            )
        for lyr in self.evol_order:
            if lyr._timestep != self._timestep * lyr._timesteps_per_network_dt:
                in_sync = False
                print(
                    "\t Network: WARNING: Layer `{}` is not in sync (t={})".format(
                        lyr.name, lyr.t
                    )
                )
        if in_sync:
            if verbose:
                print("\t Network: All layers are in sync with network.")
        else:
            raise NetworkError("Network: Not all layers are in sync with the network.")
        return in_sync

    def reset_time(self):
        """
        Reset the time of the network to zero by resetting each layer and the global network timestamp. Does not reset state.
        """
        # - Reset time for each layer
        for lyr in self.layerset:
            lyr.reset_time()

        # - Reset global network time
        self._timestep = 0

    def reset_state(self):
        """
        Reset the state of the network by resetting each layer. Does not reset time.
        """
        # - Reset state for each layer
        for lyr in self.layerset:
            lyr.reset_state()

    def reset_all(self):
        """
        Reset all state and time of the network and layers
        """
        for lyr in self.layerset:
            lyr.reset_all()

        # - Reset global network time
        self._timestep = 0

    def __repr__(self):
        return (
            "{} object with {} layers\n".format(
                self.__class__.__name__, len(self.layerset)
            )
            + "    "
            + "\n    ".join([str(lyr) for lyr in self.evol_order])
        )

    @property
    def t(self):
        """(float) Global network time"""
        return (
            0
            if not hasattr(self, "_dt") or self._dt is None
            else self._dt * self._timestep
        )

    @property
    def dt(self):
        """(float) Time step to use in layer simulations"""
        return self._dt

    def save(self, filename: str):
        """
        Save this network to a JSON file

        :param str filename:    The path to a file in which to save the network and state.
        """
        # - List with layers in their evolution order
        list_layers = []
        for lyr in self.evol_order:
            list_layers.append(lyr.to_dict())
        savedict = {"layers": list_layers}
        # - Include dt if it has been enforced at instantiation
        if self._force_dt:
            savedict["dt"] = self.dt
        with open(filename, "w") as f:
            json.dump(savedict, f)

    @staticmethod
    def load(filename: str) -> "Network":
        """
        Load a network from a JSON file

        :param str filename:    filename of a JSON filr that contains a saved network
        :return Network:        A network object with all the layers loaded from `filename`
        """

        with open(filename, "r") as f:
            loaddict: dict = json.load(f)
        # - Instantiate layers
        list_layers = loaddict["layers"]
        evol_order = []
        for lyr in list_layers:
            cls_layer = getattr(layers, lyr["class_name"])
            evol_order.append(cls_layer.load_from_dict(lyr))
        # - If dt has been stored, include as parameter for new Network object
        dt = loaddict.get("dt", None)
        return Network(*evol_order, dt=dt)

    @staticmethod
    def add_layer_class(cls_lyr: Type[layers.Layer], name: str):
        """
        Add external layer class to the namespace

        This method adds a externally-defined `.Layer` subclass to the `.layers` namespace, so that layers that are defined outside the `.rockpool.layers` module can still be loaded

        :param Layer cls:   The class that is to be added
        :param str name:    Name of the class as a string
        """
        setattr(layers, name, cls_lyr)


### --- NetworkError exception class
class NetworkError(Exception):
    """
    Define an exception class to encapsulate network errors
    """

    pass
