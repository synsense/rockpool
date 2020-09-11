#
# Implements a base class for Jax layers, that supports training
#

# - Import base classes
from rockpool.timeseries import TimeSeries

from importlib import util

if (util.find_spec("jax") is None) or (util.find_spec("jaxlib") is None):
    raise ModuleNotFoundError(
        "'jax' and 'jaxlib' backends not found. Layers that rely on Jax will not be available."
    )

from jax import jit, grad, vmap, value_and_grad
from jax.experimental.optimizers import adam

import itertools

import collections.abc

from copy import deepcopy

from abc import abstractmethod, ABC


# - Import jax elements
from jax import numpy as np
from jax.tree_util import tree_flatten, tree_unflatten

# - Import and define types
from typing import (
    Dict,
    Tuple,
    Any,
    Callable,
    Union,
    List,
    Optional,
    Collection,
    Iterable,
)

State = Any
Params = Union[Dict, Tuple, List]

__all__ = ["JaxTrainer"]


# - Define a useful default loss function
@jit
def loss_mse_reg(
    params: Params,
    states_t: Dict[str, np.ndarray],
    output_batch_t: np.ndarray,
    target_batch_t: np.ndarray,
    min_tau: float,
    lambda_mse: float = 1.0,
    reg_tau: float = 10000.0,
    reg_l2_rec: float = 1.0,
) -> float:
    """
    Loss function for target versus output

    :param Params params:               Set of packed parameters
    :param State state:                 Set of packed state values
    :param np.ndarray output_batch_t:   Output rasterised time series [TxO]
    :param np.ndarray target_batch_t:   Target rasterised time series [TxO]
    :param float min_tau:               Minimum time constant
    :param float lambda_mse:            Factor when combining loss, on mean-squared error term. Default: 1.0
    :param float reg_tau:               Factor when combining loss, on minimum time constant limit. Default: 1e5
    :param float reg_l2_rec:            Factor when combining loss, on L2-norm term of recurrent weights. Default: 1.

    :return float: Current loss value
    """
    # - Measure output-target loss
    mse = lambda_mse * np.nanmean((output_batch_t - target_batch_t) ** 2)

    # - Get loss for tau parameter constraints
    tau_loss = reg_tau * np.nanmean(
        np.where(params["tau"] < min_tau, np.exp(-(params["tau"] - min_tau)), 0)
    )

    # - Measure recurrent L2 norm
    w_res_norm = reg_l2_rec * np.nanmean(params["w_recurrent"] ** 2)

    # - Loss: target/output squared error, time constant constraint, recurrent weights norm, activation penalty
    fLoss = mse + w_res_norm + tau_loss

    # - Return loss
    return fLoss


def flatten(
    generic_collection: Union[Iterable, Collection], sep: str = "_"
) -> Collection:
    """
    Flattens a generic collection of collections into an ordered dictionary.

    ``generic_collection`` is a nested tree of inhomogeneous collections, such as `list`, `set`, `dict`, etc. This function iterates through this generic collection, and flattens all the leaf nodes into a single collection. The keys in the returned collection will be named after the orginal keys in ``generic_collection``, if any, and after the nesting level.

    :param Union[Iterable, Collection] generic_collection:  A nested tree of iterable types or collections, that will be flattened
    :param str sep:                                         The separator character to use when building keys in the flattened collection. Default: "_"

    :return Collection flattened_collection: A collection of all the items in ``generic_collection``, flattened into a single coellction.
    """
    import collections

    obj = collections.OrderedDict()

    def recurse(this, parent_key=""):
        if isinstance(this, dict):
            for k, v in this.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        elif np.size(this) == 1:
            obj[parent_key + sep + "0"] = this
        elif isinstance(this, collections.abc.Iterable):
            for ind, item in enumerate(this):
                recurse(item, parent_key + sep + str(ind) if parent_key else str(ind))
        else:
            obj[parent_key] = this

    recurse(generic_collection)
    return obj


class JaxTrainer(ABC):
    """
    Mixin class for a trainable layer, with evolution functions based on Jax

    Derive from this mixin class to implement a new Jax-backed trainable layer.

    .. rubric:: How to train a layer based on this class

    This class defines a training method `~.JaxTrainer.train_output_target`, which performs one round of optimization based on a single trial, given an input and a target signal::

        lyr.train_output_target(input_ts, target_ts)

    .. rubric:: How to use this base class

    This class defines three abstract methods that you need to define for your layer::

        def _pack(self) -> Params:
            return {
                'param_a': self._param_a,
                'param_b': self._param_b,
                ...
            }

    The :py:meth:`~.JaxTrainer.JaxTrainer._pack()` method must return a dictionary or tuple or other collection of strictly jax-compatible types, that completely define the modifiable parameters for this layer. For example: weights; biases; time constants. Included should be all parameters that one might want to perform gradient descent on. *Excluded* should be parameters that are fixed: for example ``dt``, and ``noise_std``.

    ::

        def _unpack(self, params: Params) -> None:
            (
                self._param_a,
                self._param_b,
                ...
            ) = (
                params['param_a'],
                params['param_b'],
                ...
            )

    The :py:meth:`~.JaxTrainer.JaxTrainer._unpack()` method must accept a parameters definition as returned by :py:meth:`~.JaxTrainer.JaxTrainer._pack()`, and apply those parameters to the layer.

    ::

        @property
        def _evolve_functional(self) -> Callable[]:
            return evol_func

    The property :py:attr:`~.JaxTrainer._evolve_functional` must return a *function* ``evol_func()`` with the following calling signature. This function must evolve the state of the layer, given an initial state, set of parameters and raw inputs, with *no side effects*. That means the function must not update the internal state of the layer, or update the `._t` attribute, etc. The function ``evol_func()`` must be compilable with `jax.jit`. An example property and function are shown here::

        @property
        def _evolve_functional(self) -> Callable[[Params, State, np.ndarray], Tuple[np.ndarray, State, Dict[str, np.ndarray]]:
            def evol_func(params: Params, state: State, input: np.ndarray) -> Tuple[np.ndarray, State, Dict[str, np.ndarray]]:
            '''
            :param Params params:       `params` contains the set of parameters needed to define this layer
            :param State state:         `state` contains the initial state of this layer
            :param np.ndarray input:    `input` is [TxI], T time steps by I input channels

            :return Tuple[np.ndarray, State, Dict[str, np.ndarray]]: (output, new_state, states_t)
                output:     A raw time series [TxO], T time steps by O output channels
                new_state:  The new state of the layer, after the evolution
                states_t:   A dictionary of internal state time series during this evolution
            '''
                # - Perform evolution inner loop
                output, new_state, states_t = f(input, state)

                # - Return output and state
                return output, new_state, states_t

            return evol_func
    """

    def __init__(self, *args, **kwargs):
        # - Ensure initialisation passes up the chain
        super().__init__(*args, **kwargs)

    @abstractmethod
    def _pack(self) -> Params:
        """
        Method returning a list or tuple or dict of Jax / numpy base classes, containing the tunable parameters of this layer

        You must override this abstract method when implementing your own concrete Jax layer class

        :return Params: params: list, tuple or dict of parameters
        """
        pass

    @abstractmethod
    def _unpack(self, params: Params) -> None:
        """
        Method that sets the internal parameters of this layer, given a set of parameters returned from :py:meth:`._pack`

        :param Params params:   list, tuple or dict of parameters
        """
        pass

    @property
    @abstractmethod
    def _evolve_functional(
        self,
    ) -> Callable[
        [Params, State, np.ndarray], Tuple[np.ndarray, State, Dict[str, np.ndarray]]
    ]:
        """
        Functional form of evolution for this layer

        This abstract property must return a function ``evol_func``, which evolves the dynamics of this layer given parameters, input and an initial state. The function must have the calling signature::

            def evol_func(params: Params, state: State, input: np.ndarray) -> Tuple[np.ndarray, State]:
                ...

        ``evol_func`` returns ``(outputs, new_state)``, and *must* be side-effect free. The goal is that ``evol_func`` can be compiled using `jax.jit`.

        :return Callable[[Params, State, np.ndarray)], Tuple[np.ndarray, State]]: evol_func: Evolution function
        """

        def evol_func(
            params: Params, state: State, input: np.ndarray
        ) -> Tuple[np.ndarray, State, Dict[str, np.ndarray]]:
            raise NotImplementedError(params, state, input)

        return evol_func

    # - Define a default loss function
    @property
    def _default_loss(self) -> Callable[[Any], float]:
        return loss_mse_reg

    @property
    def _default_loss_params(self) -> Dict:
        return {
            "lambda_mse": 1.0,
            "reg_tau": 10000.0,
            "reg_l2_rec": 1.0,
            "min_tau": self._dt * 11.0,
        }

    def train_output_target(
        self,
        ts_input: Union[TimeSeries, np.ndarray],
        ts_target: Union[TimeSeries, np.ndarray],
        is_first: bool = False,
        is_last: bool = False,
        debug_nans: bool = False,
        loss_fcn: Callable[[Dict, Tuple], float] = None,
        loss_params: Dict = {},
        optimizer: Callable = adam,
        opt_params: Dict = {"step_size": 1e-4},
        batch_axis=None,
    ) -> Tuple[float, Dict[str, Any], Callable[[], Tuple[np.ndarray, State]]]:
        """
        Perform one trial of Adam stochastic gradient descent to train the layer

        :param Union[TimeSeries,np.ndarray] ts_input:     `.TimeSeries` (or raw sampled signal) to use as input for this trial ``[TxI]`` (or ``[BxTxI]`` with batching)
        :param Union[TimeSeries,np.ndarray] ts_target:    `.TimeSeries` (or raw sampled signal) to use as target for this trial ``[TxO]`` (or ``[BxTxO]`` with batching)
        :param Optional[float] min_tau: Minimum time constant to permit
        :param bool is_first:           Flag to indicate this is the first trial. Resets learning and causes initialisation.
        :param bool is_last:            Flag to indicate this is the last trial. Performs clean-up (not essential)
        :param bool debug_nans:         If ``True``, ``nan`` s will raise an ``AssertionError``, and display some feedback about where and when the ``nan`` s occur. Default: ``False``, do not check for ``nan`` s. Note: Checking for ``nan`` s slows down training considerably.
        :param Callable loss_fcn:       Function that computes the loss for the currently configured layer. Default: :py:func:`loss_mse_reg`
        :param Dict loss_params:        A dictionary of loss function parameters to pass to the loss function. Must be configured on the very first call to `.train_output_target`; subsequent changes will be ignored. Default: Appropriate parameters for :py:func:`loss_mse_reg`.
        :param Callable optimizer:      A JAX-style optimizer function. See the JAX docs for details. Default: :py:func:`jax.experimental.optimizers.adam`
        :param Dict opt_params:         A dictionary of parameters passed to :py:func:`optimizer`. Default: ``{"step_size": 1e-4}``
        :param Optional[int] batch_axis: Axis over which to extract batch samples and map through the gradient and loss measurements. To use batches, you must pre-rasterise ``ts_input`` and ``ts_target``. If ``None`` (default), no batching is performed. If not ``None``, `batch_axis` defines the axis of ``ts_input`` and ``ts_target`` to pass to :py:func:`jax.vmap` as the batch axis.

        Use this function to train the output of the reservoir to match a target, given an input stimulus. This function can be called in a loop, passing in randomly-chosen training examples on each call. Parameters of the layer are updated on each call of `~.JaxTrainer.train_output_target`, but the layer time and state are *not* updated.

        .. rubric:: Batching

        A batch of samples can be provided for ``ts_input`` and ``ts_target``, where losses and gradients are averaged over the several samples in a batch. In this case, ``ts_input`` and ``ts_target`` must be provided as `ndarray` s, such that they are rasterised with the layer :py:attr:`~.Layer.dt`. You can use :py:meth:`~.Layer._prepare_input` to obtain a time base for the rasterisation / sampling.

        To perform batching, you must provide the ``batch_axis`` argument to :py:meth:`.train_output_target`. This specifies which axis of ``ts_input`` and ``ts_target`` is the batch axis::

            ts_input = np.random.rand(samples_per_batch, num_time_steps, num_inputs)
            ts_target = np.random.rand(samples_per_batch, num_time_steps, num_outputs)
            lyr.train_output_target(ts_input, ts_target, batch_axis = 0)

        .. rubric:: Writing your own loss function

        The argument ``loss_fcn`` can be used to pass in your own loss function for use in optimisation. The default loss function computes a mean-squared error between output and target signals, and provides several forms of regularisation::

            def loss_mse_reg(
                params: Params,
                states_t: States,
                output_batch_t: np.ndarray,
                target_batch_t: np.ndarray,
                min_tau: float,
                lambda_mse: float = 1.0,
                reg_tau: float = 10000.0,
                reg_l2_rec: float = 1.0,
            ) -> float:
                '''
                Loss function for target versus output

                :param Params params:               Set of packed parameters
                :param States states_t:             Dict of layer internal state tme series
                :param np.ndarray output_batch_t:   Output rasterised time series [TxO]
                :param np.ndarray target_batch_t:   Target rasterised time series [TxO]
                :param float min_tau:               Minimum time constant
                :param float lambda_mse:            Factor when combining loss, on mean-squared error term. Default: 1.0
                :param float reg_tau:               Factor when combining loss, on minimum time constant limit. Default: 1e5
                :param float reg_l2_rec:            Factor when combining loss, on L2-norm term of recurrent weights. Default: 1.

                :return float: Current loss value
                '''
                # - Measure output-target loss
                mse = lambda_mse * np.mean((output_batch_t - target_batch_t) ** 2)

                # - Get loss for tau parameter constraints
                tau_loss = reg_tau * np.mean(
                    np.where(params["tau"] < min_tau, np.exp(-(params["tau"] - min_tau)), 0)
                )

                # - Measure recurrent L2 norm
                w_res_norm = reg_l2_rec * np.mean(params["w_recurrent"] ** 2)

                # - Loss: target/output squared error, time constant constraint, recurrent weights norm, activation penalty
                fLoss = mse + tau_loss + w_res_norm

                # - Return loss
                return fLoss

        You can replace this with your own loss function, as long as it obeys the calling signature::

            def loss(params: Params, output_batch_t: np.ndarray, target_batch_t: np.ndarray, **loss_params) -> float:

        ``loss_params`` will be a dictionary of whatever arguments you would like to pass to your loss function. ``params`` will be a set of parameters returned by the layer :py:meth:`_pack` method.

        ``output_batch_t`` and ``target_batch_t`` will be rasterised versions of time series, for the output of the layer given the current parameters, and for the target signal, respectively. Both are computed for the current batch, and are aligned in time.

        :py:func:`.loss` must return a scalar float of the calculated loss value for the current batch. You can use the values in ``params`` to compute regularisation terms. You may not modify anything in ``params``. You *must* implement :py:func:`.loss` using `jax.numpy`, and :py:func:`.loss` *must* be compilable by `jax.jit`.

        :return (loss, grads, output_fcn):
                                ``loss``:   float The current loss for this batch/sample
                                ``grad_fcn``:   Dict[str,Any] PyTree of gradients for the this batch/sample
                                ``output_fcn``: Callable[[], Tuple[np.ndarray, State, Dict]] Function that returns the layer output for the current batch, the new internal states, and a dictionary of internal state time series for the evolution
        """

        # - Initialise training
        initialise = is_first or not hasattr(self, "_JaxTrainer__in_training_sgd_adam")

        if initialise:
            # - Get optimiser
            (opt_init, opt_update, get_params) = optimizer(**deepcopy(opt_params))
            self.__get_params = get_params

            # - Make update function
            @jit
            def update_fcn(
                i: int,
                opt_state: Any,
                input_batch_t: np.ndarray,
                target_batch_t: np.ndarray,
            ) -> Any:
                """
                Perform one round of optimizer update

                :param int i:                       Current optimization iteration
                :param Any opt_state:               Current optimizer state
                :param np.ndarray input_batch_t:    Input signal for this batch. Rasterized time series [BxTxI]
                :param np.ndarray target_batch_t:   Target signal for this batch. Rasterized time series [BxTxO]

                :return Any: new_opt_state
                """
                # - Get layer parameters
                opt_params = get_params(opt_state)

                # - Get the loss function gradients
                l, g = self.__grad_fcn(
                    opt_params, input_batch_t, target_batch_t, self._state
                )

                # - Call optimiser update function
                return opt_update(i, g, opt_state), l, g

            # - If using default loss, set up parameters
            if loss_fcn is None:
                loss_fcn = self._default_loss
                default_loss_params = self._default_loss_params
                default_loss_params.update(loss_params)
                loss_params = default_loss_params

            # - Get functional evolution function
            evol_func = self._evolve_functional

            # - Make a curried loss function, incorporating static loss parameters and evolution
            @jit
            def loss_curried(
                opt_params: Params,
                input_batch_t: np.ndarray,
                target_batch_t: np.ndarray,
                state: State,
            ) -> float:
                """
                Curried loss function; absorbs loss parameters

                :param Params opt_params:           Current values of the layer parameters, modified by optimization
                :param np.ndarray input_batch_t:    Input rasterized time series for this batch [BxTxO]
                :param np.ndarray target_batch_t:   Target rasterized time series for this batch [BxTxO]
                :param State state:                 Initial state for the layer

                :return float:                      Loss value for the parameters in `opt_params`, for the current batch
                """
                # - Call the layer evolution function
                output_batch_t, new_state, states_t = evol_func(
                    opt_params, state, input_batch_t
                )

                # - Call loss function and return loss
                return loss_fcn(
                    opt_params, states_t, output_batch_t, target_batch_t, **loss_params
                )

            # - Assign update, loss and gradient functions
            self.__update_fcn = update_fcn

            if batch_axis is not None:
                # - Use `vmap` to map over batches
                def loss_batch(*args, **kwargs):
                    """ Batch mean loss function """
                    loss_b = vmap(
                        loss_curried, in_axes=(None, batch_axis, batch_axis, None)
                    )(*args, **kwargs)
                    return np.mean(loss_b, axis=batch_axis)

                def grad_batch(*args, **kwargs):
                    """ Batch mean gradient function """
                    l, g = vmap(
                        value_and_grad(loss_curried),
                        in_axes=(None, batch_axis, batch_axis, None),
                    )(*args, **kwargs)
                    g, tree_def = tree_flatten(g)
                    g = [np.mean(g_item, axis=batch_axis) for g_item in g]
                    g = tree_unflatten(tree_def, g)

                    l = np.mean(l, axis=batch_axis)

                    return l, g

                self.__loss_fcn = jit(loss_batch)
                self.__grad_fcn = jit(grad_batch)
                self.__evolve_functional = jit(
                    vmap(self._evolve_functional, in_axes=(None, None, batch_axis))
                )
            else:
                # - No batching
                self.__loss_fcn = jit(loss_curried)
                self.__grad_fcn = jit(value_and_grad(loss_curried))
                self.__evolve_functional = jit(self._evolve_functional)

            # - Initialise optimizer
            self.__opt_state = opt_init(self._pack())
            self.__itercount = itertools.count()

            # - Assign "in training" flag
            self.__in_training_sgd_adam = True

        # - Prepare time base and inputs
        if isinstance(ts_input, TimeSeries):
            # - Check that `ts_target` is also a time series
            if not isinstance(ts_target, TimeSeries):
                raise TypeError(
                    "If `ts_input` is provided as a `TimeSeries` object, then `ts_target` must also be a `TimeSeries`."
                )

            # - Rasterize input and target time series
            time_base, inps, num_timesteps = self._prepare_input(ts_input, None, None)
            target = ts_target(time_base)
        else:
            # - Use pre-rasterized time series
            inps = ts_input
            target = ts_target

        # - Check for batch dimension, and augment if necessary
        if batch_axis is not None:
            inp_batch_shape = list(inps.shape)
            target_batch_shape = list(target.shape)

            # - Check that batch sizes are equal
            if inp_batch_shape[batch_axis] != target_batch_shape[batch_axis]:
                raise ValueError("Input and Target do not have a matching batch size.")

        # - Define functions that evaluate the loss and the gradient on this trial
        def l_fcn():
            return self.__loss_fcn(
                self.__get_params(self.__opt_state), inps, target, self._state
            )

        def g_fcn():
            return self.__grad_fcn(
                self.__get_params(self.__opt_state), inps, target, self._state
            )

        def o_fcn():
            return self.__evolve_functional(
                self.__get_params(self.__opt_state), self._state, inps
            )

        # - NaNs raise errors
        if debug_nans:
            str_error = ""

            # - Check current network state
            for k, v in flatten(self._state).items():
                if np.any(np.isnan(v)):
                    str_error += "Pre-evolve network state {} contains NaNs\n".format(k)

            # - Check network parameters
            for k, v in flatten(self._pack()).items():
                if np.any(np.isnan(v)):
                    str_error += "Pre-optimisation step network parameter {} contains NaNs\n".format(
                        k
                    )

            # - Check loss function
            if np.isnan(l_fcn()):
                str_error += "Loss function returned NaN\n"

            # - Check outputs
            output_ts, new_state, states_t = o_fcn()
            if np.any(np.isnan(output_ts)):
                str_error += "Network output contained NaNs\n"

            # - Check network states
            for k, v in flatten(new_state).items():
                if np.any(np.isnan(v)):
                    str_error += "Post-evolve network state {} contains NaNs\n".format(
                        k
                    )

            # - Check gradients
            gradients = g_fcn()
            debug_gradient = False
            for k, v in flatten(gradients).items():
                if np.any(np.isnan(v)):
                    str_error += "Gradient item {} contains NaNs\n".format(k)
                    debug_gradient = True

            # - Check gradients in detail
            if debug_gradient:
                # - Loop over time steps and compute gradients
                found_nan = False
                for step in range(inps.shape[0]):
                    gradients_limited = self.__grad_fcn(
                        self.__get_params(self.__opt_state),
                        inps[:step, :],
                        target[:step, :],
                        self._state,
                    )

                    for k, v in flatten(gradients_limited).items():
                        if np.any(np.isnan(v)):
                            str_error += "Gradient NaNs begin in step {}\n".format(step)
                            found_nan = True
                            break

                    if found_nan:
                        break

            # - Raise the error
            if str_error:
                raise ValueError(str_error)

        # - Perform one step of optimisation
        self.__opt_state, loss, grads = self.__update_fcn(
            next(self.__itercount), self.__opt_state, inps, target
        )

        # - Apply the parameter updates
        new_params = self.__get_params(self.__opt_state)
        self._unpack(new_params)

        # - Reset status, on "is_last" flag
        if is_last:
            del self.__in_training_sgd_adam

        return loss, grads, o_fcn
