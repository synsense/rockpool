#
# Implements a base class for Jax layers, that supports training
#

# - Import base classes
from rockpool.timeseries import TimeSeries

from jax import jit, grad
from jax.experimental.optimizers import adam

import itertools

from copy import deepcopy

from abc import abstractmethod, ABC


# - Import jax elements
from jax import numpy as np

# - Import and define types
from typing import Dict, Tuple, Any, Callable, Union, List

State = Any
Params = Union[Dict, Tuple, List]

__all__ = ["JaxTrainer"]


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
        def _evolve_functional(self) -> Callable[[Params, State, np.ndarray], Tuple[np.ndarray, State]]:
            def evol_func(params: Params, state: State, input: np.ndarray) -> Tuple[np.ndarray, State]:
            '''
            :param Params params:       `params` contains the set of parameters needed to define this layer
            :param State state:         `state` contains the initial state of this layer
            :param np.ndarray input:    `input` is [TxI], T time steps by I input channels

            :return Tuple[np.ndarray, State]: (output, new_state)
                output:     A raw time series [TxO], T time steps by O output channels
                new_state:  The new state of the layer, after the evolution
            '''
                # - Perform evolution inner loop
                output, new_state = f(input, state)

                # - Return output and state
                return output, new_state

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
    ) -> Callable[[Params, State, np.ndarray], Tuple[np.ndarray, State]]:
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
        ) -> Tuple[np.ndarray, State]:
            raise NotImplementedError(params, state, input)

        return evol_func

    def train_output_target(
        self,
        ts_input: Union[TimeSeries, np.ndarray],
        ts_target: Union[TimeSeries, np.ndarray],
        is_first: bool = False,
        is_last: bool = False,
        loss_fcn: Callable[[Dict, Tuple], float] = None,
        loss_params: Dict = {},
        optimizer: Callable = adam,
        opt_params: Dict = {"step_size": 1e-4},
    ) -> Tuple[Callable[[], float], Callable[[], float], Callable[[], np.ndarray]]:
        """
        Perform one trial of Adam stochastic gradient descent to train the layer

        :param TimeSeries ts_input:     TimeSeries (or raw sampled signal) to use as input for this trial [TxI]
        :param TimeSeries ts_target:    TimeSeries (or raw sampled signal) to use as target for this trial [TxO]
        :param Optional[float] min_tau: Minimum time constant to permit
        :param bool is_first:           Flag to indicate this is the first trial. Resets learning and causes initialisation.
        :param bool is_last:            Flag to indicate this is the last trial. Performs clean-up (not essential)
        :param Callable loss_fcn:       Function that computes the loss for the currently configured layer. Default: :py:func:`loss_mse_reg`
        :param Dict loss_params:        A dictionary of loss function parameters to pass to the loss function. Must be configured on the very first call to `.train_output_target`; subsequent changes will be ignored. Default: Appropriate parameters for :py:func:`loss_mse_reg`.
        :param Callable optimizer:      A JAX-style optimizer function. See the JAX docs for details. Default: :py:func:`jax.experimental.optimizers.adam`
        :param Dict opt_params:         A dictionary of parameters passed to :py:func:`optimizer`. Default: ``{"step_size": 1e-4}``

        Use this function to train the output of the reservoir to match a target, given an input stimulus. This function can
        be called in a loop, passing in randomly-chosen training examples on each call. Parameters of the layer are updated
        on each call of `~.JaxTrainer.train_output_target`, but the layer time and state are *not* updated.

        .. rubric:: Writing your own loss function

        The argument ``loss_fcn`` can be used to pass in your own loss function for use in optimisation. The default loss function computes a mean-squared error between output and target signals, and provides several forms of regularisation::

            def loss_mse_reg(
                params: Params,
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

        :return (loss_fcn, grad_fcn, output_fcn):
                                ``loss_fcn``:   Callable[[], float] Function that returns the current loss
                                ``grad_fcn``:   Callable[[], float] Function that returns the gradient for the current batch
                                ``output_fcn``: Callable[[], np.ndarray] Function that returns the layer output for the current batch
        """

        # - Define default loss function
        @jit
        def loss_mse_reg(
            params: Params,
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
            :param np.ndarray output_batch_t:   Output rasterised time series [TxO]
            :param np.ndarray target_batch_t:   Target rasterised time series [TxO]
            :param float min_tau:               Minimum time constant
            :param float lambda_mse:            Factor when combining loss, on mean-squared error term. Default: 1.0
            :param float reg_tau:               Factor when combining loss, on minimum time constant limit. Default: 1e5
            :param float reg_l2_rec:            Factor when combining loss, on L2-norm term of recurrent weights. Default: 1.

            :return float: Current loss value
            """
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

        # - Initialise training
        initialise = is_first or not hasattr(self, "_JaxTrainer__in_training_sgd_adam")

        if initialise:
            # print("initialise")

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
                :param np.ndarray input_batch_t:    Input signal for this batch. Rasterized time series [TxI]
                :param np.ndarray target_batch_t:   Target signal for this batch. Rasterized time series [TxO]

                :return Any: new_opt_state
                """
                # - Get layer parameters
                opt_params = get_params(opt_state)

                # - Get the loss function gradients
                g = self.__grad_fcn(
                    opt_params, input_batch_t, target_batch_t, self._state
                )

                # - Call optimiser update function
                return opt_update(i, g, opt_state)

            # - If using default loss, set up parameters
            if loss_fcn is None:
                # print("default loss function")
                loss_fcn = loss_mse_reg
                default_loss_params = {
                    "lambda_mse": 1.0,
                    "reg_tau": 10000.0,
                    "reg_l2_rec": 1.0,
                    "min_tau": self._dt * 11.0,
                }
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
                :param np.ndarray input_batch_t:    Input rasterized time series for this batch [TxO]
                :param np.ndarray target_batch_t:   Target rasterized time series for this batch [TxO]
                :param State state:                 Initial state for the layer

                :return float:                      Loss value for the parameters in `opt_params`, for the current batch
                """
                # - Call the layer evolution function
                output_batch_t, _ = evol_func(opt_params, state, input_batch_t)

                # - Call loss function and return loss
                return loss_fcn(
                    opt_params, output_batch_t[1:], target_batch_t, **loss_params
                )

            # print("using loss function: ", loss_fcn)
            # print("curried loss function: ", loss_curried)

            # - Assign update, loss and gradient functions
            self.__update_fcn = update_fcn
            self.__loss_fcn = loss_curried
            self.__grad_fcn = jit(grad(loss_curried))

            # - Initialise optimizer
            self.__opt_state = opt_init(self._pack())
            self.__itercount = itertools.count()

            # - Assign "in training" flag
            self.__in_training_sgd_adam = True

        # print("loss function: ", self.__loss_fcn)
        # print("grad function: ", self.__grad_fcn)
        # print("update function: ", self.__update_fcn)

        # - Prepare time base and inputs
        if isinstance(ts_input, TimeSeries):
            # - Check that `ts_target` is also a time series
            assert isinstance(
                ts_target, TimeSeries
            ), "If `ts_input` is provided as a `TimeSeries` object, then `ts_target` must also be a `TimeSeries`."

            # - Rasterize input and target time series
            time_base, inps, num_timesteps = self._prepare_input(ts_input, None, None)
            target = ts_target(time_base)
        else:
            # - Use pre-rasterized time series
            inps = ts_input
            target = ts_target

        # - Perform one step of optimisation
        self.__opt_state = self.__update_fcn(
            next(self.__itercount), self.__opt_state, inps, target
        )

        # - Apply the parameter updates
        self._unpack(self.__get_params(self.__opt_state))

        # - Reset status, on "is_last" flag
        if is_last:
            del self.__in_training_sgd_adam

        # - Return lambdas that evaluate the loss and the gradient
        return (
            lambda: self.__loss_fcn(
                self.__get_params(self.__opt_state), inps, target, self._state
            ),
            lambda: self.__grad_fcn(
                self.__get_params(self.__opt_state), inps, target, self._state
            ),
            lambda: self._evolve_functional(
                self.__get_params(self.__opt_state), self._state, inps
            ),
        )
