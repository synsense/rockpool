##
# train_jax_rate_sgd.py - Support for gradient-based training of a Jax recurrent rate-based reservoir
##

import itertools
from typing import Callable, Tuple, Union, Optional, Dict
import types

import jax.numpy as np
from jax.experimental.optimizers import adam
from jax import grad, jit

from rockpool.timeseries import TimeSeries, TSContinuous
from ...gpl import rate_jax as rj


def pack_params(self: rj.RecRateEulerJax_IO) -> Dict:
    """
    .pack_params() - Pack trainable parameters into a dictionary

    :return:    dict(
                     'w_in':        np.ndarray Input weights [IxN]
                     'w_recurrent': np.ndarray Recurrent weights [NxN]
                     'w_out':       np.ndarray Output weights [NxO]
                     'tau':         np.ndarray Time constants of recurrent layer [N]
                     'bias':        np.ndarray Bias values of recurrent layer [N]
                    )
    """
    # - Pack all pertinent parameters
    return {
        "w_in": self._w_in,
        "w_recurrent": self._weights,
        "w_out": self._w_out,
        "tau": self._tau,
        "bias": self._bias,
    }


def apply_params(self: rj.RecRateEulerJax_IO, params: Dict) -> None:
    """
    .apply_params() - Apply packed parameters to the layer

    :param params:  dict(
                         'w_in':        np.ndarray Input weights [IxN]
                         'w_recurrent': np.ndarray Recurrent weights [NxN]
                         'w_out':       np.ndarray Output weights [NxO]
                         'tau':         np.ndarray Time constants of recurrent layer [N]
                         'bias':        np.ndarray Bias values of recurrent layer [N]
                        )
    """
    # - Unpack and apply parameters
    (self._w_in, self._weights, self._w_out, self._tau, self._bias) = (
        params["w_in"],
        params["w_recurrent"],
        params["w_out"],
        params["tau"],
        params["bias"],
    )


def train_output_target(
    self,
    ts_input: Union[TSContinuous, np.ndarray],
    ts_target: Union[TSContinuous, np.ndarray],
    min_tau: Optional[float] = None,
    is_first: bool = False,
    is_last: bool = False,
    loss_fcn: Callable[[Dict, Tuple], float] = None,
    loss_params: Dict = {},
    optimizer: Callable = adam,
    opt_params: Dict = {"step_size": 1e-4},
) -> Tuple[Callable[[], float], Callable[[], float]]:
    """
    Perform one trial of Adam stochastic gradient descent to train the reservoir

    :param TimeSeries ts_input:     TimeSeries (or raw sampled signal) to use as input for this trial [TxI]
    :param TimeSeries ts_target:    TimeSeries (or raw sampled signal) to use as target for this trial [TxO]
    :param Optional[float] min_tau: Minimum time constant to permit
    :param bool is_first:           Flag to indicate this is the first trial. Resets learning and causes initialisation.
    :param bool is_last:            Flag to indicate this is the last trial. Performs clean-up (not essential)
    :param Callable loss_fcn:       Function that computes the loss for the currently configured layer. Default: :py:func:`loss_mse_reg`
    :param Dict loss_params:        A dictionary of loss function parameters to pass to the loss function. Must be configured on the very first call to `.train_output_target`; subsequent changes will be ignored. Default: Appropriate parameters for :py:func:`loss_mse_reg`.
    :param Callable optimizer:      A JAX-style optimizer function. See the JAX docs for details. Default: `jax.experimental.optimizers.adam`
    :param Dict opt_params:         A dictionary of parameters passed to `optimizer`. Default: {"step_size": 1e-4}

    Use this function to train the output of the reservoir to match a target, given an input stimulus. This function can
    be called in a loop, passing in randomly-chosen training examples on each call. Parameters of the layer are updated
    on each call of `.train_adam`, but the layer time and state are *not* updated.

    :return:            (loss_fcn, grad_fcn):
                            loss_fcn:   Callable[[], float] Function that returns the current loss
                            grad_fcn:   Callable[[], float] Function that returns the gradient for the current batch
    """

    # - Set a minimum tau, if not provided
    if min_tau is None:
        min_tau = self._dt * 10.0

    # - Get static arguments
    # x0 = self._state
    dt = self._dt
    noise_std = self._noise_std
    rng_key = self._rng_key
    evolve_func = self._evolve_jit

    # - Define loss function
    @jit
    def loss_mse_reg(params: dict, batch: Tuple, state, min_tau: float, lambda_mse: float = 1., reg_tau: float = 10000., reg_l2_rec: float = 1.) -> float:
        """
        loss_mse_reg() - Loss function for target versus output

        :param Dict params:         Dictionary of packed parameters
        :param Tuple batch:         (input_t, target_t)
        :param float min_tau:       Minimum time constant
        :param float lambda_mse:    Factor when combining loss, on mean-squared error term. Default: 1.0
        :param float reg_tau:       Factor when combining loss, on minimum time constant limit. Default: 1e5
        :param float reg_l2_rec:    Factor when combining loss, on L2-norm term of recurrent weights. Default: 1.

        :return float: Current loss value
        """

        # - Get inputs and targets for this batch
        input_batch_t, target_batch_t = batch

        # - Call compiled Euler solver to evolve reservoir
        _, _, _, _, outputs = evolve_func(
            state,
            params["w_in"],
            params["w_recurrent"],
            params["w_out"],
            params["bias"],
            params["tau"],
            input_batch_t,
            noise_std,
            rng_key,
            dt,
        )

        # - Measure output-target loss
        mse = lambda_mse * np.mean((outputs - target_batch_t) ** 2)

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
    initialise = is_first or not hasattr(self, "__in_training_sgd_adam")

    if initialise:
        # - Get optimiser
        (opt_init, opt_update, get_params) = optimizer(**opt_params)
        # opt_init, opt_update, get_params = adam(1e-4)
        self.__get_params = get_params

        # - Make update function
        @jit
        def update_fcn(i, opt_state, batch):
            """
            update_fcn() - Perform one round of optimizer update
            :param i:
            :param opt_state:
            :param batch:
            :return:
            """
            # - Get parameters
            params = get_params(opt_state)

            # - Call optimiser update function
            return opt_update(i, self.__grad_fcn(params, batch, self._state), opt_state)

        # - If using default loss, set up parameters
        if loss_fcn is None:
            loss_fcn = loss_mse_reg
            default_loss_params = {
                "lambda_mse": 1.0,
                "reg_tau": 10000.0,
                "reg_l2_rec": 1.0,
                "min_tau": self._dt * 10.0,
            }
            default_loss_params.update(loss_params)
            loss_params = default_loss_params

        def loss_curried(opt_params: Dict, batch: Tuple, state: Dict):
            return loss_fcn(opt_params, batch, state, **loss_params)

        # - Assign update, loss and gradient functions
        self.__update_fcn = update_fcn
        self.__loss_fcn = jit(loss_curried)
        self.__grad_fcn = jit(grad(loss_curried))

        # - Initialise optimimser
        self.__opt_state = opt_init(self.__pack_params())
        self.__itercount = itertools.count()

        # - Assign "in training" flag
        self.__in_training_sgd_adam = True

    # - Prepare time base and inputs
    if issubclass(ts_input.__class__, TimeSeries):
        time_base, inps, num_timesteps = self._prepare_input(ts_input, None, None)
        target = ts_target(time_base)
    else:
        inps = ts_input
        target = ts_target

    # - Perform one step of optimisation
    self.__opt_state = self.__update_fcn(
        next(self.__itercount), self.__opt_state, (inps, target)
    )

    # - Apply the parameter updates
    self.__apply_params(self.__get_params(self.__opt_state))

    # - Reset status, on "is_last" flag
    if is_last:
        del self.__in_training_sgd_adam

    # - Return lambdas that evaluate the loss and the gradient
    return (
        lambda: self.__loss_fcn(self.__get_params(self.__opt_state), (inps, target), self._state),
        lambda: self.__grad_fcn(self.__get_params(self.__opt_state), (inps, target), self._state),
    )


def add_shim_rate_jax_sgd(lyr: rj.RecRateEulerJax) -> rj.RecRateEulerJax:
    """
    add_shim_rate_jax_sgd() - Insert methods that support gradient-based training of the reservoir

    :param lyr:     RecRateEulerJax Pre-configured layer to train

    This function adds the method `.train_adam()` to the provided layer. Use this to perform training to match the
    reservoir output to a given target. See documentation for `.train_adam()` for details.

    :return: lyr:   RecRateEulerJax Layer with added functions
    """
    # - Check that we have a RecRateEulerJax object or subclass
    assert isinstance(
        lyr, rj.RecRateEulerJax
    ), "This function is only compatible with RecRateEulerJax subclass layers"

    # - Insert methods required for training
    lyr.train_output_target = types.MethodType(train_output_target, lyr)
    lyr.__pack_params = types.MethodType(pack_params, lyr)
    lyr.__apply_params = types.MethodType(apply_params, lyr)

    # - Return converted object
    return lyr
