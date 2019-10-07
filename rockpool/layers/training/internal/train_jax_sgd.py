##
# train_jax_sgd.py - Support for gradient-based training of a Jax recurrent rate-based reservoir
##

import itertools

import jax.numpy as np
from jax.experimental.optimizers import adam
from jax import grad, jit

from typing import Callable, Tuple, Union, Optional, Dict

from ... import RecRateEulerJax
from ....timeseries import TimeSeries
import types


def pack_params(self: RecRateEulerJax) -> Dict:
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
        "w_in": self._weights,
        "w_recurrent": self._w_recurrent,
        "w_out": self._w_out,
        "tau": self._tau,
        "bias": self._bias,
    }


def apply_params(self: RecRateEulerJax, params: Dict) -> None:
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
    (self._weights, self._w_recurrent, self._w_out, self._tau, self._bias) = (
        params["w_in"],
        params["w_recurrent"],
        params["w_out"],
        params["tau"],
        params["bias"],
    )


def train_adam(
    self,
    ts_input: Union[TimeSeries, np.ndarray],
    ts_target: Union[TimeSeries, np.ndarray],
    min_tau: Optional[float] = None,
    is_first: Optional[bool] = False,
    is_last: Optional[bool] = False,
) -> Tuple[Callable[[], float], Callable[[], float]]:
    """
    .train_adam() - Perform one trial of Adam stochastic gradient descent to train the reservoir

    :param ts_input:    TimeSeries (or raw sampled signal) to use as input for this trial [TxI]
    :param ts_target:   TimeSeries (or raw sampled signal) to use as target for this trial [TxO]
    :param min_tau:     Optional[float]: Minimum time constant to permit
    :param is_first:    Optional[bool]: Flag to indicate this is the first trial. Resets learning and causes initialisation.
    :param is_last:     Optional[bool]: Flag to indicate this is the last trial. Performs clean-up (not essential)

    Use this function to train the output of the reservoir to match a target, given an input stimulus. This function can
    be called in a loop, passing in randomly-chosen training examples on each call. Parameters of the layer are updated
    on each call of ``train_adam()`, but the layer time and state are *not* updated.

    :return:            (loss_fcn, grad_fcn):
                            loss_fcn:   Callable[[], float] Function that returns the current loss
                            grad_fcn:   Callable[[], float] Function that returns the gradient for the current batch
    """

    # - Set a minimum tau, if not provided
    if min_tau is None:
        min_tau = self._dt * 10.0

    # - Get static arguments
    x0 = self._state
    dt = self._dt
    noise_std = self._noise_std
    rng_key = self._rng_key
    evolve_func = self._evolve_jit

    # - Define loss function
    @jit
    def loss_output_target(params: dict, batch: Tuple) -> float:
        """
        loss_output_target() - Loss function for target versus output

        :param params:      dict Dictionary of packed parameters

        :return: float: Current loss value
        """

        # - Get inputs and targets for this batch
        input_batch_t, target_batch_t = batch

        # - Call compiled Euler solver to evolve reservoir
        _, _, _, _, outputs = evolve_func(
            x0,
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
        mse = np.mean((outputs - target_batch_t) ** 2)

        # - Get loss for tau parameter constraints
        tau_loss = 10000 * np.mean(
            np.where(params["tau"] < min_tau, np.exp(-(params["tau"] - min_tau)), 0)
        )

        # Measure w_res norm
        w_res_norm = np.mean(params["w_recurrent"] ** 2)

        # - Loss: target/output squared error, time constant constraint, recurrent weights norm, activation penalty
        fLoss = mse + tau_loss + w_res_norm

        # - Return loss
        return fLoss

    # - Initialise training
    initialise = is_first or not hasattr(self, "__in_training_sgd_adam")

    if initialise:
        # - Get optimiser
        opt_init, opt_update, get_params = adam(1e-4)
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
            return opt_update(i, grad(loss_output_target)(params, batch), opt_state)

        # - Assign update, loss and gradient functions
        self.__update_fcn = update_fcn
        self.__loss_fcn = loss_output_target
        self.__grad_fcn = jit(grad(loss_output_target))

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

    # - Return a lambda that evaluates the loss and the gradient
    return (
        lambda: self.__loss_fcn(self.__get_params(self.__opt_state), (inps, target)),
        lambda: self.__grad_fcn(self.__get_params(self.__opt_state), (inps, target)),
    )


def add_train_output(lyr: RecRateEulerJax) -> RecRateEulerJax:
    """
    add_train_output() - Insert methods that support gradient-based training of the reservoir

    :param lyr:     RecRateEulerJax Pre-configured layer to train

    This function adds the method `.train_adam()` to the provided layer. Use this to perform training to match the
    reservoir output to a given target. See documentation for `.train_adam()` for details.

    :return: lyr:   RecRateEulerJax Layer with added functions
    """
    # - Check that we have a RecRateEulerJax object
    assert (
        type(lyr) == RecRateEulerJax
    ), "This function is only compatible with RecRateEulerJax layers"

    # - Insert methods required for training
    lyr.train_adam = types.MethodType(train_adam, lyr)
    lyr.__pack_params = types.MethodType(pack_params, lyr)
    lyr.__apply_params = types.MethodType(apply_params, lyr)

    # - Return converted object
    return lyr
