##
# train_jax_lif_sgd.py - Support for gradient-based training of a Jax recurrent LIF reservoir
##

import itertools
from typing import Callable, Tuple, Union, Optional, Dict
import types

import jax.numpy as jnp
import numpy as np
from jax.experimental.optimizers import adam
from jax import grad, jit

from rockpool.timeseries import TimeSeries, TSContinuous, TSEvent
from ...gpl import lif_jax as lj


def pack_params(self) -> Dict:
    """
    .pack_params() - Pack trainable parameters into a dictionary

    :return:    dict(
                     'w_in':        np.ndarray Input weights [IxN]
                     'w_rec':       np.ndarray Recurrent weights [NxN], unused for FF layers
                     'w_out':       np.ndarray Output weights [NxO]
                     'tau_mem':     np.ndarray Membrane time constants of layer neurons [N]
                     'tau_syn':     np.ndarray Synaptic time constants of layer neurons [N]
                     'bias':        np.ndarray Bias values of recurrent layer [N]
                    )
    """
    # - Pack all pertinent parameters
    return {
        "w_in": self._w_in,
        "w_rec": self._weights,
        "w_out": self._w_out,
        "tau_mem": self._tau_mem,
        "tau_syn": self._tau_syn,
        "bias": self._bias,
    }


def apply_params(self, params: Dict) -> None:
    """
    .apply_params() - Apply packed parameters to the layer

    :param params:  dict(
                         'w_in':        np.ndarray Input weights [IxN]
                         'w_rec':       np.ndarray Recurrent weights [NxN], unused for FF layers
                         'w_out':       np.ndarray Output weights [NxO]
                         'tau_mem':     np.ndarray Membrane time constants of layer neurons [N]
                         'tau_syn':     np.ndarray Synaptic time constants of layer neurons [N]
                         'bias':        np.ndarray Bias values of recurrent layer [N]
                        )
    """
    # - Unpack and apply parameters
    (
        self._w_in,
        self._weights,
        self._w_out,
        self._tau_mem,
        self._tau_syn,
        self._bias,
    ) = (
        params["w_in"],
        params["w_rec"],
        params["w_out"],
        params["tau_mem"],
        params["tau_syn"],
        params["bias"],
    )


# - Training function
def train_output_target(
    self: lj.RecLIFJax,
    ts_input: TimeSeries,
    ts_target: TSContinuous,
    is_first: bool = True,
    is_last: bool = False,
    loss_fcn: Callable[[Dict, Tuple], float] = None,
    loss_params: Dict = {},
    optimizer: Callable = adam,
    opt_params: Dict = {"step_size": 1e-4},
):
    """
    Train the weighted output of a Jax LIF layer to match a target signal

    Call this function to evolve the current layer, and use a loss-gradient-based optimiser to push all parameters to minimise the loss. For example, and by default, use a regularised mean-squared-error based loss, along with the ADAM stochastic gradient descent with momentum optimiser.

    The calling signature for `loss_mse_reg` is ::

        def loss_mse_reg(
            params: Dict,
            batch: Tuple,
            state: Dict,

            lambda_mse: float = 2.0,
            reg_tau: float = 100.0,
            reg_l2_in: float = 0.0,
            reg_l2_rec: float = 1.0,
            reg_l2_out: float = 0.0,
            reg_act1: float = 1.0,
            reg_act2: float = 1.0,
            min_tau: float = 10.0 * self.dt,
        ):

        ...

        return loss, (state, dLoss, output_ts)

    `dLoss` is a dictionary containing individual loss values and regularisation values for the current trial. The loss and regularisation factors can be modified by passing a dictionary of parameters on the first call to :py:meth:`.train_output_target`.

    .. Providing your own `loss_fcn` function

    Your `loss_fcn` function must accept arguments `params`, `batch` and `state`. `batch` is a Tuple (`ts_input`, `ts_target`) for the current trial. `params` is a dictionary containing the layer parameters to be evaluated on this trial.

    The return signature for `loss_fcn` must be `return loss, (state, ...)`. You may return any extra variables you like in the second tuple, but this must be a tuple, and must contain the updated layer state as the first element of the tuple.

    You must evolve the layer using (probably?) the internal evolution function :py:meth:`~._evolve_jit`, and you must return the updated layer state.



    :param TimeSeries ts_input:     Either an event or continuous time series, which serves as input to the current layer
    :param TSContinuous ts_target:  A continuous time series which acts as the target for the weighted surrogate activity of the layer to be trained
    :param bool is_first:           Flag, `True` if this is the first training trial. If `True`, causes initialisation of the training algorithm. Set to `False` for subsequent trials. Default: `True`, this is the first trial
    :param bool is_last:            Flag, `True` if this is the final training trial. If `True`, cleans up after training. Default: `False`, this is not the final trial.
    :param Callable loss_fcn:           Function that computes the loss for the currently configured layer. Default: :py:func:`loss_mse_reg`
    :param Dict loss_params:        A dictionary of loss function parameters to pass to the loss function. Must be configured on the very first call to `.train_output_target`; subsequent changes will be ignored. Default: Appropriate parameters for :py:func:`loss_mse_reg`.
    :param Callable optimizer:      A JAX-style optimizer function. See the JAX docs for details. Default: `jax.experimental.optimizers.adam`
    :param Dict opt_params:         A dictionary of parameters passed to `optimizer`. Default: {"step_size": 1e-4}

    :return (Callable, Callable): (loss_fcn, grad_fcn)
        loss_fcn:   Returns the output of the loss function for the current trial
        grad_fcn:   Returns the gradient of the loss function for the current trial
    """
    # - Define default loss function
    def loss_mse_reg(
        params: Dict,
        batch: Tuple,
        state: Dict,
        lambda_mse: float = 2.0,
        reg_tau: float = 100.0,
        reg_l2_in: float = 0.0,
        reg_l2_rec: float = 1.0,
        reg_l2_out: float = 0.0,
        reg_act1: float = 1.0,
        reg_act2: float = 1.0,
        min_tau: float = 10.0 * self.dt,
    ):
        # - Access trial inputs and targets
        sp_in_trial_ts, target_trial_ts = batch

        # - Clip taus
        params["tau_mem"] = jnp.clip(params["tau_mem"], min_tau)
        params["tau_syn"] = jnp.clip(params["tau_syn"], min_tau)

        # - Evolve network
        state, _, output_ts, surrograte_ts, _, Vmem_ts, _ = self._evolve_jit(
            state,
            params["w_in"],
            params["w_rec"],
            params["w_out"],
            params["tau_mem"],
            params["tau_syn"],
            params["bias"],
            self._noise_std,
            sp_in_trial_ts,
            sp_in_trial_ts * 0.0,
            self._rng_key,
            self._dt,
        )

        # - MSE between output and target
        dLoss = dict()
        dLoss["loss_mse"] = lambda_mse * jnp.mean((output_ts - target_trial_ts) ** 2)

        # Regularisation for taus
        dLoss["loss_tau_mem"] = reg_tau * jnp.mean(
            jnp.clip(-(params["tau_mem"] - min_tau - 1e-3), 0.0) ** 2
        )
        dLoss["loss_tau_syn"] = reg_tau * jnp.mean(
            jnp.clip(-(params["tau_syn"] - min_tau - 1e-3), 0.0) ** 2
        )

        # - Regularisation for weights
        dLoss["loss_weights_l2"] = (
            reg_l2_in * jnp.mean(params["w_in"] ** 2)
            + reg_l2_rec * jnp.mean(params["w_rec"] ** 2)
            + reg_l2_out * jnp.mean(params["w_out"] ** 2)
        )

        # - Regularisation for activity
        dLoss["loss_activity1"] = reg_act1 * jnp.mean(surrograte_ts)
        dLoss["loss_activity2"] = reg_act2 * jnp.mean(Vmem_ts ** 2)

        # - Return loss, as well as components
        return sum(dLoss.values()), (state, dLoss, output_ts)

    # - Initialise training
    initialise = is_first or not hasattr(self, "__in_training_sgd_adam")

    if initialise:
        # - Get optimiser
        (opt_init, self.__opt_update, self.__get_params) = optimizer(**opt_params)

        # - If using default loss, set up parameters
        if loss_fcn is None:
            loss_fcn = loss_mse_reg
            default_loss_params = {
                "lambda_mse": 2.0,
                "reg_tau": 100.0,
                "reg_l2_in": 0.0,
                "reg_l2_rec": 1.0,
                "reg_l2_out": 0.0,
                "reg_act1": 1.0,
                "reg_act2": 1.0,
                "min_tau": self._dt * 10.0,
            }
            default_loss_params.update(loss_params)
            loss_params = default_loss_params

        # - Record loss and gradient functions
        if loss_fcn is None:
            loss_fcn = loss_mse_reg

        def loss_curried(opt_params: Dict, batch: Tuple, state: Dict):
            return loss_fcn(opt_params, batch, state, **loss_params)

        self.__loss_fcn = jit(loss_curried)
        self.__grad_fcn = jit(grad(loss_curried, has_aux=True))

        # - Make update function
        def update_fcn(i, opt_state, batch):
            # - Get parameters
            params = self.__get_params(opt_state)

            # - Get Gradient
            g, aux = self.__grad_fcn(params, batch, self._state)
            state = aux[0]

            # - Call optimiser update function
            return self.__opt_update(i, g, opt_state), state

        # - Assign update functions
        self.__update_fcn = jit(update_fcn)

        # - Initialise optimimser
        self.__opt_state = opt_init(self.__pack_params())
        self.__itercount = itertools.count()

        # - Assign "in training" flag
        self.__in_training_sgd_adam = True

    # - Prepare time base and inputs
    if ts_input.__class__ is TSEvent:
        time_base, inps_sp, num_timesteps = self._prepare_input_events(
            ts_input, None, None
        )
        target = ts_target(time_base)
    elif ts_input.__class__ is TSContinuous:
        time_base, inps_sp, num_timesteps = self._prepare_input_events(
            ts_input, None, None
        )
        target = ts_target(time_base)
    else:
        inps_sp = ts_input
        target = ts_target

    # - Perform one step of optimisation
    self.__opt_state, self._state = self.__update_fcn(
        next(self.__itercount), self.__opt_state, (inps_sp, target)
    )

    # - Apply the parameter updates
    self.__apply_params(self.__get_params(self.__opt_state))

    # - Reset status, on "is_last" flag
    if is_last:
        del self.__in_training_sgd_adam

    # - Execute loss and grad functions to ensure compilation
    if initialise:
        self.__grad_fcn(
            self.__get_params(self.__opt_state), (inps_sp, target), self._state
        )
        self.__loss_fcn(
            self.__get_params(self.__opt_state), (inps_sp, target), self._state
        )

    # - Return current loss, and lambdas that evaluate the loss and the gradient
    return (
        lambda: self.__loss_fcn(
            self.__get_params(self.__opt_state), (inps_sp, target), self._state
        ),
        lambda: self.__grad_fcn(
            self.__get_params(self.__opt_state), (inps_sp, target), self._state
        ),
    )


def add_shim_lif_jax_sgd(lyr) -> lj.RecLIFJax:
    """
    add_shim_lif_jax_sgd() - Insert methods that support gradient-based training of the reservoir

    :param lyr:     RecLIFJax subclass Pre-configured layer to train

    This function adds the method `.train_output_target()` to the provided layer. Use this to perform training to match the reservoir output to a given target. See documentation for `.train_output_target()` for details.

    :return: lyr:   RecLIFJax subclass Layer with added functions
    """
    # - Check that we have a RecLIFJax compatible object
    assert isinstance(
        lyr, lj.RecLIFJax
    ), "This function is only compatible with RecLIFJax subclass layers"

    # - Insert methods required for training
    lyr.train_output_target = types.MethodType(train_output_target, lyr)
    lyr.__pack_params = types.MethodType(pack_params, lyr)
    lyr.__apply_params = types.MethodType(apply_params, lyr)

    # - Return converted object
    return lyr
