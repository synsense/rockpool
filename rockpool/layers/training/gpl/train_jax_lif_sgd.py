##
# train_jax_sgd.py - Support for gradient-based training of a Jax recurrent LIF reservoir
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
    (self._w_in, self._weights, self._w_out, self._tau_mem, self._tau_syn, self._bias) = (
        params["w_in"],
        params["w_rec"],
        params["w_out"],
        params["tau_mem"],
        params["tau_syn"],
        params["bias"],
    )


# - Training function
def train_output_target_adam(self: lj.RecLIFJax,
                             ts_input: TSContinuous, ts_target: TSContinuous,
                             is_first: bool = True, is_last: bool = False,
                             loss: Callable[[Dict, Tuple], float] = None, loss_params: dict = {},
                             loss_has_aux: bool = True,
                             optimiser: Callable = adam, opt_params: dict = {'step_size': 1e-4}):
    # - Define default loss function
    def loss_mse_reg(params: Dict, batch: Tuple, state: Dict,
                     lambda_mse: float = 2.,
                     reg_tau: float = 100.,
                     reg_l2_in: float = 0.,
                     reg_l2_rec: float = 1.,
                     reg_l2_out: float = 0.,
                     reg_act1: float = 1.,
                     reg_act2: float = 1.,
                     min_tau: float = 10. * self.dt,
                     ):
        # - Access trial inputs and targets
        sp_in_trial_ts, target_trial_ts = batch

        # - Clip taus
        params['tau_mem'] = jnp.clip(params['tau_mem'], min_tau)
        params['tau_syn'] = jnp.clip(params['tau_syn'], min_tau)

        # - Evolve network
        state, _, output_ts, surrograte_ts, _, Vmem_ts, _ = self._evolve_jit(state,
                                                                             params['w_in'],
                                                                             params['w_rec'],
                                                                             params['w_out'],
                                                                             params['tau_mem'],
                                                                             params['tau_syn'],
                                                                             params['bias'],
                                                                             self.noise_std,
                                                                             sp_in_trial_ts,
                                                                             sp_in_trial_ts * 0.,
                                                                             self._rng_key,
                                                                             self._dt,
                                                                             )

        # - MSE between output and target
        dLoss = dict()
        dLoss['loss_mse'] = lambda_mse * jnp.mean((output_ts - target_trial_ts) ** 2)

        # Regularisation for taus
        dLoss['loss_tau_mem'] = reg_tau * jnp.mean(jnp.clip(-(params['tau_mem'] - min_tau - 1e-3), 0.) ** 2)
        dLoss['loss_tau_syn'] = reg_tau * jnp.mean(jnp.clip(-(params['tau_syn'] - min_tau - 1e-3), 0.) ** 2)

        # - Regularisation for weights
        dLoss['loss_weights_l2'] = (reg_l2_in * jnp.mean(params['w_in'] ** 2) +
                                    reg_l2_rec * jnp.mean(params['w_rec'] ** 2) +
                                    reg_l2_out * jnp.mean(params['w_out'] ** 2)
                                    )

        # - Regularisation for activity
        dLoss['loss_activity1'] = reg_act1 * jnp.mean(surrograte_ts)
        dLoss['loss_activity2'] = reg_act2 * jnp.mean(Vmem_ts ** 2)

        # - Return loss, as well as components
        return sum(dLoss.values()), (dLoss, output_ts, state)

    # - Initialise training
    initialise = is_first or not hasattr(self, "__in_training_sgd_adam")

    if initialise:
        # - Get optimiser
        (opt_init, opt_update, get_params) = optimiser(**opt_params)
        self.__get_params = get_params

        # - If using default loss, set up parameters
        if loss is None:
            loss = loss_mse_reg
            default_loss_params = {"lambda_mse": 2.,
                                   "reg_tau": 100.,
                                   "reg_l2_in": 0.,
                                   "reg_l2_rec": 1.,
                                   "reg_l2_out": 0.,
                                   "reg_act1": 1.,
                                   "reg_act2": 1.,
                                   "min_tau": self._dt * 10.,
                                   }
            default_loss_params.update(loss_params)
            loss_params = default_loss_params

        # - Record loss and gradient function
        if loss is None:
            loss = loss_mse_reg

        def loss_curried(opt_params: Dict, batch: Tuple, state: Dict):
            return loss(opt_params, batch, state, **loss_params)

        self.__loss_fcn = jit(loss_curried)
        self.__grad_fcn = jit(grad(loss_curried, has_aux=loss_has_aux))

        # - Make update function
        def update_fcn(i, opt_state, batch):
            # - Get parameters
            params = get_params(opt_state)

            # - Get Gradient
            if loss_has_aux:
                g, _ = self.__grad_fcn(params, batch, self._state)
            else:
                g = self.__grad_fcn(params, batch, self._state)

            # - Call optimiser update function
            return opt_update(i, g, opt_state)

        # - Assign update functions
        self.__update_fcn = jit(update_fcn)

        # - Initialise optimimser
        self.__opt_state = opt_init(self.__pack_params())
        self.__itercount = itertools.count()

        # - Assign "in training" flag
        self.__in_training_sgd_adam = True

    # - Prepare time base and inputs
    if ts_input.__class__ is TSEvent:
        time_base, inps_sp, num_timesteps = self._prepare_input_events(ts_input, None, None)
        target = ts_target(time_base)
    elif ts_input.__class__ is TSContinuous:
        time_base, inps_sp, num_timesteps = self._prepare_input_events(ts_input, None, None)
        target = ts_target(time_base)
    else:
        inps_sp = ts_input
        target = ts_target

    # - Perform one step of optimisation
    self.__opt_state = self.__update_fcn(
        next(self.__itercount), self.__opt_state, (inps_sp, target)
    )

    # - Apply the parameter updates
    self.__apply_params(self.__get_params(self.__opt_state))

    # - Reset status, on "is_last" flag
    if is_last:
        del self.__in_training_sgd_adam

    # - Return current loss, and lambdas that evaluate the loss and the gradient
    return (
        lambda: self.__loss_fcn(self.__get_params(self.__opt_state), (inps_sp, target), self._state),
        lambda: self.__grad_fcn(self.__get_params(self.__opt_state), (inps_sp, target), self._state),
    )

def add_train_output(lyr) -> lj.RecLIFJax:
    """
    add_train_output() - Insert methods that support gradient-based training of the reservoir

    :param lyr:     RecLIFJax subclass Pre-configured layer to train

    This function adds the method `.train_adam()` to the provided layer. Use this to perform training to match the
    reservoir output to a given target. See documentation for `.train_adam()` for details.

    :return: lyr:   RecLIFJax subclass Layer with added functions
    """
    # - Check that we have a RecLIFJax compatible object
    assert (
        isinstance(lyr, lj.RecLIFJax)
    ), "This function is only compatible with RecLIFJax subclass layers"

    # - Insert methods required for training
    lyr.train_adam = types.MethodType(train_output_target_adam, lyr)
    lyr.__pack_params = types.MethodType(pack_params, lyr)
    lyr.__apply_params = types.MethodType(apply_params, lyr)

    # - Return converted object
    return lyr
