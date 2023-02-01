"""
Dynap-SE autoencoder based quantization unsupervised weight configuration learning utilities

* Non User Facing *
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from copy import deepcopy

# JAX
import jax
from jax import jit, value_and_grad
from jax import numpy as jnp
from jax.lax import scan
from jax.example_libraries import optimizers

# Rockpool
from .loss import loss_reconstruction
from .digital import DigitalAutoEncoder
from .weight_handler import WeightHandler

__all__ = ["learn_weights"]


def learn_weights(
    ## Input
    weights_in: np.ndarray,
    weights_rec: np.ndarray,
    Iscale: float,
    n_bits: Optional[int] = 4,
    ## Optimization
    fixed_epoch: bool = False,
    num_epoch: int = int(1e7),
    num_epoch_checkpoint: int = int(1e3),
    eps: float = 1e-6,
    record_loss: bool = True,
    optimizer: str = "adam",
    step_size: Union[float, Callable[[int], float]] = 1e-4,
    opt_params: Optional[Dict[str, Any]] = {},
    *args,
    **kwargs,
) -> Dict[str, Union[np.ndarray, float]]:
    """
    learn_weights is a utility function to use the autoencoder quantization approach
    in deployment pipelines. One can experiment with the parameters to control the autoencoder training.

    :param weights_in: input layer weights used in Dynap-SE2 simulation
    :type weights_in: Optional[np.ndarray]
    :param weights_rec: recurrent layer (in-device neurons) weights used in Dynap-SE2 simulation
    :type weights_rec: Optional[np.ndarray]
    :param Iscale: base weight scaling current in Amperes used in simulation
    :type Iscale: float
    :param n_bits: number of target weight bits, defaults to 4
    :type n_bits: Optional[int], optional
    :param fixed_epoch: used fixed number of epochs or control the convergence by loss decrease, defaults to False
    :type fixed_epoch: bool, optional
    :param num_epoch: the fixed number of epochs as global limit, defaults to 10,000,000
    :type num_epoch: int, optional
    :param num_epoch_checkpoint: at this point (number of epochs), pipeline checks the loss decrease and decide to continue or not, defaults to 1,000.
    :type num_epoch_checkpoint: int, optional
    :param eps: the epsilon tolerance value. If the loss does not decrease more than this for five consecutive checkpoints, then training stops. defaults to 1e-6
    :type eps: float, optional
    :param record_loss: record the loss evolution or not, defaults to True
    :type record_loss: bool, optional
    :param optimizer: one of the optimizer defined in `jax.example_libraries.optimizers` : , defaults to "adam"
    :type optimizer: str, optional
    :param step_size: positive scalar, or a callable representing a step size schedule that maps the iteration index to a positive scalar. , defaults to 1e-4
    :type step_size: Union[float, Callable[[int], float]], optional
    :param opt_params: optimizer parameters dictionary, defaults to {}
    :type opt_params: Optional[Dict[str, Any]]
    :return: A dictionary of quantized weights and parameters, the quantization loss
    :rtype: Dict[str, Union[np.ndarray, float]]
    """

    ### --- Initial Object Construction --- ###

    if not isinstance(Iscale, float):
        raise ValueError("Iscale should be float!")

    # weights might not be defined, still return
    if weights_in is None and weights_rec is None:
        spec = {
            "weights_in": None,
            "sign_in": None,
            "weights_rec": None,
            "sign_rec": None,
            "Iw_0": 0.0,
            "Iw_1": 0.0,
            "Iw_2": 0.0,
            "Iw_3": 0.0,
            "quantization_loss": None,
        }
        return spec

    __handler = WeightHandler(weights_in, weights_rec)
    __encoder = DigitalAutoEncoder(__handler.w_flat.size, n_bits)

    ### --- Optimization Configuration --- ###

    ## - Get the optimiser functions
    init_fun, update_fun, get_params = __get_optimizer(
        optimizer, step_size, **opt_params
    )

    ## - Initialize the optimizer with the initial parameters
    opt_state = init_fun(deepcopy(__encoder.parameters()))

    ## - Preprocess
    __scale = lambda w: (1.0 / (jnp.max(w) - jnp.min(w) + 1))
    __scale_factor = __scale(__handler.w_flat)
    w_flat = __scale_factor * jnp.array(__handler.w_flat)

    ## - Get the jit compiled update and value-and-gradient functions
    loss_vgf = jit(
        value_and_grad(lambda params: loss_reconstruction(__encoder, params, w_flat))
    )
    update_fun = jit(update_fun)
    run_for = jit(
        lambda epoch, state: __run_for(epoch, state, get_params, loss_vgf, update_fun)
    )

    ### --- Optimize --- ###

    ## - Check the loss decrease and decide to stop training before it reaches to num_epochs
    if not fixed_epoch:
        count = 0
        rec_loss = []
        mean_loss = jnp.inf
        epoch = jnp.array(range(num_epoch_checkpoint)).reshape(-1, 1)

        for _ in range(0, num_epoch, num_epoch_checkpoint):
            opt_state, loss_t = run_for(epoch, opt_state)

            if record_loss:
                rec_loss += list(np.array(loss_t))

            ### Check the mean loss at each num_epoch_checkpoint
            if mean_loss - jnp.mean(loss_t) < eps:
                count += 1
                if count > 5:
                    break
            else:
                count = 0
                mean_loss = jnp.mean(loss_t)

    ## - Just repeat the process for the number of epochs
    else:
        epoch = jnp.array(range(num_epoch)).reshape(-1, 1)
        opt_state, rec_loss = run_for(epoch, opt_state)

    ### ---  Read the results --- ###

    optimized_encoder = __encoder.set_attributes(get_params(opt_state))
    __, code, bit_mask = optimized_encoder(w_flat)

    ## - Quantized weights
    q_weights = WeightHandler.bit2int_mask(n_bits, bit_mask)
    qw_in, qw_rec = __handler.reshape_flat_weights(q_weights)

    ### --- Return --- ###
    Iw = np.array(code) * Iscale / np.array(__scale_factor)

    spec = {
        "weights_in": qw_in,
        "sign_in": __handler.sign_in,
        "weights_rec": qw_rec,
        "sign_rec": __handler.sign_rec,
        "Iw_0": Iw[0],
        "Iw_1": Iw[1],
        "Iw_2": Iw[2],
        "Iw_3": Iw[3],
        "quantization_loss": float(rec_loss[-1]),
    }

    return spec


def __run_for(
    epoch: jnp.array,
    opt_state: optimizers.OptimizerState,
    get_params: optimizers.ParamsFn,
    loss_vgf: Callable[[Any], Tuple[float]],
    update_fun: optimizers.UpdateFn,
) -> Tuple[optimizers.OptimizerState, jnp.DeviceArray]:
    """
    __run_for is a utility function executing jax training workflow

    :param epoch: the dummy sequence array [0,1,2,3..] standing for epoch ids to be walked through
    :type epoch: jnp.array
    :param opt_state: the optimizer's initial state
    :type opt_state: optimizers.OptimizerState
    :param get_params: the optimizer's parameter getter
    :type get_params: optimizers.ParamsFn
    :param loss_vgf: the loss function returning the loss value and the gradient value
    :type loss_vgf: Callable[[Any], Tuple[float]]
    :param update_fun: the optimizers update functions
    :type update_fun: optimizers.UpdateFn
    :return: opt_state, loss_val
        :opt_state: the last optimized state recorded at the end of the last epoch
        :loss_val: the recorded loss values over epochs
    :rtype: Tuple[optimizers.OptimizerState, jnp.DeviceArray]
    """

    def step(
        opt_state: optimizers.OptimizerState, epoch: int
    ) -> Tuple[Dict[str, jnp.DeviceArray], optimizers.OptimizerState, jnp.DeviceArray]:
        """
        step stacks together the single iteration step operations during training

        :param opt_state: the optimizer's current state
        :type opt_state: optimizers.OptimizerState
        :param epoch: the current epoch
        :type epoch: int
        :return: params, opt_state, loss_val
            :params: the network parameters
            :opt_state: the current time step optimizer state
            :loss_val: the current loss value
        :rtype: Tuple[Dict[str, jnp.DeviceArray], optimizers.OptimizerState, jnp.DeviceArray]
        """

        params = get_params(opt_state)
        loss_val, grads = loss_vgf(params)
        opt_state = update_fun(epoch, grads, opt_state)

        # Return
        return opt_state, loss_val

    # --- Iterate over epochs --- #
    opt_state, loss_t = scan(step, opt_state, epoch)
    return opt_state, loss_t


def __get_optimizer(
    name: str, *args, **kwargs
) -> Tuple[optimizers.InitFn, optimizers.UpdateFn, optimizers.ParamsFn]:
    """
    _get_optimizer calls the name-requested optimizer and returns the jax optimizer functions

    :param name: the name of the optimizer
    :type name: str
    :raises ValueError: Requested optimizer is not available!
    :return: the optimizer functions
    :rtype: Tuple[optimizers.InitFn, optimizers.UpdateFn, optimizers.ParamsFn]
    """

    if name == "sgd":
        return optimizers.sgd(*args, **kwargs)
    elif name == "momentum":
        return optimizers.momentum(*args, **kwargs)
    elif name == "nesterov":
        return optimizers.nesterov(*args, **kwargs)
    elif name == "adagrad":
        return optimizers.adagrad(*args, **kwargs)
    elif name == "rmsprop":
        return optimizers.rmsprop(*args, **kwargs)
    elif name == "rmsprop_momentum":
        return optimizers.rmsprop_momentum(*args, **kwargs)
    elif name == "adam":
        return optimizers.adam(*args, **kwargs)
    elif name == "adamax":
        return optimizers.adamax(*args, **kwargs)
    elif name == "sm3":
        return optimizers.sm3(*args, **kwargs)
    else:
        raise ValueError(
            f"The optimizer : {name} is not available!"
            f"Try one of the optimizer defined in `jax.example_libraries.optimizers'` : sgd, momentum, nesterov, adagrad, rmsprop, rmsprop_momentum, adam, adamax, sm3"
        )
