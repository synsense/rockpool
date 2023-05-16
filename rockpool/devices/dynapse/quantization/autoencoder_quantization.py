"""
Dynap-SE autoencoder based quantization top level function calling the unsupervised learning methods per clusters
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .autoencoder import learn_weights


__all__ = ["autoencoder_quantization"]


def autoencoder_quantization(
    ## cluster
    n_cluster: int,
    core_map: List[int],
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
    autoencoder_quantization executes the unsupervised weight configuration learning approach
    `rockpool.devices.dynapse.quantization.autoencoder.learn.learn_weights` for each cluster seperately.
    The function subsets input and recurrent weights for each cluster and quantizes the weights according to regarding cluster's constraints.

    :param n_cluster: total number of clusters, neural cores allocated
    :type n_cluster: int
    :param core_map: core mapping for real hardware neurons (neuron_id : core_id)
    :type core_map: List[int]
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

    spec = {
        "weights_in": [],
        "sign_in": [],
        "weights_rec": [],
        "sign_rec": [],
        "Iw_0": [],
        "Iw_1": [],
        "Iw_2": [],
        "Iw_3": [],
        "quantization_loss": [],
    }

    for n in range(n_cluster):
        if weights_in is not None:
            w_in = np.zeros_like(weights_in)
            w_in[:, core_map == n] = weights_in[:, core_map == n]
        else:
            w_in = None

        if weights_rec is not None:
            w_rec = np.zeros_like(weights_rec)
            w_rec[:, core_map == n] = weights_rec[:, core_map == n]
        else:
            w_rec = None

        __temp = learn_weights(
            w_in,
            w_rec,
            Iscale,
            n_bits,
            fixed_epoch,
            num_epoch,
            num_epoch_checkpoint,
            eps,
            record_loss,
            optimizer,
            step_size,
            opt_params,
        )

        for key in spec:
            spec[key].append(__temp[key])

    return spec
