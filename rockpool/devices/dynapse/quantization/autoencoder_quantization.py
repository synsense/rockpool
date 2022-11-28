"""
Dynap-SE autoencoder based quantization package provides easy to use support

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

15/09/2022

"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .autoencoder import learn_weights


__all__ = ["autoencoder_quantization"]


### --- UTILITY FUNCTIONS --- ###


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
            Iscale[n],
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
