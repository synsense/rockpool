###
# net_rate_reservoir.py - Classes and functions for encapsulating simple rate-based reservoirs
###

from ..network import Network
from ...layers import FFRateEuler, RecRateEuler, PassThrough

from typing import Union, List
import numpy as np

ArrayLike = Union[List, np.ndarray, float, int]


def build_rate_reservoir(
    weights_in: ArrayLike,
    weights_res: ArrayLike,
    weights_out: ArrayLike,
    tau_in: ArrayLike = 1.0,
    tau_res: ArrayLike = 1.0,
    bias_in: ArrayLike = 0.0,
    bias_res: ArrayLike = 0.0,
    dt: float = 1.0 / 10.0,
    noise_std_in: float = 0.0,
    noise_std_res: float = 0.0,
    noise_std_out: float = 0.0,
):
    """
    build_rate_reservoir - Build a rate-based reservoir network, with the defined weights

    :param weights_in:
    :param weights_res:
    :param weights_out:
    :param tau_in:
    :param tau_res:
    :param bias_in:
    :param bias_res:
    :param dt:
    :param noise_std_in:
    :param noise_std_res:
    :param noise_std_out:
    :return:                Network - A reservoir network
    """

    # - Build the input layer
    input_layer = FFRateEuler(
        weights=weights_in,
        tau=tau_in,
        bias=bias_in,
        dt=dt,
        noise_std=noise_std_in,
        name="Input",
    )

    # - Build the recurrent layer
    lyrRes = RecRateEuler(
        weights=weights_res,
        tau=tau_res,
        bias=bias_res,
        dt=dt,
        noise_std=noise_std_res,
        name="Reservoir",
    )

    # - Build the output layer
    lyrOut = PassThrough(
        weights=weights_out, dt=dt, noise_std=noise_std_out, name="Readout"
    )

    # - Return the network
    return Network(input_layer, lyrRes, lyrOut)


def build_random_reservoir(
    input_size: int = 1,
    reservoir_size: int = 100,
    output_size: int = 1,
    weights_in_std: float = 1.0,
    weights_res_std: float = 1.0,
    weights_out_std: float = 1.0,
    weights_in_mean: float = 0.0,
    weights_res_mean: float = 0.0,
    weights_out_mean: float = 0.0,
    tau_in: ArrayLike = 1.0,
    tau_res: ArrayLike = 1.0,
    bias_in: ArrayLike = 0.0,
    bias_res: ArrayLike = 0.0,
    dt: float = 1.0 / 10.0,
    noise_std_in: float = None,
    noise_std_res: float = None,
    noise_std_out: float = None,
):
    """
    build_random_reservoir - Build a randomly-generated reservoir network

    :param input_size:
    :param reservoir_size:
    :param output_size:
    :param weights_in_std:
    :param weights_res_std:
    :param weights_out_std:
    :param weights_in_mean:
    :param weights_res_mean:
    :param weights_out_mean:
    :param tau_in:
    :param tau_res:
    :param bias_in:
    :param bias_res:
    :param dt:
    :param noise_std_in:
    :param noise_std_res:
    :param noise_std_out:
    :return:
    """

    # - Generate weights
    weights_in = np.random.normal(
        weights_in_mean, weights_in_std, (input_size, reservoir_size)
    )
    weights_res = np.random.normal(
        weights_res_mean, weights_res_std, (reservoir_size, reservoir_size)
    )
    weights_out = np.random.normal(
        weights_out_mean, weights_out_std, (reservoir_size, output_size)
    )

    # - Build reservoir network
    return build_rate_reservoir(
        weights_in,
        weights_res,
        weights_out,
        tau_in,
        tau_res,
        bias_in,
        bias_res,
        dt,
        noise_std_in,
        noise_std_res,
        noise_std_out,
    )
