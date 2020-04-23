###
# net_rate_reservoir.py - Classes and functions for encapsulating simple rate-based reservoirs
###

from ..network import Network
from ...layers import FFRateEuler, RecRateEuler, PassThrough

from typing import Union, List, Optional
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
    Build a rate-based reservoir network, with the defined weights.

    This function will return a reservoir built with `.FFRateEuler` and `.RecRateEuler` layers, encapsulating a rate-based recurrent reservoir network, with current-based inputs and outputs.

    :param ArrayLike weights_in:            Input weights [M, N]
    :param ArrayLike weights_res:           Recurrent weights [N, N]
    :param ArrayLike weights_out:           Output weights [N, O]
    :param ArrayLike tau_in:      Time constants for input layer neurons [M,]. Default: ``1.``
    :param ArrayLike tau_res:     Time constants for recurrent layer neurons [N,]. Default: ``1.``
    :param ArrayLike bias_in:     Bias values to use in input layer [M,]. Default: ``0.``
    :param ArrayLike bias_res:    Bias values to use in recurrent layer [N,]. Default: ``0.``
    :param float dt:              Time step for all layers, in s. Default: ``0.1``
    :param float noise_std_in:    Noise std. dev. to use in input layer. Default: ``0.``
    :param float noise_std_res:   Noise std. dev. to use in recurrent layer. Default: ``0.``
    :param float noise_std_out:   Noise std. dev. to use in output layer. Default: ``0.``

    :return Network:                        A reservoir network with the defined parameters
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
    return Network([input_layer, lyrRes, lyrOut])


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
    noise_std_in: Optional[float] = None,
    noise_std_res: Optional[float] = None,
    noise_std_out: Optional[float] = None,
):
    """
    Build a randomly-generated reservoir network

    This function will return a reservoir built with `.FFRateEuler` and `.RecRateEuler` layers, encapsulating a rate-based recurrent reservoir network, with current-based inputs and outputs. The input, recurrent and output weights will be set randomly, according to the defined parameters.

    :param int input_size:            Input size (M). Default: ``1``
    :param int reservoir_size:        Recurrent size (N). Default: ``100``
    :param int output_size:           Output size (O). Default: ``1``
    :param float weights_in_std:      Std. dev. of input weights. Default: ``1.``
    :param float weights_res_std:     Std. dev. of recurrent weights. Default: ``1.``
    :param float weights_out_std:     Std. dev. of output weights. Default: ``1.``
    :param float weights_in_mean:     Mean input weight. Default: ``0.``
    :param float weights_res_mean:    Mean recurrent weight. Default: ``0.``
    :param float weights_out_mean:    Mean output weight. Default: ``0.``
    :param ArrayLike tau_in:          Input layer neuron time constants [M,]. Default: ``1.``
    :param ArrayLike tau_res:         Recurrent layer neuron time constants [N,]. Default: ``1.``
    :param ArrayLike bias_in:         Input layer neuron bias currents [M,]. Default: ``0.``
    :param ArrayLike bias_res:        Recurrent layer neuron bias currents [N,]. Default: ``0.``
    :param float dt:                  Time step for all layers. Default: ``0.1``
    :param Optional[float] noise_std_in:        Noise injected into input layer. Default: ``None``
    :param Optional[float] noise_std_res:       Noise injected into recurrent layer. Default: ``None``
    :param Optional[float] noise_std_out:       Noise injected into output layer. Default: ``None``

    :return Network:                            A randomly-generated firing-rate reservoir network
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
