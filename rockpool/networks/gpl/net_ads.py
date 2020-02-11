"""
Network class for implementing networks that can learn arbitrary dynamical systems (see https://arxiv.org/pdf/1705.08026.pdf for more information)
Author: Julian Buechel
Note that ADS stands for Arbitrary Dynamical System
"""
import numpy as np
from ..network import Network
from ...layers import PassThrough, FFExpSyn, RecFSSpikeADS

from typing import Union, Callable

#! Need to import RecFSSpikeADS

__all__ = ["NetworkADS"]

class NetworkADS(Network):

    def __init__(self):
        super().__init__()

    @staticmethod
    def SpecifyNetwork(N : int,
                        Nc : int,
                        Nb : int,
                        weights_in : np.ndarray,
                        weights_out : np.ndarray,
                        weights_fast : np.ndarray,
                        weights_slow : np.ndarray,
                        M : np.ndarray,
                        theta : np.ndarray,
                        eta : float,
                        k : float,
                        noise_std : float,
                        dt : float,
                        v_thresh: Union[np.ndarray, float] = -0.055,
                        v_reset: Union[np.ndarray, float] = -0.065,
                        v_rest: Union[np.ndarray, float] = -0.065,
                        tau_mem: float = 0.02,
                        tau_syn_r_fast: float = 0.001,
                        tau_syn_r_slow: float = 0.1,
                        tau_syn_out: float = 0.1,
                        refractory: float = -np.finfo(float).eps,
                        phi : Callable[[np.ndarray],np.ndarray] = lambda x : np.tanh(x)):


        """
        @brief
            Creates a network for learning 

        @params
            Nc : Dimension of input signal
            N : Number of neurons in the recurrent layer
            Nb : Number of basis functions used

            The above are passed explicitly to ensure that the desired network size matches the matrix shapes

            weights_in : [Nc,N] Weights that connect the input to the recurrent population
            weights_out : [N,Nc] Weights that reconstruct desired output x from r of the netwok (x_hat = Dr), typically set to weights_in.T
            weights_fast : [N,N] Recurrent weights for tight E/I-balance
            weights_slow : [Nb,N] Weight matrix relating psi(r) to the neurons
            theta : [Mb,1] Bias in the non-linear activation function
            eta : Learning rate for adjusting weights_slow
            k : Gain of error feedback during learning. Typically decreased while learning for generalization
            noise_std : Standard deviation of noise added to the neurons membrane potentials
            phi : Non-linear function of type ndarray -> ndarray. Implements tanh per default
        """

        # Assertions for checking the dimensions
        assert (weights_in.shape == (Nc,N)), ("Input matrix has shape %s but should have shape (%d,%d)" % (str(weights_in.shape),N,Nc))
        assert (weights_out.shape == (N,Nc)), ("Output matrix has shape %s but should have shape (%d,%d)" % (str(weights_out.shape),Nc,N))
        assert (weights_fast.shape == (N,N)), ("Fast recurrent matrix has shape %s but should have shape (%d,%d)" % (str(weights_fast.shape),N,N))
        assert (weights_slow.shape == (Nb,N)), ("Slow recurrent matrix has shape %s but should have shape (%d,%d)" % (str(weights_slow.shape),Nb,N))
        assert (theta.shape == (Nb,1) or theta.shape == (Nb,)), ("Theta has shape %s but should have shape (%d,1)" % (str(theta.shape),Nb))

        ads_layer = RecFSSpikeADS(weights_fast=weights_fast, weights_slow=weights_slow,
                                    M=M,theta=theta,eta=eta,k=k,noise_std=noise_std,
                                    dt=dt,v_thresh=v_thresh,v_reset=v_reset,v_rest=v_rest,
                                    tau_mem=tau_mem,tau_syn_r_fast=tau_syn_r_fast,tau_syn_r_slow=tau_syn_r_slow,
                                    refractory=refractory,phi=phi,name="ADS-Layer")

        # Input layer
        input_layer = PassThrough(weights_in, dt=dt, noise_std=noise_std, name="Input")

        # Output layer
        output_layer = FFExpSyn(
            weights_out,
            dt=0.1e-4,
            noise_std=noise_std,
            tau_syn=tau_syn_out,
            name="Output"
        )

        net_ads = NetworkADS()
        net_ads.input_layer = net_ads.add_layer(input_layer, external_input=True) # External -> Input
        net_ads.lyrRes = net_ads.add_layer(ads_layer, input_layer) # Input -> ADS
        net_ads.output_layer = net_ads.add_layer(output_layer, ads_layer) # ADS -> Output

        return net_ads


def _expand_to_size(inp, size: int, var_name: str = "input") -> np.ndarray:
    """
    _expand_to_size: Replicate out a scalar to a desired size

    :param inp:          scalar or array-like (N)
    :param size:           int Desired size to return (=N)
    :param var_name:   str Name of the variable to include in error messages
    :return:                np.ndarray (N) vector
    """
    if np.size(inp) == 1:
        # - Expand input to vector
        inp = np.repeat(inp, size)

    assert np.size(inp) == size, "`{}` must be a scalar or have {} elements".format(
        var_name, size
    )

    # - Return object of correct shape
    return np.reshape(inp, size)