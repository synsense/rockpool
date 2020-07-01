"""
Network class for implementing networks that can learn arbitrary dynamical systems (see https://arxiv.org/pdf/1705.08026.pdf for more information)
Author: Julian Buechel
Note that ADS stands for Arbitrary Dynamical System
"""
import numpy as np
from ..network import Network
from ...layers import PassThrough, FFExpSyn, RecFSSpikeADS
from ...timeseries import TSContinuous
import json

from typing import Union, Callable, Tuple, List

from tqdm import tqdm

import matplotlib
matplotlib.rc('font', family='Times New Roman')
matplotlib.rc('text')
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
import matplotlib.pyplot as plt

import time

__all__ = ["NetworkADS"]

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=0), axis=0) 
    return (cumsum[N:,:] - cumsum[:-N,:]) / float(N)

def pISI_variance(sim_result):
    """
    Compute the variance of the population inter-spike intervals
    Parameters:
        sim_result : Object of type evolution Object that was returned after having called evolve
    Returns:
        variance of difference array (variance of pISI) in milliseconds
    """
    times_c = sim_result['lyrRes'].times[sim_result['lyrRes'].channels > -1]
    np.sort(times_c) # Sorts in ascending order
    diff = np.diff(times_c)
    return np.sqrt(np.var(diff * 1000))

class NetworkADS(Network):

    def __init__(self):
        super().__init__()

    @staticmethod
    def load(filename: str) -> "Network":
        """
        Load a network from a JSON file

        :param str filename:    filename of a JSON filr that contains a saved network
        :return Network:        A network object with all the layers loaded from `filename`
        """
        # - Load dict holding the parameters
        with open(filename, "r") as f:
            loaddict: dict = json.load(f)

        return NetworkADS.load_from_dict(loaddict)

    @staticmethod
    def load_from_dict(config: dict, **kwargs):
        config = dict(config, **kwargs)
        conf = config["layers"][1]
        conf["tau_syn_r_out"] = config["layers"][2]["tau_syn"]
        return NetworkADS.SpecifyNetwork(**conf)

    @staticmethod
    def SpecifyNetwork(N : int,
                        Nc : int,
                        Nb : int,
                        weights_in : np.ndarray,
                        weights_out : np.ndarray,
                        weights_fast : np.ndarray,
                        weights_slow : np.ndarray,
                        eta : float,
                        k : float,
                        noise_std : float,
                        dt : float,
                        bias : np.ndarray = 0.0,
                        v_thresh: Union[np.ndarray, float] = -0.055,
                        v_reset: Union[np.ndarray, float] = -0.065,
                        v_rest: Union[np.ndarray, float] = -0.065,
                        tau_mem: float = 0.02,
                        tau_syn_r_fast: float = 0.001,
                        tau_syn_r_slow: float = 0.1,
                        tau_syn_r_out: float = 0.1,
                        refractory: float = -np.finfo(float).eps,
                        discretize=-1,
                        discretize_dynapse=False,
                        record: bool = False,
                        **kwargs,
                        ):


        """
        :brief Creates a network for learning 

        :param Nc : Dimension of input signal
        :param N : Number of neurons in the recurrent layer
        :param Nb : Number of basis functions used
        :param weights_in : [Nc,N] Weights that connect the input to the recurrent population
        :param weights_out : [N,Nc] Weights that reconstruct desired output x from r of the netwok (x_hat = Dr), typically set to weights_in.T
        :param weights_fast : [N,N] Recurrent weights for tight E/I-balance
        :param weights_slow : [Nb,N] Weight matrix relating psi(r) to the neurons
        :param theta : [Nb,1] Bias in the non-linear activation function
        :param eta : Learning rate for adjusting weights_slow
        :param k : Gain of error feedback during learning. Typically decreased while learning for generalization
        :param noise_std : Standard deviation of noise added to the neurons membrane potentials
        :param phi : Non-linear function of type ndarray -> ndarray. Implements tanh per default

        :return : NetworkADS
        """

        # Assertions for checking the dimensions
        assert (np.asarray(weights_in).shape == (Nc,N)), ("Input matrix has shape %s but should have shape (%d,%d)" % (str(np.asarray(weights_in).shape),N,Nc))
        assert (np.asarray(weights_out).shape == (N,Nc)), ("Output matrix has shape %s but should have shape (%d,%d)" % (str(np.asarray(weights_out).shape),Nc,N))
        assert (np.asarray(weights_fast).shape == (N,N)), ("Fast recurrent matrix has shape %s but should have shape (%d,%d)" % (str(np.asarray(weights_fast).shape),N,N))
        assert (np.asarray(weights_slow).shape == (Nb,N)), ("Slow recurrent matrix has shape %s but should have shape (%d,%d)" % (str(np.asarray(weights_slow).shape),Nb,N))

        ads_layer = RecFSSpikeADS(weights_fast=np.asarray(weights_fast).astype("float"), weights_slow=np.asarray(weights_slow).astype("float"), weights_out = np.asarray(weights_out).astype("float"), weights_in=np.asarray(weights_in).astype("float"),
                                    eta=eta,k=k,bias=np.asarray(bias).astype("float"),noise_std=noise_std,
                                    dt=dt,v_thresh=np.asarray(v_thresh).astype("float"),v_reset=np.asarray(v_reset).astype("float"),v_rest=np.asarray(v_rest).astype("float"),
                                    tau_mem=tau_mem,tau_syn_r_fast=tau_syn_r_fast,tau_syn_r_slow=tau_syn_r_slow, tau_syn_r_out=tau_syn_r_out,
                                    refractory=refractory,record=record,name="lyrRes",discretize=discretize,discretize_dynapse=discretize_dynapse)

        input_layer = PassThrough(np.asarray(weights_in).astype("float"), dt=dt, noise_std=noise_std, name="input_layer")

        output_layer = FFExpSyn(
            np.asarray(weights_out).astype("float"),
            dt=dt,
            noise_std=noise_std,
            tau_syn=tau_syn_r_out,
            name="output_layer"
        )

        net_ads = NetworkADS()
        net_ads.input_layer = net_ads.add_layer(input_layer, external_input=True) # - External -> Input
        net_ads.lyrRes = net_ads.add_layer(ads_layer, input_layer) # - Input -> ADS
        net_ads.output_layer = net_ads.add_layer(output_layer, ads_layer) # - ADS -> Output

        return net_ads

    def train_step(self, ts_input, ts_target, k : float, eta : float, verbose : bool):
        """
        :brief Do a training evolution on the passed input. This will update the weights and reset the network afterwards.
        :param ts_input : TSContinuous input that the network is evolved over
        :param ts_target : TSContinuous data used to compute the error over time
        :param k : float : Scaling factor of the current which is fed back into the system during learning
        :param eta : float : Learning rate
        :param verbose : bool : Whether to evolve in verbose mode or not. If set to true, will plot various data after each evolution 
        """
        # - Do a training step
        self.lyrRes.is_training = True
        # - Adapt learning rate and k if it's time
        self.lyrRes.k = k
        self.lyrRes.eta = eta

        # - Set the target
        self.lyrRes.ts_target = ts_target
        
        train_sim = self.evolve(ts_input=ts_input, verbose=verbose)
        self.reset_all()
        self.lyrRes.is_training = False
        self.lyrRes.ts_target = None
        
        return train_sim



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
