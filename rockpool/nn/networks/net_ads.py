"""
Network class for implementing networks that can learn arbitrary dynamical systems (see https://arxiv.org/pdf/1705.08026.pdf for more information)
Author: Julian Buechel
Note that ADS stands for Arbitrary Dynamical System
"""

raise ImportError("This module needs to be ported to teh v2 API.")


import numpy as np
from rockpool.nn.networks.network import Network
from rockpool.nn.layers import PassThrough, FFExpSyn, RecFSSpikeADS
from rockpool.timeseries import TSContinuous

import json

from typing import Union

__all__ = ["NetworkADS"]


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=0), axis=0)
    return (cumsum[N:, :] - cumsum[:-N, :]) / float(N)


def pISI_variance(sim_result):
    """
    Compute the variance of the population inter-spike intervals
    Parameters:
        sim_result : Object of type evolution Object that was returned after having called evolve
    Returns:
        variance of difference array (variance of pISI) in milliseconds
    """
    times_c = sim_result["lyrRes"].times[sim_result["lyrRes"].channels > -1]
    np.sort(times_c)  # Sorts in ascending order
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
        """
        Load NetworkADS from dictionary

        :param dict config: Dictionary containing parameters obtained from JSON file
        """
        config = dict(config, **kwargs)
        conf = config["layers"][1]
        conf["tau_syn_r_out"] = config["layers"][2]["tau_syn"]
        return NetworkADS.SpecifyNetwork(**conf)

    @staticmethod
    def SpecifyNetwork(
        N: int,
        Nc: int,
        Nb: int,
        weights_in: np.ndarray,
        weights_out: np.ndarray,
        weights_fast: np.ndarray,
        weights_slow: np.ndarray,
        eta: float,
        k: float,
        noise_std: float,
        dt: float,
        bias: np.ndarray = 0.0,
        v_thresh: Union[np.ndarray, float] = 1.0,
        v_reset: Union[np.ndarray, float] = 0.0,
        v_rest: Union[np.ndarray, float] = 0.5,
        tau_mem: float = 0.05,
        tau_syn_r_fast: float = 0.07,
        tau_syn_r_slow: float = 0.07,
        tau_syn_r_out: float = 0.07,
        refractory: float = -np.finfo(float).eps,
        discretize=-1,
        discretize_dynapse=False,
        record: bool = True,
        **kwargs,
    ):

        """
        Create NetworkADS instance that can be trained to learn the dynamics of a rate based network

        :param int Nc: Dimension of input signal
        :param int N: Number of neurons in the recurrent layer
        :param int Nb: Number of basis functions used (=N)
        :param ndarray weights_in: [NcxN] matrix that projects current input into the network [not trained]
        :param ndarray weights_out: [NxNc] matrix for reading out the target [not trained]
        :param ndarray weights_fast: [NxN] matrix implementing the balanced network. Predefined given ffwd matrix (see tutorial)
        :param ndarray weights_slow: [NxN] learnable recurrent matrix implementing dynamics for the task
        :param float eta: Learning rate
        :param float k: Scaling factor determining the magnitude of error-current that is fed back into the system
        :param float noise_std: Standard deviation of Gaussian (zero mean) noise applied
        :param float dt: Euler integration timestep
        :param [ndarray,float] bias: Bias applied to the neurons membrane potential
        :param [ndarray,float] v_thresh: Spiking threshold typically at 1. Caution: Potentials are clipped at zero, meaning there can not be negative potentials
        :param [ndarray,float] v_reset: Reset potential typically at 0
        :param [ndarray,float] v_rest: Resting potential typically at v_thresh/2
        :param float tau_mem: Membrane time constant
        :param float tau_syn_r_fast: Synaptic time constant of fast connections 0.07s
        :param float tau_syn_r_slow: Synaptic time constant of slow connections typically 0.07s
        :param float tau_syn_r_out: Synaptic time constant of output filter typically 0.07s
        :param float refractory: Refractory period in seconds. Typically set to 0
        :param int discretize: Number of distinctive weights used at all times. E.g. 8 would mean a 3 bit resolution. discretize_dynapse must be set False
        :param bool discretize_dynapse: If set to True, the constraints of the DYNAP-SE II are imposed on the slow recurrent weight matrix

        :return NetworkADS: Trainable network of size N
        """

        # Assertions for checking the dimensions
        assert np.asarray(weights_in).shape == (
            Nc,
            N,
        ), "Input matrix has shape %s but should have shape (%d,%d)" % (
            str(np.asarray(weights_in).shape),
            N,
            Nc,
        )
        assert np.asarray(weights_out).shape == (
            N,
            Nc,
        ), "Output matrix has shape %s but should have shape (%d,%d)" % (
            str(np.asarray(weights_out).shape),
            Nc,
            N,
        )
        assert np.asarray(weights_fast).shape == (
            N,
            N,
        ), "Fast recurrent matrix has shape %s but should have shape (%d,%d)" % (
            str(np.asarray(weights_fast).shape),
            N,
            N,
        )
        assert np.asarray(weights_slow).shape == (
            Nb,
            N,
        ), "Slow recurrent matrix has shape %s but should have shape (%d,%d)" % (
            str(np.asarray(weights_slow).shape),
            Nb,
            N,
        )

        ads_layer = RecFSSpikeADS(
            weights_fast=np.asarray(weights_fast).astype("float"),
            weights_slow=np.asarray(weights_slow).astype("float"),
            weights_out=np.asarray(weights_out).astype("float"),
            weights_in=np.asarray(weights_in).astype("float"),
            eta=eta,
            k=k,
            bias=np.asarray(bias).astype("float"),
            noise_std=noise_std,
            dt=dt,
            v_thresh=np.asarray(v_thresh).astype("float"),
            v_reset=np.asarray(v_reset).astype("float"),
            v_rest=np.asarray(v_rest).astype("float"),
            tau_mem=tau_mem,
            tau_syn_r_fast=tau_syn_r_fast,
            tau_syn_r_slow=tau_syn_r_slow,
            tau_syn_r_out=tau_syn_r_out,
            refractory=refractory,
            record=record,
            name="lyrRes",
            discretize=discretize,
            discretize_dynapse=discretize_dynapse,
        )

        input_layer = PassThrough(
            np.asarray(weights_in).astype("float"),
            dt=dt,
            noise_std=noise_std,
            name="input_layer",
        )

        output_layer = FFExpSyn(
            np.asarray(weights_out).astype("float"),
            dt=dt,
            noise_std=noise_std,
            tau_syn=tau_syn_r_out,
            name="output_layer",
        )

        net_ads = NetworkADS()
        net_ads.input_layer = net_ads.add_layer(
            input_layer, external_input=True
        )  # - External -> Input
        net_ads.lyrRes = net_ads.add_layer(ads_layer, input_layer)  # - Input -> ADS
        net_ads.output_layer = net_ads.add_layer(
            output_layer, ads_layer
        )  # - ADS -> Output

        return net_ads

    def train_step(self, ts_input, ts_target, k: float, eta: float, verbose: bool):
        """
        Do a training evolution on the passed input. This will update the weights and reset the network afterwards

        :param TSContinuous ts_input: TSContinuous input that the network is evolved over
        :param TSContinuous ts_target: data used to compute the error over time
        :param float k: Scaling factor of the current which is fed back into the system during learning
        :param float eta: Learning rate
        :param bool verbose: Whether to evolve in verbose mode or not. If set to true, will plot various data after each evolution

        :return dict: Dictionary containing recorded states and outputs per layer
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
