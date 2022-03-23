###
# net_deneve.py - Classes and functions for encapsulating Denève reservoirs
###

raise ImportError("This module needs to be ported to the v2 API.")

from rockpool.nn.networks.network import Network
from rockpool.nn.layers import PassThrough, FFExpSyn
from rockpool.nn.layers.spike_bt import RecFSSpikeEulerBT

import numpy as np

# - Configure exports
__all__ = ["NetworkDeneve"]


class NetworkDeneve(Network):
    """
    `.Network` base class that defines and builds networks of Deneve reservoirs to solve linear problems.

    Use :py:meth:`~.NetworkDeneve.SolveLinearProblem` or :py:meth:`~.NetworkDeneve.SpecifyNetwork` to build a Deneve tight balance recurrent spiking network.
    """

    def __init__(self, **kwargs):
        # - Call super-class constructor
        super().__init__(**kwargs)

    @staticmethod
    def SolveLinearProblem(
        a: np.ndarray = None,
        net_size: int = None,
        gamma: np.ndarray = None,
        dt: float = None,
        mu: float = 1e-4,
        nu: float = 1e-3,
        noise_std: float = 0.0,
        tau_mem: float = 20e-3,
        tau_syn_fast: float = 1e-3,
        tau_syn_slow: float = 100e-3,
        v_thresh: np.ndarray = -55e-3,
        v_rest: np.ndarray = -65e-3,
        refractory: float = -np.finfo(float).eps,
    ) -> "NetworkDeneve":
        """
        Direct implementation of a linear dynamical system

        :param np.ndarray a:        [PxP] Matrix defining linear dynamical system
        :param int net_size:        Desired size of recurrent reservoir layer (Default: 100)

        :param np.ndarray gamma:    [PxN] Input kernel (Default: 50 * Normal / net_size)

        :param float dt:            Nominal time step (Default: 0.1 ms)

        :param float mu:            Linear cost parameter (Default: 1e-4)
        :param float nu:            Quadratic cost parameter (Default: 1e-3)

        :param float noise_std:     Noise std. dev. (Default: 0.)

        :param float tau_mem:       Neuron membrane time constant (Default: 20 ms)
        :param float tau_syn_fast:  Fast synaptic time constant (Default: 1 ms)
        :param float tau_syn_slow:  Slow synaptic time constant (Default: 100 ms)

        :param float v_thresh:      Threshold membrane potential (Default: -55 mV)
        :param float v_rest:        Rest potential (Default: -65 mV)
        :param float refractory:    Refractory time for neuron spiking (Default: 0 ms)

        :return `.NetworkDeneve`:   A configured `.NetworkDeneve` object, containing input, reservoir and output layers
        """
        # - Get input parameters
        nJ = a.shape[0]

        # - Generate random input weights if not provided
        if gamma is None:
            # - Default net size
            if net_size is None:
                net_size = 100

            # - Create random input kernels
            gamma = np.random.randn(nJ, net_size)
            gamma /= np.sum(np.abs(gamma), 0, keepdims=True)
            gamma /= net_size
            gamma *= 50

        else:
            assert (net_size is None) or net_size == gamma.shape[
                1
            ], "`net_size` must match the size of `gamma`."
            net_size = gamma.shape[1]

        # - Generate network
        lambda_d = np.asarray(1 / tau_syn_slow)

        v_t = (
            nu * lambda_d
            + mu * lambda_d**2
            + np.sum(abs(gamma.T), -1, keepdims=True) ** 2
        ) / 2

        omega_f = gamma.T @ gamma + mu * lambda_d**2 * np.identity(net_size)
        omega_s = gamma.T @ (a + lambda_d * np.identity(nJ)) @ gamma

        # - Scale problem to arbitrary membrane potential ranges and time constants
        t_dash = _expand_to_size(v_thresh, net_size)
        b = np.reshape(_expand_to_size(v_rest, net_size), (net_size, -1))
        a = np.reshape(_expand_to_size(v_thresh, net_size), (net_size, -1)) - b

        gamma_dash = a * gamma.T / v_t

        omega_f_dash = a * omega_f / v_t
        omega_s_dash = a * omega_s / v_t

        # - Pull out reset voltage from fast synaptic weights
        v_reset = t_dash - np.diag(omega_f_dash)
        np.fill_diagonal(omega_f_dash, 0)

        # - Scale omega_f_dash by fast TC
        omega_f_dash /= tau_syn_fast

        # - Scale everything by membrane TC
        omega_f_dash *= tau_mem
        omega_s_dash *= tau_mem
        gamma_dash *= tau_mem

        # - Build and return network
        return NetworkDeneve.SpecifyNetwork(
            weights_fast=-omega_f_dash,
            weights_slow=omega_s_dash,
            weights_in=gamma_dash.T,
            weights_out=gamma.T,
            dt=dt,
            noise_std=noise_std,
            v_rest=v_rest,
            v_reset=v_reset,
            v_thresh=v_thresh,
            tau_mem=tau_mem,
            tau_syn_r_fast=tau_syn_fast,
            tau_syn_r_slow=tau_syn_slow,
            tau_syn_out=tau_syn_slow,
            refractory=refractory,
        )

    @staticmethod
    def SpecifyNetwork(
        weights_fast,
        weights_slow,
        weights_in,
        weights_out,
        dt: float = None,
        noise_std: float = 0.0,
        v_thresh: np.ndarray = -55e-3,
        v_reset: np.ndarray = -65e-3,
        v_rest: np.ndarray = -65e-3,
        tau_mem: float = 20e-3,
        tau_syn_r_fast: float = 1e-3,
        tau_syn_r_slow: float = 100e-3,
        tau_syn_out: float = 100e-3,
        refractory: float = -np.finfo(float).eps,
    ):
        """
        Directly configure all layers of a reservoir

        :param np.ndarray weights_fast: [NxN] Matrix of fast synaptic weights
        :param np.ndarray weights_slow: [NxN] Matrix of slow synaptic weights
        :param np.ndarray weights_in:   [LxN] Matrix of input kernels
        :param np.ndarray weights_out:  [NxM] Matrix of output kernels

        :param float dt:                Nominal time step
        :param float noise_std:         Noise Std. Dev.

        :param np.ndarray v_rest:       [Nx1] Vector of rest potentials (spiking layer)
        :param np.ndarray v_reset:      [Nx1] Vector of reset potentials (spiking layer)
        :param np.ndarray v_thresh:     [Nx1] Vector of threshold potentials (spiking layer)

        :param float tau_mem:           Neuron membrane time constant (spiking layer)
        :param float tau_syn_r_fast:    Fast recurrent synaptic time constant
        :param float tau_syn_r_slow:    Slow recurrent synaptic time constant
        :param float tau_syn_out:       Synaptic time constant for output layer

        :param float refractory:        Refractory time for spiking layer

        :return `.NetworkDeneve`:       Configured `.NetworkDeneve` spiking network
        """
        # - Construct reservoir
        reservoir_layer = RecFSSpikeEulerBT(
            weights_fast,
            weights_slow,
            dt=dt,
            noise_std=noise_std,
            tau_mem=tau_mem,
            tau_syn_r_fast=tau_syn_r_fast,
            tau_syn_r_slow=tau_syn_r_slow,
            v_thresh=v_thresh,
            v_rest=v_rest,
            v_reset=v_reset,
            refractory=refractory,
            name="Deneve_Reservoir",
        )

        # - Ensure time step is consistent across layers
        if dt is None:
            dt = reservoir_layer.dt

        # - Construct input layer
        input_layer = PassThrough(weights_in, dt=dt, noise_std=noise_std, name="Input")

        # - Construct output layer
        output_layer = FFExpSyn(
            weights_out,
            dt=0.1e-4,
            noise_std=noise_std,
            tau_syn=tau_syn_out,
            name="Output",
        )

        # - Build network
        net_deneve = NetworkDeneve()
        net_deneve.input_layer = net_deneve.add_layer(input_layer, external_input=True)
        net_deneve.lyrRes = net_deneve.add_layer(reservoir_layer, input_layer)
        net_deneve.output_layer = net_deneve.add_layer(output_layer, reservoir_layer)

        # - Return constructed network
        return net_deneve


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
