"""
Implement the layer for the NetworkADS (Arbitrary Dynamical System), which is capable of learning
an arbitrary dynamical system.
"""

from ..layer import Layer
from rockpool.timeseries import TSEvent, TSContinuous
import numpy as np
from typing import Union, Callable, Any, Tuple, Optional
import copy
from numba import njit

import matplotlib
matplotlib.rc('font', family='Times New Roman')
matplotlib.rc('text')
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
import matplotlib.pyplot as plt # For quick plottings

__all__ = ["RecFSSpikeADS"]

class RecFSSpikeADS(Layer):
    """
    Implement the layer for the NetworkADS (Arbitrary Dynamical System), which is capable of learning
    an arbitrary dynamical system.
    See rockpool/networks/gpl/net_as.py for the parameters passed here.
    """
    def __init__(self,
                weights_fast : np.ndarray,
                weights_slow : np.ndarray,
                M : np.ndarray,
                theta : np.ndarray,
                eta : float,
                k : float,
                noise_std : float,
                dt : float,
                v_thresh : Union[np.ndarray,float],
                v_reset : Union[np.ndarray,float],
                v_rest : Union[np.ndarray,float],
                tau_mem : float,
                tau_syn_r_fast : float,
                tau_syn_r_slow : float,
                refractory : float,
                phi : Callable[[np.ndarray],np.ndarray],
                name : str):
        
        super().__init__(weights=weights_fast, noise_std=noise_std, name=name)

        # Fast weights, noise_std and name are access. via self.XX or self._XX
        self.weights_slow = weights_slow
        self.M = M
        self.theta = theta
        self.eta = eta
        self.k = k
        self.v_thresh = np.asarray(v_thresh).astype("float")
        self.v_reset = np.asarray(v_reset).astype("float")
        self.v_rest = np.asarray(v_rest).astype("float")
        self.tau_mem = np.asarray(tau_mem).astype("float")
        self.tau_syn_r_fast = np.asarray(tau_syn_r_fast).astype("float")
        self.tau_syn_r_slow = np.asarray(tau_syn_r_slow).astype("float")
        self.refractory = float(refractory)
        self.phi = phi
        self.learning_callback = None
        self.is_training = False
        self._ts_target = None

        # - Set a reasonable dt
        if dt is None:
            self.dt = self._min_tau / 10
        else:
            self.dt = np.asarray(dt).astype("float")

        # Initialise the network
        self.reset_all()

    def reset_state(self):
        """
        Reset the internal state of the network
        """
        self.state = self.v_rest.copy()
        self.I_s_S = np.zeros(self.size)
        self.I_s_F = np.zeros(self.size)


    def evolve(self,
                ts_input: Optional[TSContinuous] = None,
                duration: Optional[float] = None,
                num_timesteps: Optional[int] = None,
                verbose: bool = False,
                min_delta: Optional[float] = None,) -> TSEvent:
        """
        Evolve the function on the input c(t). This function simply feeds the input through the network and does not perform any learning
        """

        # This is a dummy evolution so no errors are produced when executing net.evolve(..)
        self.t = num_timesteps * self.dt
        return TSEvent([0.5, 0.6],[0, 0])

    def to_dict(self):
        NotImplemented

    @property
    def _min_tau(self):
        """
        (float) Smallest time constant of the layer
        """
        return min(np.min(self.tau_syn_r_slow), np.min(self.tau_syn_r_fast))

    @property
    def output_type(self):
        """ (`TSEvent`) Output `TimeSeries` class (`TSEvent`) """
        return TSEvent


    @property
    def tau_syn_r_f(self):
        """ (float) Fast synaptic time constant (s) """
        return self.__tau_syn_r_f

    @tau_syn_r_f.setter
    def tau_syn_r_f(self, tau_syn_r_f):
        self.__tau_syn_r_f = self._expand_to_net_size(tau_syn_r_f, "tau_syn_r_f")

    @property
    def tau_syn_r_s(self):
        """ (float) Slow synaptic time constant (s) """
        return self.__tau_syn_r_s

    @tau_syn_r_s.setter
    def tau_syn_r_s(self, tau_syn_r_s):
        self.__tau_syn_r_s = self._expand_to_net_size(tau_syn_r_s, "tau_syn_r_s")

    @property
    def v_thresh(self):
        """ (float) Threshold potential """
        return self.__thresh

    @v_thresh.setter
    def v_thresh(self, v_thresh):
        self.__thresh = self._expand_to_net_size(v_thresh, "v_thresh")

    @property
    def v_rest(self):
        """ (float) Resting potential """
        return self.__rest

    @v_rest.setter
    def v_rest(self, v_rest):
        self.__rest = self._expand_to_net_size(v_rest, "v_rest")

    @property
    def v_reset(self):
        """ (float) Reset potential"""
        return self.__reset

    @v_reset.setter
    def v_reset(self, v_reset):
        self.__reset = self._expand_to_net_size(v_reset, "v_reset")

    @Layer.dt.setter
    def dt(self, new_dt):
        assert (
            new_dt <= self._min_tau / 10
        ), "`new_dt` must be shorter than 1/10 of the shortest time constant, for numerical stability."

        # - Call super-class setter
        super(RecFSSpikeADS, RecFSSpikeADS).dt.__set__(self, new_dt)

    @property
    def ts_target(self):
        return self._ts_target

    # TODO Need to implement a setter for self.ts_target :TSContinuous
    @ts_target.setter
    def ts_target(self, t):
        self._ts_target = t
        print("ts_target setter called")



###### Convenience functions

# - Convenience method to return a nan array
def full_nan(shape: Union[tuple, int]):
    a = np.empty(shape)
    a.fill(np.nan)
    return a


### --- Compiled concenience functions


@njit
def min_argmin(data: np.ndarray) -> Tuple[float, int]:
    """
    Accelerated function to find minimum and location of minimum

    :param data:  np.ndarray of data

    :return (float, int):        min_val, min_loc
    """
    n = 0
    min_loc = -1
    min_val = np.inf
    for x in data:
        if x < min_val:
            min_loc = n
            min_val = x
        n += 1

    return min_val, min_loc


@njit
def argwhere(data: np.ndarray) -> list:
    """
    Accelerated argwhere function

    :param np.ndarray data:  Boolean array

    :return list:         vnLocations where data = True
    """
    vnLocs = []
    n = 0
    for val in data:
        if val:
            vnLocs.append(n)
        n += 1

    return vnLocs


@njit
def clip_vector(v: np.ndarray, f_min: float, f_max: float) -> np.ndarray:
    """
    Accelerated vector clip function

    :param np.ndarray v:
    :param float min:
    :param float max:

    :return np.ndarray: Clipped vector
    """
    v[v < f_min] = f_min
    v[v > f_max] = f_max
    return v


@njit
def clip_scalar(val: float, f_min: float, f_max: float) -> float:
    """
    Accelerated scalar clip function

    :param float val:
    :param float min:
    :param float max:

    :return float: Clipped value
    """
    if val < f_min:
        return f_min
    elif val > f_max:
        return f_max
    else:
        return val


def rep_to_net_size(data: Any, size: Tuple):
    """
    Repeat some data to match the layer size

    :param Any data:
    :param Tuple size:

    :return np.ndarray:
    """
    if np.size(data) == 1:
        return np.repeat(data, size)
    else:
        return data.flatten()