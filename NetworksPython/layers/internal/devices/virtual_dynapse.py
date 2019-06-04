##########
# virtual_dynapse.py - This module defines a Layer class that simulates a DynapSE
#                      processor. Its purpose is to provide an understanding of
#                      which operations are possible with the hardware. The
#                      implemented neuron model is a simplification of the actual
#                      circuits and therefore only serves as a rough approximation.
#                      Accordingly, hyperparameters such as time constants or
#                      baseweights give an idea on the parameters that can be set
#                      but there is no direct correspondence to the hardware biases.
# Author: Felix Bauer, aiCTX AG, felix.bauer@ai-ctx.com
# TODO: Mention that ways of extending connectivity exist, but complex.
##########

### --- Imports

# Built-in modules
from warnings import warn
from typing import Tuple, List, Union, Optional

# Third-party modules
import numpy as np

# NetworksPython modules
from NetworksPython.layers import Layer, ArrayLike, RecIAFSpkInNest
from . import params

### --- Constants
CONNECTIONS_VALID = 0
FANIN_EXCEEDED = 1
FANOUT_EXCEEDED = 2

### --- Class definition


class VirtualDynapse(Layer):

    self._num_chips = params.NUM_CHIPS
    self._num_cores_chip = params.NUM_CORES_CHIP
    self._num_neurons_core = params.NUM_NEURONS_CORE

    def __init__(
        self,
        dt: float = 1e-5,
        connections: Union[np.ndarray, dict, None] = None,
        tau_mem_1=0.05,  # - Array of size 16
        tau_mem_2=0.05,
        tau_syn_e=0.05,
        tau_syn_i=0.05,
        baseweight_e=0.05,
        baseweight_i=0.05,
        bias=0,
        t_refractory=0.005,
        threshold=0.01,
        has_tau2=False,  # - Binary array with size number of neurons
        name: str = "unnamed",
    ):
        # - Set up weights
        self.baseweight_e = baseweight_e
        self.baseweight_i = baseweight_i
        self.connections = connections

        # - Instantiate super class.
        super().__init__(weights=self.weights, dt=dt, name=name)

        # - Store remaining parameters
        self.bias = bias
        self.tau_mem_1 = tau_mem_1
        self.tau_mem_2 = tau_mem_2
        self.tau_syn_e = tau_syn_e
        self.tau_syn_i = tau_syn_i
        self.t_refractory = t_refractory
        self.threshold = threshold
        self.has_tau2 = has_tau2

    def _generate_simulator(self):
        # - Nest-layer for approximate simulation of neuron dynamics
        self.simulator = RecIAFSpkInNest(
            weights_in=self.weights_in,
            weights_rec=self.weights_rec,
            bias=bias,
            tau_n=self.tau_n,
            tau_s=self.tau_s,
            tau_s_inh=self.tau_s_inh,
            dt=dt,
        )

    def set_connections(
        self,
        connections: Union[dict, np.ndarray],
        neurons_pre: np.ndarray,
        neurons_post: Optional[np.ndarray] = None,
        add: bool = False,
    ):
        """
        set_connections - Set connections between specific neuron populations.
                          Verify that connections are supported by the hardware.
        :params connections:  2D np.ndarray: Will assume positive (negative) values
                              correspond to excitatory (inhibitory) synapses.
                              Axis 0 (1) corresponds to pre- (post-) synaptic neurons.
                              Sizes must match `neurons_pre` and `neurons_post`.
        :params neurons_pre:   Array-like with IDs of presynaptic neurons that `connections`
                               refer to. If None, use all neurons (from 0 to self.size - 1).
        :params neurons_post:  Array-like with IDs of postsynaptic neurons that `connections`
                               refer to. If None, use same IDs as presynaptic neurons.
        """

        if neurons_pre is None:
            neurons_pre = np.arange(self.size)

        if neurons_post is None:
            neuron_post = neurons_pre

        # - Complete connectivity array after adding new connections
        connections_new = self.connections.copy()

        # - Indices of specified neurons in full connectivity matrix
        idcs_row, idcs_col = np.meshgrid(neurons_pre, neurons_post, indexing="ij")

        if add:
            # - Add new connections to connectivity array
            connections_new[idcs_row, idcs_col] += connections
        else:
            # - Replace connections between given neurons with new ones
            connections_new[idcs_row, idcs_col] = connections

        self.connections = connections_new

    def validate_connections(
        self, connections: Union[np.ndarray, dict], verbose: bool
    ) -> int:
        """
        validate:
            - limited fan in, by summing connections over dim 0 and then dim 1
            - fan-out limit to 3 chips -> how to check? (sum over dim 0 and check rows ->
            np.unique(v_row // 16).size < 4
        """
        # - Get 2D matrix with number of connections between neurons
        conn_count_2d = self.get_connection_counts(connections)

        result = 0

        if verbose:
            print(self.start_print + "Testing provided connections:")

        # - Test fan-in
        exceeds_fanin: np.ndarray = (
            np.sum(conn_count_2d, axis=1) > params.NUM_CAMS_NEURON
        )
        if exceeds_fanin.any():
            result += FANIN_EXCEEDED
            if verbose:
                "\tFan-in ({}) exceeded for neurons: {}".format(
                    params.NUM_CAMS_NEURON, np.where(exceeds_fanin)[0]
                )
        # - Test fan-out
        # List with target chips for each (presynaptic) neuron
        tgtchip_list = [
            np.nonzero(row)[0] // params.NUM_NEURONS_CHIP for row in conn_count_2d
        ]
        # Number of different target chips per neuron
        nums_tgtchips = np.array([np.unique(tgtchips) for tgtchips in tgtchip_list])
        exceeds_fanout = nums_tgtchips > params.NUM_SRAMS_NEURON
        if exceeds_fanout.any():
            result += FANOUT_EXCEEDED
            if verbose:
                "\tFan-out ({}) exceeded for neurons: {}".format(
                    params.NUM_SRAMS_NEURON, np.where(exceeds_fanout)[0]
                )

        if verbose and result == CONNECTIONS_VALID:
            "\tConnections ok."

        return result

    def get_connection_counts(self, connections: Union[dict, np.ndarray]) -> np.ndarray:
        """
        get_connection_counts - From a dict or array of connections for different
                                synapse types, generate a 2D matrix with the total
                                number of connections between each neuron pair.
        :params connections:  2D np.ndarray: Will assume positive (negative) values
                              correspond to excitatory (inhibitory) synapses.
        :return:
            2D np.ndarray with total number of connections between neurons
        """
        return np.abs(connections)

    def evolve(self, ts_input, duration, num_timesteps, verbose):
        pass

    def _update_weights(self):
        """
        _update_weights - Update internal representation of weights by multiplying
                          baseweights with number of connections for each core and
                          synapse type.
        """
        connections_exc = np.clip(self.connections, 0, None)
        connections_inh = np.clip(self.connections, None, 0)
        factor_exc = np.repeat(self.baseweight_e, self.num_neurons_core)
        factor_inh = np.repeat(self.baseweight_i, self.num_neurons_core)
        self._weights = connections_exc * factor_exc + connections_inh * factor_inh

    def _process_parameter(
        self,
        parameter: Union[bool, int, float, ArrayLike],
        name: str,
        nonnegative: bool = True,
    ) -> np.ndarray:
        """
        _process_parameter - Reshape parameter to array of size `self.num_cores`.
                             If `nonnegative` is `True`, clip negative values to 0.
        :param parameter:    Parameter to be reshaped and possibly clipped.
        :param name:         Name of the paramter (for print statements).
        :param nonnegative:  If `True`, clip negative values to 0 and warn if
                             there are any.
        """
        parameter = self._expand_to_size(parameter, self.num_cores, name, False)
        if nonnegative:
            # - Make sure that parameters are nonnegative
            if (np.array(parameter) < 0).any():
                warn(
                    self.start_print
                    + f"`{name}` must be at least 0. Negative values are clipped to 0."
                )
            parameter = np.clip(parameter, 0, None)
        return parameter

    @property
    def connections(self):
        return self._vtRefractoryTime

    @connections.setter
    def connections(self, connections_new: Optional[np.ndarray]):
        if connections_new is None:
            # - Remove all connections
            self._connections = np.zeros((self.num_neurons, self.num_neurons))
            self._update_weights()

        else:
            # - Make sure that connections have match hardware specifications
            if (
                self.validate_connections(connections_new, verbose=True)
                != CONNECTIONS_VALID
            ):
                raise ValueError(
                    self.start_print + "Connections not compatible with hardware."
                )
            else:
                # - Update connections
                self._connections = connections_new
                # - Update weights accordingly
                self._update_weights()

    @property
    def weights(self):
        return self._weights

    @property
    def baseweight_e(self):
        return self._baseweight_e

    @baseweight_e.setter
    def baseweight_e(self, baseweight_new: Union[float, ArrayLike]):
        self._baseweight_e = self._process_parameter(
            baseweight_new, "baseweight_e", nonnegative=True
        )
        self._update_weights()

    @property
    def baseweight_i(self):
        return self._baseweight_i

    @baseweight_i.setter
    def baseweight_i(self, baseweight_new: Union[float, ArrayLike]):
        self._baseweight_i = self._process_parameter(
            baseweight_new, "baseweight_i", nonnegative=True
        )
        self._update_weights()

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias_new: Union[float, ArrayLike]):
        self._bias = self._process_parameter(bias_new, "bias", nonnegative=True)

    @property
    def t_refractory(self):
        return self._t_refractory

    @t_refractory.setter
    def t_refractory(self, t_refractory_new: Union[float, ArrayLike]):
        self._t_refractory = self._process_parameter(
            t_refractory_new, "t_refractory", nonnegative=True
        )

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, threshold_new: Union[float, ArrayLike]):
        self._threshold = self._process_parameter(
            threshold_new, "threshold", nonnegative=True
        )

    @property
    def tau_mem_1(self):
        return self._tau_mem_1

    @tau_mem_1.setter
    def tau_mem_1(self, tau_mem_1_new: Union[float, ArrayLike]):
        self._tau_mem_1 = self._process_parameter(
            tau_mem_1_new, "tau_mem_1", nonnegative=True
        )

    @property
    def tau_mem_2(self):
        return self._tau_mem_2

    @tau_mem_2.setter
    def tau_mem_2(self, tau_mem_2_new: Union[float, ArrayLike]):
        self._tau_mem_2 = self._process_parameter(
            tau_mem_2_new, "tau_mem_2", nonnegative=True
        )

    @property
    def tau_syn_e(self):
        return self._tau_syn_e

    @tau_syn_e.setter
    def tau_syn_e(self, tau_syn_e_new: Union[float, ArrayLike]):
        self._tau_syn_e = self._process_parameter(
            tau_syn_e_new, "tau_syn_e", nonnegative=True
        )

    @property
    def tau_syn_i(self):
        return self._tau_syn_i

    @tau_syn_i.setter
    def tau_syn_i(self, tau_syn_i_new: Union[float, ArrayLike]):
        self._tau_syn_i = self._process_parameter(
            tau_syn_i_new, "tau_syn_i", nonnegative=True
        )

    @property
    def has_tau2(self):
        return self._has_tau2

    @has_tau2.setter
    def has_tau2(self, new_has_tau2: Union[bool, ArrayLike]):
        new_has_tau2 = self._expand_to_size(
            new_has_tau2, self.num_neurons, "has_tau2", False
        )
        if not new_has_tau2.dtype == bool:
            raise ValueError(self.start_print + "`has_tau2` must consist of booleans.")
        else:
            self._has_tau2 = new_has_tau2

    @property
    def num_chips(self):
        return self._num_chips

    @property
    def num_cores_chip(self):
        return self._num_cores_chip

    @property
    def num_neurons_core(self):
        return self._num_neurons_core

    @property
    def num_cores(self):
        return self._num_cores_chip * self._num_chips

    @property
    def num_neurons_chip(self):
        return self.num_cores * self._num_neurons_core

    @property
    def num_neurons(self):
        return self.num_neurons_chip * self._num_chips

    @property
    def start_print(self):
        return f"VirtualDynapse '{self.name}': "

    @property
    def input_type(self):
        return TSEvent

    @property
    def output_type(self):
        return TSEvent


# Functions:
# - Different synapse types?
# - Adaptation?
# - NMDA synapses?
# - BUF_P???
# - Syn-thresholds???
# Limitations
# Mismatch between parameters
# TODO
# Define tau_n, tau_s(inh), rec and inp weights
