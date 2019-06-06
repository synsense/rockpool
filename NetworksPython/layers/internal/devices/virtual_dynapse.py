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
CONNECTION_ALIASING = 4

### --- Class definition


class VirtualDynapse(Layer):

    self._num_chips = params.NUM_CHIPS
    self._num_cores_chip = params.NUM_CORES_CHIP
    self._num_neurons_core = params.NUM_NEURONS_CORE

    def __init__(
        self,
        dt: float = 1e-5,
        connections_in: Optional[np.ndarray] = None,
        connections_rec: Optional[np.ndarray] = None,
        tau_mem_1: Union[float, np.ndarray] = 0.05,
        tau_mem_2: Union[float, np.ndarray] = 0.05,
        has_tau2: Union[bool, np.ndarray] = False,
        tau_syn_e: Union[float, np.ndarray] = 0.05,
        tau_syn_i: Union[float, np.ndarray] = 0.05,
        baseweight_e: Union[float, np.ndarray] = 0.05,
        baseweight_i: Union[float, np.ndarray] = 0.05,
        bias: Union[float, np.ndarray] = 0,
        t_refractory: Union[float, np.ndarray] = 0.005,
        threshold: Union[float, np.ndarray] = 0.01,
        name: str = "unnamed",
        num_cores: int = 1,
    ):
        """
        VritualDynapse - Simulation of DynapSE neurmorphic processor.
        :param dt:        Time step size in seconds
        :param connections_in:   2D-array defining connections from external input
                                 Size at most 1024x4096. Will be filled with 0s if smaller.
        :param connections_rec:  2D-array defining connections between neurons
                                 Size at most 4096x4096. Will be filled with 0s if smaller.
        :param tau_mem_1:        float or 1D-array of size 16, with membrane time constant
                                 for each core, in seconds. If float, same for all cores.
        :param tau_mem_2:        float or 1D-array of size 16 with alternative membrane time
                                 constant for each core, in seconds. If float, same for all
                                 cores.
        :param has_tau2:         bool or 1D-array of size 4096, indicating which neuron
                                 usees the alternative membrane time constant. If bool,
                                 same for all cores.
        :param tau_syn_e:        float or 1D-array of size 16 with time constant for
                                 excitatory synapses for each core, in seconds. If float,
                                 same for all cores.
        :param tau_syn_i:        float or 1D-array of size 16 with time constant for
                                 inhibitory synapses for each core, in seconds. If float,
                                 same for all cores.
        :param baseweight_e:     float or 1D-array of size 16 with multiplicator (>=0) for
                                 binary excitatory weights for each core. If float, same
                                 for all cores.
        :param baseweight_i:     float or 1D-array of size 16 with multiplicator (>=0) for
                                 binary inhibitory weights for each core. If float, same
                                 for all cores.
        :param bias:             float or 1D-array of size 16 with constant neuron bias (>=0)
                                 for each core. If float, same for all cores.
        :param t_refractory:     float or 1D-array of size 16 with refractory time in
                                 secondsfor each core. If float, same for all cores.
        :param threshold:        float or 1D-array of size 16 with neuron firing threshold
                                 for each core. If float, same for all cores.
        :param name:             Name for this object instance.
        :param num_cores:        Number of cpu cores available for simulation.
        """

        # - Store internal parameters
        self.name = name
        self.tau_mem_1 = tau_mem_1
        self.tau_mem_2 = tau_mem_2
        self.has_tau2 = has_tau2

        # - Set up connections and weights
        self._baseweight_e = self._process_parameter(baseweight_e, "baseweight_e", True)
        self._baseweight_i = self._process_parameter(baseweight_i, "baseweight_i", True)
        if (
            self.validate_connections(connections_rec, connections_new, verbose=True)
            != CONNECTIONS_VALID
        ):
            raise ValueError(
                self.start_print + "Connections not compatible with hardware."
            )
        self._connections_rec = self._process_connections(
            connections_rec, external=False
        )
        self._connections_in = self._process_connections(connections_in, external=True)
        weights_rec = _generate_weights()
        weights_in = _generate_weights(external=True)

        # - Remaining parameters
        bias = self._process_parameter(bias, "bias", True)
        threshold = self._process_parameter(threshold, "threshold", True)
        t_refractory = self._process_parameter(t_refractory, "t_refractory", True)
        tau_syn_e = self._process_parameter(tau_syn_e, "tau_syn_e", True)
        tau_syn_i = self._process_parameter(tau_syn_i, "tau_syn_i", True)

        # - Nest-layer for approximate simulation of neuron dynamics
        self._simulator = RecIAFSpkInNest(
            weights_in=weights_in,
            weights_rec=weights_rec,
            bias=np.repeat(bias, self.num_neurons_core),
            threshold=np.repeat(threshold, self.num_neurons_core),
            v_reset=0,
            v_rest=0,
            tau_mem=self.tau_mem_all,
            tau_syn=np.repeat(self.tau_syn_e, self.num_neurons_core),
            tau_syn_inh=np.repeat(self.tau_syn_i, self.num_neurons_core),
            dt=dt,
        )

    def set_connections(
        self,
        connections: Union[dict, np.ndarray],
        neurons_pre: np.ndarray,
        neurons_post: Optional[np.ndarray] = None,
        external: bool = False,
        add: bool = False,
    ):
        """
        set_connections - Set connections between specific neuron populations.
                          Verify that connections are supported by the hardware.
        :param connections:    2D np.ndarray: Will assume positive (negative) values
                               correspond to excitatory (inhibitory) synapses.
                               Axis 0 (1) corresponds to pre- (post-) synaptic neurons.
                               Sizes must match `neurons_pre` and `neurons_post`.
        :param neurons_pre:    Array-like with IDs of presynaptic neurons that `connections`
                               refer to. If None, use all neurons (from 0 to
                               self.num_neurons - 1).
        :param neurons_post:   Array-like with IDs of postsynaptic neurons that `connections`
                               refer to. If None, use same IDs as presynaptic neurons, unless
                               `external` is True. In this case use all neurons.
        :param add:            If True, new connections are added to exising ones, otherweise
                               connections between given neuron populations are replaced.
        :param external:       If True, presynaptic neurons are external.
        """

        if neurons_pre is None:
            neurons_pre = np.arange(self.num_external if external else self.size)

        if neurons_post is None:
            neuron_post = neurons_pre if not external else np.arange(self.size)

        # - Complete connectivity array after adding new connections
        connections = self.connections_in if external else self.connections_rec

        # - Indices of specified neurons in full connectivity matrix
        idcs_row, idcs_col = np.meshgrid(neurons_pre, neurons_post, indexing="ij")

        if add:
            # - Add new connections to connectivity array
            connections[idcs_row, idcs_col] += connections
        else:
            # - Replace connections between given neurons with new ones
            connections[idcs_row, idcs_col] = connections

    def validate_connections(
        self,
        connections_rec: np.ndarray,
        connections_in: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> int:
        """
        validate_connections - Check whether connections are compatible with the
                               following constraints:
                                 - Fan-in per neuron is limited to 64
                                 - Fan-out of each neuron can only comprise 3 chips.
                                 - If any two neurons with presynaptic connections
                                   to any of the neurons on a given core are on
                                   different chips but have the same IDs within their
                                   respective chips, aliasing of events may occur.
                                   Therefore this scenario is considered invalid.
        :param connections_rec:     2D np.ndarray (NxN): Connectivity matrix to be.
                                validated. Will assume positive (negative) values
                                correspond to excitatory (inhibitory) synapses.
        :param connections_in:  If not `None`, 2D np.ndarray (MxN) that is considered
                                as external input connections to the population
                                that `connections_rec` refers to. This is considered
                                for validaiton of the fan-in of `connections_rec`.
                                Positive (negative) values correspond to excitatory
                                (inhibitory) synapses.
        :param verbose:         If `True`, print out detailed information about
                                validity of connections.
        return
            Integer indicating the result of the validation.
        """
        # - Get 2D matrix with number of connections between neurons
        conn_count_2d = self.get_connection_counts(connections_rec)

        result = 0

        if verbose:
            print(self.start_print + "Testing provided connections:")

        # - Test fan-in
        if connections_in is not None:
            conn_count_in = np.vstack(
                (self.get_connection_counts(connections_in), conn_count_2d)
            )
        else:
            conn_count_in = conn_count_2d
        exceeds_fanin: np.ndarray = (
            np.sum(conn_count_in, axis=0) > params.NUM_CAMS_NEURON
        )
        if exceeds_fanin.any():
            result += FANIN_EXCEEDED
            if verbose:
                print(
                    "\tFan-in ({}) exceeded for neurons: {}".format(
                        params.NUM_CAMS_NEURON, np.where(exceeds_fanin)[0]
                    )
                )
        # - Test fan-out
        # List with target chips for each (presynaptic) neuron
        tgtchip_list = [
            np.nonzero(row)[0] // self.num_neurons_chip for row in conn_count_2d
        ]
        # Number of different target chips per neuron
        nums_tgtchips = np.array([np.unique(tgtchips) for tgtchips in tgtchip_list])
        exceeds_fanout = nums_tgtchips > params.NUM_SRAMS_NEURON
        if exceeds_fanout.any():
            result += FANOUT_EXCEEDED
            if verbose:
                print(
                    "\tFan-out ({}) exceeded for neurons: {}".format(
                        params.NUM_SRAMS_NEURON, np.where(exceeds_fanout)[0]
                    )
                )

        # - Test for connection aliasing
        # Lists for collecting affected presyn. neurons (chips, IDs) and postsyn. cores
        alias_pre_chips: List[List[np.ndarray]] = []
        alias_pre_ids: List[List[int]] = []
        alias_post_cores: List[int] = []
        for core_id in range(self.num_cores):
            # - IDs of neurons where core starts and ends
            id_start = core_id * self.num_neurons_core
            id_end = (core_id + 1) * self.num_neurons_core
            # - Connections with common postsynaptic core
            conns_samecore = conn_count_2d[:, id_start:id_end]
            # - Lists for collecting affected chip and neuron IDs for specific core
            alias_pre_ids_core = []
            alias_pre_chips_core = []
            for neuron_id in range(self.num_neurons_chip):
                # - Connections with `neuron_id`-th neuron of each chip as presyn. neuron
                conns_samecore_sameid = conns_samecore[
                    neuron_id :: self.num_neurons_chip
                ]
                # - IDs of chips from which presynaptic connecitons originate
                connected_presyn_chips = np.unique(np.nonzero(conns_samecore_sameid)[0])
                # - Only one presynaptic neuron with `neuron_id` is allowed for each core
                #   If there are more, collect information
                if len(connected_presyn_chips) > 1:
                    alias_pre_ids_core.append(neuron_id)
                    alias_pre_chips_core.append(connected_presyn_chips)
            if alias_pre_ids_core:
                alias_post_cores.append(core_id)
                alias_pre_ids.append(alias_pre_ids_core)
                alias_pre_chips.append(alias_pre_chips_core)

        if alias_pre_chips:
            result += CONNECTION_ALIASING
            if verbose:
                print_output = (
                    "\tConnection aliasing detected: Neurons on the same core should not "
                    + "have presynaptic connections with neurons that have same IDs (within "
                    + "their respective chips) but are on different chips. Affected "
                    + "postsynaptic cores are: "
                )
                for core_id, neur_ids_core, chips_core in zip(
                    alias_post_cores, alias_pre_ids, alias_pre_chips
                ):
                    core_print = f"\tCore {core_id}:"
                    for id_neur, chips in zip(neur_ids_core, chips_core):
                        id_print = "\t\t Presynaptic ID {} on chips {}".format(
                            id_neur, ", ".join(str(id_ch) for id_ch in chips)
                        )
                        core_print += "\n" + id_print
                    print_output += "\n" + core_print
                print(print_output)

        if verbose and result == CONNECTIONS_VALID:
            print("\tConnections ok.")

        return result

    def get_connection_counts(self, connections: np.ndarray) -> np.ndarray:
        """
        get_connection_counts - From a dict or array of connections for different
                                synapse types, generate a 2D matrix with the total
                                number of connections between each neuron pair.
        :params connections:  2D np.ndarray (NxN): Will assume positive (negative)
                              values correspond to excitatory (inhibitory) synapses.
        :return:
            2D np.ndarray with total number of connections between neurons
        """
        return np.abs(connections)

    def evolve(self, ts_input, duration, num_timesteps, verbose):
        pass

    def _generate_weights(self, external: bool = False) -> np.ndarray:
        """
        _generate_weights: Generate weight matrix from connections and base weights
        :param external: If `True`, generate input weights, otherwise internal weights
        :return:
            2D-array with generated weights
        """
        # - Choose between input and internal connections
        connections = self.connections_in if external else self.connections_rec
        # - Separate excitatory and inhibitory connections
        connections_exc = np.clip(connections, 0, None)
        connections_inh = np.clip(connections, None, 0)
        factor_exc = np.repeat(self.baseweight_e, self.num_neurons_core)
        factor_inh = np.repeat(self.baseweight_i, self.num_neurons_core)
        # - Calculate weights
        return connections_exc * factor_exc + connections_inh * factor_inh

    def _update_weights(self, external: bool = False):
        """
        _update_weights - Update internal representation of weights by multiplying
                          baseweights with number of connections for each core and
                          synapse type.
        :param external:  If `True`, update input weights, otherwise internal weights
        """
        # - Generate weights from connections and base weights
        weights = self._generate_weights(external)
        # - Weight update
        if external:
            self._simulator.weights_in = weights
        else:
            self._simulator.weights_rec = weights

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

    def _process_connections(
        connections: Optional[np.ndarray] = None, external: bool = False
    ) -> np.ndarray:
        """
        _process_connections - Bring connectivity matric into correct shape
        :param connections:  Connectivity matrix. If None, generate 0-matrix.
        :external:           If True, generate external input connection matrix.
        :return:
            2D-np.ndarray - Generated connection matrix.
        """
        num_rows = self.num_external if external else self.num_neurons
        if connections is None:
            # - Remove all connections
            return np.zeros((num_rows, self.num_neurons))
        else:
            # - Handle smaller connectivity matrices by filling up with zeros
            conn_shape = connections.shape
            if conn_shape != (num_rows, self.num_neurons):
                connections0 = np.zeros((num_rows, self.num_neurons))
                connections0[: conn_shape[0], : conn_shape[1]] = connections
                connections = connections0
            return connections

    @property
    def connections_rec(self):
        return self._connections_rec

    @connections_rec.setter
    def connections_rec(self, connections_new: Optional[np.ndarray]):
        # - Bring connections into correct shape
        connections_new = _process_connections(connections_new, external=True)
        # - Make sure that connections have match hardware specifications
        if (
            self.validate_connections(
                connections_new, self.connections_in, verbose=True
            )
            == CONNECTIONS_VALID
        ):
            # - Update connections
            self._connections_rec = connections_new
            # - Update weights accordingly
            self._update_weights()
        else:
            raise ValueError(
                self.start_print + "Connections not compatible with hardware."
            )

    @property
    def connections_in(self):
        return self.connections_in

    @connections_in.setter
    def connections_in(self, connections_new: Optional[np.ndarray]):
        # - Bring connections into correct shape
        connections_new = _process_connections(connections_new, external=True)
        # - Make sure that connections have match hardware specifications
        if (
            self.validate_connections(
                self.connections_rec, connections_new, verbose=True
            )
            == CONNECTIONS_VALID
        ):
            raise ValueError(
                self.start_print + "Connections not compatible with hardware."
            )
        else:
            # - Update connections
            self._connections_in = connections_new
            # - Update weights accordingly
            self._update_weights(external=True)

    @property
    def weights(self):
        return self._simulator.weights

    @property
    def weights_rec(self):
        return self._simulator.weights_rec

    @property
    def weights_in(self):
        return self._simulator.weights_in

    @property
    def baseweight_e(self):
        return self._baseweight_e

    @baseweight_e.setter
    def baseweight_e(self, baseweight_new: Union[float, ArrayLike]):
        self._baseweight_e = self._process_parameter(
            baseweight_new, "baseweight_e", nonnegative=True
        )
        self._update_weights()
        self._update_weights(external=True)

    @property
    def baseweight_i(self):
        return self._baseweight_i

    @baseweight_i.setter
    def baseweight_i(self, baseweight_new: Union[float, ArrayLike]):
        self._baseweight_i = self._process_parameter(
            baseweight_new, "baseweight_i", nonnegative=True
        )
        self._update_weights()
        self._update_weights(external=True)

    @property
    def bias(self):
        return self._simulator.bias[:: self.num_neurons_core]

    @bias.setter
    def bias(self, bias_new: Union[float, ArrayLike]):
        bias_new = self._process_parameter(bias_new, "bias", nonnegative=True)
        self._simulator.bias = np.repeat(bias_new, self.num_neurons_core)

    @property
    def t_refractory(self):
        return self._simulator.t_refractory[:: self.num_neurons_core]

    @t_refractory.setter
    def t_refractory(self, t_refractory_new: Union[float, ArrayLike]):
        t_refractory_new = self._process_parameter(
            t_refractory_new, "t_refractory", nonnegative=True
        )
        self._simulator.t_refractory = np.repeat(
            t_refractory_new, self.num_neurons_core
        )

    @property
    def threshold(self):
        return self._simulator.threshold[:: self.num_neurons_core]

    @threshold.setter
    def threshold(self, threshold_new: Union[float, ArrayLike]):
        threshold_new = self._process_parameter(
            threshold_new, "threshold", nonnegative=True
        )
        self._simulator.threshold = np.repeat(threshold_new, self.num_neurons_core)

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
    def _tau_mem_all(self):
        # - Expand arrays of time constants from core-wise to neuron-wise
        tau_mem_1_all = (
            np.repeat(self.tau_mem_1, self.num_neurons_core) * self.has_tau2 == False
        )
        tau_mem_2_all = np.repeat(self.tau_mem_2, self.num_neurons_core) * self.has_tau2
        return tau_mem_1_all + tau_mem_2_all

    @property
    def tau_mem_all(self):
        return self._simulator.tau_mem

    @property
    def tau_syn_e(self):
        return self._simulator.tau_syn[:: self.num_neurons_core]

    @tau_syn_e.setter
    def tau_syn_e(self, tau_syn_e_new: Union[float, ArrayLike]):
        tau_syn_e_new = self._process_parameter(
            tau_syn_e_new, "tau_syn_e", nonnegative=True
        )
        self._simulator.tau_syn = np.repeat(tau_syn_e_new, self.num_neurons_core)

    @property
    def tau_syn_i(self):
        return self._simulator.tau_syn_inh[:: self.num_neurons_core]

    @tau_syn_i.setter
    def tau_syn_i(self, tau_syn_i_new: Union[float, ArrayLike]):
        tau_syn_i_new = self._process_parameter(
            tau_syn_i_new, "tau_syn_i", nonnegative=True
        )
        self._simulator.tau_syn_inh = np.repeat(tau_syn_i_new, self.num_neurons_core)

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
    def num_external(self):
        return self.num_neurons_chip

    @property
    def start_print(self):
        return f"VirtualDynapse '{self.name}': "

    @property
    def input_type(self):
        return TSEvent

    @property
    def output_type(self):
        return TSEvent

    @property
    def dt(self):
        return self._simulator.dt

    @property
    def _timestep(self):
        return self._simulator._timestep

    @property
    def size(self):
        return self.num_neurons

    @property
    def size_in(self):
        return self.num_external


# Functions:
# - Different synapse types?
# - Adaptation?
# - NMDA synapses?
# - BUF_P???
# - Syn-thresholds???
# Limitations
# Mismatch between parameters
