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
from NetworksPython.layers import Layer
from . import params

### --- Constants
CONNECTIONS_OK = 0
FANIN_EXCEEDED = 1
FANOUT_EXCEEDED = 2

### --- Class definition


class VirtualDynapse(Layer):
    def __init__(
        self,
        dt,
        connections,  # - 3D array, with 0th dim corresponding to synapse type, others connections (pos. integers)
        neuron_ids,  # IDs of neurons to be used -> must match last dim. of connecitons in size
        tau_mem,  # - Array of size 16
        tau_mem_alt,
        tau_syn_fe,
        tau_syn_se,
        tau_syn_fi,
        tau_syn_si,
        t_refractory,
        weights_fe,
        weights_se,
        weights_fi,
        weights_si,
        thresholds,
        # syn_thresholds???
        tau_alt,  # - Binary array with size number of neurons
    ):
        pass

    def set_connections(
        self,
        connections: Union[dict, np.ndarray],
        neurons_pre: np.ndarray,
        neurons_post: Optional[np.ndarray],
    ):

        if neurons_post is None:
            neuron_post = neurons_pre

        if isinstance(connections, dict):
            connections = self._connection_dict_to_3darray(connections)

        # - Complete connectivity array after adding new connections
        connections_new = self.connections.copy()

        # - Add new connections to connectivity array
        for i, presyn_id in enumerate(neurons_pre):
            connections_new[:, presyn_id, neurons_post] += connections[:, i]

        # - Test if

    def _connections_to_3darray(
        self, connections: Union[dict, np.ndarray]
    ) -> np.ndarray:
        """
        _connections_to_3darray - Bring connections into 3D array with axis 0
                                  corresponding to synapse type.
        :params connections:  Must be one of the following:
                                - dict with synapse types (see params.SYNAPSE_TYPES) as keys
                                       and connectivity matrix for each type as values
                                - 3D np.ndarray with axis 0 corresponding to synapse types.
                                  First dimension must match size of params.SYNAPSE_TYPES.
                                - 2D np.ndarray: Will assume positive values correspond to
                                  params.SYNAPSE_TYPES[0], negative values to
                                  params.SYNAPSE_TYPES[1].
        :return:
            3D np.ndarray with axis 0 corresponding to synapse types. Order as
            in params.SYNAPSE_TYPES
        """

        # - Handle dicts
        if isinstance(connections, dict):
            conn_3d = self._connection_dict_to_3darray(connections)

        # - Handle arrays
        else:
            num_syn_types = len(params.SYNAPSE_TYPES)
            # - Convert connections to array
            connections = np.asarray(connections)
            if connections.ndim == 3:
                # - Make sure all synapse types are defined
                if connections.shape[0] != num_syn_types:
                    raise ValueError(
                        self.start_print
                        + f"First dimension of 3D connectivity array must be {num_syn_types}."
                    )
                else:
                    # - Use connections as they are given
                    conn_3d = connections
            elif connections.ndim == 2:
                warn(
                    self.start_print
                    + "No synapse types specified. Will Assume "
                    + "{} for positive, {} for negative weights.".format(
                        params.SYNAPSE_TYPES[0], params.SYNAPSE_TYPES[2]
                    )
                )
                conn_3d = np.zeros((num_syn_types,) + connections.shape)
                conn_3d[0] = np.clip(connections, 0, None)
                conn_3d[2] = -np.clip(connections, None, 0)
            else:
                raise ValueError(
                    self.start_print + f"`connections` must be dict, 3D- or 2D-array."
                )

        return conn_3d

    def _connection_dict_to_3darray(self, connections: dict) -> np.ndarray:
        """
        _connection_dict_to_3darray - Convert dict with synapse types as keys to
                                      3D array with axis 0 corresponding to syn type
        :params connectiosn:  dict with synapse types (see params.SYNAPSE_TYPES) as
                              keys and connectivity matrix for each type as values
        :return:
            3D np.ndarray with axis 0 corresponding to synapse types. Order as
            in params.SYNAPSE_TYPES
        """

        # - Warn, if unrecognized synapse types are presented:
        unknown_syntypes: set = set(connections.keys()).difference(params.SYNAPSE_TYPES)
        if unknown_syntypes:
            warn(
                self.start_print
                + "The following synapse types have not been recognized and will "
                + f"be ignored: {unknown_syntypes}. "
                + f"Supported types are: {params.SYNAPSE_TYPES}"
            )

        # - Collect connectivity matrices in list for different syn. types
        connection_list: List[np.ndarray] = []
        type_presented: List[bool] = []  # List indicating types defined in dict
        for syn_type in params.SYNAPSE_TYPES:
            try:
                connection_list.append(connections[syn_type])
                type_presented.append(True)
            except KeyError:
                # - `None` to represent missing syn. types
                type_presented.append(False)

        num_syn_types = len(params.SYNAPSE_TYPES)

        # - Infer matrix dimensions from first defined connectivity matrix
        try:
            conn_dims: Tuple[int] = connection_list[0].shape
        except IndexError:
            warn(self.start_print + "No connections defined.")
            return np.zeros((num_syn_types, 0, 0))

        # - 3D connectivity array
        conn_3d = np.zeros((num_syn_types,) + conn_dims)
        for idx_type, conn_2d in zip(np.where(type_presented)[0], connection_list):
            try:
                conn_3d[idx_type] = conn_2d
            except ValueError as e:
                if conn_2d.shape != conn_dims:
                    raise ValueError(
                        self.start_print
                        + "All connectivity matrices must have the same shape."
                    )
                else:
                    raise e

        return conn_3d

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

        if verbose and result == CONNECTIONS_OK:
            "\tConnections ok."

        return result

    def get_connection_counts(self, connections: Union[dict, np.ndarray]) -> np.ndarray:
        """
        get_connection_counts - From a dict or array of connections for different
                                synapse types, generate a 2D matrix with the total
                                number of connections between each neuron pair.
        :params connections:  Must be one of the following:
                                - dict with synapse types (see params.SYNAPSE_TYPES) as keys
                                       and connectivity matrix for each type as values
                                - 3D np.ndarray with axis 0 corresponding to synapse types.
                                  First dimension must match size of params.SYNAPSE_TYPES.
                                - 2D np.ndarray: Will assume positive values correspond to
                                  params.SYNAPSE_TYPES[0], negative values to
                                  params.SYNAPSE_TYPES[1].
        :return:
            2D np.ndarray with total number of connections between neurons
        """

        # - Convert connections to 3D array
        connections = self._connections_to_3darray(connections)

        # - Collect full number of connections between neurons in a 2D matrix
        connections_2d = np.sum(connections, axis=0)

        return connections_2d

    def evolve(self, ts_input, duration, num_timesteps, verbose):
        pass

    @property
    def start_print(self):
        return f"VirtualDynapse '{self.name}': "


# Functions:
# - Adaptivity?
# - NMDA synapses?
# - BUF_P???
# - Syn-thresholds???
# Limitations
# - Isi time step, event limit and isi limit??
