"""
Dynap-SE2 graph mapper package
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np

from copy import deepcopy

from rockpool.graph import GraphModule
from rockpool.typehints import FloatVector

from rockpool.devices.dynapse.lookup import NUM_NEURONS

from .utils import lifnet_to_dynapsim
from .container import DynapseGraphContainer

__all__ = ["mapper"]


def mapper(
    graph: GraphModule,
    in_place=False,
    clustering_method: str = "greedy",
    n_cluster: Optional[int] = None,
    max_neuron_per_cluster: int = NUM_NEURONS,
) -> Dict[str, float]:
    """
    mapper maps a computational graph onto Dynap-SE2 architecture.
    Each neural core shares a parameter group in Dynap-SE2.
    Therefore neurons inside the same core operates with the same current setting.
    `mapper` function cluster parameters in groups and allocates an hardware posititon for neurons.

    See Also:
        The tutorials in :ref:`/devices/DynapSE/post-training.ipynb`

    :param graph: Any graph (constraints apply) aimed to be deployed to Dynap-SE2
    :type graph: GraphModule
    :param clustering_method: the clustering approach. ``"random"`` or ``"greedy"``, defaults to ``"greedy"``
    :type clustering_method: str, optional
    :param n_cluster: number of clusters. minimum number of clusters possible if None, defaults to None
    :type n_cluster: int, optional
    :param max_neuron_per_cluster: maximum number of neurons allowed per cluster, defaults to ``NUM_NEURONS``
    :type max_neuron_per_cluster: int, optional

    :return: a specification object which can be used to create a config object
    :rtype: Dict[str, float]
    """

    ## -- Convert the network if necessary -- ##
    try:
        wrapper = DynapseGraphContainer.from_graph_holder(graph)
    except:
        graph = (
            lifnet_to_dynapsim(deepcopy(graph))
            if not in_place
            else lifnet_to_dynapsim(graph)
        )
        wrapper = DynapseGraphContainer.from_graph_holder(graph)

    ## --  Compute and check the number of neurons and number of clusters -- ##
    n_neurons = __get_num_neurons(*wrapper.current_dict.values())
    n_cluster, core_map = __get_cluster_map(
        n_neurons, clustering_method, n_cluster, max_neuron_per_cluster
    )

    ## -- Compute the centers -- ##

    def __cluster_center(param: FloatVector) -> FloatVector:
        """
        __cluster_center computes the cluster center of a single parameter

        :param param: any parameter, each index indicating a neuron's individual current
        :type param: FloatVector
        :return: a list of cluster centers
        :rtype: FloatVector
        """

        param = np.array(param)
        centers = [np.mean(param[core_map == c]) for c in range(n_cluster)]
        return centers

    clustered_currents = {
        key: __cluster_center(val) for key, val in wrapper.current_dict.items()
    }

    ## -- Create the spec dictionary -- ##

    specs = {
        "mapped_graph": graph,
        "weights_in": wrapper.w_in,
        "weights_rec": wrapper.w_rec,
        "Iscale": wrapper.Iscale,
        "n_cluster": n_cluster,
        "core_map": core_map,
        "n_neuron": n_neurons,
        "unclustered": wrapper.current_dict,
    }
    specs.update(clustered_currents)

    return specs


### --- Private Section --- ###


def __get_num_neurons(*args: List[FloatVector]) -> int:
    """
    __get_num_neurons process the given list of FloatVectors and obtains the common length

    :raises ValueError: Number of neurons indicated does not match!
    :return: the elements of all the vectors provided
    :rtype: int
    """
    n_neurons = len(args[0])
    for arg in args:
        if len(arg) != n_neurons:
            raise ValueError(f"Number of neurons indicated does not match!")

    return n_neurons


def __get_cluster_map(
    n_neurons: int,
    clustering_method: str,
    n_cluster: Optional[int],
    max_neuron_per_cluster: int = NUM_NEURONS,
) -> Tuple[int, np.ndarray]:
    """
    __get_cluster_map _summary_

    :param n_neurons: _description_
    :type n_neurons: int
    :param clustering_method: the clustering approach. "random" or "greedy"
    :type clustering_method: str
    :param n_cluster: number of clusters. minimum number of clusters possible if None
    :type n_cluster: int
    :param max_neuron_per_cluster: maximum number of neurons allowed per cluster, defaults to NUM_NEURONS
    :type max_neuron_per_cluster: int, optional
    :raises ValueError: At least {min_cluster} clusters required to allocate {n_neurons} neurons!
    :raises ValueError: Number of clusters {n_cluster} cannot be greater than number of neurons {n_neurons}
    :raises ValueError: Unknown clustering method!
    :return: _description_
    :rtype: Tuple[int, np.ndarray]
    """

    min_cluster = n_neurons // max_neuron_per_cluster + 1

    ### --- Check if the cluster size and maximum number of neurons per cluster match --- ###

    if n_cluster is None:
        n_cluster = min_cluster

    elif n_cluster < min_cluster:
        raise ValueError(
            f"At least {min_cluster} clusters required to allocate {n_neurons} neurons!"
        )

    if n_cluster > n_neurons:
        raise ValueError(
            f"Number of clusters {n_cluster} cannot be greater than number of neurons {n_neurons}"
        )

    ## -- Get a clustered core map in accordance with the clustering method chosen -- ##

    if clustering_method == "greedy":
        neuron_per_cluster = np.floor(n_neurons / n_cluster)
        core_map = np.array([n // neuron_per_cluster for n in range(n_neurons)])
    elif clustering_method == "random":
        core_map = np.random.permutation([n % n_cluster for n in range(n_neurons)])
    else:
        raise ValueError("Unknown clustering method!")

    return n_cluster, core_map
