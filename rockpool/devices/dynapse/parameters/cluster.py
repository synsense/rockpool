"""
Dynap-SE parameter clustering utilities

Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

09/11/2022
"""
from __future__ import annotations

from rockpool.typehints import FloatVector

import numpy as np

__all__ = ["parameter_clustering"]

### --- Utility Functions --- ###


def get_num_neurons(*args) -> int:
    n_neurons = len(args[0])
    for arg in args:
        if len(arg) != n_neurons:
            raise ValueError(f"Number of neurons indicated does not match!")

    return n_neurons


def parameter_clustering(
    Idc: FloatVector,
    If_nmda: FloatVector,
    Igain_ahp: FloatVector,
    Igain_mem: FloatVector,
    Igain_syn: FloatVector,
    Ipulse_ahp: FloatVector,
    Ipulse: FloatVector,
    Iref: FloatVector,
    Ispkthr: FloatVector,
    Itau_ahp: FloatVector,
    Itau_mem: FloatVector,
    Itau_syn: FloatVector,
    Iw_ahp: FloatVector,
    Iscale: float,
    n_cluster: int = 1,
    *args,
    **kwargs,
):

    n_neurons = get_num_neurons(
        Idc,
        If_nmda,
        Igain_ahp,
        Igain_mem,
        Igain_syn,
        Ipulse_ahp,
        Ipulse,
        Iref,
        Ispkthr,
        Itau_ahp,
        Itau_mem,
        Itau_syn,
        Iw_ahp,
    )

    if n_cluster > n_neurons:
        raise ValueError(
            f"Number of clusters {n_cluster} cannot be greater than number of neurons {n_neurons}"
        )

    # Cluster Parameters randomly
    core_map = np.random.permutation([n % n_cluster for n in range(n_neurons)])

    def __cluster(param: FloatVector) -> FloatVector:
        param = np.array(param)
        centers = [np.mean(param[core_map == c]) for c in range(n_cluster)]
        return centers

    return {
        "n_cluster": n_cluster,
        "core_map": core_map,
        "n_neuron": n_neurons,
        "Idc": __cluster(Idc),
        "If_nmda": __cluster(If_nmda),
        "Igain_ahp": __cluster(Igain_ahp),
        "Igain_mem": __cluster(Igain_mem),
        "Igain_syn": __cluster(Igain_syn),
        "Ipulse_ahp": __cluster(Ipulse_ahp),
        "Ipulse": __cluster(Ipulse),
        "Iref": __cluster(Iref),
        "Ispkthr": __cluster(Ispkthr),
        "Itau_ahp": __cluster(Itau_ahp),
        "Itau_mem": __cluster(Itau_mem),
        "Itau_syn": __cluster(Itau_syn),
        "Iw_ahp": __cluster(Iw_ahp),
        "Iscale": [Iscale] * n_cluster,
    }
