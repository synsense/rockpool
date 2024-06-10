"""
Obtain a DynapSim network from the spec output of the mapper
See also `rockpool.devices.dynapse.mapper`
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np

from rockpool.nn.modules.module import ModuleBase
from rockpool.nn.modules import LinearJax
from rockpool.nn.combinators import JaxSequential

from rockpool.devices.dynapse.simulation.dynapsim import DynapSim
from rockpool.devices.dynapse.lookup import default_weights

from rockpool.typehints import FloatVector

__all__ = ["dynapsim_net_from_spec"]


def dynapsim_net_from_spec(
    n_cluster: int,
    core_map: List[int],
    weights_in: Optional[FloatVector],
    weights_rec: Optional[FloatVector],
    # params
    Idc: List[FloatVector],
    If_nmda: List[FloatVector],
    Igain_ahp: List[FloatVector],
    Igain_mem: List[FloatVector],
    Igain_syn: List[FloatVector],
    Ipulse_ahp: List[FloatVector],
    Ipulse: List[FloatVector],
    Iref: List[FloatVector],
    Ispkthr: List[FloatVector],
    Itau_ahp: List[FloatVector],
    Itau_mem: List[FloatVector],
    Itau_syn: List[FloatVector],
    Iw_ahp: List[FloatVector],
    # definitions
    Iscale: float = default_weights["Iscale"],
    percent_mismatch: Optional[float] = None,
    dt: float = 1e-3,
    *args,
    **kwargs,
) -> ModuleBase:
    """
    dynapsim_net_from_specification gets a specification and creates a sequential dynapsim network consisting of a linear layer (virtual connections) and a recurrent layer (hardware connections)

    :param n_cluster: total number of clusters, neural cores allocated
    :type n_cluster: int
    :param core_map: core map (neuron_id : core_id) for in-device neurons, defaults to CORE_MAP
    :type core_map: List[int]
    :param weights_in: a list of quantized input weight matrices
    :type weights_in: Optional[FloatVector]
    :param weights_rec: a list of quantized recurrent weight matrices
    :type weights_rec: Optional[FloatVector]
    :param Idc: a list of Constant DC current injected to membrane in Amperes
    :type Idc: List[FloatVector]
    :param If_nmda: a list of NMDA gate soft cut-off current setting the NMDA gating voltage in Amperes
    :type If_nmda: List[FloatVector]
    :param Igain_ahp: a list of gain bias current of the spike frequency adaptation block in Amperes
    :type Igain_ahp: List[FloatVector]
    :param Igain_mem: a list of gain bias current for neuron membrane in Amperes
    :type Igain_mem: List[FloatVector]
    :param Igain_syn: a list of gain bias current of synaptic gates (AMPA, GABA, NMDA, SHUNT) combined in Amperes
    :type Igain_syn: List[FloatVector]
    :param Ipulse_ahp: a list of bias current setting the pulse width for spike frequency adaptation block ```t_pulse_ahp``` in Amperes
    :type Ipulse_ahp: List[FloatVector]
    :param Ipulse: a list of bias current setting the pulse width for neuron membrane ```t_pulse``` in Amperes
    :type Ipulse: List[FloatVector]
    :param Iref: a list of bias current setting the refractory period ```t_ref``` in Amperes
    :type Iref: List[FloatVector]
    :param Ispkthr: a list of spiking threshold current, neuron spikes if :math:`I_{mem} > I_{spkthr}` in Amperes
    :type Ispkthr: List[FloatVector]
    :param Itau_ahp: a list of Spike frequency adaptation leakage current setting the time constant ```tau_ahp``` in Amperes
    :type Itau_ahp: List[FloatVector]
    :param Itau_mem: a list of Neuron membrane leakage current setting the time constant ```tau_mem``` in Amperes
    :type Itau_mem: List[FloatVector]
    :param Itau_syn: a list of (AMPA, GABA, NMDA, SHUNT) synapses combined leakage current setting the time constant ```tau_syn``` in Amperes
    :type Itau_syn: List[FloatVector]
    :param Iw_ahp: a list of spike frequency adaptation weight current of the neurons of the core in Amperes
    :type Iw_ahp: List[FloatVector]
    :param Iscale: network weight scaling current, defaults to default_weights["Iscale"]
    :type Iscale: float, optional
    :param percent_mismatch: Gaussian parameter mismatch percentage (check `transform.mismatch_generator` implementation), defaults to None
    :type percent_mismatch: Optional[float], optional
    :param dt: The time step for the forward-Euler ODE solver, defaults to 1e-3
    :type dt: float, optional
    :return: a `nn.combinators.Sequential` combinator possibly encapsulating a `nn.modules.LinearJax` layer and a `DynapSim` layer, or just a `DynapSim` layer in the case that no input weights defined
    :rtype: `nn.modules.JaxModule`
    """
    Idc_unc = np.zeros_like(core_map, dtype=float)
    If_nmda_unc = np.zeros_like(core_map, dtype=float)
    Igain_ahp_unc = np.zeros_like(core_map, dtype=float)
    Igain_mem_unc = np.zeros_like(core_map, dtype=float)
    Igain_syn_unc = np.zeros_like(core_map, dtype=float)
    Ipulse_ahp_unc = np.zeros_like(core_map, dtype=float)
    Ipulse_unc = np.zeros_like(core_map, dtype=float)
    Iref_unc = np.zeros_like(core_map, dtype=float)
    Ispkthr_unc = np.zeros_like(core_map, dtype=float)
    Itau_ahp_unc = np.zeros_like(core_map, dtype=float)
    Itau_mem_unc = np.zeros_like(core_map, dtype=float)
    Itau_syn_unc = np.zeros_like(core_map, dtype=float)
    Iw_ahp_unc = np.zeros_like(core_map, dtype=float)

    for i in range(n_cluster):
        np.place(Idc_unc, core_map == i, Idc[i])
        np.place(If_nmda_unc, core_map == i, If_nmda[i])
        np.place(Igain_ahp_unc, core_map == i, Igain_ahp[i])
        np.place(Igain_mem_unc, core_map == i, Igain_mem[i])
        np.place(Igain_syn_unc, core_map == i, Igain_syn[i])
        np.place(Ipulse_ahp_unc, core_map == i, Ipulse_ahp[i])
        np.place(Ipulse_unc, core_map == i, Ipulse[i])
        np.place(Iref_unc, core_map == i, Iref[i])
        np.place(Ispkthr_unc, core_map == i, Ispkthr[i])
        np.place(Itau_ahp_unc, core_map == i, Itau_ahp[i])
        np.place(Itau_mem_unc, core_map == i, Itau_mem[i])
        np.place(Itau_syn_unc, core_map == i, Itau_syn[i])
        np.place(Iw_ahp_unc, core_map == i, Iw_ahp[i])

    weights_in = np.array(weights_in) if weights_in is not None else None
    weights_rec = np.array(weights_rec) if weights_rec is not None else None

    # Construct the layers
    in_layer = (
        LinearJax(weights_in.shape, weights_in, has_bias=False)
        if weights_in is not None
        else None
    )

    n_rec = len(core_map)

    dynapsim_layer = DynapSim(
        shape=(n_rec, n_rec),
        Idc=Idc_unc,
        If_nmda=If_nmda_unc,
        Igain_ahp=Igain_ahp_unc,
        Igain_mem=Igain_mem_unc,
        Igain_syn=Igain_syn_unc,
        Ipulse_ahp=Ipulse_ahp_unc,
        Ipulse=Ipulse_unc,
        Iref=Iref_unc,
        Ispkthr=Ispkthr_unc,
        Itau_ahp=Itau_ahp_unc,
        Itau_mem=Itau_mem_unc,
        Itau_syn=Itau_syn_unc,
        has_rec=True if weights_rec is not None else False,
        w_rec=weights_rec,
        percent_mismatch=percent_mismatch,
        Iscale=Iscale,
        dt=dt,
    )

    # The resulting sequential module ! :tada:

    if in_layer is None:
        mod = dynapsim_layer
    else:
        mod = JaxSequential(in_layer, dynapsim_layer)

    return mod
