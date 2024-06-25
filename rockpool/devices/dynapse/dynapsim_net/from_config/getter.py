"""
Dynap-SE simulator generator from the samna config object
"""

from typing import Dict, Optional

import numpy as np

from rockpool.nn.modules import JaxModule
from rockpool.nn.modules import LinearJax
from rockpool.nn.combinators import JaxSequential

from rockpool.devices.dynapse.samna_alias import (
    Dynapse2Configuration,
    Dynapse2Destination,
)
from rockpool.devices.dynapse.parameters import DynapSimCore
from rockpool.devices.dynapse.lookup import default_weights
from rockpool.devices.dynapse.simulation import DynapSim

from .parameter import ParameterHandler

__all__ = ["dynapsim_net_from_config"]


def dynapsim_net_from_config(
    config: Dynapse2Configuration,
    input_channel_map: Optional[Dict[int, Dynapse2Destination]] = None,
    Iscale: float = default_weights["Iscale"],
    percent_mismatch: Optional[float] = None,
    dt: float = 1e-3,
    *args,
    **kwargs,
) -> JaxModule:
    """
    dynapsim_net_from_config constructs a `DynapSim` network by processing a samna configuration object

    :param config: a samna configuration object used to configure all the system level properties
    :type config: Dynapse2Configuration
    :param input_channel_map: the mapping between input timeseries channels and the destinations, Providing an input channel map restores the zero rows that may occur in the simulation input weight matrices. The exact same dimensional weight matrix can only be restored in this way
    :type input_channel_map: Dict[int, Dynapse2Destination]
    :param Iscale: network weight scaling current, defaults to default_weights["Iscale"]
    :type Iscale: float, optional
    :param percent_mismatch: Gaussian parameter mismatch percentage (check `transforms.mismatch_generator` implementation), defaults to None
    :type percent_mismatch: Optional[float], optional
    :param dt: The time step for the forward-Euler ODE solver, defaults to 1e-3
    :type dt: float, optional
    :return: a `nn.combinators.Sequential` combinator possibly encapsulating a `nn.modules.LinearJax` layer and a `DynapSim` layer, or just a `DynapSim` layer in the case that no input weights defined
    :rtype: `nn.modules.JaxModule`
    """

    # Empty parameter lists
    Idc = []
    If_nmda = []
    Igain_ahp = []
    Igain_mem = []
    Igain_syn = []
    Ipulse_ahp = []
    Ipulse = []
    Iref = []
    Ispkthr = []
    Itau_ahp = []
    Itau_mem = []
    Itau_syn = []
    Iw_ahp = []
    Iw_trace = []

    # Get a parameter handler object which will lead the simulation network configuration
    param_handler = ParameterHandler.from_config(config)

    # Get a simulation core object for each represented core
    sim_cores = {}
    for h, c in param_handler.core_list:
        sim_cores[(h, c)] = DynapSimCore.from_Dynapse2Core(config.chips[h].cores[c])

    # Collect currents of neurons of respective cores in parameter lists
    for n, (h, c) in enumerate(param_handler.core_map):
        Idc.append(sim_cores[(h, c)].Idc)
        If_nmda.append(sim_cores[(h, c)].If_nmda)
        Igain_ahp.append(sim_cores[(h, c)].Igain_ahp)
        Igain_mem.append(sim_cores[(h, c)].Igain_mem)
        Igain_syn.append(param_handler.compose_Igain_syn(sim_cores[(h, c)], n))
        Ipulse_ahp.append(sim_cores[(h, c)].Ipulse_ahp)
        Ipulse.append(sim_cores[(h, c)].Ipulse)
        Iref.append(sim_cores[(h, c)].Iref)
        Ispkthr.append(sim_cores[(h, c)].Ispkthr)
        Itau_ahp.append(sim_cores[(h, c)].Itau_ahp)
        Itau_mem.append(sim_cores[(h, c)].Itau_mem)
        Itau_syn.append(param_handler.compose_Itau_syn(sim_cores[(h, c)], n))
        Iw_ahp.append(sim_cores[(h, c)].Iw_ahp)
        Iw_trace.append([sim_cores[(h, c)].weight_bits.Iw])

    # Get restored and scaled weight matrices using the Iw traces of the neurons
    w_in_scaled = param_handler.get_scaled_weights_in(Iw_trace, Iscale)
    w_rec_scaled = param_handler.get_scaled_weights_rec(Iw_trace, Iscale)

    # Input layer (external world -> hardware)
    if w_in_scaled.any():
        # Restore zero rows
        if input_channel_map is not None:
            w_in_scaled = __restore_zero_rows(w_in_scaled, input_channel_map)

        in_layer = LinearJax(w_in_scaled.shape, w_in_scaled, has_bias=False)
    else:
        in_layer = None

    # Recurrent layer (hardware -> hardware)
    dynapsim_layer = DynapSim(
        shape=param_handler.n_rec,
        Idc=np.array(Idc),
        If_nmda=np.array(If_nmda),
        Igain_ahp=np.array(Igain_ahp),
        Igain_mem=np.array(Igain_mem),
        Igain_syn=np.array(Igain_syn),
        Ipulse_ahp=np.array(Ipulse_ahp),
        Ipulse=np.array(Ipulse),
        Iref=np.array(Iref),
        Ispkthr=np.array(Ispkthr),
        Itau_ahp=np.array(Itau_ahp),
        Itau_mem=np.array(Itau_mem),
        Itau_syn=np.array(Itau_syn),
        Iw_ahp=np.array(Iw_ahp),
        w_rec=w_rec_scaled,
        has_rec=w_rec_scaled.any(),
        percent_mismatch=percent_mismatch,
        Iscale=Iscale,
        dt=dt,
    )

    # The resulting sequential module ! :tada:
    if in_layer is not None:
        mod = JaxSequential(in_layer, dynapsim_layer)
    else:
        mod = dynapsim_layer

    return mod


### --- Private Section --- ###
def __restore_zero_rows(
    trimmed_weights: np.ndarray,
    channel_map: Dict[int, Dynapse2Destination],
) -> np.ndarray:
    """
    __restore_zero_rows restores the zero rows that occur in the simulation weights but lost in config object
    Configuration object does not store a connection information if all weights from one input channel to all neurons are zero
    However, input channel map stores. This function exploits the input channel map and restores the zero rows

    :param trimmed_weights: the weight matrix contatining only rows with at least one non-zero entitiy
    :type trimmed_weights: np.ndarray
    :param channel_map: the mapping between input timeseries channels and the hardware destinations
    :type channel_map: Dict[int, Dynapse2Destination]
    :return: the restored weight matrix with (possibly) zero rows.
    :rtype: np.ndarray
    """

    # Obtain an empty matrix
    shape = (len(channel_map), trimmed_weights.shape[1])
    weights = np.zeros(shape)

    # Re-shape the trimmed weight matrix
    non_zero_idx = [key for key, val in channel_map.items() if val]
    weights[non_zero_idx, :] = trimmed_weights

    return weights
