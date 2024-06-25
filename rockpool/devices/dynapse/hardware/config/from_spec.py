"""
Dynap-SE2 samna configuration getter
Process a quantized specification dictionary and obtain a deployable object

See also `devices.dynapse.mapper`
See also `devices.dynapse.autoencoder_quantization`
"""

from __future__ import annotations
import logging
import numpy as np

from typing import Any, Dict, List, Optional, Tuple
from rockpool.devices.dynapse.samna_alias import Dynapse2Configuration

from rockpool.typehints import FloatVector, IntVector
from rockpool.devices.dynapse.parameters import DynapSimCore

from rockpool.devices.dynapse.lookup import NUM_CORES, NUM_NEURONS, CHIP_MAP, CHIP_POS

# Try to import samna for device interfacing
SAMNA_AVAILABLE = True
try:
    import samna
except:
    samna = Any
    logging.warning(
        "Device interface requires `samna` package which is not installed on the system"
    )
    SAMNA_AVAILABLE = False

from .allocator import WeightAllocator

# - Configure exports
__all__ = ["config_from_specification"]


def config_from_specification(
    n_cluster: int,
    core_map: List[int],
    weights_in: Optional[List[Optional[IntVector]]],
    weights_rec: Optional[List[Optional[IntVector]]],
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
    # Optionals
    sign_in: Optional[List[Optional[IntVector]]] = None,
    sign_rec: Optional[List[Optional[IntVector]]] = None,
    Iw_0: Optional[List[FloatVector]] = None,
    Iw_1: Optional[List[FloatVector]] = None,
    Iw_2: Optional[List[FloatVector]] = None,
    Iw_3: Optional[List[FloatVector]] = None,
    # definitions
    chip_map: Dict[int, int] = CHIP_MAP,
    chip_pos: Dict[int, Tuple[int]] = CHIP_POS,
    num_cores: int = NUM_CORES,
    num_neurons: int = NUM_NEURONS,
    *args,
    **kwargs,
) -> Dynapse2Configuration:
    """
    config_from_specification gets a specification and creates a samna configuration object for Dynap-SE2 chip.
    All the parameteres and weight matrices are provided as lists, indices indicating the exact cluster(core id).

    :param n_cluster: total number of clusters, neural cores allocated
    :type n_cluster: int
    :param core_map: core map (neuron_id : core_id) for in-device neurons, defaults to CORE_MAP
    :type core_map: List[int]
    :param weights_in: a list of quantized input weight matrices
    :type weights_in: Optional[List[Optional[IntVector]]]
    :param weights_rec: a list of quantized recurrent weight matrices
    :type weights_rec: Optional[List[Optional[IntVector]]]
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
    :param Ipulse_ahp: a list of bias current setting the pulse width for spike frequency adaptation block ``t_pulse_ahp`` in Amperes
    :type Ipulse_ahp: List[FloatVector]
    :param Ipulse: a list of bias current setting the pulse width for neuron membrane ``t_pulse`` in Amperes
    :type Ipulse: List[FloatVector]
    :param Iref: a list of bias current setting the refractory period ``t_ref`` in Amperes
    :type Iref: List[FloatVector]
    :param Ispkthr: a list of spiking threshold current, neuron spikes if :math:`I_{mem} > I_{spkthr}` in Amperes
    :type Ispkthr: List[FloatVector]
    :param Itau_ahp: a list of Spike frequency adaptation leakage current setting the time constant ``tau_ahp`` in Amperes
    :type Itau_ahp: List[FloatVector]
    :param Itau_mem: a list of Neuron membrane leakage current setting the time constant ``tau_mem`` in Amperes
    :type Itau_mem: List[FloatVector]
    :param Itau_syn: a list of (AMPA, GABA, NMDA, SHUNT) synapses combined leakage current setting the time constant ``tau_syn`` in Amperes
    :type Itau_syn: List[FloatVector]
    :param Iw_ahp: a list of spike frequency adaptation weight current of the neurons of the core in Amperes
    :type Iw_ahp: List[FloatVector]
    :param sign_in: a list of input weight directions (+1 : excitatory, -1 : inhibitory) matrices, defaults to None
    :type sign_in: Optional[List[Optional[IntVector]]]
    :param sign_rec: a list of recurrent weight directions (+1 : excitatory, -1 : inhibitory) matrices, defaults to None
    :type sign_rec: Optional[List[Optional[IntVector]]]
    :param Iw_0: a list of weight bit 0 current of the neurons of the core in Amperes, defaults to None
    :type Iw_0: Optional[List[FloatVector]]
    :param Iw_1: a list of weight bit 1 current of the neurons of the core in Amperes, defaults to None
    :type Iw_1: Optional[List[FloatVector]]
    :param Iw_2: a list of weight bit 2 current of the neurons of the core in Amperes, defaults to None
    :type Iw_2: Optional[List[FloatVector]]
    :param Iw_3: a list of weight bit 3 current of the neurons of the core in Amperes, defaults to None
    :type Iw_3: Optional[List[FloatVector]]
    :param chip_map: chip map (core_id : chip_id) for all cores, defaults to CHIP_MAP
    :type chip_map: Dict[int, int], optional
    :param chip_pos: global chip position dictionary (chip_id : (xpos,ypos)), defaults to CHIP_POS
    :type chip_pos: Dict[int, Tuple[int]], optional
    :param num_cores: the number of cores per chip, defaults to NUM_CORES
    :type num_cores: int, optional
    :param num_neurons: the number of neurons per core, defaults to NUM_NEURONS
    :type num_neurons: int, optional
    :return: config, input_channel_map
        :config: a modified samna ``Dynapse2Configuration`` object
        :input_channel_map: the mapping between input timeseries channels and the destinations
    :rtype: Tuple[Dynapse2Configuration, Dict[int, Dynapse2Destination]]
    """

    new_config = samna.dynapse2.Dynapse2Configuration()
    core_map = np.array(core_map)

    if len(core_map.shape) != 1:
        raise ValueError("Core_map should be one dimensional!")

    ## -- Get cores one by one -- ##
    for c in range(n_cluster):
        # Get the right chip and the indicated core config
        ch = chip_map[c]
        core_config = new_config.chips[ch].cores[c % num_cores]

        # Convert the core parameters
        core = DynapSimCore(
            Idc=Idc[c],
            If_nmda=If_nmda[c],
            Igain_ahp=Igain_ahp[c],
            Igain_ampa=Igain_syn[c],
            Igain_gaba=Igain_syn[c],
            Igain_nmda=Igain_syn[c],
            Igain_shunt=Igain_syn[c],
            Igain_mem=Igain_mem[c],
            Ipulse_ahp=Ipulse_ahp[c],
            Ipulse=Ipulse[c],
            Iref=Iref[c],
            Ispkthr=Ispkthr[c],
            Itau_ahp=Itau_ahp[c],
            Itau_ampa=Itau_syn[c],
            Itau_gaba=Itau_syn[c],
            Itau_nmda=Itau_syn[c],
            Itau_shunt=Itau_syn[c],
            Itau_mem=Itau_mem[c],
            Iw_ahp=Iw_ahp[c],
            Iw_0=Iw_0[c] if Iw_0 is not None else 0.0,
            Iw_1=Iw_1[c] if Iw_1 is not None else 0.0,
            Iw_2=Iw_2[c] if Iw_2 is not None else 0.0,
            Iw_3=Iw_3[c] if Iw_3 is not None else 0.0,
        )

        # Allocate memory for the weights
        allocator = WeightAllocator(
            weights_in=weights_in[c] if weights_in is not None else None,
            weights_rec=weights_rec[c] if weights_rec is not None else None,
            sign_in=sign_in[c] if sign_in is not None else None,
            sign_rec=sign_rec[c] if sign_rec is not None else None,
            core_map=core_map,
            chip_map=chip_map,
            chip_pos=chip_pos,
        )

        # SRAM blocks are responsible for storing the destinations of the neurons
        sram = allocator.SRAM_content(
            use_samna=True,
            monitor_neurons=list(range(0, num_neurons)),
        )

        # CAM blocks are responsible for storing the incoming connections of the neurons
        cam = allocator.CAM_content(use_samna=True)

        # Neural parameters are shared across all neurons inside the core
        params = core.export_Dynapse2Parameters()

        # Update the configuration object

        ## DC excitation
        if Idc[c] > 0:
            nidx = np.where(core_map == c)[0]
            for n in nidx:
                core_config.neurons[n].latch_so_dc = True

        ## Receiving connections
        for n, cam_content in cam.items():
            core_config.neurons[n].synapses = cam_content

        ## Broadcasting connections
        for n, sram_content in sram.items():
            core_config.neurons[n].destinations = sram_content

        ## Parameters
        for key, (coarse, fine) in params.items():
            core_config.parameters[key].coarse_value = coarse
            core_config.parameters[key].fine_value = fine

        ## Returns
        input_channel_map = allocator.input_channel_map()

    return {"config": new_config, "input_channel_map": input_channel_map}
