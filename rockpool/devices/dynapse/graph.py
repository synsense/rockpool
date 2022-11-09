"""
Dynap-SE graph modules implementing conversion and translation methods

Note : Existing modules are reconstructed considering consistency with Xylo support.

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

15/09/2022
"""
from typing import Optional
from rockpool.graph import GenericNeurons

from dataclasses import dataclass, field

__all__ = ["DynapseNeurons"]


@dataclass(eq=False, repr=False)
class DynapseNeurons(GenericNeurons):
    """
    DynapseNeurons stores the core computational properties of the Dynap-SE network

    :param Idc: Constant DC current injected to membrane in Amperes
    :type Idc: FloatVector
    :param If_nmda: NMDA gate soft cut-off current setting the NMDA gating voltage in Amperes
    :type If_nmda: FloatVector
    :param Igain_ahp: gain bias current of the spike frequency adaptation block in Amperes
    :type Igain_ahp: FloatVector
    :param Igain_ampa: gain bias current of excitatory AMPA synapse in Amperes
    :type Igain_ampa: FloatVector
    :param Igain_gaba: gain bias current of inhibitory GABA synapse in Amperes
    :type Igain_gaba: FloatVector
    :param Igain_nmda: gain bias current of excitatory NMDA synapse in Amperes
    :type Igain_nmda: FloatVector
    :param Igain_shunt: gain bias current of the inhibitory SHUNT synapse in Amperes
    :type Igain_shunt: FloatVector
    :param Igain_mem: gain bias current for neuron membrane in Amperes
    :type Igain_mem: FloatVector
    :param Ipulse_ahp: bias current setting the pulse width for spike frequency adaptation block `t_pulse_ahp` in Amperes
    :type Ipulse_ahp: FloatVector
    :param Ipulse: bias current setting the pulse width for neuron membrane `t_pulse` in Amperes
    :type Ipulse: FloatVector
    :param Iref: bias current setting the refractory period `t_ref` in Amperes
    :type Iref: FloatVector
    :param Ispkthr: spiking threshold current, neuron spikes if :math:`Imem > Ispkthr` in Amperes
    :type Ispkthr: FloatVector
    :param Itau_ahp: Spike frequency adaptation leakage current setting the time constant `tau_ahp` in Amperes
    :type Itau_ahp: FloatVector
    :param Itau_ampa: AMPA synapse leakage current setting the time constant `tau_ampa` in Amperes
    :type Itau_ampa: FloatVector
    :param Itau_gaba: GABA synapse leakage current setting the time constant `tau_gaba` in Amperes
    :type Itau_gaba: FloatVector
    :param Itau_nmda: NMDA synapse leakage current setting the time constant `tau_nmda` in Amperes
    :type Itau_nmda: FloatVector
    :param Itau_shunt: SHUNT synapse leakage current setting the time constant `tau_shunt` in Amperes
    :type Itau_shunt: FloatVector
    :param Itau_mem: Neuron membrane leakage current setting the time constant `tau_mem` in Amperes
    :type Itau_mem: FloatVector
    :param Iw_ahp: spike frequency adaptation weight current of the neurons of the core in Amperes
    :type Iw_ahp: FloatVector
    """

    Idc: Union[IntVector, FloatVector] = field(default_factory=list)
    If_nmda: Union[IntVector, FloatVector] = field(default_factory=list)
    Igain_ahp: Union[IntVector, FloatVector] = field(default_factory=list)
    Igain_mem: Union[IntVector, FloatVector] = field(default_factory=list)
    Igain_syn: Union[IntVector, FloatVector] = field(default_factory=list)
    Ipulse_ahp: Union[IntVector, FloatVector] = field(default_factory=list)
    Ipulse: Union[IntVector, FloatVector] = field(default_factory=list)
    Iref: Union[IntVector, FloatVector] = field(default_factory=list)
    Ispkthr: Union[IntVector, FloatVector] = field(default_factory=list)
    Itau_ahp: Union[IntVector, FloatVector] = field(default_factory=list)
    Itau_mem: Union[IntVector, FloatVector] = field(default_factory=list)
    Itau_syn: Union[IntVector, FloatVector] = field(default_factory=list)
    Iw_ahp: Union[IntVector, FloatVector] = field(default_factory=list)
    Iscale: Optional[float] = None
    dt: Optional[float] = None

