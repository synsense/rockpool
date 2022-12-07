"""
Dynap-SE2 full board configuration classes and methods

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
03/05/2022
"""

from dataclasses import dataclass

from rockpool.devices.dynapse.lookup import default_currents
from rockpool.typehints import FloatVector

from ..base import DynapSimProperty

__all__ = ["DynapSimCurrents"]


@dataclass
class DynapSimCurrents(DynapSimProperty):
    """
    DynapSimCurrents contains the common simulation current values of Dynap-SE chips

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

    Idc: FloatVector = default_currents["Idc"]
    If_nmda: FloatVector = default_currents["If_nmda"]
    Igain_ahp: FloatVector = default_currents["Igain_ahp"]
    Igain_ampa: FloatVector = default_currents["Igain_ampa"]
    Igain_gaba: FloatVector = default_currents["Igain_gaba"]
    Igain_nmda: FloatVector = default_currents["Igain_nmda"]
    Igain_shunt: FloatVector = default_currents["Igain_shunt"]
    Igain_mem: FloatVector = default_currents["Igain_mem"]
    Ipulse_ahp: FloatVector = default_currents["Ipulse_ahp"]
    Ipulse: FloatVector = default_currents["Ipulse"]
    Iref: FloatVector = default_currents["Iref"]
    Ispkthr: FloatVector = default_currents["Ispkthr"]
    Itau_ahp: FloatVector = default_currents["Itau_ahp"]
    Itau_ampa: FloatVector = default_currents["Itau_ampa"]
    Itau_gaba: FloatVector = default_currents["Itau_gaba"]
    Itau_nmda: FloatVector = default_currents["Itau_nmda"]
    Itau_shunt: FloatVector = default_currents["Itau_shunt"]
    Itau_mem: FloatVector = default_currents["Itau_mem"]
    Iw_ahp: FloatVector = default_currents["Iw_ahp"]
