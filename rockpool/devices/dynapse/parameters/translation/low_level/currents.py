"""
Dynap-SE2 all simulated currents

* Non User facing *
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
    """

    Idc: FloatVector = default_currents["Idc"]
    """Constant DC current injected to membrane in Amperes"""

    If_nmda: FloatVector = default_currents["If_nmda"]
    """NMDA gate soft cut-off current setting the NMDA gating voltage in Amperes"""

    Igain_ahp: FloatVector = default_currents["Igain_ahp"]
    """gain bias current of the spike frequency adaptation block in Amperes"""

    Igain_ampa: FloatVector = default_currents["Igain_ampa"]
    """gain bias current of excitatory AMPA synapse in Amperes"""

    Igain_gaba: FloatVector = default_currents["Igain_gaba"]
    """gain bias current of inhibitory GABA synapse in Amperes"""

    Igain_nmda: FloatVector = default_currents["Igain_nmda"]
    """gain bias current of excitatory NMDA synapse in Amperes"""

    Igain_shunt: FloatVector = default_currents["Igain_shunt"]
    """gain bias current of the inhibitory SHUNT synapse in Amperes"""

    Igain_mem: FloatVector = default_currents["Igain_mem"]
    """gain bias current for neuron membrane in Amperes"""

    Ipulse_ahp: FloatVector = default_currents["Ipulse_ahp"]
    """bias current setting the pulse width for spike frequency adaptation block ``t_pulse_ahp`` in Amperes"""

    Ipulse: FloatVector = default_currents["Ipulse"]
    """bias current setting the pulse width for neuron membrane ``t_pulse`` in Amperes"""

    Iref: FloatVector = default_currents["Iref"]
    """bias current setting the refractory period ``t_ref`` in Amperes"""

    Ispkthr: FloatVector = default_currents["Ispkthr"]
    """spiking threshold current, neuron spikes if :math:`I_{mem} > I_{spkthr}` in Amperes"""

    Itau_ahp: FloatVector = default_currents["Itau_ahp"]
    """Spike frequency adaptation leakage current setting the time constant ``tau_ahp`` in Amperes"""

    Itau_ampa: FloatVector = default_currents["Itau_ampa"]
    """AMPA synapse leakage current setting the time constant ``tau_ampa`` in Amperes"""

    Itau_gaba: FloatVector = default_currents["Itau_gaba"]
    """GABA synapse leakage current setting the time constant ``tau_gaba`` in Amperes"""

    Itau_nmda: FloatVector = default_currents["Itau_nmda"]
    """NMDA synapse leakage current setting the time constant ``tau_nmda`` in Amperes"""

    Itau_shunt: FloatVector = default_currents["Itau_shunt"]
    """SHUNT synapse leakage current setting the time constant ``tau_shunt`` in Amperes"""

    Itau_mem: FloatVector = default_currents["Itau_mem"]
    """Neuron membrane leakage current setting the time constant ``tau_mem`` in Amperes"""

    Iw_ahp: FloatVector = default_currents["Iw_ahp"]
    """spike frequency adaptation weight current of the neurons of the core in Amperes"""
