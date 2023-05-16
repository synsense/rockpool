"""
Dynap-SE2 layout parameters

* Non User Facing *
"""

from dataclasses import dataclass

from rockpool.devices.dynapse.lookup import default_layout
from rockpool.typehints import FloatVector

from ..base import DynapSimProperty

__all__ = ["DynapSimLayout"]


@dataclass
class DynapSimLayout(DynapSimProperty):
    """
    DynapSimLayout contains the constant values used in simulation that are related to the exact silicon layout of a Dynap-SE chips.
    """

    C_ahp: FloatVector = default_layout["C_ahp"]
    """AHP synapse capacitance in Farads"""

    C_ampa: FloatVector = default_layout["C_ampa"]
    """AMPA synapse capacitance in Farads"""

    C_gaba: FloatVector = default_layout["C_gaba"]
    """GABA synapse capacitance in Farads"""

    C_nmda: FloatVector = default_layout["C_nmda"]
    """NMDA synapse capacitance in Farads"""

    C_pulse_ahp: FloatVector = default_layout["C_pulse_ahp"]
    """spike frequency adaptation circuit pulse-width creation sub-circuit capacitance in Farads"""

    C_pulse: FloatVector = default_layout["C_pulse"]
    """pulse-width creation sub-circuit capacitance in Farads"""

    C_ref: FloatVector = default_layout["C_ref"]
    """refractory period sub-circuit capacitance in Farads"""

    C_shunt: FloatVector = default_layout["C_shunt"]
    """SHUNT synapse capacitance in Farads"""

    C_mem: FloatVector = default_layout["C_mem"]
    """neuron membrane capacitance in Farads"""

    Io: FloatVector = default_layout["Io"]
    """Dark current in Amperes that flows through the transistors even at the idle state"""

    kappa_n: FloatVector = default_layout["kappa_n"]
    """Subthreshold slope factor (n-type transistor)"""

    kappa_p: FloatVector = default_layout["kappa_p"]
    """Subthreshold slope factor (p-type transistor)"""

    Ut: FloatVector = default_layout["Ut"]
    """Thermal voltage in Volts"""

    Vth: FloatVector = default_layout["Vth"]
    """The cut-off Vgs potential of the transistors in Volts (not type specific)"""
