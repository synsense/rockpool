"""
Dynap-SE2 full board configuration classes and methods

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
03/05/2022
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

    :param C_ahp: AHP synapse capacitance in Farads
    :type C_ahp: FloatVector, optional
    :param C_ampa: AMPA synapse capacitance in Farads
    :type C_ampa: FloatVector, optional
    :param C_gaba: GABA synapse capacitance in Farads
    :type C_gaba: FloatVector, optional
    :param C_nmda: NMDA synapse capacitance in Farads
    :type C_nmda: FloatVector, optional
    :param C_pulse_ahp: spike frequency adaptation circuit pulse-width creation sub-circuit capacitance in Farads
    :type C_pulse_ahp: FloatVector, optional
    :param C_pulse: pulse-width creation sub-circuit capacitance in Farads
    :type C_pulse: FloatVector, optional
    :param C_ref: refractory period sub-circuit capacitance in Farads
    :type C_ref: FloatVector, optional
    :param C_shunt: SHUNT synapse capacitance in Farads
    :type C_shunt: FloatVector, optional
    :param C_mem: neuron membrane capacitance in Farads
    :type C_mem: FloatVector, optional
    :param Io: Dark current in Amperes that flows through the transistors even at the idle state
    :type Io: FloatVector, optional
    :param kappa_n: Subthreshold slope factor (n-type transistor)
    :type kappa_n: FloatVector, optional
    :param kappa_p: Subthreshold slope factor (p-type transistor)
    :type kappa_p: FloatVector, optional
    :param Ut: Thermal voltage in Volts
    :type Ut: FloatVector, optional
    :param Vth: The cut-off Vgs potential of the transistors in Volts (not type specific)
    :type Vth: FloatVector, optional
    """

    C_ahp: FloatVector = default_layout["C_ahp"]
    C_ampa: FloatVector = default_layout["C_ampa"]
    C_gaba: FloatVector = default_layout["C_gaba"]
    C_nmda: FloatVector = default_layout["C_nmda"]
    C_pulse_ahp: FloatVector = default_layout["C_pulse_ahp"]
    C_pulse: FloatVector = default_layout["C_pulse"]
    C_ref: FloatVector = default_layout["C_ref"]
    C_shunt: FloatVector = default_layout["C_shunt"]
    C_mem: FloatVector = default_layout["C_mem"]
    Io: FloatVector = default_layout["Io"]
    kappa_n: FloatVector = default_layout["kappa_n"]
    kappa_p: FloatVector = default_layout["kappa_p"]
    Ut: FloatVector = default_layout["Ut"]
    Vth: FloatVector = default_layout["Vth"]
