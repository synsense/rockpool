"""
Dynap-SE circuit layout classes to configure DynapSim simulation modules

renamed : dynapse1_simconfig.py -> simconfig.py @ 211208
split_from : simconfig.py -> layout.py @ 220114

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
24/08/2021
Split : 14/01/2022

TODOS:
[] TODO: Capacitance ratio or exact value flexibility
[] TODO: SE2 Capacitance values
"""
from dataclasses import dataclass


@dataclass
class DynapSECapacitance:
    """
    DynapSE1Capacitance stores the ratio between capacitance values in the membrane subcircuit and
    in the DPI synapse subcircuits. It also has a base capacitance value to multiply with the factors stored.
    After the post-initialization, user sees the exact capacitance values in Farads.

    :param Co: the base capacitance value in Farads, defaults to 5e-13
    :type Co: float, optional
    :param mem: membrane capacitance, defaults to 10.0
    :type mem: float, optional
    :param ref: refractory period sub-circuit capacitance in Co units, defaults to 1.0
    :type ref: float, optional
    :param pulse: pulse-width creation sub-circuit capacitance in Co units, defaults to 1.0
    :type pulse: float, optional
    :param gaba_b: GABA_B synapse capacitance in Co units, defaults to 50.0
    :type gaba_b: float, optional
    :param gaba_a: GABA_A synapse capacitance in Co units, defaults to 49.0
    :type gaba_a: float, optional
    :param nmda: NMDA synapse capacitance in Co units, defaults to 50.0
    :type nmda: float, optional
    :param ampa: AMPA synapse capacitance in Co units, defaults to 49.0
    :type ampa: float, optional
    :param ahp: AHP synapse capacitance in Co units, defaults to 80.0
    :type ahp: float, optional
    """

    Co: float = 5e-13
    mem: float = 6.0
    ref: float = 3.0
    pulse: float = 1.0
    gaba_b: float = 50.0
    gaba_a: float = 49.0  # shunt
    nmda: float = 50.0
    ampa: float = 49.0
    ahp: float = 80.0

    def __post_init__(self) -> None:
        """
        __post_init__ runs after __init__ and sets the variables depending on the initial values
        """
        for name, value in self.__dict__.items():
            if name != "Co":
                self.__setattr__(name, float(value * self.Co))


@dataclass
class DynapSELayout:
    """
    DynapSE1Layout contains the constant values used in simulation that are related to the exact silicon layout of a Dynap-SE1 chip.

    :param kappa_n: Subthreshold slope factor (n-type transistor), defaults to 0.75
    :type kappa_n: float, optional
    :param kappa_p: Subthreshold slope factor (p-type transistor), defaults to 0.66
    :type kappa_p: float, optional
    :param Ut: Thermal voltage in Volts, defaults to 25e-3
    :type Ut: float, optional
    :param Io: Dark current in Amperes that flows through the transistors even at the idle state, defaults to 5e-13
    :type Io: float, optional
    :param Von: The opening Vgs potential of the transistors (not type specific), defaults to 7e-1
    :type Von: float, optional

    :Instance Variables:

    :ivar kappa: Mean kappa
    :type kappa: float
    """

    kappa_n: float = 0.75
    kappa_p: float = 0.66
    Ut: float = 25e-3
    Io: float = 5e-13
    Von: float = 7e-1

    def __post_init__(self) -> None:
        """
        __post_init__ runs after __init__ and sets the variables depending on the initial values
        """
        self.kappa = (self.kappa_n + self.kappa_p) / 2.0
