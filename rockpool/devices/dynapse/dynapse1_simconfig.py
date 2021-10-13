"""
Dynap-SE1 Parameter classes to be used in initial configuration of DynapSE1NeuronSynapseJax module

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
24/08/2021
"""

from rockpool.typehints import Value
from jax import numpy as np

import numpy as onp
from dataclasses import dataclass

from typing import (
    Tuple,
    Any,
    Optional,
)

_SAMNA_AVAILABLE = True

try:
    from samna.dynapse1 import (
        Dynapse1Configuration,
        Dynapse1ParameterGroup,
    )
except ModuleNotFoundError as e:
    Dynapse1Configuration = Any
    Dynapse1ParameterGroup = Any

    print(
        e,
        "\nDynapSE1SimulationConfiguration object cannot be factored from a samna config object!",
    )
    _SAMNA_AVAILABLE = False


@dataclass
class DynapSE1Capacitance:
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
    mem: float = 10.0
    ref: float = 1.0
    pulse: float = 1.0
    gaba_b: float = 50.0
    gaba_a: float = 49.0
    nmda: float = 50.0
    ampa: float = 49.0
    ahp: float = 80.0

    def __post_init__(self) -> None:
        """
        __post_init__ runs after __init__ and sets the variables depending on the initial values
        """
        for name, value in self.__dict__.items():
            if name != "Co":
                self.__setattr__(name, onp.float32(value * self.Co))


@dataclass
class DynapSE1Layout:
    """
    DynapSE1Layout contains the constant values used in simulation that are related to the exact silicon layout of a Dynap-SE1 chip.

    :param kappa_n: Subthreshold slope factor (n-type transistor), defaults to 0.75
    :type kappa_n: float, optional
    :param kappa_p: Subthreshold slope factor (n-type transistor), defaults to 0.66
    :type kappa_p: float, optional
    :param Ut: Thermal voltage in Volts, defaults to 25e-3
    :type Ut: float, optional
    :param Io: Dark current in Amperes that flows through the transistors even at the idle state, defaults to 5e-13
    :type Io: float, optional

    :Instance Variables:

    :ivar kappa: Mean kappa
    :type kappa: float
    """

    kappa_n: float = 0.75
    kappa_p: float = 0.66
    Ut: float = 25e-3
    Io: float = 5e-13

    def __post_init__(self) -> None:
        """
        __post_init__ runs after __init__ and sets the variables depending on the initial values
        """
        self.kappa = (self.kappa_n + self.kappa_p) / 2.0


@dataclass
class DPIParameters:
    """
    DPIParameters encapsulates DPI-specific bias current parameters and calculates their logical reciprocals.

    :param Itau: Synaptic time constant current in Amperes, that is inversely proportional to time constant tau, defaults to 10e-12
    :type Itau: float, optional
    :param Ith:  DPI's threshold(gain) current in Amperes, scaling factor for the synaptic weight (typically x2, x4 of I_tau) :math:`I_{th} = f_{gain} \cdot I_{\\tau}`, defaults to None
    :type Ith: float, optional
    :param tau: Synaptic time constant, co-depended to Itau. In the case its provided, Itau is infered from tau, defaults to None
    :type tau: Optional[float], optional
    :param f_gain: the gain ratio for the steady state solution. :math:`f_{gain}= \dfrac{I_{th}}{I_{\\tau}}`. In the case its provided, Ith is infered from f_gain, defaults to 4
    :type f_gain: float, optional
    :param C: DPI synaptic capacitance in Farads, fixed at layout time, defaults to 1e-12
    :type C: float
    :param layout: constant values that are related to the exact silicon layout of a chip, defaults to None
    :type layout: Optional[DynapSE1Layout], optional

    :Instance Variables:

    :ivar f_tau: Tau factor for DPI circuit. :math:`f_{\\tau} = \\dfrac{U_T}{\\kappa \\cdot C}`, :math:`f_{\\tau} = I_{\\tau} \\cdot \\tau`
    :type f_tau: float
    """

    Itau: Optional[float] = 7e-12
    Ith: Optional[float] = None
    tau: Optional[float] = None
    f_gain: Optional[float] = 4
    C: float = 1e-12
    layout: Optional[DynapSE1Layout] = None

    def __post_init__(self) -> None:
        """
        __post_init__ runs after __init__ and initializes the DPIParameters block, check if given values are in legal bounds.

        :raises ValueError: Illegal capacitor value given. It cannot be less than 0.
        """

        # Silicon features
        if self.C < 0:
            raise ValueError(f"Illegal capacitor value C : {self.C}F. It should be > 0")

        if self.layout is None:
            self.layout = DynapSE1Layout()

        # Find the tau factor using the layout parameters
        self.f_tau = (self.layout.Ut / self.layout.kappa) * self.C

        # Infere Itau from tau
        if self.tau is not None:
            self.Itau = self.f_tau / self.tau

        # Infere tau from Itau
        elif self.Itau is not None:
            self.tau = self.f_tau / self.Itau

        self.depended_param_check(
            self.tau, self.Itau, current_limits=(self.layout.Io, None)
                )

        # Infere Ith from f_gain and Itau
        if self.f_gain is not None:
            self.Ith = self.Itau * self.f_gain

        # Infere f_gain from Ith and Itau
        elif self.Ith is not None:
            self.f_gain = self.Ith / self.Itau

        self.depended_param_check(
            self.f_gain, self.Ith, current_limits=(self.layout.Io, None)
        )

    def depended_param_check(
        param: float,
        current: float,
        param_limits: Tuple[float] = (0, None),
        current_limits: Tuple[float] = (0, None),
    ) -> None:
        """
        depended_param_check checks the co-depended parameter and current values if they are None and
        if they are in the valid range.

        :param param: The current depended parameter value to set the current or to be set by the current.
        :type param: float
        :param current: The parameter depended current value to set the parameter or to be set by the parameter
        :type current: float
        :param param_limits: lower and upper limits of the current depended parameter Tuple(lower, upper), defaults to (0, None)
        :type param_limits: Tuple[float], optional
        :param current_limits: lower and upper limits of the parameter depended current value Tuple(lower, upper), defaults to (0, None)
        :type current_limits: Tuple[float], optional
        :raises ValueError: At least one of  `param` or `current` should be defined
        :raises ValueError: Illegal parameter value given. It should be greater then param_limits[0]
        :raises ValueError: Illegal parameter value given. It should be less then param_limits[1]
        :raises ValueError: Illegal current value given. It should be greater then current_limits[0]
        :raises ValueError: Illegal current value given. It should be less then current_limits[1]
        """

        param_lower, param_upper = param_limits
        current_lower, current_upper = current_limits

        if param is None and current is None:
                raise ValueError(
                "At least one of  `param` or `current` should be defined. `param` has priority over `current`"
                )

        if param_lower is not None and param < param_lower:
            raise ValueError(
                f"Illegal parameter value : {param:.1e}. It should be > {param_lower:.1e}"
            )

        if param_upper is not None and param > param_upper:
            raise ValueError(
                f"Illegal parameter value : {param:.1e}. It should be < {param_upper:.1e}"
            )

        if current_lower is not None and current < current_lower:
            raise ValueError(
                f"Desired parameter : {param:.1e} is unachievable with this parameter set. "
                f"Current value: {current:.1e} A should have been greater than the dark current: {current_lower:.1e}A"
            )

        if current_upper is not None and current > current_upper:
            raise ValueError(
                f"Desired parameter : {param:.1e} is unachievable with this parameter set. "
                f"Current value: {current:.1e} A should have been less than the upper limit: {current_upper:.1e}A"
            )


@dataclass
class SynapseParameters(DPIParameters):
    """
    DPIParameters contains DPI specific parameter and state variables

    :param Iw: Synaptic weight current in Amperes, determines how strong the response is in terms of amplitude, defaults to 1e-7
    :type Iw: float, optional
    :param Isyn: DPI output current in Amperes (state variable), defaults to Io
    :type Isyn: Optional[float], optional
    """

    Iw: float = 1e-7
    Isyn: Optional[float] = None

    def __post_init__(self):
        """
        __post_init__ runs after __init__ and initializes the SynapseParameters, check if Isyn value is in legal bounds.

        :raises ValueError: Illegal Isyn value desired. It cannot be less than Io
        """
        super().__post_init__()

        if self.Isyn is None:
            self.Isyn = self.layout.Io

        if self.Isyn < self.layout.Io:
            raise ValueError(
                f"Illegal Isyn : {self.Isyn}A. It should be greater than Io : {self.layout.Io}"
            )


@dataclass
class FeedbackParameters:
    """
    FeedbackParameters contains the positive feedback circuit heuristic parameters of Dynap-SE1 membrane.
    Parameters are used to calculate positive feedback current with respect to the formula below.

    .. math ::
        I_{a} = \\dfrac{I_{a_{gain}}}{1+ exp\\left(-\\dfrac{I_{mem}+I_{a_{th}}}{I_{a_{norm}}}\\right)}

    :param Igain: Feedback gain current, heuristic parameter, defaults to 5e-11
    :type Igain: float, optional
    :param Ith: Feedback threshold current, typically a fraction of Ispk_th, defaults to 5e-10
    :type Ith: float, optional
    :param Inorm: Feedback normalization current, heuristic parameter, defaults to 1e-11
    :type Inorm: float, optional
    """

    Igain: float = 5e-11
    Ith: float = 5e-10
    Inorm: float = 1e-11


@dataclass
class MembraneParameters(DPIParameters):
    """
    MembraneParameters contains membrane specific parameters and state variables

    :param Imem: The sub-threshold current that represents the real neuron’s membrane potential variable, defaults to Io
    :type Imem: Optional[float], optional
    :param feedback: positive feedback circuit heuristic parameters:Ia_gain, Ia_th, and Ia_norm, defaults to None
    :type feedback: Optional[FeedbackParameters], optional
    :param r_Cref: The ratio of refractory and membrane capacitance values :math:`\\dfrac{C_{ref}}{C_{mem}}`
    :type r_Cref: float, optional
    :param r_Cpulse: The ratio of pulse and membrane capacitance values :math:`\\dfrac{C_{pulse}}{C_{mem}}`
    :type r_Cpulse: float, optional

    :Instance Variables:

    :ivar Cref: the capacitance value of the circuit that implements the refractory period
    :type Cref: float
    :ivar Cpulse: the capacitance value of the circuit that converts the spikes to pulses
    :type Cpulse: float
    :ivar f_ref: the capacitance value of the circuit that implements the refractory period
    :type f_ref: float
    :ivar f_pulse: the capacitance value of the circuit that converts the spikes to pulses
    :type f_pulse: float
    """

    C: float = 3.2e-12
    r_Cref: float = 0.1
    r_Cpulse: float = 0.1
    Imem: Optional[float] = None
    feedback: Optional[FeedbackParameters] = None

    def __post_init__(self) -> None:
        """
        __post_init__ runs after __init__ and initializes the feedback block with default values in the case that it's not specified, check if Imem value is in legal bounds.

        :raises ValueError: Illegal Imem value desired. It cannot be less than Io
        """
        super().__post_init__()

        if self.Imem is None:
            self.Imem = self.layout.Io

        if self.Imem < self.layout.Io:
            raise ValueError(
                f"Illegal Imem : {self.Imem}A. It should be greater than Io : {self.layout.Io}"
            )

        self.Cref = self.C * self.r_Cref
        self.Cpulse = self.C * self.r_Cpulse

        self.f_ref = (self.layout.Ut / self.layout.kappa) * self.Cref
        self.f_pulse = (self.layout.Ut / self.layout.kappa) * self.Cpulse

        if self.feedback is None:
            self.feedback = FeedbackParameters()


@dataclass
class AHPParameters(SynapseParameters):
    """
    AHPParameters inherits SynapseParameters and re-arrange the default parameters for AHP circuit
    """

    C: float = 40e-12


@dataclass
class NMDAParameters(SynapseParameters):
    """
    NMDAParameters inherits SynapseParameters and re-arrange the default parameters for NMDA circuit
    """

    C: float = 28e-12


@dataclass
class AMPAParameters(SynapseParameters):
    """
    AMPAParameters inherits SynapseParameters and re-arrange the default parameters for AMPA circuit
    """

    C: float = 28e-12


@dataclass
class GABAAParameters(SynapseParameters):
    """
    GABAAParameters inherits SynapseParameters and re-arrange the default parameters for GABA_A circuit
    """

    C: float = 27e-12
    # Iw: float = 0


@dataclass
class GABABParameters(SynapseParameters):
    """
    GABABParameters inherits SynapseParameters and re-arrange the default parameters for GABA_B circuit
    """

    C: float = 27e-12
    # Iw: float = 0


@dataclass
class DynapSE1SimulationConfiguration:
    """
    DynapSE1SimulationConfiguration encapsulates the DynapSE1 circuit parameters and provides an easy access.

    :param t_ref: refractory period in seconds, limits maximum firing rate, defaults to 15e-3
    :type t_ref: float, optional
    :param t_pulse: the width of the pulse in seconds produced by virtue of a spike, defaults to 1e-5
    :type t_pulse: float, optional
    :param fpulse_ahp: the decrement factor for the pulse widths arriving in AHP circuit, defaults to 0.1
    :type fpulse_ahp: float, optional
    :param Ispkthr: Spiking threshold current in Amperes, depends on layout (see chip for details), defaults to 1e-9
    :type Ispkthr: float, optional
    :param Ireset: Reset current after spike generation in Amperes, defaults to Io
    :type Ireset: Optional[float], optional
    :param Idc: Constant DC current in Amperes, injected to membrane, defaults to Io
    :type Idc: Optional[float], optional
    :param If_nmda: The NMDA gate current in Amperes setting the NMDA gating voltage. If :math:`V_{mem} > V_{nmda}` : The :math:`I_{syn_{NMDA}}` current is added up to the input current, else it cannot. defaults to Io
    :type If_nmda: Optional[float], optional
    :param mem: Membrane block parameters (Imem, Itau, Ith, feedback(Igain, Ith, Inorm)), defaults to None
    :type mem: Optional[MembraneParameters], optional
    :param ahp: Spike frequency adaptation block parameters (Isyn, Itau, Ith, Iw), defaults to None
    :type ahp: Optional[SynapseParameters], optional
    :param nmda: NMDA synapse paramters (Isyn, Itau, Ith, Iw), defaults to None
    :type nmda: Optional[SynapseParameters], optional
    :param ampa: AMPA synapse paramters (Isyn, Itau, Ith, Iw), defaults to None
    :type ampa: Optional[SynapseParameters], optional
    :param gaba_a: GABA_A synapse paramters (Isyn, Itau, Ith, Iw), defaults to None
    :type gaba_a: Optional[SynapseParameters], optional
    :param gaba_b: GABA_B (shunt) synapse paramters (Isyn, Itau, Ith, Iw), defaults to None
    :type gaba_b: Optional[SynapseParameters], optional

    :Instance Variables:

    :ivar t_pulse_ahp: reduced pulse width also look at ``t_pulse`` and ``fpulse_ahp``
    :type t_pulse_ahp: float
    :ivar f_tau_mem: Tau factor for membrane circuit. :math:`f_{\\tau} = \\dfrac{U_T}{\\kappa \\cdot C}`, :math:`f_{\\tau} = I_{\\tau} \\cdot \\tau`
    :type f_tau_mem: float
    :ivar f_tau_syn: A vector of tau factors in the following order: [GABA_B, GABA_A, NMDA, AMPA, AHP]
    :type f_tau_syn: np.ndarray
    :ivar f_t_ref: time factor for refractory period circuit. :math:`f_{\\t} = \\dfrac{U_T}{\\kappa \\cdot C}`, :math:`f_{\\t} = I_{\\t} \\cdot \\t`
    :type f_t_ref: float
    :ivar f_t_pulse: time factor for pulse width generation circuit. :math:`f_{\\t} = \\dfrac{U_T}{\\kappa \\cdot C}`, :math:`f_{\\t} = I_{\\t} \\cdot \\t`
    :type f_t_pulse: float

    [] TODO : Implement get bias currents utility
    """

    t_ref: float = 10e-3
    t_pulse: float = 1e-5
    fpulse_ahp: float = 0.1
    Ispkthr: float = 1e-9
    Ireset: Optional[float] = None
    Idc: Optional[float] = None
    If_nmda: Optional[float] = None
    layout: Optional[DynapSE1Layout] = None
    mem: Optional[MembraneParameters] = None
    ahp: Optional[SynapseParameters] = None
    nmda: Optional[SynapseParameters] = None
    ampa: Optional[SynapseParameters] = None
    gaba_a: Optional[SynapseParameters] = None
    gaba_b: Optional[SynapseParameters] = None

    def __post_init__(self) -> None:
        """
        __post_init__ runs after __init__ and initializes the DPI and membrane blocks with default values in the case that they are not specified.
        """
        if self.layout is None:
            self.layout = DynapSE1Layout()

        # Set the bias currents to Io by default
        if self.Idc is None:
            self.Idc = self.layout.Io
        if self.If_nmda is None:
            self.If_nmda = self.layout.Io
        if self.Ireset is None:
            self.Ireset = self.layout.Io

        _Co = 1e-12

        # Initialize the subcircuit blocks with the same layout
        if self.mem is None:
            self.mem = MembraneParameters(
                C=_Co * 4, r_Cref=0.1, r_Cpulse=0.1, layout=self.layout
            )

        if self.gaba_b is None:
            self.gaba_b = GABABParameters(C=_Co * 20, layout=self.layout)
        if self.gaba_a is None:
            self.gaba_a = GABAAParameters(C=_Co * 2, layout=self.layout)
        if self.nmda is None:
            self.nmda = NMDAParameters(C=_Co * 20, layout=self.layout)
        if self.ampa is None:
            self.ampa = AMPAParameters(C=_Co * 2, layout=self.layout)
        if self.ahp is None:
            self.ahp = AHPParameters(C=_Co * 10, layout=self.layout)

        self.t_pulse_ahp = self.t_pulse * self.fpulse_ahp

        # Membrane
        self.f_tau_mem = self.mem.f_tau
        self.f_t_ref = self.mem.f_ref
        self.f_t_pulse = self.mem.f_pulse

        # All DPI synapses together
        self.f_tau_syn = np.array(
            [
                self.gaba_b.f_tau,
                self.gaba_a.f_tau,
                self.nmda.f_tau,
                self.ampa.f_tau,
                self.ahp.f_tau,
            ]
        )
