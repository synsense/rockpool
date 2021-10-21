"""
Dynap-SE1 Parameter classes to be used in initial configuration of DynapSE1NeuronSynapseJax module

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
24/08/2021
"""
from __future__ import annotations

from rockpool.devices.dynapse.router import NeuronKey

from rockpool.devices.dynapse.biasgen import DynapSE1BiasGen
from jax import numpy as np

import numpy as onp
from dataclasses import dataclass

from typing import Tuple, Any, Optional, Union, List, Dict

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
        "\nDynapSE1SimCore object cannot be factored from a samna config object!",
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

        self.tau, self.Itau = self.time_current_dependence(
            self.tau, self.Itau, self.f_tau
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

    @staticmethod
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

    def time_current_dependence(
        self, tx: Optional[float], Ix: Optional[float], factor: float
    ) -> Tuple[float, float]:
        """
        time_current_dependence calculates the time constant from the current or the current from the time constant
        depending on which one is provided using the factor calculated. In general :math:`f = I_{x} \cdot t_{x}`

        :param tx: the current depended time constant, calculated using Ix if None.
        :type tx: Optional[float]
        :param Ix: the time constant depended current, calculated using tx if None.
        :type Ix: Optional[float]
        :param factor: the f value in :math:`f = I_{x} \cdot t_{x}`
        :type factor: float
        :return: the time constant and the current value calculated and controlled
        :rtype: Tuple[float, float]
        """

        # Infere Ix from tx
        if tx is not None:
            Ix = factor / tx

        # Infere tx from Ix
        elif self.Itau is not None:
            tx = factor / Ix

        self.depended_param_check(tx, Ix, current_limits=(self.layout.Io, None))

        return tx, Ix


@dataclass
class SynapseParameters(DPIParameters):
    """
    SynapseParameters contains DPI specific parameter and state variables

    :param Iw: Synaptic weight current in Amperes, determines how strong the response is in terms of amplitude, defaults to 1e-7
    :type Iw: float, optional
    :param Isyn: DPI output current in Amperes (state variable), defaults to Io
    :type Isyn: Optional[float], optional
    """

    __doc__ += DPIParameters.__doc__

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

    @classmethod
    def _from_parameter_group(
        cls,
        parameter_group: Dynapse1ParameterGroup,
        layout: DynapSE1Layout,
        Itau_name: str,
        Ithr_name: str,
        Iw_name: str,
        *args,
        **kwargs,
    ) -> SynapseParameters:
        """
        _from_parameter_group is a common factory method to construct a `SynapseParameters` object using a `Dynapse1ParameterGroup` object
        Each individual synapse is expected to provide their individual Itau, Ithr and Iw bias names

        :param parameter_group: samna config object for setting the parameter group within one core
        :type parameter_group: Dynapse1ParameterGroup
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSE1Layout
        :param Itau_name: the name of the leak bias current
        :type Itau_name: str
        :param Ithr_name: the name of the gain bias current
        :type Ithr_name: str
        :param Iw_name: the name of the base weight bias current
        :type Iw_name: str
        :return: a `SynapseParameters` object, whose parameters obtained from the hardware configuration
        :rtype: SynapseParameters
        """

        bias = lambda name: DynapSE1BiasGen.param_to_bias(
            parameter_group.get_parameter_by_name(name), Io=layout.Io
        )

        mod = cls(
            Itau=bias(Itau_name),
            Ith=bias(Ithr_name),
            f_gain=None,  # deduced from Ith/Itau
            tau=None,  # deduced from Itau
            layout=layout,
            Iw=bias(Iw_name),
            *args,
            **kwargs,
        )
        return mod


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

    :param Cref: the capacitance value of the circuit that implements the refractory period
    :type Cref: float
    :param Cpulse: the capacitance value of the circuit that converts the spikes to pulses
    :type Cpulse: float
    :param Imem: The sub-threshold current that represents the real neuron’s membrane potential variable, defaults to Io
    :type Imem: Optional[float], optional
    :param Iref: [description], defaults to None
    :type Iref: Optional[float], optional
    :param t_ref: refractory period in seconds, limits maximum firing rate. The value co-depends on `Iref` and `t_ref` definition has priority over `Iref`, defaults to 10e-3
    :type t_ref: Optional[float], optional
    :param Ipulse: the bias current setting `t_pulse`, defaults to None
    :type Ipulse: Optional[float], optional
    :param t_pulse: the width of the pulse in seconds produced by virtue of a spike, The value co-depends on `Ipulse` and `t_pulse` definition has priority over `Ipulse`, defaults to 10e-6
    :type t_pulse: Optional[float], optional
    :param feedback: positive feedback circuit heuristic parameters:Ia_gain, Ia_th, and Ia_norm, defaults to None
    :type feedback: Optional[FeedbackParameters], optional

    :Instance Variables:

    :ivar f_ref: the capacitance value of the circuit that implements the refractory period
    :type f_ref: float
    :ivar f_pulse: the capacitance value of the circuit that converts the spikes to pulses
    :type f_pulse: float
    """

    __doc__ += DPIParameters.__doc__

    C: float = 3.2e-12
    Cref: float = 5e-13
    Cpulse: float = 5e-13
    tau: Optional[float] = 20e-3
    Imem: Optional[float] = None
    Iref: Optional[float] = None
    t_ref: Optional[float] = 10e-3
    Ipulse: Optional[float] = None
    t_pulse: Optional[float] = 10e-6
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

        self.f_ref = (self.layout.Ut / self.layout.kappa) * self.Cref
        self.f_pulse = (self.layout.Ut / self.layout.kappa) * self.Cpulse

        self.t_ref, self.Iref = self.time_current_dependence(
            self.t_ref, self.Iref, self.f_ref
        )
        self.t_pulse, self.Ipulse = self.time_current_dependence(
            self.t_pulse, self.Ipulse, self.f_pulse
        )

        if self.feedback is None:
            self.feedback = FeedbackParameters()

    @classmethod
    def from_parameter_group(
        cls,
        parameter_group: Dynapse1ParameterGroup,
        layout: DynapSE1Layout,
        *args,
        **kwargs,
    ) -> MembraneParameters:
        """
        from_parameter_group is a `MembraneParameters` factory method with hardcoded bias parameter names

        :param parameter_group: samna config object for setting the parameter group within one core
        :type parameter_group: Dynapse1ParameterGroup
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSE1Layout
        :return: a `MembraneParameters` object whose parameters obtained from the hardware configuration
        :rtype: MembraneParameters
        """

        bias = lambda name: DynapSE1BiasGen.param_to_bias(
            parameter_group.get_parameter_by_name(name), Io=layout.Io
        )

        mod = cls(
            Itau=bias("IF_TAU1_N"),
            Ith=bias("IF_THR_N"),
            tau=None,  # deduced from Itau
            f_gain=None,  # deduced from Ith/Itau
            Iref=bias("IF_RFR_N"),
            t_ref=None,  # deduced from Iref
            Ipulse=bias("PULSE_PWLK_P"),
            t_pulse=None,  # deduced from Ipulse
            layout=layout,
            *args,
            **kwargs,
        )
        return mod


@dataclass
class GABABParameters(SynapseParameters):
    """
    GABABParameters inherits SynapseParameters and re-arrange the default parameters for GABA_B circuit
    """

    __doc__ += SynapseParameters.__doc__

    tau: Optional[float] = 100e-3
    # Iw: float = 0

    @classmethod
    def from_parameter_group(
        cls,
        parameter_group: Dynapse1ParameterGroup,
        layout: DynapSE1Layout,
        *args,
        **kwargs,
    ) -> GABABParameters:
        """
        from_parameter_group is a `GABABParameters` factory method implemented by customizing `SynapseParameter._from_parameter_group()`

        :param parameter_group: samna config object for setting the parameter group within one core
        :type parameter_group: Dynapse1ParameterGroup
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSE1Layout
        :return: a `GABABParameters` object, whose parameters obtained from the hardware configuration
        :rtype: GABABParameters
        """

        return cls._from_parameter_group(
            parameter_group,
            layout,
            Itau_name="NPDPII_TAU_F_P",
            Ithr_name="NPDPII_THR_F_P",
            Iw_name="PS_WEIGHT_INH_F_N",
            *args,
            **kwargs,
        )


@dataclass
class GABAAParameters(SynapseParameters):
    """
    GABAAParameters inherits SynapseParameters and re-arrange the default parameters for GABA_A circuit
    """

    __doc__ += SynapseParameters.__doc__

    tau: Optional[float] = 10e-3
    # Iw: float = 0

    @classmethod
    def from_parameter_group(
        cls,
        parameter_group: Dynapse1ParameterGroup,
        layout: DynapSE1Layout,
        *args,
        **kwargs,
    ) -> GABAAParameters:
        """
        from_parameter_group is a `GABAAParameters` factory method implemented by customizing `SynapseParameter._from_parameter_group()`

        :param parameter_group: samna config object for setting the parameter group within one core
        :type parameter_group: Dynapse1ParameterGroup
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSE1Layout
        :return: a `GABAAParameters` object, whose parameters obtained from the hardware configuration
        :rtype: GABAAParameters
        """

        return cls._from_parameter_group(
            parameter_group,
            layout,
            Itau_name="NPDPII_TAU_S_P",
            Ithr_name="NPDPII_THR_S_P",
            Iw_name="PS_WEIGHT_INH_S_N",
            *args,
            **kwargs,
        )


@dataclass
class NMDAParameters(SynapseParameters):
    """
    NMDAParameters inherits SynapseParameters and re-arrange the default parameters for NMDA circuit
    """

    __doc__ += SynapseParameters.__doc__

    tau: Optional[float] = 100e-3

    @classmethod
    def from_parameter_group(
        cls,
        parameter_group: Dynapse1ParameterGroup,
        layout: DynapSE1Layout,
        *args,
        **kwargs,
    ) -> NMDAParameters:
        """
        from_parameter_group is a `NMDAParameters` factory method implemented by customizing `SynapseParameter._from_parameter_group()`

        :param parameter_group: samna config object for setting the parameter group within one core
        :type parameter_group: Dynapse1ParameterGroup
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSE1Layout
        :return: a `NMDAParameters` object, whose parameters obtained from the hardware configuration
        :rtype: NMDAParameters
        """

        return cls._from_parameter_group(
            parameter_group,
            layout,
            Itau_name="NPDPIE_TAU_S_P",
            Ithr_name="NPDPIE_THR_S_P",
            Iw_name="PS_WEIGHT_EXC_S_N",
            *args,
            **kwargs,
        )


@dataclass
class AMPAParameters(SynapseParameters):
    """
    AMPAParameters inherits SynapseParameters and re-arrange the default parameters for AMPA circuit
    """

    __doc__ += SynapseParameters.__doc__

    tau: Optional[float] = 10e-3

    @classmethod
    def from_parameter_group(
        cls,
        parameter_group: Dynapse1ParameterGroup,
        layout: DynapSE1Layout,
        *args,
        **kwargs,
    ) -> AMPAParameters:
        """
        from_parameter_group is a `AMPAParameters` factory method implemented by customizing `SynapseParameter._from_parameter_group()`

        :param parameter_group: samna config object for setting the parameter group within one core
        :type parameter_group: Dynapse1ParameterGroup
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSE1Layout
        :return: a `AMPAParameters` object, whose parameters obtained from the hardware configuration
        :rtype: AMPAParameters
        """

        return cls._from_parameter_group(
            parameter_group,
            layout,
            Itau_name="NPDPIE_TAU_F_P",
            Ithr_name="NPDPIE_THR_F_P",
            Iw_name="PS_WEIGHT_EXC_F_N",
            *args,
            **kwargs,
        )


@dataclass
class AHPParameters(SynapseParameters):
    """
    AHPParameters inherits SynapseParameters and re-arrange the default parameters for AHP circuit
    """

    __doc__ += SynapseParameters.__doc__

    tau: Optional[float] = 50e-3

    @classmethod
    def from_parameter_group(
        cls,
        parameter_group: Dynapse1ParameterGroup,
        layout: DynapSE1Layout,
        *args,
        **kwargs,
    ) -> AHPParameters:
    """
        from_parameter_group is a `AHPParameters` factory method implemented by customizing `SynapseParameter._from_parameter_group()`

        :param parameter_group: samna config object for setting the parameter group within one core
        :type parameter_group: Dynapse1ParameterGroup
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSE1Layout
        :return: a `AHPParameters` object, whose parameters obtained from the hardware configuration
        :rtype: AHPParameters
    """

        return cls._from_parameter_group(
            parameter_group,
            layout,
            Itau_name="IF_AHTAU_N",
            Ithr_name="IF_AHTHR_N",
            Iw_name="IF_AHW_P",
            *args,
            **kwargs,
        )


@dataclass
class DynapSE1SimCore:
    """
    DynapSE1SimCore encapsulates the DynapSE1 circuit parameters and provides an easy access.

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
    :param layout: constant values that are related to the exact silicon layout of a chip, defaults to None
    :type layout: Optional[DynapSE1Layout], optional
    :param capacitance: subcircuit capacitance values that are related to each other and depended on exact silicon layout of a chip, defaults to None
    :type capacitance: Optional[DynapSE1Capacitance], optional
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

    :type t_ref: refractory period in seconds, limits maximum firing rate
    :type t_ref: float, optional
    :type t_pulse: the width of the pulse in seconds produced by virtue of a spike
    :type t_pulse: float, optional
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
    """

    fpulse_ahp: float = 0.1
    Ispkthr: float = 1e-9
    Ireset: Optional[float] = None
    Idc: Optional[float] = None
    If_nmda: Optional[float] = None
    layout: Optional[DynapSE1Layout] = None
    capacitance: Optional[DynapSE1Capacitance] = None
    mem: Optional[MembraneParameters] = None
    gaba_b: Optional[SynapseParameters] = None
    gaba_a: Optional[SynapseParameters] = None
    nmda: Optional[SynapseParameters] = None
    ampa: Optional[SynapseParameters] = None
    ahp: Optional[SynapseParameters] = None

    def __post_init__(self) -> None:
        """
        __post_init__ runs after __init__ and initializes the DPI and membrane blocks with default values in the case that they are not specified.
        """
        if self.layout is None:
            self.layout = DynapSE1Layout()
        if self.capacitance is None:
            self.capacitance = DynapSE1Capacitance()

        # Set the bias currents to Io by default
        if self.Idc is None:
            self.Idc = self.layout.Io
        if self.If_nmda is None:
            self.If_nmda = self.layout.Io
        if self.Ireset is None:
            self.Ireset = self.layout.Io

        # Initialize the subcircuit blocks with the same layout
        if self.mem is None:
            self.mem = MembraneParameters(
                C=self.capacitance.mem,
                Cref=self.capacitance.ref,
                Cpulse=self.capacitance.pulse,
                layout=self.layout,
            )

        if self.gaba_b is None:
            self.gaba_b = GABABParameters(C=self.capacitance.gaba_b, layout=self.layout)
        if self.gaba_a is None:
            self.gaba_a = GABAAParameters(C=self.capacitance.gaba_a, layout=self.layout)
        if self.nmda is None:
            self.nmda = NMDAParameters(C=self.capacitance.nmda, layout=self.layout)
        if self.ampa is None:
            self.ampa = AMPAParameters(C=self.capacitance.ampa, layout=self.layout)
        if self.ahp is None:
            self.ahp = AHPParameters(C=self.capacitance.ahp, layout=self.layout)

        self.t_pulse = self.mem.t_pulse
        self.t_ref = self.mem.t_ref
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

    @classmethod
    def from_samna_parameter_group(
        cls,
        parameter_group: Dynapse1ParameterGroup,
        fpulse_ahp: float = 0.1,
        Ispkthr: float = 1e-9,
        Ireset: Optional[float] = None,
        layout: Optional[DynapSE1Layout] = None,
        capacitance: Optional[DynapSE1Capacitance] = None,
    ) -> DynapSE1SimCore:
        """
        from_samna_parameter_group create a simulation configuration object a the samna config object.
        21/25 parameters used in the configuration of the simulator object.
        "IF_BUF_P", "IF_CASC_N", "R2R_P", and "IF_TAU2_N" has no effect on the simulator.
        The parameters which cannot be obtained from the parameter_group object should be defined explicitly

        :param parameter_group: samna config object for setting the parameter group within one core
        :type parameter_group: Dynapse1ParameterGroup
        :param fpulse_ahp: the decrement factor for the pulse widths arriving in AHP circuit, defaults to 0.1
        :type fpulse_ahp: float, optional
        :param Ispkthr: Spiking threshold current in Amperes, depends on layout (see chip for details), defaults to 1e-9
        :type Ispkthr: float, optional
        :param Ireset: Reset current after spike generation in Amperes, defaults to Io
        :type Ireset: Optional[float], optional
        :param layout: constant values that are related to the exact silicon layout of a chip, defaults to None
        :type layout: Optional[DynapSE1Layout], optional
        :param capacitance: subcircuit capacitance values that are related to each other and depended on exact silicon layout of a chip, defaults to None
        :type capacitance: Optional[DynapSE1Capacitance], optional
        :return: simulator config object to construct a `DynapSE1NeuronSynapseJax` object
        :rtype: DynapSE1SimCore
        """

        if layout is None:
            layout = DynapSE1Layout()

        if capacitance is None:
            capacitance = DynapSE1Capacitance()

        bias = lambda name: DynapSE1BiasGen.param_to_bias(
            parameter_group.get_parameter_by_name(name), Io=layout.Io
        )

        mem = MembraneParameters.from_parameter_group(
            parameter_group,
            layout,
            C=capacitance.mem,
            Cref=capacitance.ref,
            Cpulse=capacitance.pulse,
        )

        # Fast inhibitory (shunt)
        gaba_b = GABABParameters.from_parameter_group(
            parameter_group,
            layout,
            C=capacitance.gaba_b,
        )

        # Slow inhibitory
        gaba_a = GABAAParameters.from_parameter_group(
            parameter_group,
            layout,
            C=capacitance.gaba_a,
        )

        # Slow Excitatory
        nmda = NMDAParameters.from_parameter_group(
            parameter_group,
            layout,
            C=capacitance.nmda,
        )

        # Fast Excitatory
        ampa = AMPAParameters.from_parameter_group(
            parameter_group,
            layout,
            C=capacitance.ampa,
        )

        ahp = AHPParameters.from_parameter_group(
            parameter_group,
            layout,
            C=capacitance.ahp,
        )

        mod = cls(
            fpulse_ahp=fpulse_ahp,
            Ispkthr=Ispkthr,
            Ireset=Ireset,
            Idc=bias("IF_DC_P"),
            If_nmda=bias("IF_NMDA_N"),
            layout=layout,
            capacitance=capacitance,
            mem=mem,
            gaba_b=gaba_b,
            gaba_a=gaba_a,
            nmda=nmda,
            ampa=ampa,
            ahp=ahp,
        )
        return mod
