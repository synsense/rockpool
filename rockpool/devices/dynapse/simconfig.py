"""
Dynap-SE Parameter classes to be used in initial configuration of DynapSEAdExpLIFJax module

renamed : dynapse1_simconfig.py -> simconfig.py @ 211208

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
24/08/2021
"""
from __future__ import annotations

from rockpool.devices.dynapse.router import NeuronKey

from rockpool.devices.dynapse.biasgen import DynapSE1BiasGen
from rockpool.devices.dynapse.router import Router
from jax import numpy as np

import numpy as onp
from dataclasses import dataclass

from typing import Tuple, Any, Optional, List, Dict, Generator

import itertools

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
                self.__setattr__(name, onp.float32(value * self.Co))


@dataclass
class DynapSE1Layout:
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

    @staticmethod
    def bias(
        layout: DynapSE1Layout, parameter_group: Dynapse1ParameterGroup, name: str
    ) -> float:
        """
        extract and obtain the bias current from the samna parameter group using bias generator

        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSE1Layout
        :param parameter_group: samna config object for setting the parameter group within one core
        :type parameter_group: Dynapse1ParameterGroup
        :param name: the parameter name of the bias current
        :type name: str
        :return: biasgen corrected bias value by multiplying a correction factor
        :rtype: float
        """
        return DynapSE1BiasGen.param_to_bias(
            parameter_group.get_parameter_by_name(name), Io=layout.Io
        )

    @property
    def f_tau(self) -> float:
        """
        f_tau is the tau factor for DPI circuit. :math:`f_{\\tau} = \\dfrac{U_T}{\\kappa \\cdot C}`, :math:`f_{\\tau} = I_{\\tau} \\cdot \\tau`
        """
        return (self.layout.Ut / self.layout.kappa) * self.C


@dataclass
class WeightParameters:
    """
    WeightParameters encapsulates weight currents of the configurable synapses between neurons. It provides a general way of handling SE1 and SE2 base weight currents.

    :param Iw_0: the first base weight current corresponding to the 0th bit of the bit-mask, in Amperes. In DynapSE1, it's GABA_B base weigth.
    :type Iw_0: float
    :param Iw_1: the second base weight current corresponding to the 1st bit of the bit-mask, in Amperes. In DynapSE1, it's GABA_A base weigth.
    :type Iw_1: float
    :param Iw_2: the third base weight current corresponding to the 2nd bit of the bit-mask, in Amperes. In DynapSE1, it's NMDA base weigth.
    :type Iw_2: float
    :param Iw_3: the fourth base weight current corresponding to the 3rd bit of the bit-mask, in Amperes. In DynapSE1, it's AMPA base weigth.
    :type Iw_3: float
    :param layout: constant values that are related to the exact silicon layout of a chip, defaults to None
    :type layout: Optional[DynapSE1Layout], optional
    """

    Iw_0: float = 1e-6  # GABA_B
    Iw_1: float = 2e-6  # GABA_A
    Iw_2: float = 4e-6  # NMDA
    Iw_3: float = 8e-6  # AMPA
    layout: Optional[DynapSE1Layout] = None

    def __post_init__(self) -> None:
        """
        __post_init__ runs after __init__ and initializes the WeightParameters block, checks if given values are in legal bounds.
        """

        if self.layout is None:
            self.layout = DynapSE1Layout()

    @classmethod
    def from_parameter_group(
        cls,
        parameter_group: Dynapse1ParameterGroup,
        layout: DynapSE1Layout,
        *args,
        **kwargs,
    ) -> WeightParameters:
        """
        from_parameter_group is a factory method to construct a `WeightParameters` object using a `Dynapse1ParameterGroup` object

        :param parameter_group: samna config object for setting the parameter group within one core
        :type parameter_group: Dynapse1ParameterGroup
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSE1Layout
        :return: a `WeightParameters` object, whose parameters obtained from the hardware configuration
        :rtype: WeightParameters
        """

        bias = lambda name: DynapSE1BiasGen.param_to_bias(
            parameter_group.get_parameter_by_name(name), Io=layout.Io
        )

        mod = cls(
            Iw_0=bias("PS_WEIGHT_INH_S_N"),  # GABA_B
            Iw_1=bias("PS_WEIGHT_INH_F_N"),  # GABA_A
            Iw_2=bias("PS_WEIGHT_EXC_S_N"),  # NMDA
            Iw_3=bias("PS_WEIGHT_EXC_F_N"),  # AMPA
            layout=layout,
            *args,
            **kwargs,
        )
        return mod

    def get_vector(self) -> np.ndarray:
        """
        get_vector gather the base weight currents together and creates a base weight vector

        :return: a base weight vector each index representing the same index bit. i.e. Iw[0] = Iw_0, Iw[1] = Iw_1 et
        :rtype: np.ndarray
        """
        weights = np.array([self.Iw_0, self.Iw_1, self.Iw_2, self.Iw_3])
        return weights


@dataclass
class SynapseParameters(DPIParameters):
    """
    SynapseParameters contains DPI specific parameter and state variables
    :param Isyn: DPI output current in Amperes (state variable), defaults to Io
    :type Isyn: Optional[float], optional
    """

    __doc__ += DPIParameters.__doc__

    Isyn: Optional[float] = 1e-10

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
        *args,
        **kwargs,
    ) -> SynapseParameters:
        """
        _from_parameter_group is a common factory method to construct a `SynapseParameters` object using a `Dynapse1ParameterGroup` object
        Each individual synapse is expected to provide their individual Itau and Ithr bias names

        :param parameter_group: samna config object for setting the parameter group within one core
        :type parameter_group: Dynapse1ParameterGroup
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSE1Layout
        :param Itau_name: the name of the leak bias current
        :type Itau_name: str
        :param Ithr_name: the name of the gain bias current
        :type Ithr_name: str
        :return: a `SynapseParameters` object, whose parameters obtained from the hardware configuration
        :rtype: SynapseParameters
        """

        bias = lambda name: cls.bias(layout, parameter_group, name)

        mod = cls(
            Itau=bias(Itau_name),
            Ith=bias(Ithr_name),
            f_gain=None,  # deduced from Ith/Itau
            tau=None,  # deduced from Itau
            layout=layout,
            *args,
            **kwargs,
        )
        return mod


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
    :param tau2: Secondary membrane time constant, co-depended to Itau2. In the case its provided, Itau2 is infered from tau. A neuron's time constant can be switched to tau or tau2. defaults to None
    :type tau2: Optional[float], optional
    :param Itau2: Secondary membrane time constant current in Amperes. A neuron's time constant can be switched to tau or tau2. defaults to 2.4e-5
    :type Itau2: float, optional
    :param Iref: the bias current setting the `t_ref`, defaults to None
    :type Iref: Optional[float], optional
    :param t_ref: refractory period in seconds, limits maximum firing rate. In the refractory period the synaptic input current of the membrane is the dark current. The value co-depends on `Iref` and `t_ref` definition has priority over `Iref`, defaults to 10e-3
    :type t_ref: Optional[float], optional
    :param Ipulse: the bias current setting `t_pulse`, defaults to None
    :type Ipulse: Optional[float], optional
    :param t_pulse: the width of the pulse in seconds produced by virtue of a spike, The value co-depends on `Ipulse` and `t_pulse` definition has priority over `Ipulse`, defaults to 10e-6
    :type t_pulse: Optional[float], optional
    :param Ispkthr: Spiking threshold current in Amperes, depends on layout (see chip for details), defaults to 1e-9
    :type Ispkthr: float, optional
    :param Ireset: Reset current after spike generation in Amperes, defaults to Io
    :type Ireset: Optional[float], optional
    :param Idc: Constant DC current in Amperes, injected to membrane, defaults to Io
    :type Idc: Optional[float], optional
    :param If_nmda: The NMDA gate current in Amperes setting the NMDA gating voltage. If :math:`V_{mem} > V_{nmda}` : The :math:`I_{syn_{NMDA}}` current is added up to the input current, else it cannot. defaults to Io
    :type If_nmda: Optional[float], optional
    """

    __doc__ += DPIParameters.__doc__

    C: float = 3e-12
    Cref: float = 1.5e-12
    Cpulse: float = 5e-13
    tau: Optional[float] = 20e-3
    tau2: Optional[float] = None
    Itau2: Optional[float] = 2.4e-5  # Max bias current possible
    f_gain: float = 2
    Imem: Optional[float] = None
    Iref: Optional[float] = None
    t_ref: Optional[float] = 2e-3
    Ipulse: Optional[float] = None
    t_pulse: Optional[float] = 10e-6
    Ispkthr: float = 1e-6
    Ireset: Optional[float] = None
    Idc: Optional[float] = None
    If_nmda: Optional[float] = None

    def __post_init__(self) -> None:
        """
        __post_init__ runs after __init__ and initializes the with default values in the case that it's not specified, check if Imem value is in legal bounds.

        :raises ValueError: Illegal Imem value desired. It cannot be less than Io
        """
        super().__post_init__()

        if self.Imem is None:
            self.Imem = self.layout.Io
        if self.Idc is None:
            self.Idc = self.layout.Io
        if self.If_nmda is None:
            self.If_nmda = self.layout.Io
        if self.Ireset is None:
            self.Ireset = self.layout.Io

        if self.Imem < self.layout.Io:
            raise ValueError(
                f"Illegal Imem : {self.Imem}A. It should be greater than Io : {self.layout.Io}"
            )

        self.tau2, self.Itau2 = self.time_current_dependence(
            self.tau2, self.Itau2, self.f_tau
        )

        self.t_ref, self.Iref = self.time_current_dependence(
            self.t_ref, self.Iref, self.f_ref
        )
        self.t_pulse, self.Ipulse = self.time_current_dependence(
            self.t_pulse, self.Ipulse, self.f_pulse
        )

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

        bias = lambda name: cls.bias(layout, parameter_group, name)

        mod = cls(
            Itau=bias("IF_TAU1_N"),
            Ith=bias("IF_THR_N"),
            tau=None,  # deduced from Itau
            tau2=None,  # deduced from Itau2
            Itau2=bias("IF_TAU2_N"),
            f_gain=None,  # deduced from Ith/Itau
            Iref=bias("IF_RFR_N"),
            t_ref=None,  # deduced from Iref
            Ipulse=bias("PULSE_PWLK_P"),
            t_pulse=None,  # deduced from Ipulse
            Idc=bias("IF_DC_P"),
            If_nmda=bias("IF_NMDA_N"),
            layout=layout,
            *args,
            **kwargs,
        )
        return mod

    @property
    def f_ref(self) -> float:
        """
        f_ref is a the recractory period factor for DPI circuit. :math:`f_{ref} = V_{on} \\cdot C_{ref}`, :math:`f_{ref} = I_{ref} \\cdot \\t_{ref}`
        """
        return self.layout.Von * self.Cref

    @property
    def f_pulse(self) -> float:
        """
        f_pulse is a the pulse width factor for DPI circuit. :math:`f_{pulse} = V_{on} \\cdot C_{pulse}`, :math:`f_{pulse} = I_{pulse} \\cdot \\t_{pulse}`
        """
        return self.layout.Von * self.Cpulse


@dataclass
class GABABParameters(SynapseParameters):
    """
    GABABParameters inherits SynapseParameters and re-arrange the default parameters for GABA_B circuit
    """

    __doc__ += SynapseParameters.__doc__

    C: float = 2.5e-11
    tau: Optional[float] = 100e-3

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
            Itau_name="NPDPII_TAU_S_P",
            Ithr_name="NPDPII_THR_S_P",
            *args,
            **kwargs,
        )


@dataclass
class GABAAParameters(SynapseParameters):
    """
    GABAAParameters inherits SynapseParameters and re-arrange the default parameters for GABA_A circuit
    """

    __doc__ += SynapseParameters.__doc__

    C: float = 2.45e-11
    tau: Optional[float] = 10e-3

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
            Itau_name="NPDPII_TAU_F_P",
            Ithr_name="NPDPII_THR_F_P",
            *args,
            **kwargs,
        )


@dataclass
class NMDAParameters(SynapseParameters):
    """
    NMDAParameters inherits SynapseParameters and re-arrange the default parameters for NMDA circuit
    """

    __doc__ += SynapseParameters.__doc__

    C: float = 2.5e-11
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
            *args,
            **kwargs,
        )


@dataclass
class AMPAParameters(SynapseParameters):
    """
    AMPAParameters inherits SynapseParameters and re-arrange the default parameters for AMPA circuit
    """

    __doc__ += SynapseParameters.__doc__

    C: float = 2.45e-11
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
            *args,
            **kwargs,
        )


@dataclass
class AHPParameters(SynapseParameters):
    """
    AHPParameters inherits SynapseParameters and re-arrange the default parameters for AHP circuit
    """

    __doc__ += SynapseParameters.__doc__

    C: float = 4e-11
    tau: Optional[float] = 50e-3
    Iw: float = 1e-6

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
            Iw=cls.bias(layout, parameter_group, "IF_AHW_P"),
            *args,
            **kwargs,
        )


@dataclass
class DynapSE1SimCore:
    """
    DynapSE1SimCore encapsulates the DynapSE1 circuit parameters and provides an easy access.

    :param size: the number of neurons allocated in the simulation core, the length of the property arrays, defaults to None
    :type size: Optional[int], optional
    :param core_key: the chip_id and core_id tuple uniquely defining the core, defaults to None
    :type core_key: Optional[Tuple[np.uint8]], optional
    :param neuron_idx_map: the neuron index map used in the case that the matrix indexes of the neurons and the device indexes are different.
    :type neuron_idx_map: Dict[np.uint8, np.uint16]
    :param fpulse_ahp: the decrement factor for the pulse widths arriving in AHP circuit, defaults to 0.1
    :type fpulse_ahp: float, optional
    :param layout: constant values that are related to the exact silicon layout of a chip, defaults to None
    :type layout: Optional[DynapSE1Layout], optional
    :param capacitance: subcircuit capacitance values that are related to each other and depended on exact silicon layout of a chip, defaults to None
    :type capacitance: Optional[DynapSE1Capacitance], optional
    :param mem: Membrane block parameters (Imem, Itau, Ith), defaults to None
    :type mem: Optional[MembraneParameters], optional
    :param ahp: Spike frequency adaptation block parameters (Isyn, Itau, Ith, Iw), defaults to None
    :type ahp: Optional[SynapseParameters], optional
    :param nmda: NMDA synapse paramters (Isyn, Itau, Ith, Iw), defaults to None
    :type nmda: Optional[SynapseParameters], optional
    :param ampa: AMPA synapse paramters (Isyn, Itau, Ith, Iw), defaults to None
    :type ampa: Optional[SynapseParameters], optional
    :param gaba_a: GABA_A (shunt) synapse paramters (Isyn, Itau, Ith, Iw), defaults to None
    :type gaba_a: Optional[SynapseParameters], optional
    :param gaba_b: GABA_B synapse paramters (Isyn, Itau, Ith, Iw), defaults to None
    :type gaba_b: Optional[SynapseParameters], optional

    :Instance Variables:

    :ivar matrix_id_iter: static matrix ID counter. Increase whenever a new object created with the default initialization routine. Increment by 1 for each and every neruon.
    :type matrix_id_iter: itertools.count
    :ivar chip_core_iter: static chip_core key counter. Increase whenever a new object created with the default initialization routine. Increment by NUM_NEURONS(256), then decode the UID to obtain the chip and core IDs.
    :type chip_core_iter: itertools.count
    """

    size: Optional[int] = None
    core_key: Optional[Tuple[np.uint8]] = None
    neuron_idx_map: Dict[np.uint8, np.uint16] = None
    fpulse_ahp: float = 0.1
    layout: Optional[DynapSE1Layout] = None
    capacitance: Optional[DynapSE1Capacitance] = None
    mem: Optional[MembraneParameters] = None
    gaba_b: Optional[SynapseParameters] = None
    gaba_a: Optional[SynapseParameters] = None
    nmda: Optional[SynapseParameters] = None
    ampa: Optional[SynapseParameters] = None
    ahp: Optional[SynapseParameters] = None

    # Static counters for default construction
    matrix_id_iter = itertools.count()
    chip_core_iter = itertools.count(step=Router.NUM_NEURONS)

    def __post_init__(self) -> None:
        """
        __post_init__ runs after __init__ and initializes the DPI and membrane blocks with default values in the case that they are not specified.

        :raises ValueError: Either provide the size or neuron index map!
        """
        if self.size is None:
            if self.neuron_idx_map is not None:
                self.size = len(self.neuron_idx_map)
            else:
                raise ValueError("Either provide the size or neuron index map!")

        if self.layout is None:
            self.layout = DynapSE1Layout()

        if self.capacitance is None:
            self.capacitance = DynapSE1Capacitance()

        if self.core_key is None:
            # Create a neuron key depending on the static chip_core counter then check if respective neuron ID is valid
            _neuron_key = Router.decode_UID(next(self.chip_core_iter))
            Router.get_UID(*_neuron_key[0:2], self.size - 1)
            self.core_key = _neuron_key[0:2]

        if self.neuron_idx_map is None:
            # Create a neuron index map depending on the static matrix ID counter
            matrix_ids = [next(self.matrix_id_iter) for i in range(self.size)]
            self.neuron_idx_map = dict(zip(matrix_ids, range(self.size)))

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

    def __len__(self):
        return self.size

    @staticmethod
    def reset(cls: DynapSE1SimCore):
        """
        reset resets the static counters of a `DynapSE1SimCore` class to 0

        :param cls: the class to reset
        :type cls: DynapSE1SimCore

        Usage:
        from rockpool.devices.dynapse.simconfig import DynapSE1SimCore as simcore
        simcore.reset(simcore)
        """
        cls.matrix_id_iter = itertools.count()
        cls.chip_core_iter = itertools.count(step=Router.NUM_NEURONS)

    @classmethod
    def from_samna_parameter_group(
        cls,
        parameter_group: Dynapse1ParameterGroup,
        size: int,
        core_key: Tuple[np.uint8] = (0, 0),
        neuron_idx_map: Optional[Dict[np.uint8, np.uint16]] = None,
        fpulse_ahp: float = 0.1,
        Ispkthr: float = 1e-6,
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
        :param size: the number of neurons allocated in the simulation core, the length of the property arrays.
        :type size: int
        :param core_key: the chip_id and core_id tuple uniquely defining the core
        :type core_key: Tuple[np.uint8]
        :param neuron_idx_map: the neuron index map used in the case that the matrix indexes of the neurons and the device indexes are different, defaults to None
        :type neuron_idx_map: Optional[Dict[np.uint8, np.uint16]], optional
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
        :return: simulator config object to construct a `DynapSEAdExpLIFJax` object
        :rtype: DynapSE1SimCore
        """

        if layout is None:
            layout = DynapSE1Layout()

        if capacitance is None:
            capacitance = DynapSE1Capacitance()

        mem = MembraneParameters.from_parameter_group(
            parameter_group,
            layout,
            C=capacitance.mem,
            Cref=capacitance.ref,
            Cpulse=capacitance.pulse,
            Ispkthr=Ispkthr,
            Ireset=Ireset,
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
            size=size,
            core_key=core_key,
            neuron_idx_map=neuron_idx_map,
            fpulse_ahp=fpulse_ahp,
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

    ## Property Utils ##

    def _get_syn_vector(self, target: str) -> np.ndarray:
        """
        _get_syn_vector lists the parameters traversing the different object instances of the same class

        :param target: target parameter to be extracted and listed in the order of object list
        :type target: str
        :return: An array of target parameter values obtained from the objects
        :rtype: np.ndarray
        """

        object_list = [self.gaba_b, self.gaba_a, self.nmda, self.ampa, self.ahp]

        param_list = [
            obj.__getattribute__(target) if obj is not None else None
            for obj in object_list
        ]

        return np.array(param_list)

    def mem_property(self, attr: str) -> np.ndarray:
        """
        mem_property fetches an attribute from the membrane subcircuit and create a property array covering all the neurons allocated

        :param attr: the target attribute
        :type attr: str
        :return: 1D an array full of the value of the target attribute (`size`,)
        :rtype: np.ndarray
        """
        return np.full(self.size, self.mem.__getattribute__(attr), dtype=np.float32)

    def layout_property(self, attr: str) -> np.ndarray:
        """
        layout_property fetches an attribute from the circuit layout and create a property array covering all the neurons allocated

        :param attr: the target attribute
        :type attr: str
        :return: 1D an array full of the value of the target attribute (`size`,)
        :rtype: np.ndarray
        """
        return np.full(self.size, self.layout.__getattribute__(attr), dtype=np.float32)

    def syn_property(self, attr: str) -> np.ndarray:
        """
        syn_property fetches a attributes from the synaptic gate subcircuits in [GABA_B, GABA_A, NMDA, AMPA, AHP] order and create a property array covering all the neurons allocated

        :param attr: the target attribute
        :type attr: str
        :return: 2D an array full of the values of the target attributes of all 5 synapses (`size`, 5)
        :rtype: np.ndarray
        """
        return np.full((self.size, 5), self._get_syn_vector(attr), dtype=np.float32).T

    ## -- Property Utils End -- ##

    @property
    def Imem(self) -> np.ndarray:
        """
        Imem is an array of membrane currents of the neurons with shape = (Nrec,)
        """
        return self.mem_property("Imem")

    @property
    def Itau_mem(self) -> np.ndarray:
        """
        Itau_mem is an array of membrane leakage currents of the neurons with shape = (Nrec,)
        """
        return self.mem_property("Itau")

    @property
    def Itau2_mem(self) -> np.ndarray:
        """
        Itau2_mem is an array of secondary membrane leakage currents of the neurons with shape = (Nrec,)
        """
        return self.mem_property("Itau2")

    @property
    def f_gain_mem(self) -> np.ndarray:
        """
        f_gain_mem is an array of membrane gain parameter of the neurons with shape = (Nrec,)
        """
        return self.mem_property("f_gain")

    @property
    def Isyn(self) -> np.ndarray:
        """
        Isyn is a 2D array of synapse currents of the neurons in the order of [GABA_B, GABA_A, NMDA, AMPA, AHP] with shape = (5,Nrec)
        """
        return self.syn_property("Isyn")

    @property
    def Itau_syn(self) -> np.ndarray:
        """
        Itau_syn is a 2D array of synapse leakage currents of the neurons in the order of [GABA_B, GABA_A, NMDA, AMPA, AHP] with shape = (5,Nrec)
        """
        return self.syn_property("Itau")

    @property
    def f_gain_syn(self) -> np.ndarray:
        """
        f_gain_syn is a 2D array of synapse gain parameters of the neurons in the order of [GABA_B, GABA_A, NMDA, AMPA, AHP] with shape = (5,Nrec)
        """
        return self.syn_property("f_gain")

    @property
    def Iw(self) -> np.ndarray:
        """
        Iw is a 2D array of synapse weight currents of the neurons in the order of [GABA_B, GABA_A, NMDA, AMPA, AHP] with shape = (5,Nrec)
        """
        return self.syn_property("Iw")

    @property
    def kappa(self) -> np.ndarray:
        """
        kappa is the mean subthreshold slope factor of the transistors with shape (Nrec,)
        """
        return self.layout_property("kappa")

    @property
    def Ut(self) -> np.ndarray:
        """
        Ut is the thermal voltage in Volts with shape (Nrec,)
        """
        return self.layout_property("Ut")

    @property
    def Io(self) -> np.ndarray:
        """
        Io is the dark current in Amperes that flows through the transistors even at the idle state with shape (Nrec,)
        """
        return self.layout_property("Io")

    @property
    def f_tau_mem(self) -> np.ndarray:
        """
        f_tau_mem is an array of tau factor for membrane circuit. :math:`f_{\\tau} = \\dfrac{U_T}{\\kappa \\cdot C}`, :math:`f_{\\tau} = I_{\\tau} \\cdot \\tau` with shape (Nrec,)
        """
        return self.mem_property("f_tau")

    @property
    def f_tau_syn(self) -> np.ndarray:
        """
        f_tau_syn is a 2D array of tau factors in the following order: [GABA_B, GABA_A, NMDA, AMPA, AHP] with shape (5, Nrec)
        """
        return self.syn_property("f_tau")

    @property
    def f_t_ref(self) -> np.ndarray:
        """
        f_t_ref is an array of the factor of conversion from the refractory period current to the refractory period shape (Nrec,)
        """
        return self.mem_property("f_ref")

    @property
    def f_t_pulse(self) -> np.ndarray:
        """
        f_t_pulse is an array of the factor of conversion from pulse width in seconds to pulse width bias current in Amperes with shape (Nrec,)
        """
        return self.mem_property("f_pulse")

    @property
    def t_pulse(self) -> np.ndarray:
        """
        t_pulse is an array of the width of the pulse in seconds produced by virtue of a spike with shape (Nrec,)
        """
        return self.mem_property("t_pulse")

    @property
    def t_pulse_ahp(self) -> np.ndarray:
        """
        t_pulse_ahp is an array of reduced pulse width also look at ``t_pulse`` and ``fpulse_ahp`` with shape (Nrec,)
        """
        return self.t_pulse * self.fpulse_ahp

    @property
    def t_ref(self) -> np.ndarray:
        """
        t_ref is an array of the refractory period in seconds, limits maximum firing rate. In the refractory period the synaptic input current of the membrane is the dark current. with shape (Nrec,)
        """
        return self.mem_property("t_ref")

    @property
    def Ispkthr(self) -> np.ndarray:
        """
        Ispkthr is an array of the spiking threshold current in Amperes with shape  shape (Nrec,)
        """
        return self.mem_property("Ispkthr")

    @property
    def Ireset(self) -> np.ndarray:
        """
        Ireset is an array of the reset current after spike generation with shape (Nrec,)
        """
        return self.mem_property("Ireset")

    @property
    def Idc(self) -> np.ndarray:
        """
        Idc is an array of the constant DC current in Amperes, injected to membrane with shape (Nrec,)
        """
        return self.mem_property("Idc")

    @property
    def If_nmda(self) -> np.ndarray:
        """
        If_nmda is an array of the The NMDA gate current in Amperes setting the NMDA gating voltage. If :math:`V_{mem} > V_{nmda}` : The :math:`I_{syn_{NMDA}}` current is added up to the input current, else it cannot with shape (Nrec,)
        """
        return self.mem_property("If_nmda")


@dataclass
class DynapSE1SimBoard:
    """
    DynapSE1SimBoard encapsulates the DynapSE1 core configuration objects and help merging simulation
    parameters implicitly defined in the cores. It makes it easier to run a multi-core simulation and simulate the mismatch

    :param size: the number of neurons allocated in the simulation board, the length of the property arrays. If the size is greater than the capacity of one core, then the neurons are splitted across different cores. If `cores` are defined, then the size are obtained from `cores` list. defaults to 1
    :type size: Optional[int], optional
    :param cores: a list of `DynapSE1SimCore` objects whose parameters are to be merged into one simulation base, defaults to None
    :type cores: Optional[List[DynapSE1SimCore]], optional
    :param idx_map: a dictionary of the mapping between matrix indexes of the neurons and their global unique neuron keys
    :type idx_map: Dict[int, NeuronKey]
    """

    size: Optional[int] = 1
    cores: Optional[List[DynapSE1SimCore]] = None
    idx_map: Optional[Dict[int, NeuronKey]] = None

    def __post_init__(self) -> None:
        """
        __post_init__ runs after __init__ and merges DynapSE1SimCore properties in the given order

        :raises ValueError: Core size does not match the number of neurons to allocate!
        :raises ValueError: Core key in the index map and core key in the `DynapSE1SimCore` does not match!
        :raises ValueError: Neuron index map for core in the `idx_map`, and the neuron index map in `DynapSE1SimCore` does not match!
        :raises ValueError: The board configuration object size and number of device neruons indicated in idx_map does not match!
        :raises ValueError: size is required for default construction!
        """

        if self.cores is None:
            # Default construction of the simulation cores given the size
            if self.size is None:
                raise ValueError("size is required for default construction!")
            split_list = DynapSE1SimBoard.split_across_cores(self.size)
            DynapSE1SimCore.reset(DynapSE1SimCore)
            self.cores = [DynapSE1SimCore(size) for size in split_list]
            # 600 -> [256, 256, 88]

        if self.idx_map is None:
            # Collect the index map from the simulation cores
            self.idx_map = self.collect_idx_map_from_cores(self.cores)

        if self.size is None:
            self.size = len(self.idx_map)

        core_dict = DynapSE1SimBoard.idx_map_to_core_dict(self.idx_map)
        self.size = self.__len__()

        # Check if simulation core's meta-information match with the global maps
        for core, (core_key, neuron_map) in zip(self.cores, core_dict.items()):
            if len(core) != len(neuron_map):
                raise ValueError(
                    f"Core size = {len(core)} does not match the number of neurons to allocate! {neuron_map}"
                )
            if core.core_key != core_key:
                raise ValueError(
                    f"Core key in the index map {core_key} and core key in the `DynapSE1SimCore` {core.core_key} does not match!"
                )
            if core.neuron_idx_map != neuron_map:
                raise ValueError(
                    f"Neuron index map for core {core_key} in the `idx_map` {neuron_map}, and the neuron index map in `DynapSE1SimCore` {core.neuron_idx_map } does not match!"
                )

        if self.__len__() != len(self.idx_map):
            raise ValueError(
                f"The board configuration object size {self.__len__()} and number of device neruons indicated in idx_map {len(self.idx_map)} does not match!"
            )

        attr_list = [
            "Imem",
            "Itau_mem",
            "Itau2_mem",
            "f_gain_mem",
            "Isyn",
            "Itau_syn",
            "f_gain_syn",
            "Iw",
            "kappa",
            "Ut",
            "Io",
            "f_tau_mem",
            "f_tau_syn",
            "f_t_ref",
            "f_t_pulse",
            "t_pulse",
            "t_pulse_ahp",
            "t_ref",
            "Ispkthr",
            "Ireset",
            "Idc",
            "If_nmda",
        ]

        for attr in attr_list:
            self.__setattr__(attr, self.merge_core_properties(attr))

    def __len__(self):
        size = 0
        for core in self.cores:
            size += len(core)
        return size

    @staticmethod
    def split_across_cores(
        size: int, neruon_per_core: int = Router.NUM_NEURONS
    ) -> List[int]:
        """
        split_across_cores is a helper function to split some random number of neurons across different cores
        600 -> [256, 256, 88]

        :param size: desired simulation size, the total number of neurons
        :type size: int
        :param neruon_per_core: maximum number of neruons to be allocated in a single core, defaults to Router.NUM_NEURONS
        :type neruon_per_core: int, optional
        :return: a list of number of neurons to split across different cores
        :rtype: List[int]
        """

        def it(n_neurons: int) -> Generator[int]:
            """
            it a while iterator to be used in list comprehension.
            Yield maksimum number of neurons to allocate in one core `neruon_per_core`
            if the number of neurons left is greater than the maksimum number of neurons.
            Else yield the number of neruons left if the number of neurons left greater than 0.

            :param n_neurons: total number of neurons to allocate
            :type n_neurons: int
            :yield: exact number of neurons for next step
            :rtype: Generator[int]
            """
            while n_neurons > 0:
                yield neruon_per_core if n_neurons > neruon_per_core else n_neurons
                n_neurons -= neruon_per_core

        split_list = [i for i in it(size)]
        return split_list

    def merge_core_properties(self, attr: str) -> np.ndarray:
        """
        merge_core_properties help merging property arrays of the cores given into one array.
        In this way, the simulator runs in the neruon level and it only knows the location of the
        neruon by looking at the index map. In this way, the simulator module can compute the evolution of
        the currents and spikes efficiently.

        :param attr: the target attribute to be merged
        :type attr: str
        :return: The parameter or state to be used in the simulation
        :rtype: np.ndarray
        """

        prop = np.empty((*self.cores[0].__getattribute__(attr).shape[0:-1], 0))
        for core in self.cores:
            prop = np.hstack((prop, core.__getattribute__(attr)))
        return prop

    @staticmethod
    def collect_idx_map_from_cores(
        cores: List[DynapSE1SimCore],
    ) -> Dict[int, NeuronKey]:
        """
        collect_idx_map_from_cores obtain an index map using the neuron index maps and core keys of the cores given in the `cores` list.

        :param cores: a list of `DynapSE1SimCore` objects whose parameters are to be merged into one simulation base
        :type cores: List[DynapSE1SimCore]
        :return: a dictionary of the mapping between matrix indexes of the neurons and their global unique neuron keys
        :rtype: Dict[int, NeuronKey]
        """
        core_dict = {}
        for core in cores:
            core_dict[core.core_key] = core.neuron_idx_map
        idx_map = DynapSE1SimBoard.core_dict_to_idx_map(core_dict)
        return idx_map

    @staticmethod
    def check_neuron_id_order(nidx: List[int]) -> None:
        """
        check_neuron_id_order check if the neuron indices are successive and in order.

        :param nidx: a list of neuron indices
        :type nidx: List[int]
        :raises ValueError: Neuron indices are not ordered
        :raises ValueError: The neuron indices are not successive numbers between 0 and N
        """

        if nidx != sorted(nidx):
            raise ValueError(f"Neuron indices are not ordered!\n" f"{nidx}\n")

        if onp.sum(nidx) * 2 != nidx[-1] * (nidx[-1] + 1):
            raise ValueError(
                f"Missing neuron indices! The neuron indices should be successive numbers from 0 to n\n"
                f"{nidx}\n"
            )

    @staticmethod
    def idx_map_to_core_dict(
        idx_map: Dict[int, NeuronKey]
    ) -> Dict[Tuple[int], Dict[int, int]]:
        """
        idx_map_to_core_dict converts an index map to a core dictionary. In the index map, the neuron
        matric indices are mapped to nerun keys(chipID, coreID, neuronID). In the core dictionary,
        the the neruon matric indices are mapped to neuronIDs for each individual core keys(chipID, coreID)

        idx_map = {
            0: (1, 0, 20),
            1: (1, 0, 36),
            2: (1, 0, 60),
            3: (2, 3, 107),
            4: (2, 3, 110),
            5: (3, 1, 152)
        }

        core_dict = {
            (1, 0): {0: 20, 1: 36, 2: 60},
            (2, 3): {3: 107, 4: 110},
            (3, 1): {5: 152}
        }

        :param idx_map: a dictionary of the mapping between matrix indexes of the neurons and their global unique neuron keys
        :type idx_map: Dict[int, NeuronKey]
        :return: a dictionary from core keys (chipID, coreID) to an index map of neruons (neuron index : local neuronID) that the core allocates.
        :rtype: Dict[Tuple[int], Dict[int, int]]
        """
        # Find unique chip-core ID pairs
        chip_core = onp.unique(
            list(map(lambda nid: nid[0:2], idx_map.values())),
            axis=0,
        )
        core_index = list(map(lambda t: tuple(t), chip_core))

        # {(chipID, coreID):{neuron_index: neuronID}} -> {(0,2) : {0:60, 1: 78}}
        core_dict = dict(zip(core_index, [{} for i in range(len(core_index))]))

        # Fill core dictionary
        for neuron_index, neuron_key in idx_map.items():
            core_key = neuron_key[0:2]
            core_dict[core_key][neuron_index] = neuron_key[2]

        # Sort the core dictionary values(neuron index maps) w.r.t. keys {0: 20, 2: 60, 1: 36} -> {0: 20, 1: 36, 2: 60}
        for core_key, nidx_map in core_dict.items():
            core_dict[core_key] = dict(
                sorted(nidx_map.items(), key=lambda item: item[0])
            )

        # Sort the core dictionary values(neuron index maps) between each others so that neuron indices are successive
        core_dict = dict(
            sorted(core_dict.items(), key=lambda item: list(item[1].keys())[0])
        )

        # Check if everything is all right
        nidx = [nid for nid_map in core_dict.values() for nid in nid_map]
        DynapSE1SimBoard.check_neuron_id_order(nidx)

        return core_dict

    @staticmethod
    def core_dict_to_idx_map(
        core_dict: Dict[Tuple[int], Dict[int, int]]
    ) -> Dict[int, NeuronKey]:
        """
        core_dict_to_idx_map converts a core dictionary to an index map. In the index map, the neuron
        matric indices are mapped to nerun keys(chipID, coreID, neuronID). In the core dictionary,
        the the neruon matric indices are mapped to neuronIDs for each individual core keys(chipID, coreID)

        core_dict = {
            (1, 0): {0: 20, 1: 36, 2: 60},
            (2, 3): {3: 107, 4: 110},
            (3, 1): {5: 152}
        }

        idx_map = {
            0: (1, 0, 20),
            1: (1, 0, 36),
            2: (1, 0, 60),
            3: (2, 3, 107),
            4: (2, 3, 110),
            5: (3, 1, 152)
        }

        :param core_dict: a dictionary from core keys (chipID, coreID) to an index map of neruons (neuron index : local neuronID) that the core allocates.
        :type core_dict: Dict[Tuple[int], Dict[int, int]]
        :return: a dictionary of the mapping between matrix indexes of the neurons and their global unique neuron keys
        :rtype: Dict[int, NeuronKey]
        """
        idx_map = {}
        for core_key, neuron_idx in core_dict.items():
            chip_core_id = onp.array(onp.tile(core_key, (len(neuron_idx), 1)))
            device_nid = onp.array(list(neuron_idx.values())).reshape(-1, 1)
            values = onp.hstack((chip_core_id, device_nid))
            temp = dict(zip(list(neuron_idx.keys()), map(lambda t: tuple(t), values)))
            idx_map = {**idx_map, **temp}
        return idx_map

    @classmethod
    def from_config(
        cls,
        config: Dynapse1Configuration,
        idx_map: Optional[Dict[int, NeuronKey]] = None,
    ) -> DynapSE1SimBoard:
        """
        from_config is a class factory method for DynapSE1SimBoard object such that the parameters
        are obtained from a samna device configuration object.

        If an idx_map is not given, then it's extracted from the config object. However, it's not recommended
        because the index map should come along with the weight matrix. So, when costructing a configuration
        object, Use the Router utilities and get the index map along with the recurrent weight matrix
        extracted from the samna config object. Then provide it externally. Simulation board is responsible for
        gathering the biases together. However, using the active neruon information inside the index map is
        beneficial since we can know where to look at. For example if there is no neuron allocated in a core,
        then we do not need to investigate the bias currents related that core.

        :param config: samna Dynapse1 configuration object used to configure a network on the chip
        :type config: Dynapse1Configuration, optional
        :param idx_map: a dictionary of the mapping between matrix indexes of the neurons and their global unique neuron keys, defaults to None
        :type idx_map: Optional[Dict[int, NeuronKey]], optional
        :return: `DynapSE1SimBoard` object ready to configure a simulator
        :rtype: DynapSE1SimBoard
        """

        sim_cores = []
        if idx_map is None:
            _, idx_map = Router.w_rec_from_config(config, return_maps=True)
        core_dict = DynapSE1SimBoard.idx_map_to_core_dict(idx_map)

        # Gather `DynapSE1SimCore` objects
        for (chipID, coreID), neuron_map in core_dict.items():
            # Traverse the chip for core parameter group
            pg = config.chips[chipID].cores[coreID].parameter_group
            sim_core = DynapSE1SimCore.from_samna_parameter_group(
                pg, len(neuron_map), (chipID, coreID), neuron_map
            )
            sim_cores.append(sim_core)

        # Helps to order the idx_map if it's not in proper format
        idx_map = DynapSE1SimBoard.core_dict_to_idx_map(core_dict)
        mod = cls(None, sim_cores, idx_map)
        return mod

    @classmethod
    def from_idx_map(
        cls,
        idx_map: Dict[int, NeuronKey],
    ) -> DynapSE1SimBoard:
        """
        from_idx_map is a class factory method for DynapSE1SimBoard object such that the default parameters
        are used but the neurons ids and the shape is obtained from the idx_map.

        :param idx_map: a dictionary of the mapping between matrix indexes of the neurons and their global unique neuron keys, defaults to None
        :type idx_map: Optional[Dict[int, NeuronKey]], optional
        :return: `DynapSE1SimBoard` object ready to configure a simulator
        :rtype: DynapSE1SimBoard
        """

        sim_cores = []

        core_dict = DynapSE1SimBoard.idx_map_to_core_dict(idx_map)

        # Gather `DynapSE1SimCore` objects
        for (chipID, coreID), neuron_map in core_dict.items():
            # Traverse the chip for core parameter group
            sim_core = DynapSE1SimCore(len(neuron_map), (chipID, coreID), neuron_map)
            sim_cores.append(sim_core)

        # Helps to order the idx_map if it's not in proper format
        idx_map = DynapSE1SimBoard.core_dict_to_idx_map(core_dict)
        mod = cls(None, sim_cores, idx_map)
        return mod
