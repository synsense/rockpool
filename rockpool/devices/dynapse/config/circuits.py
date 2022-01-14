"""
Dynap-SE Parameter classes to be used to configure DynapSEAdExpLIFJax simulation modules

renamed : dynapse1_simconfig.py -> simconfig.py @ 211208

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
24/08/2021
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

from dataclasses import dataclass

import numpy as np

from rockpool.devices.dynapse.infrastructure.biasgen import DynapSE1BiasGen
from rockpool.devices.dynapse.config.layout import DynapSELayout


_SAMNA_AVAILABLE = True

try:
    from samna.dynapse1 import Dynapse1ParameterGroup
except ModuleNotFoundError as e:
    Dynapse1ParameterGroup = Any

    print(
        e, "\nDynapSE1SimCore object cannot be factored from a samna config object!",
    )
    _SAMNA_AVAILABLE = False


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
    :type layout: Optional[DynapSELayout], optional
    """

    Itau: Optional[float] = 7e-12
    Ith: Optional[float] = None
    tau: Optional[float] = None
    f_gain: Optional[float] = 4
    C: float = 1e-12
    layout: Optional[DynapSELayout] = None

    def __post_init__(self) -> None:
        """
        __post_init__ runs after __init__ and initializes the DPIParameters block, check if given values are in legal bounds.

        :raises ValueError: Illegal capacitor value given. It cannot be less than 0.
        """

        # Silicon features
        if self.C < 0:
            raise ValueError(f"Illegal capacitor value C : {self.C}F. It should be > 0")

        if self.layout is None:
            self.layout = DynapSELayout()

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
        layout: DynapSELayout, parameter_group: Dynapse1ParameterGroup, name: str
    ) -> float:
        """
        extract and obtain the bias current from the samna parameter group using bias generator

        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSELayout
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
    :type layout: Optional[DynapSELayout], optional
    """

    Iw_0: float = 1e-6  # GABA_B
    Iw_1: float = 1e-6  # GABA_A
    Iw_2: float = 1e-6  # NMDA
    Iw_3: float = 1e-6  # AMPA
    layout: Optional[DynapSELayout] = None

    def __post_init__(self) -> None:
        """
        __post_init__ runs after __init__ and initializes the WeightParameters block, checks if given values are in legal bounds.
        """

        if self.layout is None:
            self.layout = DynapSELayout()

    @classmethod
    def from_parameter_group(
        cls,
        parameter_group: Dynapse1ParameterGroup,
        layout: DynapSELayout,
        *args,
        **kwargs,
    ) -> WeightParameters:
        """
        from_parameter_group is a factory method to construct a `WeightParameters` object using a `Dynapse1ParameterGroup` object

        :param parameter_group: samna config object for setting the parameter group within one core
        :type parameter_group: Dynapse1ParameterGroup
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSELayout
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

        IMPORTANT : get_vector does not return `jnp.DeviceArray` beacuse it needs preprocessing
        before its used in the simulator

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
        layout: DynapSELayout,
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
        :type layout: DynapSELayout
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
        layout: DynapSELayout,
        *args,
        **kwargs,
    ) -> MembraneParameters:
        """
        from_parameter_group is a `MembraneParameters` factory method with hardcoded bias parameter names

        :param parameter_group: samna config object for setting the parameter group within one core
        :type parameter_group: Dynapse1ParameterGroup
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSELayout
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
        layout: DynapSELayout,
        *args,
        **kwargs,
    ) -> GABABParameters:
        """
        from_parameter_group is a `GABABParameters` factory method implemented by customizing `SynapseParameter._from_parameter_group()`

        :param parameter_group: samna config object for setting the parameter group within one core
        :type parameter_group: Dynapse1ParameterGroup
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSELayout
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
        layout: DynapSELayout,
        *args,
        **kwargs,
    ) -> GABAAParameters:
        """
        from_parameter_group is a `GABAAParameters` factory method implemented by customizing `SynapseParameter._from_parameter_group()`

        :param parameter_group: samna config object for setting the parameter group within one core
        :type parameter_group: Dynapse1ParameterGroup
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSELayout
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
        layout: DynapSELayout,
        *args,
        **kwargs,
    ) -> NMDAParameters:
        """
        from_parameter_group is a `NMDAParameters` factory method implemented by customizing `SynapseParameter._from_parameter_group()`

        :param parameter_group: samna config object for setting the parameter group within one core
        :type parameter_group: Dynapse1ParameterGroup
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSELayout
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
        layout: DynapSELayout,
        *args,
        **kwargs,
    ) -> AMPAParameters:
        """
        from_parameter_group is a `AMPAParameters` factory method implemented by customizing `SynapseParameter._from_parameter_group()`

        :param parameter_group: samna config object for setting the parameter group within one core
        :type parameter_group: Dynapse1ParameterGroup
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSELayout
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
        layout: DynapSELayout,
        *args,
        **kwargs,
    ) -> AHPParameters:
        """
        from_parameter_group is a `AHPParameters` factory method implemented by customizing `SynapseParameter._from_parameter_group()`

        :param parameter_group: samna config object for setting the parameter group within one core
        :type parameter_group: Dynapse1ParameterGroup
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSELayout
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
