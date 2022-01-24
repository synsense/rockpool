"""
Dynap-SE Parameter classes to be used to configure DynapSEAdExpLIFJax simulation modules

renamed : dynapse1_simconfig.py -> simconfig.py @ 211208

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
24/08/2021
[] TODO: Change scaling factor operation
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Union

from dataclasses import dataclass

import numpy as np

from rockpool.devices.dynapse.infrastructure.biasgen import BiasGenSE1, BiasGenSE2
from rockpool.devices.dynapse.config.layout import DynapSELayout
from rockpool.devices.dynapse.lookup import param_name

_SAMNA_SE1_AVAILABLE = True
_SAMNA_SE2_AVAILABLE = True

param_name_table = param_name.table

try:
    from samna.dynapse1 import Dynapse1Parameter
except ModuleNotFoundError as e:
    Dynapse1Parameter = Any

    print(
        e, "\nDynapSE1SimCore object cannot be factored from a samna config object!",
    )
    _SAMNA_SE1_AVAILABLE = False

try:
    from samna.dynapse2 import Dynapse2Parameter
except ModuleNotFoundError as e:
    Dynapse2Parameter = Any
    print(
        e, "\nDynapSE2SimCore object cannot be factored from a samna config object!",
    )
    _SAMNA_SE2_AVAILABLE = False


@dataclass
class DynapSEParamGen:
    """
    DynapSEParamGen encapsulates common Dynap-SE1/SE2 paramter configuration -> bias current 
    conversion methods and serve as a top level class.

    :param version: the processor version. either 1 or 2
    :type version: int
    """

    version: int

    def __post_init__(self) -> None:
        """
        __post_init__ runs after __init__ and initializes the DynapSEParamGen object, check if the processor version supported.

        :raises ValueError: Only Dynap-SE1 and Dynap-SE2 versions are available. (version 1 or 2)
        """
        if self.version == 1:
            self.bias_gen = BiasGenSE1()
        elif self.version == 2:
            self.bias_gen = BiasGenSE2()
        else:
            raise ValueError(
                f"Currently Dynap-SE1 and Dynap-SE2 are available. (version 1 or 2)!\n"
                f"No support for version: {self.version}!"
            )

    def _check_param_version(self, param: Union[Dynapse1Parameter, Dynapse2Parameter]):
        """
        _check_param_version Check if the samna parameter version and the simulated processor version is compatible

        :param param: a parameter inside samna config object of interest to check if the samna parameter version matches the simulation configuration version
        :type param: Union[Dynapse1Parameter, Dynapse2Parameter]
        :raises TypeError: The parameter type and processor version indicated is incompatible!"
        """
        if isinstance(param, Dynapse1Parameter) and self.version == 1:
            None
        elif isinstance(param, Dynapse2Parameter) and self.version == 2:
            None
        else:
            raise TypeError(
                f"The parameter type {type(param)} and processor version Dynap-SE{self.version} indicated is incompatible!"
            )

    def bias(
        self,
        samna_parameters: Dict[str, Union[Dynapse1Parameter, Dynapse2Parameter]],
        name: str,
    ) -> float:
        """
        extract and obtain the bias current from the samna parameter group using bias generator

        :param samna_parameters: a parameter dictionary inside samna config object for setting the parameter group within one core
        :type samna_parameters: Dict[str, Union[Dynapse1Parameter, Dynapse2Parameter]]
        :param name: the parameter name of the bias current
        :type name: str
        :return: biasgen corrected bias value by multiplying a correction factor
        :rtype: float
        """
        param = samna_parameters[name]
        self._check_param_version(param)
        bias_current = self.bias_gen.param_to_bias(name, param)
        return bias_current


@dataclass
class SimulationParameters:
    """
    SimulationParameters encapsulates a samna parameter dictionary, creates right version
    of the BiasGen and returns the right bias parameter objects given a simulation parameter name

    :param version: the processor version. either 1 or 2
    :type version: int
    """

    samna_parameters: Dict[str, Union[Dynapse1Parameter, Dynapse2Parameter]]

    def __post_init__(self) -> None:
        """
        __post_init__ runs after __init__ and initializes the SimulationParameters object, check if the processor version supported.

        :raises TypeError: The dictionary type does not refer to any processor version!
        """
        _param_type = list(self.samna_parameters.values())[0]
        if isinstance(_param_type, Dynapse1Parameter):
            self.version = 1
        elif isinstance(_param_type, Dynapse2Parameter):
            self.version = 2
        else:
            raise TypeError(
                f"The dictionary type {_param_type} does not refer to any processor version!"
            )

        self.paramgen = DynapSEParamGen(self.version)

    def nominal(self, name: str):
        """
        nominal abstracts the invisible conversion to find the nominal current value that 
        the simulation paramter need to have depending on the device configuration.

        * Provided the name of the simulation current parameter, find the respected device parameter name
        * Using the right version bias generator, find the nominal current value

        :param name: the simulation parameter name of the desired parameter (like Itau_ampa)
        :type name: str
        :return: the BiasGen generated current value of the parameter of interest
        :rtype: [type]
        """
        device_name = param_name_table[name][self.version]
        return self.paramgen.bias(self.samna_parameters, device_name)


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
    Iw_1: float = 2e-6  # GABA_A
    Iw_2: float = 4e-6  # NMDA
    Iw_3: float = 8e-6  # AMPA
    layout: Optional[DynapSELayout] = None

    def __post_init__(self) -> None:
        """
        __post_init__ runs after __init__ and initializes the WeightParameters block, checks if given values are in legal bounds.
        """

        if self.layout is None:
            self.layout = DynapSELayout()

    @classmethod
    def from_samna_parameters(
        cls,
        samna_parameters: Dict[str, Union[Dynapse1Parameter, Dynapse2Parameter]],
        layout: DynapSELayout,
        *args,
        **kwargs,
    ) -> WeightParameters:
        """
        from_samna_parameters is a factory method to construct a `WeightParameters` object using a samna config object

        :param samna_parameters: a parameter dictionary inside samna config object for setting the parameter group within one core
        :type samna_parameters: Dict[str, Union[Dynapse1Parameter, Dynapse2Parameter]]
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSELayout
        :return: a `WeightParameters` object, whose parameters obtained from the hardware configuration
        :rtype: WeightParameters
        """

        simparam = SimulationParameters(samna_parameters)

        mod = cls(
            Iw_0=simparam.nominal("Iw_0"),  # GABA_B - se1
            Iw_1=simparam.nominal("Iw_1"),  # GABA_A - se1
            Iw_2=simparam.nominal("Iw_2"),  # NMDA - se1
            Iw_3=simparam.nominal("Iw_3"),  # AMPA - se1
            layout=layout,
            *args,
            **kwargs,
        )
        return mod

    # [] TODO : Remove
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
    def _from_samna_parameters(
        cls,
        samna_parameters: Dict[str, Union[Dynapse1Parameter, Dynapse2Parameter]],
        layout: DynapSELayout,
        Itau_name: str,
        Ith_name: str,
        *args,
        **kwargs,
    ) -> SynapseParameters:
        """
        _from_samna_parameters is a common factory method to construct a `SynapseParameters` object using a `Dynapse1ParameterGroup` object
        Each individual synapse is expected to provide their individual Itau and Ithr bias names

        :param samna_parameters: a parameter dictionary inside samna config object for setting the parameter group within one core
        :type samna_parameters: Dict[str, Union[Dynapse1Parameter, Dynapse2Parameter]]
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSELayout
        :param Itau_name: the name of the leak bias current
        :type Itau_name: str
        :param Ith_name: the name of the gain bias current
        :type Ith_name: str
        :return: a `SynapseParameters` object, whose parameters obtained from the hardware configuration
        :rtype: SynapseParameters
        """

        simparam = SimulationParameters(samna_parameters)

        mod = cls(
            Itau=simparam.nominal(Itau_name),
            Ith=simparam.nominal(Ith_name),
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
    def from_samna_parameters(
        cls,
        samna_parameters: Dict[str, Union[Dynapse1Parameter, Dynapse2Parameter]],
        layout: DynapSELayout,
        *args,
        **kwargs,
    ) -> MembraneParameters:
        """
        from_samna_parameters is a `MembraneParameters` factory method with hardcoded bias parameter names
        
        :param samna_parameters: a parameter dictionary inside samna config object for setting the parameter group within one core
        :type samna_parameters: Dict[str, Union[Dynapse1Parameter, Dynapse2Parameter]]
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSELayout
        :return: a `MembraneParameters` object whose parameters obtained from the hardware configuration
        :rtype: MembraneParameters
        """

        simparam = SimulationParameters(samna_parameters)

        mod = cls(
            Itau=simparam.nominal("Itau_mem"),
            Ith=simparam.nominal("Ith_mem"),
            tau=None,  # deduced from Itau
            tau2=None,  # deduced from Itau2
            Itau2=simparam.nominal("Itau2_mem"),
            f_gain=None,  # deduced from Ith/Itau
            Iref=simparam.nominal("Iref"),
            t_ref=None,  # deduced from Iref
            Ipulse=simparam.nominal("Ipulse"),
            t_pulse=None,  # deduced from Ipulse
            Idc=simparam.nominal("Idc"),
            If_nmda=simparam.nominal("If_nmda"),
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
    def from_samna_parameters(
        cls,
        samna_parameters: Dict[str, Union[Dynapse1Parameter, Dynapse2Parameter]],
        layout: DynapSELayout,
        *args,
        **kwargs,
    ) -> GABABParameters:
        """
        from_samna_parameters is a `GABABParameters` factory method implemented by customizing `SynapseParameter._from_samna_parameters()`

        :param samna_parameters: a parameter dictionary inside samna config object for setting the parameter group within one core
        :type samna_parameters: Dict[str, Union[Dynapse1Parameter, Dynapse2Parameter]]
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSELayout
        :return: a `GABABParameters` object, whose parameters obtained from the hardware configuration
        :rtype: GABABParameters
        """

        return cls._from_samna_parameters(
            samna_parameters,
            layout,
            Itau_name="Itau_gaba_b",
            Ith_name="Ith_gaba_b",
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
    def from_samna_parameters(
        cls,
        samna_parameters: Dict[str, Union[Dynapse1Parameter, Dynapse2Parameter]],
        layout: DynapSELayout,
        *args,
        **kwargs,
    ) -> GABAAParameters:
        """
        from_samna_parameters is a `GABAAParameters` factory method implemented by customizing `SynapseParameter._from_samna_parameters()`

        :param samna_parameters: a parameter dictionary inside samna config object for setting the parameter group within one core
        :type samna_parameters: Dict[str, Union[Dynapse1Parameter, Dynapse2Parameter]]
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSELayout
        :return: a `GABAAParameters` object, whose parameters obtained from the hardware configuration
        :rtype: GABAAParameters
        """

        return cls._from_samna_parameters(
            samna_parameters,
            layout,
            Itau_name="Itau_gaba_a",
            Ith_name="Ith_gaba_a",
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
    def from_samna_parameters(
        cls,
        samna_parameters: Dict[str, Union[Dynapse1Parameter, Dynapse2Parameter]],
        layout: DynapSELayout,
        *args,
        **kwargs,
    ) -> NMDAParameters:
        """
        from_samna_parameters is a `NMDAParameters` factory method implemented by customizing `SynapseParameter._from_samna_parameters()`

        :param samna_parameters: a parameter dictionary inside samna config object for setting the parameter group within one core
        :type samna_parameters: Dict[str, Union[Dynapse1Parameter, Dynapse2Parameter]]
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSELayout
        :return: a `NMDAParameters` object, whose parameters obtained from the hardware configuration
        :rtype: NMDAParameters
        """

        return cls._from_samna_parameters(
            samna_parameters,
            layout,
            Itau_name="Itau_nmda",
            Ith_name="Ith_nmda",
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
    def from_samna_parameters(
        cls,
        samna_parameters: Dict[str, Union[Dynapse1Parameter, Dynapse2Parameter]],
        layout: DynapSELayout,
        *args,
        **kwargs,
    ) -> AMPAParameters:
        """
        from_samna_parameters is a `AMPAParameters` factory method implemented by customizing `SynapseParameter._from_samna_parameters()`

        :param samna_parameters: a parameter dictionary inside samna config object for setting the parameter group within one core
        :type samna_parameters: Dict[str, Union[Dynapse1Parameter, Dynapse2Parameter]]
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSELayout
        :return: a `AMPAParameters` object, whose parameters obtained from the hardware configuration
        :rtype: AMPAParameters
        """

        return cls._from_samna_parameters(
            samna_parameters,
            layout,
            Itau_name="Itau_ampa",
            Ith_name="Ith_ampa",
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
    def from_samna_parameters(
        cls,
        samna_parameters: Dict[str, Union[Dynapse1Parameter, Dynapse2Parameter]],
        layout: DynapSELayout,
        *args,
        **kwargs,
    ) -> AHPParameters:
        """
        from_samna_parameters is a `AHPParameters` factory method implemented by customizing `SynapseParameter._from_samna_parameters()`

        :param samna_parameters: a parameter dictionary inside samna config object for setting the parameter group within one core
        :type samna_parameters: Dict[str, Union[Dynapse1Parameter, Dynapse2Parameter]]
        :param layout: constant values that are related to the exact silicon layout of a chip
        :type layout: DynapSELayout
        :return: a `AHPParameters` object, whose parameters obtained from the hardware configuration
        :rtype: AHPParameters
        """

        simparam = SimulationParameters(samna_parameters)

        mod = cls(
            Itau=simparam.nominal("Itau_ahp"),
            Ith=simparam.nominal("Ith_ahp"),
            f_gain=None,  # deduced from Ith/Itau
            tau=None,  # deduced from Itau
            Iw=simparam.nominal("Iw_ahp"),
            layout=layout,
            *args,
            **kwargs,
        )

        return mod
