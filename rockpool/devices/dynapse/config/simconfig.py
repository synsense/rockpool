"""
Dynap-SE1/SE2 full board configuration classes and methods

renamed : dynapse1_simconfig.py -> simconfig.py @ 211208
split_from : simconfig.py -> layout.py @ 220114
split_from : simconfig.py -> circuits.py @ 220114
merged from : layout.py -> simcore.py @ 220505
merged from : circuits.py -> simcore.py @ 220505
merged from : board.py -> simcore.py @ 220531
renamed : simcore.py -> simconfig.py @ 220531

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
03/05/2022

[] TODO : Add r_spkthr to gain
[] TODO : add from_bias methods to samna aliases
"""

from __future__ import annotations
import logging

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from dataclasses import dataclass, replace, field
import numpy as np

from rockpool.devices.dynapse.definitions import (
    CoreKey,
    NeuronKey,
    NUM_CHIPS,
    NUM_CORES,
    NUM_NEURONS,
)

from rockpool.devices.dynapse.infrastructure.biasgen import (
    BiasGen,
    BiasGenSE1,
    BiasGenSE2,
)

from rockpool.devices.dynapse.lookup import param_name
from rockpool.devices.dynapse.default import dlayout, dweight, dtime, dgain, dcurrents
from rockpool.devices.dynapse.samna_alias.dynapse1 import (
    Dynapse1Parameter,
    Dynapse1Core,
    Dynapse1Configuration,
)
from rockpool.devices.dynapse.samna_alias.dynapse2 import (
    Dynapse2Parameter,
    Dynapse2Core,
    Dynapse2Configuration,
)
from rockpool.devices.dynapse.ref.weights_old import WeightParameters
from rockpool.devices.dynapse.infrastructure.router import Router, Connector
from rockpool.typehints import FloatVector


@dataclass
class DynapSimProperty:
    """
    DynapSimProperty stands for a commmon base class for all implementations
    """

    def get_full(self, size: int) -> Dict[str, np.ndarray]:
        """
        get_full creates a dictionary with respect to the object, with arrays of current values

        :param size: the lengths of the current arrays
        :type size: int
        :return: the object dictionary with current arrays given the size
        :rtype: Dict[str, np.ndarray]
        """
        __get__ = lambda name: np.full(size, self.__getattribute__(name))
        _dict = {name: __get__(name) for name in self.__dict__.keys()}
        return _dict


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

    Idc: FloatVector = dcurrents["Idc"]
    If_nmda: FloatVector = dcurrents["If_nmda"]
    Igain_ahp: FloatVector = dcurrents["Igain_ahp"]
    Igain_ampa: FloatVector = dcurrents["Igain_ampa"]
    Igain_gaba: FloatVector = dcurrents["Igain_gaba"]
    Igain_nmda: FloatVector = dcurrents["Igain_nmda"]
    Igain_shunt: FloatVector = dcurrents["Igain_shunt"]
    Igain_mem: FloatVector = dcurrents["Igain_mem"]
    Ipulse_ahp: FloatVector = dcurrents["Ipulse_ahp"]
    Ipulse: FloatVector = dcurrents["Ipulse"]
    Iref: FloatVector = dcurrents["Iref"]
    Ispkthr: FloatVector = dcurrents["Ispkthr"]
    Itau_ahp: FloatVector = dcurrents["Itau_ahp"]
    Itau_ampa: FloatVector = dcurrents["Itau_ampa"]
    Itau_gaba: FloatVector = dcurrents["Itau_gaba"]
    Itau_nmda: FloatVector = dcurrents["Itau_nmda"]
    Itau_shunt: FloatVector = dcurrents["Itau_shunt"]
    Itau_mem: FloatVector = dcurrents["Itau_mem"]
    Iw_ahp: FloatVector = dcurrents["Iw_ahp"]


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

    C_ahp: FloatVector = dlayout["C_ahp"]
    C_ampa: FloatVector = dlayout["C_ampa"]
    C_gaba: FloatVector = dlayout["C_gaba"]
    C_nmda: FloatVector = dlayout["C_nmda"]
    C_pulse_ahp: FloatVector = dlayout["C_pulse_ahp"]
    C_pulse: FloatVector = dlayout["C_pulse"]
    C_ref: FloatVector = dlayout["C_ref"]
    C_shunt: FloatVector = dlayout["C_shunt"]
    C_mem: FloatVector = dlayout["C_mem"]
    Io: FloatVector = dlayout["Io"]
    kappa_n: FloatVector = dlayout["kappa_n"]
    kappa_p: FloatVector = dlayout["kappa_p"]
    Ut: FloatVector = dlayout["Ut"]
    Vth: FloatVector = dlayout["Vth"]


@dataclass
class DynapSimWeightBits(DynapSimProperty):
    """
    DynapSimWeightBits encapsulates weight bit current parameters of Dynap-SE chips

    :param Iw_0: weight bit 0 current of the neurons of the core in Amperes
    :type Iw_0: FloatVector
    :param Iw_1: weight bit 1 current of the neurons of the core in Amperes
    :type Iw_1: FloatVector
    :param Iw_2: weight bit 2 current of the neurons of the core in Amperes
    :type Iw_2: FloatVector
    :param Iw_3: weight bit 3 current of the neurons of the core in Amperes
    :type Iw_3: FloatVector
    """

    Iw_0: FloatVector = dweight["Iw_0"]
    Iw_1: FloatVector = dweight["Iw_1"]
    Iw_2: FloatVector = dweight["Iw_2"]
    Iw_3: FloatVector = dweight["Iw_3"]

    @property
    def Iw(self) -> np.ndarray:
        return np.stack([self.Iw_0, self.Iw_1, self.Iw_2, self.Iw_3]).T


@dataclass
class DynapSimCore(DynapSimCurrents, DynapSimLayout, DynapSimWeightBits):
    """
    DynapSE1SimCore stores the simulation currents and manages the conversion from configuration objects
    It also provides easy update mechanisms using coarse&fine values, high-level parameter representations and etc.
    """

    __doc__ += "\nDynapSimCurrents" + DynapSimCurrents.__doc__
    __doc__ += "\nDynapSimLayout" + DynapSimLayout.__doc__

    @classmethod
    def from_specification(
        cls,
        Idc: FloatVector = dcurrents["Idc"],
        If_nmda: FloatVector = dcurrents["If_nmda"],
        r_gain_ahp: FloatVector = dgain["r_gain_ahp"],
        r_gain_ampa: FloatVector = dgain["r_gain_ampa"],
        r_gain_gaba: FloatVector = dgain["r_gain_gaba"],
        r_gain_nmda: FloatVector = dgain["r_gain_nmda"],
        r_gain_shunt: FloatVector = dgain["r_gain_shunt"],
        r_gain_mem: FloatVector = dgain["r_gain_mem"],
        t_pulse_ahp: FloatVector = dtime["t_pulse_ahp"],
        t_pulse: FloatVector = dtime["t_pulse"],
        t_ref: FloatVector = dtime["t_ref"],
        Ispkthr: FloatVector = dcurrents["Ispkthr"],
        tau_ahp: FloatVector = dtime["tau_ahp"],
        tau_ampa: FloatVector = dtime["tau_ampa"],
        tau_gaba: FloatVector = dtime["tau_gaba"],
        tau_nmda: FloatVector = dtime["tau_nmda"],
        tau_shunt: FloatVector = dtime["tau_shunt"],
        tau_mem: FloatVector = dtime["tau_mem"],
        Iw_0: FloatVector = dweight["Iw_0"],
        Iw_1: FloatVector = dweight["Iw_1"],
        Iw_2: FloatVector = dweight["Iw_2"],
        Iw_3: FloatVector = dweight["Iw_3"],
        Iw_ahp: FloatVector = dcurrents["Iw_ahp"],
        C_ahp: FloatVector = dlayout["C_ahp"],
        C_ampa: FloatVector = dlayout["C_ampa"],
        C_gaba: FloatVector = dlayout["C_gaba"],
        C_nmda: FloatVector = dlayout["C_nmda"],
        C_pulse_ahp: FloatVector = dlayout["C_pulse_ahp"],
        C_pulse: FloatVector = dlayout["C_pulse"],
        C_ref: FloatVector = dlayout["C_ref"],
        C_shunt: FloatVector = dlayout["C_shunt"],
        C_mem: FloatVector = dlayout["C_mem"],
        Io: FloatVector = dlayout["Io"],
        kappa_n: FloatVector = dlayout["kappa_n"],
        kappa_p: FloatVector = dlayout["kappa_p"],
        Ut: FloatVector = dlayout["Ut"],
        Vth: FloatVector = dlayout["Vth"],
    ) -> DynapSimCore:
        """
        from_specification is a class factory method helping DynapSimCore object construction
        using higher level representaitons of the currents like gain ratio or time constant whenever applicable.

        :param Idc: Constant DC current injected to membrane in Amperes
        :type Idc: FloatVector, optional
        :param If_nmda: NMDA gate soft cut-off current setting the NMDA gating voltage in Amperes
        :type If_nmda: FloatVector, optional
        :param r_gain_ahp: spike frequency adaptation block gain ratio :math:`Igain_ahp/Itau_ahp`
        :type r_gain_ahp: FloatVector, optional
        :param r_gain_ampa: excitatory AMPA synpse gain ratio :math:`Igain_ampa/Itau_ampa`
        :type r_gain_ampa: FloatVector, optional
        :param r_gain_gaba: inhibitory GABA synpse gain ratio :math:`Igain_gaba/Itau_gaba `
        :type r_gain_gaba: FloatVector, optional
        :param r_gain_nmda: excitatory NMDA synpse gain ratio :math:`Igain_nmda/Itau_nmda`
        :type r_gain_nmda: FloatVector, optional
        :param r_gain_shunt: inhibitory SHUNT synpse gain ratio :math:`Igain_shunt/Itau_shunt`
        :type r_gain_shunt: FloatVector, optional
        :param r_gain_mem: neuron membrane gain ratio :math:`Igain_mem/Itau_mem`
        :type r_gain_mem: FloatVector, optional
        :param t_pulse_ahp: the spike pulse width for spike frequency adaptation circuit in seconds
        :type t_pulse_ahp: FloatVector, optional
        :param t_pulse: the spike pulse width for neuron membrane in seconds
        :type t_pulse: FloatVector, optional
        :param t_ref: refractory period of the neurons in seconds
        :type t_ref: FloatVector, optional
        :param Ispkthr: spiking threshold current, neuron spikes if :math:`Imem > Ispkthr` in Amperes
        :type Ispkthr: FloatVector, optional
        :param tau_ahp: Spike frequency leakage time constant in seconds
        :type tau_ahp: FloatVector, optional
        :param tau_ampa: AMPA synapse leakage time constant in seconds
        :type tau_ampa: FloatVector, optional
        :param tau_gaba: GABA synapse leakage time constant in seconds
        :type tau_gaba: FloatVector, optional
        :param tau_nmda: NMDA synapse leakage time constant in seconds
        :type tau_nmda: FloatVector, optional
        :param tau_shunt:SHUNT synapse leakage time constant in seconds
        :type tau_shunt: FloatVector, optional
        :param tau_mem: Neuron membrane leakage time constant in seconds
        :type tau_mem: FloatVector, optional
        :param Iw_0: weight bit 0 current of the neurons of the core in Amperes
        :type Iw_0: FloatVector
        :param Iw_1: weight bit 1 current of the neurons of the core in Amperes
        :type Iw_1: FloatVector
        :param Iw_2: weight bit 2 current of the neurons of the core in Amperes
        :type Iw_2: FloatVector
        :param Iw_3: weight bit 3 current of the neurons of the core in Amperes
        :type Iw_3: FloatVector
        :param Iw_ahp: spike frequency adaptation weight current of the neurons of the core in Amperes
        :type Iw_ahp: FloatVector
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
        :return: DynapSimCore object
        :rtype: DynapSimCore
        """

        # Depended default parameter initialization
        Idc = Io if Idc is None else Idc
        If_nmda = Io if If_nmda is None else If_nmda

        # Construct the core with compulsory low level current parameters
        _core = cls(
            Idc=Idc,
            If_nmda=If_nmda,
            Ispkthr=Ispkthr,
            Iw_0=Iw_0,
            Iw_1=Iw_1,
            Iw_2=Iw_2,
            Iw_3=Iw_3,
            Iw_ahp=Iw_ahp,
            C_ahp=C_ahp,
            C_ampa=C_ampa,
            C_gaba=C_gaba,
            C_nmda=C_nmda,
            C_pulse_ahp=C_pulse_ahp,
            C_pulse=C_pulse,
            C_ref=C_ref,
            C_shunt=C_shunt,
            C_mem=C_mem,
            Io=Io,
            kappa_n=kappa_n,
            kappa_p=kappa_p,
            Ut=Ut,
            Vth=Vth,
        )

        # Set the Itau currents
        _time = DynapSimTime(
            t_pulse_ahp,
            t_pulse,
            t_ref,
            tau_ahp,
            tau_ampa,
            tau_gaba,
            tau_nmda,
            tau_shunt,
            tau_mem,
        )
        _core = _time.update_DynapSimCore(_core)

        # Set Igain currents depending on the ratio between related Itau currents
        _gain = DynapSimGain(
            r_gain_ahp,
            r_gain_ampa,
            r_gain_gaba,
            r_gain_nmda,
            r_gain_shunt,
            r_gain_mem,
        )
        _core = _gain.update_DynapSimCore(_core)

        return _core

    @classmethod
    def __from_samna(
        cls,
        biasgen: BiasGen,
        param_getter: Callable[[str], Union[Dynapse1Parameter, Dynapse2Parameter]],
        param_map: Dict[str, str],
    ) -> DynapSimCore:
        """
        __from_samna is a class factory method which uses samna configuration objects to extract the simulation currents

        :param biasgen: the bias generator to convert the device parameters with coarse and fine values to bias currents
        :type biasgen: BiasGen
        :param param_getter: a function wich returns a samna parameter object given a name
        :type param_getter: Callable[[str], Union[Dynapse1Parameter, Dynapse2Parameter]]
        :param param_map: the dictionary of simulated currents and their respective device configaration parameter names like {"Idc": "SOIF_DC_P"}
        :type param_map: Dict[str, str]
        :return: a dynapse core simulation object whose parameters are imported from a samna configuration object
        :rtype: DynapSimCore
        """
        _current = lambda name: biasgen.param_to_bias(name, param_getter(name))
        _dict = {sim: _current(param) for sim, param in param_map.items()}
        _mod = cls(**_dict)
        return _mod

    @classmethod
    def from_Dynapse1Core(cls, core: Dynapse1Core) -> DynapSimCore:
        """
        from_Dynapse1Core is a class factory method which uses a samna Dynapse1Core object to extract the simualation current parameters

        :param core: a samna Dynapse1Core configuration object used to configure the core properties
        :type core: Dynapse1Core
        :return: a dynapse core simulation object whose parameters are imported from a samna configuration object
        :rtype: DynapSimCore
        """
        _mod = cls.__from_samna(
            biasgen=BiasGenSE1(),
            param_getter=lambda name: core.parameter_group.param_map[name],
            param_map=param_name.se1,
        )

        return _mod

    @classmethod
    def from_Dynapse2Core(cls, core: Dynapse2Core) -> DynapSimCore:
        """
        from_Dynapse2Core is a class factory method which uses a samna Dynapse2Core object to extract the simualation current parameters

        :param core: a samna Dynapse2Core configuration object used to configure the core properties
        :type core: Dynapse2Core
        :return: a dynapse core simulation object whose parameters are imported from a samna configuration object
        :rtype: DynapSimCore
        """
        _mod = cls.__from_samna(
            biasgen=BiasGenSE2(),
            param_getter=lambda name: core.parameters[name],
            param_map=param_name.se2,
        )
        return _mod

    def __export_parameters(
        self,
        biasgen: BiasGen,
        param_map: Dict[str, str],
    ) -> Dict[str, Tuple[np.uint8, np.uint8]]:
        """
        __export_parameters is the common export method for Dynap-SE1 and Dynap-SE2.
        It converts all current values to their coarse-fine value representations for device configuration

        :param biasgen: the device specific bias generator
        :type biasgen: BiasGen
        :param param_map: the simulation current -> parameter name conversion table
        :type param_map: Dict[str, str]
        :return: a dictionary of mapping between parameter names and respective coarse-fine values
        :rtype: Dict[str, Tuple[np.uint8, np.uint8]]
        """

        converter = lambda sim, param: biasgen.get_coarse_fine(
            param, self.__getattribute__(sim)
        )

        param_dict = {param: converter(sim, param) for sim, param in param_map.items()}
        return param_dict

    def export_Dynapse1Parameters(self) -> Dict[str, Tuple[np.uint8, np.uint8]]:
        """
        export_Dynapse1Parameters is Dynap-SE1 specific parameter extraction method using `DynapSimCore.__export_parameters()` method.
        It converts all current values to their coarse-fine value representations for device configuration

        :return: a dictionary of mapping between parameter names and respective coarse-fine values
        :rtype: Dict[str, Tuple[np.uint8, np.uint8]]
        """
        return self.__export_parameters(
            biasgen=BiasGenSE1(),
            param_map=param_name.se1,
        )

    def export_Dynapse2Parameters(self) -> Dict[str, Tuple[np.uint8, np.uint8]]:
        """
        export_Dynapse2Parameters is Dynap-SE2 specific parameter extraction method using `DynapSimCore.__export_parameters()` method.
        It converts all current values to their coarse-fine value representations for device configuration

        :return: a dictionary of mapping between parameter names and respective coarse-fine values
        :rtype: Dict[str, Tuple[np.uint8, np.uint8]]
        """
        return self.__export_parameters(
            biasgen=BiasGenSE2(),
            param_map=param_name.se2,
        )

    def update(self, attr: str, value: Any) -> DynapSimCore:
        """
        update_current updates an attribute and returns a new object, does not change the original object.

        :param attr: any attribute that belongs to DynapSimCore object
        :type attr: str
        :param value: the new value to set
        :type value: Any
        :return: updated DynapSimCore object
        :rtype: DynapSimCore
        """
        if attr in list(self.__dict__.keys()):
            _updated = replace(self)
            _updated.__setattr__(attr, value)
            self.compare(self, _updated)

        return _updated

    def __update_high_level(
        self,
        obj: DynapSimCoreHigh,
        attr_getter: Callable[[str], Any],
        attr: str,
        value: Any,
    ) -> DynapSimCore:
        """
        __update_high_level updates high level representations of the current values like time constants and gain ratios.
        The current values are updated accordingly without changing the original object.

        :param obj: the high level object that stores the projections of the current values
        :type obj: DynapSimCoreHigh
        :param attr_getter: a function to get the high level attribute from the high level object
        :type attr_getter: Callable[[str], Any]
        :param attr: any attribute that belongs to any DynapSimCoreHigh object
        :type attr: str
        :param value: the new value to set
        :type value: Any
        :return: updated DynapSimCore object
        :rtype: DynapSimCore
        """
        if attr in list(obj.__dict__.keys()):
            obj.__setattr__(attr, value)
            _updated = obj.update_DynapSimCore(self)
            logging.info(
                f" {attr} value changed from {attr_getter(attr)} to {obj.__getattribute__(attr)}"
            )
            self.compare(self, _updated)

        return _updated

    def update_time_constant(self, attr: str, value: Any) -> DynapSimCore:
        """
        update_time_constant updates currents setting time constant attributes that have a representation in `DynapSimTime()` class instances

        :param attr: any attribute that belongs to any DynapSimTime object
        :type attr: str
        :param value: the new value to set
        :type value: Any
        :return: updated DynapSimCore object
        :rtype: DynapSimCore
        """
        return self.__update_high_level(
            obj=DynapSimTime(),
            attr_getter=lambda name: self.time.__getattribute__(name),
            attr=attr,
            value=value,
        )

    def update_gain_ratio(self, attr: str, value: Any) -> DynapSimCore:
        """
        update_gain_ratio updates currents setting gain ratio (Igain/Itau) attributes that have a representation in `DynapSimGain()` class instances

        :param attr: any attribute that belongs to any DynapSimGain object
        :type attr: str
        :param value: the new value to set
        :type value: Any
        :return: updated DynapSimCore object
        :rtype: DynapSimCore
        """
        return self.__update_high_level(
            obj=DynapSimGain(),
            attr_getter=lambda name: self.gain.__getattribute__(name),
            attr=attr,
            value=value,
        )

    @staticmethod
    def compare(core1: DynapSimCore, core2: DynapSimCore) -> Dict[str, Tuple[Any]]:
        """
        compare compares two DynapSimCore objects detects the different values set

        :param core1: the first core object
        :type core1: DynapSimCore
        :param core2: the second core object to compare against the first one
        :type core2: DynapSimCore
        :return: a dictionary of changed values
        :rtype: Dict[str, Tuple[Any]]
        """

        changed = {}
        for key in core1.__dict__:
            val1 = core1.__getattribute__(key)
            val2 = core2.__getattribute__(key)
            if val1 != val2:
                changed[key] = (val1, val2)
                logging.info(f" {key} value changed from {val1} to {val2}")

        return changed

    @property
    def layout(self) -> DynapSimLayout:
        """
        layout returns a subset of object which belongs to DynapSimLayout
        """
        __dict = dict.fromkeys(DynapSimLayout.__annotations__.keys())
        for key in __dict:
            __dict[key] = self.__getattribute__(key)
        return DynapSimLayout(**__dict)

    @property
    def currents(self) -> DynapSimCurrents:
        """
        currents returns a subset of object which belongs to DynapSimCurrents
        """
        __dict = dict.fromkeys(DynapSimCurrents.__annotations__.keys())
        for key in __dict:
            __dict[key] = self.__getattribute__(key)
        return DynapSimCurrents(**__dict)

    @property
    def weight_bits(self) -> DynapSimWeightBits:
        """
        weight_bits returns a subset of object which belongs to DynapSimWeightBits
        """
        __dict = dict.fromkeys(DynapSimWeightBits.__annotations__.keys())
        for key in __dict:
            __dict[key] = self.__getattribute__(key)
        return DynapSimWeightBits(**__dict)

    @property
    def time(self) -> DynapSimTime:
        """
        time creates the high level time constants set by currents
        Ipulse_ahp, Ipulse, Iref, Itau_ahp, Itau_ampa, Itau_gaba, Itau_nmda, Itau_shunt, Itau_mem
        """
        return DynapSimTime.from_DynapSimCore(self)

    @property
    def gain(self) -> DynapSimGain:
        """
        gain creates the high level gain ratios set by currents
        Igain_ahp, Igain_ampa, Igain_gaba, Igain_nmda, Igain_shunt, Igain_mem
        """
        return DynapSimGain.from_DynapSimCore(self)


@dataclass
class DynapSimConfig(DynapSimCore):
    """
    DynapSimConfig stores the simulation currents, layout parameters and weight matrices necessary to
    configure a DynapSE1/SE2 simulator

    :param shape: the network shape (n_input, n_hidden, n_output)
    :type shape: Optional[Union[Tuple[int], int]], optional
    :param w_in: input weight matrix
    :type w_in: np.ndarray, optional
    :param w_rec: recurrent weight matrix
    :type w_rec: np.ndarray, optional
    :param w_out: output weight matrix
    :type w_out: np.ndarray, optional
    :param cores: dictionary of simulation cores
    :type cores: Optional[Dict[str, DynapSimCore]], optional
    :param router: the router object reading the memory content to create the weight masks
    """

    shape: Optional[Union[Tuple[int], int]] = None
    w_in: np.ndarray = None
    w_rec: np.ndarray = None
    w_out: np.ndarray = None
    cores: Optional[Dict[str, DynapSimCore]] = field(default=None, repr=False)
    router: Optional[Router] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.shape = self.__shape_check(self.shape)

    @staticmethod
    def __shape_check(shape: Tuple[int]) -> Tuple[int]:
        """
        __shape_check checks the shape and fixes the dimensionality in case it's necessary

        :param shape: the network shape (n_input, n_hidden, n_output)
        :type shape: Tuple[int]
        :raises ValueError: Shape dimensions should represent (input, hidden, output) layers
        :return: the fixed shape
        :rtype: Tuple[int]
        """
        if isinstance(shape, int):
            shape = (None, shape, None)
        if isinstance(shape, tuple) and len(shape) != 3:
            raise ValueError(
                f"Shape dimensions should represent (input, hidden, output) layers"
            )
        return shape

    @staticmethod
    def __merge_cores(
        cores: Dict[CoreKey, DynapSimCore],
        core_map: Dict[CoreKey, List[np.uint8]],
    ) -> Dict[str, np.ndarray]:
        """
        __merge_cores merge core properties in arrays, number of elements representing the number of neurons

        :param cores: a dictionary of simualtion cores
        :type cores: Dict[CoreKey, DynapSimCore]
        :param core_map: a dictionary of the mapping between active cores and list of active neurons
        :type core_map: Dict[CoreKey, List[np.uint8]]
        :return: a dictionary of merged attributes
        :rtype: Dict[str, np.ndarray]
        """
        attr_dict = dict.fromkeys(DynapSimCore().__dict__.keys(), np.empty((0)))

        for (h, c), _core in cores.items():
            core_dict = _core.get_full(len(core_map[h, c]))
            for __attr in attr_dict:
                attr_dict[__attr] = np.concatenate(
                    (attr_dict[__attr], core_dict[__attr])
                )

        return attr_dict

    @classmethod
    def from_specification(
        cls,
        shape: Optional[Union[Tuple[int], int]],
        w_in: Optional[np.ndarray] = None,
        w_rec: Optional[np.ndarray] = None,
        w_in_mask: Optional[np.ndarray] = None,
        w_rec_mask: Optional[np.ndarray] = None,
        w_out: Optional[np.ndarray] = None,
        Idc: float = dcurrents["Idc"],
        If_nmda: float = dcurrents["If_nmda"],
        r_gain_ahp: float = dgain["r_gain_ahp"],
        r_gain_ampa: float = dgain["r_gain_ampa"],
        r_gain_gaba: float = dgain["r_gain_gaba"],
        r_gain_nmda: float = dgain["r_gain_nmda"],
        r_gain_shunt: float = dgain["r_gain_shunt"],
        r_gain_mem: float = dgain["r_gain_mem"],
        t_pulse_ahp: float = dtime["t_pulse_ahp"],
        t_pulse: float = dtime["t_pulse"],
        t_ref: float = dtime["t_ref"],
        Ispkthr: float = dcurrents["Ispkthr"],
        tau_ahp: float = dtime["tau_ahp"],
        tau_ampa: float = dtime["tau_ampa"],
        tau_gaba: float = dtime["tau_gaba"],
        tau_nmda: float = dtime["tau_nmda"],
        tau_shunt: float = dtime["tau_shunt"],
        tau_mem: float = dtime["tau_mem"],
        Iw_0: float = dweight["Iw_0"],
        Iw_1: float = dweight["Iw_1"],
        Iw_2: float = dweight["Iw_2"],
        Iw_3: float = dweight["Iw_3"],
        Iw_ahp: float = dcurrents["Iw_ahp"],
        C_ahp: float = dlayout["C_ahp"],
        C_ampa: float = dlayout["C_ampa"],
        C_gaba: float = dlayout["C_gaba"],
        C_nmda: float = dlayout["C_nmda"],
        C_pulse_ahp: float = dlayout["C_pulse_ahp"],
        C_pulse: float = dlayout["C_pulse"],
        C_ref: float = dlayout["C_ref"],
        C_shunt: float = dlayout["C_shunt"],
        C_mem: float = dlayout["C_mem"],
        Io: float = dlayout["Io"],
        kappa_n: float = dlayout["kappa_n"],
        kappa_p: float = dlayout["kappa_p"],
        Ut: float = dlayout["Ut"],
        Vth: float = dlayout["Vth"],
    ) -> DynapSimConfig:
        """
        from_specification is a class factory method using the weight specifications and the current/layout parameters
        One can directly define w_in/w_rec/w_out or one can define the weight masks.
        If weight masks are defined, weight matrices are calculated using the simulation currents and weight masks together
        All the simulation currents and layout parameter can be passed by kwargs. For more info, please check `DynapSimCore.from_specification()`

        :param shape: the network shape (n_input, n_hidden, n_output). If shape is int, then it means that the input and output should not be considered.
        :type shape: Optional[Union[Tuple[int], int]]
        :param w_in: input weight matrix
        :type w_in: Optional[np.ndarray], optional
        :param w_rec: recurrent weight matrix
        :type w_rec: Optional[np.ndarray], optional
        :param w_in_mask: the weight mask to set the input weight matrix. Used if `w_in` is None
        :type w_in_mask: Optional[np.ndarray], optional
        :param w_rec_mask: the weight mask to set the recurrent weight matrix. Used if `w_rec` is None
        :type w_rec_mask: Optional[np.ndarray], optional
        :param w_out: the output weight mask (binary in general)
        :type w_out: Optional[np.ndarray], optional
        :param Idc: Constant DC current injected to membrane in Amperes
        :type Idc: float, optional
        :param If_nmda: NMDA gate soft cut-off current setting the NMDA gating voltage in Amperes
        :type If_nmda: float, optional
        :param r_gain_ahp: spike frequency adaptation block gain ratio :math:`Igain_ahp/Itau_ahp`
        :type r_gain_ahp: float, optional
        :param r_gain_ampa: excitatory AMPA synpse gain ratio :math:`Igain_ampa/Itau_ampa`
        :type r_gain_ampa: float, optional
        :param r_gain_gaba: inhibitory GABA synpse gain ratio :math:`Igain_gaba/Itau_gaba `
        :type r_gain_gaba: float, optional
        :param r_gain_nmda: excitatory NMDA synpse gain ratio :math:`Igain_nmda/Itau_nmda`
        :type r_gain_nmda: float, optional
        :param r_gain_shunt: inhibitory SHUNT synpse gain ratio :math:`Igain_shunt/Itau_shunt`
        :type r_gain_shunt: float, optional
        :param r_gain_mem: neuron membrane gain ratio :math:`Igain_mem/Itau_mem`
        :type r_gain_mem: float, optional
        :param t_pulse_ahp: the spike pulse width for spike frequency adaptation circuit in seconds
        :type t_pulse_ahp: float, optional
        :param t_pulse: the spike pulse width for neuron membrane in seconds
        :type t_pulse: float, optional
        :param t_ref: refractory period of the neurons in seconds
        :type t_ref: float, optional
        :param Ispkthr: spiking threshold current, neuron spikes if :math:`Imem > Ispkthr` in Amperes
        :type Ispkthr: float, optional
        :param tau_ahp: Spike frequency leakage time constant in seconds
        :type tau_ahp: float, optional
        :param tau_ampa: AMPA synapse leakage time constant in seconds
        :type tau_ampa: float, optional
        :param tau_gaba: GABA synapse leakage time constant in seconds
        :type tau_gaba: float, optional
        :param tau_nmda: NMDA synapse leakage time constant in seconds
        :type tau_nmda: float, optional
        :param tau_shunt:SHUNT synapse leakage time constant in seconds
        :type tau_shunt: float, optional
        :param tau_mem: Neuron membrane leakage time constant in seconds
        :type tau_mem: float, optional
        :param Iw_0: weight bit 0 current of the neurons of the core in Amperes
        :type Iw_0: float
        :param Iw_1: weight bit 1 current of the neurons of the core in Amperes
        :type Iw_1: float
        :param Iw_2: weight bit 2 current of the neurons of the core in Amperes
        :type Iw_2: float
        :param Iw_3: weight bit 3 current of the neurons of the core in Amperes
        :type Iw_3: float
        :param Iw_ahp: spike frequency adaptation weight current of the neurons of the core in Amperes
        :type Iw_ahp: float
        :param C_ahp: AHP synapse capacitance in Farads
        :type C_ahp: float, optional
        :param C_ampa: AMPA synapse capacitance in Farads
        :type C_ampa: float, optional
        :param C_gaba: GABA synapse capacitance in Farads
        :type C_gaba: float, optional
        :param C_nmda: NMDA synapse capacitance in Farads
        :type C_nmda: float, optional
        :param C_pulse_ahp: spike frequency adaptation circuit pulse-width creation sub-circuit capacitance in Farads
        :type C_pulse_ahp: float, optional
        :param C_pulse: pulse-width creation sub-circuit capacitance in Farads
        :type C_pulse: float, optional
        :param C_ref: refractory period sub-circuit capacitance in Farads
        :type C_ref: float, optional
        :param C_shunt: SHUNT synapse capacitance in Farads
        :type C_shunt: float, optional
        :param C_mem: neuron membrane capacitance in Farads
        :type C_mem: float, optional
        :param Io: Dark current in Amperes that flows through the transistors even at the idle state
        :type Io: Union[float, np.ndarray], optional
        :param kappa_n: Subthreshold slope factor (n-type transistor)
        :type kappa_n: Union[float, np.ndarray], optional
        :param kappa_p: Subthreshold slope factor (p-type transistor)
        :type kappa_p: Union[float, np.ndarray], optional
        :param Ut: Thermal voltage in Volts
        :type Ut: Union[float, np.ndarray], optional
        :param Vth: The cut-off Vgs potential of the transistors in Volts (not type specific)
        :type Vth: Union[float, np.ndarray], optional
        :return: a `DynapSimConfig` object created from specifications
        :rtype: DynapSimConfig
        """

        # Create the default maps
        tag_in, n_rec, tag_out = cls.__shape_check(shape)
        idx_map = cls.default_idx_map(n_rec)
        core_map = Connector.core_map_from_idx_map(idx_map)

        # Fill the core dictionary with simulated cores generated by the SAME specifications
        cores = {}
        for h, c in core_map:
            cores[(h, c)] = DynapSimCore.from_specification(
                Idc=Idc,
                If_nmda=If_nmda,
                r_gain_ahp=r_gain_ahp,
                r_gain_ampa=r_gain_ampa,
                r_gain_gaba=r_gain_gaba,
                r_gain_nmda=r_gain_nmda,
                r_gain_shunt=r_gain_shunt,
                r_gain_mem=r_gain_mem,
                t_pulse_ahp=t_pulse_ahp,
                t_pulse=t_pulse,
                t_ref=t_ref,
                Ispkthr=Ispkthr,
                tau_ahp=tau_ahp,
                tau_ampa=tau_ampa,
                tau_gaba=tau_gaba,
                tau_nmda=tau_nmda,
                tau_shunt=tau_shunt,
                tau_mem=tau_mem,
                Iw_0=Iw_0,
                Iw_1=Iw_1,
                Iw_2=Iw_2,
                Iw_3=Iw_3,
                Iw_ahp=Iw_ahp,
                C_ahp=C_ahp,
                C_ampa=C_ampa,
                C_gaba=C_gaba,
                C_nmda=C_nmda,
                C_pulse_ahp=C_pulse_ahp,
                C_pulse=C_pulse,
                C_ref=C_ref,
                C_shunt=C_shunt,
                C_mem=C_mem,
                Io=Io,
                kappa_n=kappa_n,
                kappa_p=kappa_p,
                Ut=Ut,
                Vth=Vth,
            )

        attr_dict = cls.__merge_cores(cores, core_map)

        def get_weight(mask: np.ndarray, n_in: int, n_rec: int) -> np.ndarray:
            """
            get_weight creates the weight matrix using the mask given

            :param mask: the binary encoded weight mask
            :type mask: np.ndarray
            :param n_in: axis=0 length
            :type n_in: int
            :param n_rec: axis=1 length
            :type n_rec: int
            :return: the weight matrix
            :rtype: np.ndarray
            """
            if mask is None:
                mask = cls.poisson_mask((n_in, n_rec, 4))
            wparam = cls.__get_weight_params(mask, attr_dict)
            return mask, wparam.weights

        ## Get Weights
        if w_in is None and tag_in is not None:
            w_in_mask, w_in = get_weight(w_in_mask, tag_in, n_rec)

        if w_rec is None:
            w_rec_mask, w_rec = get_weight(w_rec_mask, n_rec, n_rec)

        if w_out is None and tag_out is not None:
            w_out = np.eye(n_rec, tag_out)

        # Store the router as well
        router = Router(
            n_chips=(len(core_map) // NUM_CHIPS) + 1,
            shape=shape,
            core_map=core_map,
            idx_map=idx_map,
            w_in_mask=w_in_mask,
            w_rec_mask=w_rec_mask,
            w_out_mask=w_out,
        )

        _mod = cls(
            shape=shape,
            w_in=w_in,
            w_rec=w_rec,
            w_out=w_out,
            cores=cores,
            router=router,
            **attr_dict,
        )
        return _mod

    @classmethod
    def __from_samna(
        cls,
        config: Union[Dynapse1Configuration, Dynapse2Configuration],
        router_constructor: Callable[
            [Union[Dynapse1Configuration, Dynapse2Configuration]], Router
        ],
        simcore_constructor: Callable[
            [Union[Dynapse1Core, Dynapse2Core]], DynapSimCore
        ],
        **kwargs,
    ) -> DynapSimConfig:
        """
        __from_samna is the common class factory method for Dynap-SE1 and Dynap-SE2 samna configuration objects
        One can overwrite any simulation parameter by passing them in kwargs like (...tau_mem = 1e-3)

        :param config: the samna device configuration object
        :type config: Union[Dynapse1Configuration, Dynapse2Configuration]
        :param router_constructor: the device specific router constructor method
        :type router_constructor: Callable[ [Union[Dynapse1Configuration, Dynapse2Configuration]], Router ]
        :param simcore_constructor: the device specific simcore constructor method
        :type simcore_constructor: Callable[ [Union[Dynapse1Core, Dynapse2Core]], DynapSimCore ]
        :return: a DynapSimConfig object constructed using samna configuration objects
        :rtype: DynapSimConfig
        """

        router = router_constructor(config)
        core_map = router.core_map

        cores: Dict[CoreKey, DynapSimCore] = {}

        for h, c in core_map:
            simcore = simcore_constructor(config.chips[h].cores[c])
            cores[(h, c)] = simcore

        # Overwrite the simulation the cores if kwargs given
        if kwargs:
            for key, value in kwargs.items():
                if key in DynapSimCurrents.__annotations__:
                    for c in cores:
                        cores[c] = cores[c].update(key, value)

                if key in DynapSimTime.__annotations__:
                    for c in cores:
                        cores[c] = cores[c].update_time_constant(key, value)

                if key in DynapSimGain.__annotations__:
                    for c in cores:
                        cores[c] = cores[c].update_gain_ratio(key, value)

        # Merge the cores, get the weights, return the module
        attr_dict = cls.__merge_cores(cores, core_map)
        w_in_param = cls.__get_weight_params(router.w_in_mask, attr_dict)
        w_rec_param = cls.__get_weight_params(router.w_rec_mask, attr_dict)

        _mod = cls(
            shape=router.shape,
            w_in=w_in_param.weights,
            w_rec=w_rec_param.weights,
            w_out=router.w_out_mask,
            cores=cores,
            router=router,
            **attr_dict,
        )

        return _mod

    @classmethod
    def from_Dynapse1Configuration(
        cls, config: Dynapse1Configuration, **kwargs
    ) -> DynapSimConfig:
        """
        from_Dynapse1Configuration is the Dynap-SE1 specific class factory method exploiting `.__from_samna()`

        :param config: the samna configuration object
        :type config: Dynapse1Configuration
        :return: a DynapSimConfig object obtained using samna `Dynapse1Configuration` object
        :rtype: DynapSimConfig
        """
        return cls.__from_samna(
            config=config,
            router_constructor=Router.from_Dynapse1Configuration,
            simcore_constructor=DynapSimCore.from_Dynapse1Core,
            Ispkthr=1e-6,
            Ipulse_ahp=3.5e-7,
            **kwargs,
        )

    @classmethod
    def from_Dynapse2Configuration(
        cls, config: Dynapse2Configuration, **kwargs
    ) -> DynapSimConfig:
        """
        from_Dynapse2Configuration is the Dynap-SE2 specific class factory method exploiting `.__from_samna()`

        :param config: the samna configuration object
        :type config: Dynapse2Configuration
        :return: a DynapSimConfig object obtained using samna `Dynapse2Configuration` object
        :rtype: DynapSimConfig
        """
        return cls.__from_samna(
            config=config,
            router_constructor=Router.from_Dynapse2Configuration,
            simcore_constructor=DynapSimCore.from_Dynapse2Core,
            **kwargs,
        )

    ### --- Utilities --- ###

    @staticmethod
    def __get_weight_params(
        mask: np.ndarray, attr_dict: Dict[str, np.ndarray]
    ) -> WeightParameters:
        """
        __get_weight_params creates a weight parameter object using a merged attribute dictionary

        :param mask: the weight mask obtained from router
        :type mask: np.ndarray
        :param attr_dict: a merged attribute dictioary obtained from several simulation cores
        :type attr_dict: Dict[str, np.ndarray]
        :return: a trainable weight parameter object
        :rtype: WeightParameters
        """
        _wparam = WeightParameters(
            Iw_0=attr_dict["Iw_0"],
            Iw_1=attr_dict["Iw_1"],
            Iw_2=attr_dict["Iw_2"],
            Iw_3=attr_dict["Iw_3"],
            mux=mask,
        )
        return _wparam

    @staticmethod
    def default_idx_map(size: int) -> Dict[int, NeuronKey]:
        """
        default_idx_map creates the default index map with a random number of neurons splitting the neurons across different cores

        :param size: desired simulation size, the total number of neurons
        :type size: int
        :return: a dictionary of the mapping between matrix indexes of the neurons and their neuron keys
        :rtype: Dict[int, NeuronKey]
        """

        h = lambda idx: idx // (NUM_CORES * NUM_NEURONS)
        c = lambda idx: idx // NUM_NEURONS
        n = lambda idx: idx - c(idx) * NUM_NEURONS
        idx_map = {idx: (h(idx), c(idx), n(idx)) for idx in range(size)}

        return idx_map

    @staticmethod
    def poisson_mask(
        shape: Tuple[int],
        fill_rate: Union[float, List[float]] = [0.25, 0.2, 0.04, 0.06],
        n_bits: int = 4,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        poisson_mask creates a three-dimensional weight mask using a poisson distribution
        The function takes desired fill rates of the matrices and converts it to a poisson lambda.
        The analytical solution is here:

        .. math ::
            f(X=x) = \\dfrac{\\lambda^{x}\\cdot e^{-\\lambda}}{x!}
            f(X=0) = e^{-\\lambda}
            p = 1 - f(X=0) = 1 - e^{-\\lambda}
            e^{-\\lambda} = 1-p
            \\lambda = -ln(1-p) ; 0<p<1

        :param shape: the three-dimensional shape of the weight matrix
        :type shape: Tuple[int]
        :param fill_rate: the fill rates desired to be converted to a list of posisson rates of the weights specific to synaptic-gates (3rd dimension)
        :type fill_rate: Union[float, List[float]]
        :param n_bits: number of weight parameter bits used
        :type n_bits: int
        :raises ValueError: The possion rate list given does not have the same shape with the 3rd dimension
        :return: 3D numpy array representing a Dynap-SE connectivity matrix
        :rtype: np.ndarray
        """
        np.random.seed(seed)

        if isinstance(shape, int):
            shape = (shape,)

        if isinstance(fill_rate, float):
            fill_rate = [fill_rate] * shape[-1]

        if len(fill_rate) != shape[-1]:
            raise ValueError(
                "The possion rate list given does not have the same shape with the last dimension"
            )

        lambda_list = -np.log(1 - np.array(fill_rate))

        # First create a base weight matrix
        w_shape = [s for s in shape]
        w_shape[-1] = 1
        columns = [np.random.poisson(l, w_shape) for l in lambda_list]

        # Scale the position mask
        pos_mask = np.concatenate(columns, axis=-1)
        pos_mask = np.clip(pos_mask, 0, 1)
        weight = pos_mask * np.random.randint(0, 2 ** n_bits, shape)

        return weight


@dataclass
class DynapSimCoreHigh:
    """
    DynapSimCoreHigh is an abstract class to be used as a boiler-plate for high-level projection classes
    """

    @classmethod
    def from_DynapSimCore(cls, core: DynapSimCore) -> DynapSimCoreHigh:
        NotImplementedError("Abstract method not implemented!")

    def update_DynapSimCore(self, core: DynapSimCore) -> DynapSimCore:
        NotImplementedError("Abstract method not implemented!")


@dataclass
class DynapSimTime(DynapSimCoreHigh):
    """
    DynapSimTime stores the high-level projections of the currents setting time consant values

    :param t_pulse_ahp: the spike pulse width for spike frequency adaptation circuit in seconds
    :type t_pulse_ahp: FloatVector, optional
    :param t_pulse: the spike pulse width for neuron membrane in seconds
    :type t_pulse: FloatVector, optional
    :param t_ref: refractory period of the neurons in seconds
    :type t_ref: FloatVector, optional
    :param tau_ahp: Spike frequency leakage time constant in seconds
    :type tau_ahp: FloatVector, optional
    :param tau_ampa: AMPA synapse leakage time constant in seconds
    :type tau_ampa: FloatVector, optional
    :param tau_gaba: GABA synapse leakage time constant in seconds
    :type tau_gaba: FloatVector, optional
    :param tau_nmda: NMDA synapse leakage time constant in seconds
    :type tau_nmda: FloatVector, optional
    :param tau_shunt:SHUNT synapse leakage time constant in seconds
    :type tau_shunt: FloatVector, optional
    :param tau_mem: Neuron membrane leakage time constant in seconds
    :type tau_mem: FloatVector, optional
    """

    t_pulse_ahp: FloatVector = dtime["t_pulse_ahp"]
    t_pulse: FloatVector = dtime["t_pulse"]
    t_ref: FloatVector = dtime["t_ref"]
    tau_ahp: FloatVector = dtime["tau_ahp"]
    tau_ampa: FloatVector = dtime["tau_ampa"]
    tau_gaba: FloatVector = dtime["tau_gaba"]
    tau_nmda: FloatVector = dtime["tau_nmda"]
    tau_shunt: FloatVector = dtime["tau_shunt"]
    tau_mem: FloatVector = dtime["tau_mem"]

    @classmethod
    def from_DynapSimCore(cls, core: DynapSimCore) -> DynapSimTime:
        """
        from_DynapSimCore is a class factory method using DynapSimCore object

        :param core: the `DynapSimCore` object contatining the current values setting the time constants
        :type core: DynapSimCore
        :return: a `DynapSimTime` object, that stores the time constants set by a `DynapSimCore`
        :rtype: DynapSimTime
        """

        def _tau(name: str) -> float:
            """
            _tau converts a current value to a tau parameter

            :param name: the name of the subcircuit
            :type name: str
            :return: time constant in seconds
            :rtype: float
            """
            return cls.tau_converter(
                tau=core.__getattribute__(f"Itau_{name}"),
                Ut=core.Ut,
                kappa=(core.kappa_n + core.kappa_p) / 2,
                C=core.__getattribute__(f"C_{name}"),
            )

        def _pw(name: str) -> float:
            """
            _pw converts a current value to a pulse width parameter

            :param name: the name of the subcircuit
            :type name: str
            :return: pulse width in seconds
            :rtype: float
            """
            return cls.pw_converter(
                pw=core.__getattribute__(f"I{name}"),
                Vth=core.Vth,
                C=core.__getattribute__(f"C_{name}"),
            )

        # Construct the object
        _mod = cls(
            t_pulse_ahp=_pw("pulse_ahp"),
            t_pulse=_pw("pulse"),
            t_ref=_pw("ref"),
            tau_ahp=_tau("ahp"),
            tau_ampa=_tau("ampa"),
            tau_gaba=_tau("gaba"),
            tau_nmda=_tau("nmda"),
            tau_shunt=_tau("shunt"),
            tau_mem=_tau("mem"),
        )
        return _mod

    def update_DynapSimCore(self, core: DynapSimCore) -> DynapSimCore:
        """
        update_DynapSimCore updates a `DynapSimCore` object using the defined attirbutes in `DynapSimTime` object
        It does not change the original core object and returns an updated copy

        :param core: a `DynapSimCore` object to be updated
        :type core: DynapSimCore
        :return: an updated copy of DynapSimCore object
        :rtype: DynapSimCore
        """

        _core = replace(core)

        def _tau(name: str) -> FloatVector:
            """
            _tau converts a time constant to a representative current

            :param name: the name of the subcircuit of interest
            :type name: str
            :return: the current in Amperes setting the time constant
            :rtype: FloatVector
            """
            tau = self.__getattribute__(f"tau_{name}")
            if tau is None:
                __value = _core.__getattribute__(f"Itau_{name}")
            else:
                __value = self.tau_converter(
                    tau=tau,
                    Ut=_core.Ut,
                    kappa=(_core.kappa_n + _core.kappa_p) / 2,
                    C=_core.__getattribute__(f"C_{name}"),
                )

            return __value

        def _pw(name: str) -> FloatVector:
            """
            _pw converts a pulse width to a representative current

            :param name: the name of the subcircuit of interest
            :type name: str
            :return: the current in Amperes setting the pulse width
            :rtype: FloatVector
            """
            pw = self.__getattribute__(f"t_{name}")
            if pw is None:
                __value = _core.__getattribute__(f"I{name}")
            else:
                __value = self.pw_converter(
                    pw=pw,
                    Vth=_core.Vth,
                    C=_core.__getattribute__(f"C_{name}"),
                )

            return __value

        # Update
        for time in ["pulse_ahp", "pulse", "ref"]:
            _core.__setattr__(f"I{time}", _pw(time))

        for syn in ["ahp", "ampa", "gaba", "nmda", "shunt", "mem"]:
            _core.__setattr__(f"Itau_{syn}", _tau(syn))

        return _core


    @staticmethod
    def tau_converter(tau: FloatVector, Ut: FloatVector, kappa: FloatVector, C: FloatVector) -> FloatVector:
        """
        tau_converter converts a time constant to a current value or a current value to a time constant using the conversion above:

        .. math ::

            \\tau = \\dfrac{C U_{T}}{\\kappa I_{\\tau}}

        :param tau: a time constant or a current setting the time constant
        :type tau: FloatVector
        :param Ut: Thermal voltage in Volts
        :type Ut: FloatVector, optional
        :param kappa: Subthreshold slope factor of the responsible transistor
        :type kappa: FloatVector
        :param C: the capacitance value of the subcircuit
        :type C: FloatVector
        :return: a time constant or a current setting the time constant. If a time constant provided as input, the current is returned and vice versa
        :rtype: FloatVector
        """
        if tau is None or (np.array(tau) <= np.array(0.0)).any():
            return None
        _tau = ((Ut / kappa) * C) / tau
        return _tau

    @staticmethod
    def pw_converter(pw: FloatVector, Vth: FloatVector, C: FloatVector) -> FloatVector:
        """
        pw_converter converts a pulse width to a current value or a current value to a pulse width using the conversion above:

        .. math ::

            pw = \\dfrac{C V_{th}}{\\kappa I_{pw}}

        :param pw: a pulse width or a current setting the pulse width
        :type pw: FloatVector
        :param Vth: The cut-off Vgs potential of the respective transistor in Volts
        :type Vth: FloatVector
        :param C: the capacitance value of the subcircuit
        :type C: FloatVector
        :return: a pulse width or a current setting the pulse width. If a pulse width provided as input, the current is returned and vice versa
        :rtype: FloatVector
        """
        if pw is None or (np.array(pw) <= np.array(0.0)).any():
            return None
        _pw = (Vth * C) / pw
        return _pw


@dataclass
class DynapSimGain(DynapSimCoreHigh):
    """
    DynapSimGain stores the ratio between gain and tau current values

    :param r_gain_ahp: spike frequency adaptation block gain ratio :math:`Igain_ahp/Itau_ahp`
    :type r_gain_ahp: FloatVector, optional
    :param r_gain_ampa: excitatory AMPA synpse gain ratio :math:`Igain_ampa/Itau_ampa`
    :type r_gain_ampa: FloatVector, optional
    :param r_gain_gaba: inhibitory GABA synpse gain ratio :math:`Igain_gaba/Itau_gaba `
    :type r_gain_gaba: FloatVector, optional
    :param r_gain_nmda: excitatory NMDA synpse gain ratio :math:`Igain_nmda/Itau_nmda`
    :type r_gain_nmda: FloatVector, optional
    :param r_gain_shunt: inhibitory SHUNT synpse gain ratio :math:`Igain_shunt/Itau_shunt`
    :type r_gain_shunt: FloatVector, optional
    :param r_gain_mem: neuron membrane gain ratio :math:`Igain_mem/Itau_mem`
    :type r_gain_mem: FloatVector, optional
    """

    r_gain_ahp: FloatVector = dgain["r_gain_ahp"]
    r_gain_ampa: FloatVector = dgain["r_gain_ampa"]
    r_gain_gaba: FloatVector = dgain["r_gain_gaba"]
    r_gain_nmda: FloatVector = dgain["r_gain_nmda"]
    r_gain_shunt: FloatVector = dgain["r_gain_shunt"]
    r_gain_mem: FloatVector = dgain["r_gain_mem"]

    @classmethod
    def from_DynapSimCore(cls, core: DynapSimCore) -> DynapSimGain:
        """
        from_DynapSimCore is a class factory method using DynapSimCore object

        :param core: the `DynapSimCore` object contatining the current values setting the gain ratios
        :type core: DynapSimCore
        :return: a `DynapSimGain` object, that stores the gain ratios set by a `DynapSimCore`
        :rtype: DynapSimGain
        """
        _r_gain = lambda name: cls.ratio_gain(
            Igain=core.__getattribute__(f"Igain_{name}"),
            Itau=core.__getattribute__(f"Itau_{name}"),
        )

        # Construct the object
        _mod = cls(
            r_gain_ahp=_r_gain("ahp"),
            r_gain_ampa=_r_gain("ampa"),
            r_gain_gaba=_r_gain("gaba"),
            r_gain_nmda=_r_gain("nmda"),
            r_gain_shunt=_r_gain("shunt"),
            r_gain_mem=_r_gain("mem"),
        )
        return _mod

    def update_DynapSimCore(self, core: DynapSimCore) -> DynapSimCore:
        """
        update_DynapSimCore updates a `DynapSimCore` object using the defined attirbutes in `DynapSimGain` object
        It does not change the original core object and returns an updated copy

        :param core: a `DynapSimCore` object to be updated
        :type core: DynapSimCore
        :return: an updated copy of DynapSimCore object
        :rtype: DynapSimCore
        """
        _I_gain = lambda name: self.gain_current(
            Igain=_core.__getattribute__(f"Igain_{name}"),
            r_gain=self.__getattribute__(f"r_gain_{name}"),
            Itau=_core.__getattribute__(f"Itau_{name}"),
        )
        _core = replace(core)

        for syn in ["ahp", "ampa", "gaba", "nmda", "shunt", "mem"]:
            _core.__setattr__(f"Igain_{syn}", _I_gain(syn))

        return _core

    @staticmethod
    def ratio_gain(Igain: Optional[FloatVector], Itau: Optional[FloatVector]) -> FloatVector:
        """
        ratio_gain checks the parameters and divide Igain by Itau

        :param Igain: any gain bias current in Amperes
        :type Igain: Optional[FloatVector]
        :param Itau: any leakage current in Amperes
        :type Itau: Optional[FloatVector]
        :return: the ratio between the currents if the currents are properly set
        :rtype: FloatVector
        """

        if (
            Itau is not None
            and (np.array(Itau) > 0).all()
            and Igain is not None
            and (np.array(Igain) > 0).all()
        ):
            return Igain / Itau
        else:
            return None

    @staticmethod
    def gain_current(
        Igain: Optional[FloatVector], r_gain: Optional[FloatVector], Itau: Optional[FloatVector]
    ) -> FloatVector:
        """
        gain_current checks the ratio and Itau to deduce Igain out of them

        :param Igain: any gain bias current in Amperes
        :type Igain: Optional[FloatVector]
        :param r_gain: the ratio between Igain and Itau
        :type r_gain: Optional[FloatVector]
        :param Itau: any leakage current in Amperes
        :type Itau: Optional[FloatVector]
        :return: the gain bias current Igain in Amperes obtained from r_gain and Itau
        :rtype: FloatVector
        """
        if r_gain is None:
            return Igain
        elif Itau is not None:
            return Itau * r_gain
        else:
            return Igain


if __name__ == "__main__":
    config = DynapSimCore()
    print(config.__dict__)
