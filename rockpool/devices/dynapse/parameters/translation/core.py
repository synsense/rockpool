"""
Dynap-SE2 simulation core implementation.
Use this module converting a hardware configuration to a simulation setting

* Non User facing *
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple
from dataclasses import dataclass, replace

import logging
import numpy as np


from rockpool.devices.dynapse.parameters import param_to_analog, analog_to_param

from rockpool.devices.dynapse.lookup import (
    sim2device_se2,
    default_layout,
    default_weights,
    default_time_constants,
    default_gain_ratios,
    default_currents,
)

from rockpool.devices.dynapse.samna_alias import Dynapse2Core

from rockpool.typehints import FloatVector

from .low_level import DynapSimCurrents, DynapSimLayout, DynapSimWeightBits
from .high_level import DynapSimTime, DynapSimGain
from .high_level.high import DynapSimCoreHigh

__all__ = ["DynapSimCore"]


@dataclass
class DynapSimCore(DynapSimCurrents, DynapSimLayout, DynapSimWeightBits):
    """
    DynapSimCore stores the simulation currents and manages the conversion from configuration objects.
    It also provides easy update mechanisms using coarse&fine values, high-level parameter representations and etc.

    ..  code-block:: python
        :caption: Device -> Simulation current (pseudo-code)

        simcore = DynapSimCore.from_Dynapse2Core(config.chips[0].cores[0])
        Itau_ampa = simcore.Itau_ampa

    """

    @classmethod
    def from_specification(
        cls,
        Idc: FloatVector = default_currents["Idc"],
        If_nmda: FloatVector = default_currents["If_nmda"],
        r_gain_ahp: FloatVector = default_gain_ratios["r_gain_ahp"],
        r_gain_ampa: FloatVector = default_gain_ratios["r_gain_ampa"],
        r_gain_gaba: FloatVector = default_gain_ratios["r_gain_gaba"],
        r_gain_nmda: FloatVector = default_gain_ratios["r_gain_nmda"],
        r_gain_shunt: FloatVector = default_gain_ratios["r_gain_shunt"],
        r_gain_mem: FloatVector = default_gain_ratios["r_gain_mem"],
        t_pulse_ahp: FloatVector = default_time_constants["t_pulse_ahp"],
        t_pulse: FloatVector = default_time_constants["t_pulse"],
        t_ref: FloatVector = default_time_constants["t_ref"],
        Ispkthr: FloatVector = default_currents["Ispkthr"],
        tau_ahp: FloatVector = default_time_constants["tau_ahp"],
        tau_ampa: FloatVector = default_time_constants["tau_ampa"],
        tau_gaba: FloatVector = default_time_constants["tau_gaba"],
        tau_nmda: FloatVector = default_time_constants["tau_nmda"],
        tau_shunt: FloatVector = default_time_constants["tau_shunt"],
        tau_mem: FloatVector = default_time_constants["tau_mem"],
        Iw_0: FloatVector = default_weights["Iw_0"],
        Iw_1: FloatVector = default_weights["Iw_1"],
        Iw_2: FloatVector = default_weights["Iw_2"],
        Iw_3: FloatVector = default_weights["Iw_3"],
        Iw_ahp: FloatVector = default_currents["Iw_ahp"],
        C_ahp: FloatVector = default_layout["C_ahp"],
        C_ampa: FloatVector = default_layout["C_ampa"],
        C_gaba: FloatVector = default_layout["C_gaba"],
        C_nmda: FloatVector = default_layout["C_nmda"],
        C_pulse_ahp: FloatVector = default_layout["C_pulse_ahp"],
        C_pulse: FloatVector = default_layout["C_pulse"],
        C_ref: FloatVector = default_layout["C_ref"],
        C_shunt: FloatVector = default_layout["C_shunt"],
        C_mem: FloatVector = default_layout["C_mem"],
        Io: FloatVector = default_layout["Io"],
        kappa_n: FloatVector = default_layout["kappa_n"],
        kappa_p: FloatVector = default_layout["kappa_p"],
        Ut: FloatVector = default_layout["Ut"],
        Vth: FloatVector = default_layout["Vth"],
    ) -> DynapSimCore:
        """
        from_specification is a class factory method helping DynapSimCore object construction
        using higher level representaitons of the currents like gain ratio or time constant whenever applicable.

        :param Idc: Constant DC current injected to membrane in Amperes, defaults to default_currents["Idc"]
        :type Idc: FloatVector, optional
        :param If_nmda: NMDA gate soft cut-off current setting the NMDA gating voltage in Amperes, defaults to default_currents["If_nmda"]
        :type If_nmda: FloatVector, optional
        :param r_gain_ahp: spike frequency adaptation block gain ratio, defaults to default_gain_ratios["r_gain_ahp"]
        :type r_gain_ahp: FloatVector, optional
        :param r_gain_ampa: xcitatory AMPA synpse gain ratio, defaults to default_gain_ratios["r_gain_ampa"]
        :type r_gain_ampa: FloatVector, optional
        :param r_gain_gaba: inhibitory GABA synpse gain ratio, defaults to default_gain_ratios["r_gain_gaba"]
        :type r_gain_gaba: FloatVector, optional
        :param r_gain_nmda: excitatory NMDA synpse gain ratio, defaults to default_gain_ratios["r_gain_nmda"]
        :type r_gain_nmda: FloatVector, optional
        :param r_gain_shunt: inhibitory SHUNT synpse gain ratio, defaults to default_gain_ratios["r_gain_shunt"]
        :type r_gain_shunt: FloatVector, optional
        :param r_gain_mem: neuron membrane gain ratio, defaults to default_gain_ratios["r_gain_mem"]
        :type r_gain_mem: FloatVector, optional
        :param t_pulse_ahp: the spike pulse width for spike frequency adaptation circuit in seconds, defaults to default_time_constants["t_pulse_ahp"]
        :type t_pulse_ahp: FloatVector, optional
        :param t_pulse: the spike pulse width for neuron membrane in seconds, defaults to default_time_constants["t_pulse"]
        :type t_pulse: FloatVector, optional
        :param t_ref: refractory period of the neurons in seconds, defaults to default_time_constants["t_ref"]
        :type t_ref: FloatVector, optional
        :param Ispkthr: spiking threshold current, neuron spikes if :math:`I_{mem} > I_{spkthr}` in Amperes, defaults to default_currents["Ispkthr"]
        :type Ispkthr: FloatVector, optional
        :param tau_ahp: Spike frequency leakage time constant in seconds, defaults to default_time_constants["tau_ahp"]
        :type tau_ahp: FloatVector, optional
        :param tau_ampa: AMPA synapse leakage time constant in seconds, defaults to default_time_constants["tau_ampa"]
        :type tau_ampa: FloatVector, optional
        :param tau_gaba: GABA synapse leakage time constant in seconds, defaults to default_time_constants["tau_gaba"]
        :type tau_gaba: FloatVector, optional
        :param tau_nmda: NMDA synapse leakage time constant in seconds, defaults to default_time_constants["tau_nmda"]
        :type tau_nmda: FloatVector, optional
        :param tau_shunt: SHUNT synapse leakage time constant in seconds, defaults to default_time_constants["tau_shunt"]
        :type tau_shunt: FloatVector, optional
        :param tau_mem: Neuron membrane leakage time constant in seconds, defaults to default_time_constants["tau_mem"]
        :type tau_mem: FloatVector, optional
        :param Iw_0: weight bit 0 current of the neurons of the core in Amperes, defaults to default_weights["Iw_0"]
        :type Iw_0: FloatVector, optional
        :param Iw_1: weight bit 1 current of the neurons of the core in Amperes, defaults to default_weights["Iw_1"]
        :type Iw_1: FloatVector, optional
        :param Iw_2: weight bit 2 current of the neurons of the core in Amperes, defaults to default_weights["Iw_2"]
        :type Iw_2: FloatVector, optional
        :param Iw_3: weight bit 3 current of the neurons of the core in Amperes, defaults to default_weights["Iw_3"]
        :type Iw_3: FloatVector, optional
        :param Iw_ahp: spike frequency adaptation weight current of the neurons of the core in Amperes, defaults to default_currents["Iw_ahp"]
        :type Iw_ahp: FloatVector, optional
        :param C_ahp: AHP synapse capacitance in Farads, defaults to default_layout["C_ahp"]
        :type C_ahp: FloatVector, optional
        :param C_ampa: AMPA synapse capacitance in Farads, defaults to default_layout["C_ampa"]
        :type C_ampa: FloatVector, optional
        :param C_gaba: GABA synapse capacitance in Farads, defaults to default_layout["C_gaba"]
        :type C_gaba: FloatVector, optional
        :param C_nmda: NMDA synapse capacitance in Farads, defaults to default_layout["C_nmda"]
        :type C_nmda: FloatVector, optional
        :param C_pulse_ahp: spike frequency adaptation circuit pulse-width creation sub-circuit capacitance in Farads, defaults to default_layout["C_pulse_ahp"]
        :type C_pulse_ahp: FloatVector, optional
        :param C_pulse: pulse-width creation sub-circuit capacitance in Farads, defaults to default_layout["C_pulse"]
        :type C_pulse: FloatVector, optional
        :param C_ref: refractory period sub-circuit capacitance in Farads, defaults to default_layout["C_ref"]
        :type C_ref: FloatVector, optional
        :param C_shunt: SHUNT synapse capacitance in Farads, defaults to default_layout["C_shunt"]
        :type C_shunt: FloatVector, optional
        :param C_mem: neuron membrane capacitance in Farads, defaults to default_layout["C_mem"]
        :type C_mem: FloatVector, optional
        :param Io: Dark current in Amperes that flows through the transistors even at the idle state, defaults to default_layout["Io"]
        :type Io: FloatVector, optional
        :param kappa_n: Subthreshold slope factor (n-type transistor), defaults to default_layout["kappa_n"]
        :type kappa_n: FloatVector, optional
        :param kappa_p: Subthreshold slope factor (p-type transistor), defaults to default_layout["kappa_p"]
        :type kappa_p: FloatVector, optional
        :param Ut: Thermal voltage in Volts, defaults to default_layout["Ut"]
        :type Ut: FloatVector, optional
        :param Vth: The cut-off Vgs potential of the transistors in Volts (not type specific), defaults to default_layout["Vth"]
        :type Vth: FloatVector, optional
        :return: DynapSimCore object instance
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
    def from_Dynapse2Core(cls, core: Dynapse2Core) -> DynapSimCore:
        """
        from_Dynapse2Core is a class factory method which uses samna configuration objects to extract the simulation currents

        :return: a dynapse core simulation object whose parameters are imported from a samna configuration object
        :rtype: DynapSimCore
        """
        _current = lambda name: param_to_analog(name, core.parameters[name])
        _dict = {sim: _current(param) for sim, param in sim2device_se2.items()}
        _mod = cls(**_dict)
        return _mod

    def export_Dynapse2Parameters(self) -> Dict[str, Tuple[np.uint8, np.uint8]]:
        """
        export_Dynapse2Parameters converts all current values to their coarse-fine value representations for device configuration

        :return: a dictionary of mapping between parameter names and respective coarse-fine values
        :rtype: Dict[str, Tuple[np.uint8, np.uint8]]
        """
        converter = lambda sim, param: analog_to_param(
            param, self.__getattribute__(sim)
        )
        param_dict = {
            param: converter(sim, param) for sim, param in sim2device_se2.items()
        }
        return param_dict

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
        update_time_constant updates currents setting time constant attributes

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
        update_gain_ratio updates currents setting gain ratio (Igain/Itau) attributes

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
        """layout returns a subset of object which belongs to DynapSimLayout"""
        __dict = dict.fromkeys(DynapSimLayout.__annotations__.keys())
        for key in __dict:
            __dict[key] = self.__getattribute__(key)
        return DynapSimLayout(**__dict)

    @property
    def currents(self) -> DynapSimCurrents:
        """currents returns a subset of object which belongs to DynapSimCurrents"""
        __dict = dict.fromkeys(DynapSimCurrents.__annotations__.keys())
        for key in __dict:
            __dict[key] = self.__getattribute__(key)
        return DynapSimCurrents(**__dict)

    @property
    def weight_bits(self) -> DynapSimWeightBits:
        """weight_bits returns a subset of object which belongs to DynapSimWeightBits"""
        __dict = dict.fromkeys(DynapSimWeightBits.__annotations__.keys())
        for key in __dict:
            __dict[key] = self.__getattribute__(key)
        return DynapSimWeightBits(**__dict)

    @property
    def time(self) -> DynapSimTime:
        """time creates the high level time constants set by currents Ipulse_ahp, Ipulse, Iref, Itau_ahp, Itau_ampa, Itau_gaba, Itau_nmda, Itau_shunt, Itau_mem"""
        return DynapSimTime.from_DynapSimCore(self)

    @property
    def gain(self) -> DynapSimGain:
        """gain creates the high level gain ratios set by currents : Igain_ahp, Igain_ampa, Igain_gaba, Igain_nmda, Igain_shunt, Igain_mem"""
        return DynapSimGain.from_DynapSimCore(self)
