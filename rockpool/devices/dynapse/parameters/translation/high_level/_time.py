"""
Dynap-SE2 time constant computation and and conversion utils

* Non User Facing * 
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from rockpool.devices.dynapse.lookup import default_time_constants

import numpy as np

from rockpool.typehints import FloatVector
from .high import DynapSimCoreHigh, DynapSimCore


__all__ = ["DynapSimTime"]


@dataclass
class DynapSimTime(DynapSimCoreHigh):
    """
    DynapSimTime stores the high-level projections of the currents setting time consant values
    """

    t_pulse_ahp: FloatVector = default_time_constants["t_pulse_ahp"]
    """the spike pulse width for spike frequency adaptation circuit in seconds"""

    t_pulse: FloatVector = default_time_constants["t_pulse"]
    """the spike pulse width for neuron membrane in seconds"""

    t_ref: FloatVector = default_time_constants["t_ref"]
    """refractory period of the neurons in seconds"""

    tau_ahp: FloatVector = default_time_constants["tau_ahp"]
    """Spike frequency leakage time constant in seconds"""

    tau_ampa: FloatVector = default_time_constants["tau_ampa"]
    """AMPA synapse leakage time constant in seconds"""

    tau_gaba: FloatVector = default_time_constants["tau_gaba"]
    """GABA synapse leakage time constant in seconds"""

    tau_nmda: FloatVector = default_time_constants["tau_nmda"]
    """NMDA synapse leakage time constant in seconds"""

    tau_shunt: FloatVector = default_time_constants["tau_shunt"]
    """synapse leakage time constant in seconds"""

    tau_mem: FloatVector = default_time_constants["tau_mem"]
    """Neuron membrane leakage time constant in seconds"""

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
    def tau_converter(
        tau: FloatVector, Ut: FloatVector, kappa: FloatVector, C: FloatVector
    ) -> FloatVector:
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
