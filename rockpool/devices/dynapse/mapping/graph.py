"""
Dynap-SE2 graph implementation
"""

from __future__ import annotations
from typing import Any, Dict, Optional, List, Tuple, Union

import logging
import numpy as np
from dataclasses import dataclass, field

from rockpool.typehints import FloatVector, IntVector
from rockpool.graph import (
    GenericNeurons,
    GraphModule,
    LIFNeuronWithSynsRealValue,
    replace_module,
)

from rockpool.devices.dynapse.lookup import (
    default_layout,
    default_weights,
    default_time_constants,
    default_gain_ratios,
)
from rockpool.devices.dynapse.parameters import DynapSimGain, DynapSimTime

try:
    from torch import Tensor
except:
    logging.warn("torch is not available!")
    Tensor = np.ndarray


__all__ = ["DynapseNeurons"]


@dataclass(eq=False, repr=False)
class DynapseNeurons(GenericNeurons):
    """
    DynapseNeurons stores the core computational properties of a Dynap-SE network
    """

    Idc: Union[IntVector, FloatVector] = field(default_factory=list)
    """Constant DC current injected to membrane in Amperes"""

    If_nmda: Union[IntVector, FloatVector] = field(default_factory=list)
    """NMDA gate soft cut-off current setting the NMDA gating voltage in Amperes"""

    Igain_ahp: Union[IntVector, FloatVector] = field(default_factory=list)
    """gain bias current of the spike frequency adaptation block in Amperes"""

    Igain_mem: Union[IntVector, FloatVector] = field(default_factory=list)
    """gain bias current for neuron membrane in Amperes"""

    Igain_syn: Union[IntVector, FloatVector] = field(default_factory=list)
    """gain bias current of synaptic gates (AMPA, GABA, NMDA, SHUNT) combined in Amperes"""

    Ipulse_ahp: Union[IntVector, FloatVector] = field(default_factory=list)
    """bias current setting the pulse width for spike frequency adaptation block ``t_pulse_ahp`` in Amperes"""

    Ipulse: Union[IntVector, FloatVector] = field(default_factory=list)
    """bias current setting the pulse width for neuron membrane ``t_pulse`` in Amperes"""

    Iref: Union[IntVector, FloatVector] = field(default_factory=list)
    """bias current setting the refractory period ``t_ref`` in Amperes"""

    Ispkthr: Union[IntVector, FloatVector] = field(default_factory=list)
    """spiking threshold current, neuron spikes if :math:`I_{mem} > I_{spkthr}` in Amperes"""

    Itau_ahp: Union[IntVector, FloatVector] = field(default_factory=list)
    """Spike frequency adaptation leakage current setting the time constant ``tau_ahp`` in Amperes"""

    Itau_mem: Union[IntVector, FloatVector] = field(default_factory=list)
    """Neuron membrane leakage current setting the time constant ``tau_mem`` in Amperes"""

    Itau_syn: Union[IntVector, FloatVector] = field(default_factory=list)
    """AMPA, GABA, NMDA, SHUNT) synapses combined leakage current setting the time constant ``tau_syn`` in Amperes"""

    Iw_ahp: Union[IntVector, FloatVector] = field(default_factory=list)
    """spike frequency adaptation weight current of the neurons of the core in Amperes"""

    Iscale: Optional[float] = None
    """the scaling current"""

    dt: Optional[float] = None
    """the time step for the forward-Euler ODE solver"""

    @classmethod
    def _convert_from(
        cls,
        mod: GraphModule,
        r_gain_mem: FloatVector = default_gain_ratios["r_gain_mem"],
        r_gain_syn: FloatVector = default_gain_ratios["r_gain_ampa"],
        t_pulse: FloatVector = default_time_constants["t_pulse"],
        t_ref: FloatVector = default_time_constants["t_ref"],
        C_pulse: FloatVector = default_layout["C_pulse"],
        C_ref: FloatVector = default_layout["C_ref"],
        C_mem: FloatVector = default_layout["C_mem"],
        C_syn: FloatVector = default_layout["C_syn"],
        Iscale: float = default_weights["Iscale"],
    ) -> DynapseNeurons:
        """
        _convert_from converts a graph module to DynapseNeuron structure. Uses default parameter

        NOTE:

        LIF does not have equivalent computation for
            * AHP (After-Hyper-Polarization)
            * NMDA Voltage Depended Gating

        Therefore : Itau_ahp, If_nmda, Igain_ahp, Ipulse_ahp, and, Iw_ahp currents are zero.

        :param mod: the reference graph module
        :type mod: GraphModule
        :param r_gain_mem: neuron membrane gain ratio, defaults to default_gain_ratios["r_gain_mem"]
        :type r_gain_mem: FloatVector, optional
        :param r_gain_syn: synapse gain ratio, defaults to default_gain_ratios["r_gain_ampa"]
        :type r_gain_syn: FloatVector, optional
        :param t_pulse: the spike pulse width for neuron membrane in seconds, defaults to default_time_constants["t_pulse"]
        :type t_pulse: FloatVector, optional
        :param t_ref: refractory period of the neurons in seconds, defaults to default_time_constants["t_ref"]
        :type t_ref: FloatVector, optional
        :param C_pulse: pulse-width creation sub-circuit capacitance in Farads, defaults to default_layout["C_pulse"]
        :type C_pulse: FloatVector, optional
        :param C_ref: refractory period sub-circuit capacitance in Farads, defaults to default_layout["C_ref"]
        :type C_ref: FloatVector, optional
        :param C_mem: neuron membrane capacitance in Farads, defaults to default_layout["C_mem"]
        :type C_mem: FloatVector, optional
        :param C_syn: synaptic capacitance in Farads, defaults to default_layout["C_syn"]
        :type C_syn: FloatVector, optional
        :param Iscale: the scaling current, defaults to default_weights["Iscale"]
        :type Iscale: float, optional
        :raises ValueError: graph module cannot be converted
        :return: DynapseNeurons object instance
        :rtype: DynapseNeurons
        """

        if isinstance(mod, cls):
            # - No need to do anything
            return mod

        elif isinstance(mod, LIFNeuronWithSynsRealValue):
            # Some lambda functions for clean computation
            shape = cls.__get_equal_shape(mod.threshold, mod.bias, mod.tau_mem)
            zero_param = lambda: np.zeros(shape, dtype=float).tolist()

            nonzero_param = lambda val: np.full(shape, val, dtype=float).tolist()

            # Tau currents has to be re-usable
            Itau_mem = cls.__leakage_current(mod.tau_mem, C_mem)
            Itau_syn = cls.__leakage_current(mod.tau_syn, C_syn)

            # - Build a new neurons module to insert into the graph
            neurons = cls._factory(
                size_in=len(mod.input_nodes),
                size_out=len(mod.output_nodes),
                name=mod.name,
                computational_module=mod.computational_module,
                Ispkthr=cls.__scale(mod.threshold, Iscale),
                Idc=cls.__scale(mod.bias, Iscale),
                Itau_mem=Itau_mem,
                Itau_syn=Itau_syn,
                Itau_ahp=zero_param(),
                If_nmda=zero_param(),
                Igain_ahp=zero_param(),
                Igain_mem=cls.__gain_current(r_gain_mem, Itau_mem),
                Igain_syn=cls.__gain_current(r_gain_syn, Itau_syn),
                Ipulse_ahp=zero_param(),
                Ipulse=nonzero_param(cls.__pulse_current(t_pulse, C_pulse)),
                Iref=nonzero_param(cls.__pulse_current(t_ref, C_ref)),
                Iw_ahp=zero_param(),
                Iscale=Iscale,
                dt=mod.dt,
            )

            # - Replace the target module and return
            replace_module(mod, neurons)
            return neurons

        else:
            raise ValueError(
                f"Graph module of type {type(mod).__name__} cannot be converted to a {cls.__name__}"
            )

    @classmethod
    def merge(cls, graph_list: List[DynapseNeurons]) -> DynapseNeurons:
        """
        merge combines a list of computational ``DynapseNeuron`` objects into one. The length of attributes are equal to the
        number of output nodes. Even though the submodules has single valued attributes, they are repeated as many times as
        the number of their output neurons.

        NOTE : Returned single object is neither connected to the input nor the outputs of the any of the modules given.

        :param graph_list: an ordered list of DynapseNeuron objects
        :type graph_list: List[DynapseNeurons]
        :return: a single ``DynapseNeuron`` object with combined arrays of attributes
        :rtype: DynapseNeurons
        """

        if not graph_list:
            raise ValueError("The merge list is empty!")

        # Initial values
        n_in = 0
        n_out = 0
        name = ""
        comp_mod_list = []
        attributes = {attr: [] for attr in cls.current_attrs()}

        # Save the reference values
        Iscale = graph_list[0].Iscale
        dt = graph_list[0].dt

        for g in graph_list:
            # Increment the length
            n_in += len(g.input_nodes)
            n_out += len(g.output_nodes)

            # Control
            if g.Iscale != Iscale:
                raise ValueError(
                    "Iscale should be the same in all modules in the merge list!"
                )
            if g.dt != dt:
                raise ValueError(
                    "dt should be the same in all modules in the merge list!"
                )

            new_attributes = g.get_full()

            # Extend attributes
            for attr in attributes:
                attributes[attr].extend(new_attributes[attr])

            # Append to name and the computational module list
            name += f"__{g.name}__"
            comp_mod_list.append(g.computational_module)

        # Get the merged module
        neurons = cls._factory(
            size_in=n_in,
            size_out=n_out,
            name=name,
            computational_module=comp_mod_list,
            Iscale=Iscale,
            dt=dt,
            **attributes,
        )

        return neurons

    def get_full(self) -> Dict[str, np.ndarray]:
        """
        get_full creates a dictionary of parameteric current attributes with extended current values

        :return: the object dictionary with extended current arrays
        :rtype: Dict[str, np.ndarray]
        """

        def __extend__(__name: str) -> Union[FloatVector, IntVector]:
            temp = self.__getattribute__(__name)

            # extend if not FloatVector
            if isinstance(temp, (int, float)):
                temp = np.full(len(self.output_nodes), temp).tolist()

            # Check the size if extended
            elif isinstance(temp, (np.ndarray, list, Tensor)):
                if len(temp) != len(self.output_nodes):
                    raise ValueError(
                        "Extended property does not match with the number of neurons!"
                    )
            else:
                raise TypeError(f"Unrecognized attribute type {type(temp)}!")

            return temp

        return {attr: __extend__(attr) for attr in self.current_attrs()}

    @staticmethod
    def current_attrs() -> List[str]:
        """
        current_attrs lists all current paramters stored inside DynapseNeurons computational graph

        :return: a list of parametric curents
        :rtype: List[str]
        """
        return list(
            DynapseNeurons.__dataclass_fields__.keys()
            - GenericNeurons.__dataclass_fields__.keys()
            - {"Iscale", "dt"}
        )

    ### --- Private Section --- ###

    @staticmethod
    def __get_equal_shape(*args) -> Tuple[int]:
        """
        get_equal_shape makes sure that the all arguments have the same shape

        :raises ValueError: Attribute shapes does not match!
        :return: the equal shape of all the arguments
        :rtype: Tuple[int]
        """

        shape = np.array(args[0]).shape if len(args) > 0 else ()

        if len(args) > 1:
            for arg in args[1:]:
                if np.array(arg).shape != shape:
                    raise ValueError(
                        f"Attribute shapes does not match! {np.array(arg).shape} != {shape}"
                    )

        return shape

    @staticmethod
    def __scale(v: FloatVector, scale: float) -> List[float]:
        """
        __scale converts any FloatVector to list and scales

        :param v: the float vector of interest
        :type v: FloatVector
        :param scale: the scaling factor
        :type scale: float
        :return: a float list
        :rtype: List[float]
        """
        return (np.array(v, dtype=float).flatten() * scale).tolist()

    @staticmethod
    def __leakage_current(tc: FloatVector, C: float) -> FloatVector:
        """
        __leakage_current uses default layout configuration and converts a time constant to a leakage current using the conversion method defined in ``DynapSimTime`` module

        :param tc: the time constant in seconds
        :type tc: FloatVector
        :param C: the capacitance value in Farads
        :type C: float
        :return: the leakage current
        :rtype: FloatVector
        """
        kappa = (default_layout["kappa_n"] + default_layout["kappa_p"]) / 2.0
        Itau = DynapSimTime.tau_converter(
            np.array(tc).flatten(), default_layout["Ut"], kappa, C
        )
        Itau = np.array(Itau, dtype=float).tolist()
        return Itau

    @staticmethod
    def __gain_current(r: float, Itau: FloatVector) -> FloatVector:
        """
        __gain_current converts a gain ratio to a amplifier gain current using the leakage current provided

        :param r: the desired amplifier gain ratio
        :type r: float
        :param Itau: the depended leakage current
        :type Itau: FloatVector
        :return: an amplifier gain current
        :rtype: FloatVector
        """
        Igain = DynapSimGain.gain_current(
            Igain=None, r_gain=r, Itau=np.array(Itau).flatten()
        )
        Igain = np.array(Igain, dtype=float).tolist()

        return Igain

    @staticmethod
    def __pulse_current(t_pw: FloatVector, C: float) -> FloatVector:
        """
        __pulse_current uses default layout configuration and converts a pulse width to a pulse current using the conversion method defined in ``DynapSimTime`` module

        :param t_pw: the pulse width in seconds
        :type t_pw: FloatVector
        :param C: the capacitance value in Farads
        :type C: float
        :return: a pulse current
        :rtype: FloatVector
        """
        Ipw = DynapSimTime.pw_converter(np.array(t_pw), default_layout["Vth"], C)
        Ipw = np.array(Ipw, dtype=float).tolist()
        return Ipw
