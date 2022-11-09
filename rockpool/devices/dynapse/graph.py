"""
Dynap-SE graph modules implementing conversion and translation methods

Note : Existing modules are reconstructed considering consistency with Xylo support.

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

15/09/2022
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

from rockpool.devices.dynapse.default import dlayout, dweight, dtime, dgain
from rockpool.devices.dynapse.config.simconfig import DynapSimGain, DynapSimTime

try:
    from torch import Tensor
except:
    logging.warn("torch is not available!")
    Tensor = np.ndarray


__all__ = ["DynapseNeurons"]


@dataclass(eq=False, repr=False)
class DynapseNeurons(GenericNeurons):
    """
    DynapseNeurons stores the core computational properties of the Dynap-SE network

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

    Idc: Union[IntVector, FloatVector] = field(default_factory=list)
    If_nmda: Union[IntVector, FloatVector] = field(default_factory=list)
    Igain_ahp: Union[IntVector, FloatVector] = field(default_factory=list)
    Igain_mem: Union[IntVector, FloatVector] = field(default_factory=list)
    Igain_syn: Union[IntVector, FloatVector] = field(default_factory=list)
    Ipulse_ahp: Union[IntVector, FloatVector] = field(default_factory=list)
    Ipulse: Union[IntVector, FloatVector] = field(default_factory=list)
    Iref: Union[IntVector, FloatVector] = field(default_factory=list)
    Ispkthr: Union[IntVector, FloatVector] = field(default_factory=list)
    Itau_ahp: Union[IntVector, FloatVector] = field(default_factory=list)
    Itau_mem: Union[IntVector, FloatVector] = field(default_factory=list)
    Itau_syn: Union[IntVector, FloatVector] = field(default_factory=list)
    Iw_ahp: Union[IntVector, FloatVector] = field(default_factory=list)
    Iscale: Optional[float] = None
    dt: Optional[float] = None

    @classmethod
    def _convert_from(
        cls,
        mod: GraphModule,
        r_gain_mem: FloatVector = dgain["r_gain_mem"],
        r_gain_syn: FloatVector = dgain["r_gain_ampa"],
        t_pulse: FloatVector = dtime["t_pulse"],
        t_ref: FloatVector = dtime["t_ref"],
        C_pulse: FloatVector = dlayout["C_pulse"],
        C_ref: FloatVector = dlayout["C_ref"],
        C_mem: FloatVector = dlayout["C_mem"],
        C_syn: FloatVector = dlayout["C_syn"],
        Iscale: float = dweight["Iscale"],
    ) -> DynapseNeurons:
        """
        _convert_from converts a graph module to DynapseNeuron structure. Uses default parameter

        NOTE

        LIF does not have equivalent computation for
        * AHP (After-Hyper-Polarization)
        * NMDA Voltage Depended Gating

        Therefore : Itau_ahp, If_nmda, Igain_ahp, Ipulse_ahp, and, Iw_ahp currents are zero.

        :param mod: the reference graph module
        :type mod: GraphModule
        :param r_gain_mem: neuron membrane gain ratio :math:`Igain_mem/Itau_mem`
        :type r_gain_mem: FloatVector, optional
        :param r_gain_syn: _description_, defaults to dgain["r_gain_ampa"]
        :type r_gain_syn: float, optional
        :param t_pulse: the spike pulse width for neuron membrane in seconds
        :type t_pulse: FloatVector, optional
        :param t_ref: refractory period of the neurons in seconds
        :type t_ref: FloatVector, optional
        :param C_pulse: pulse-width creation sub-circuit capacitance in Farads, defaults to dlayout["C_pulse"]
        :type C_pulse: FloatVector, optional
        :param C_ref: refractory period sub-circuit capacitance in Farads, defaults to dlayout["C_ref"]
        :type C_ref: FloatVector, optional
        :param C_mem: neuron membrane capacitance in Farads, defaults to dlayout["C_mem"]
        :type C_mem: FloatVector, optional
        :param C_syn: synaptic capacitance in Farads, defaults to dlayout["C_syn"]
        :type C_syn: FloatVector, optional
        :param Iscale: the scaling current, defaults to dweight["Iscale"]
        :type Iscale: float, optional
        :raises ValueError: _description_
        :return: _description_
        :rtype: DynapseNeurons
        """

        if isinstance(mod, cls):
            # - No need to do anything
            return mod

        elif isinstance(mod, LIFNeuronWithSynsRealValue):

            # Some lambda functions for clean computation
            shape = cls.get_equal_shape(mod.threshold, mod.bias, mod.tau_mem)
            zero_param = lambda: cls.zero_param(shape)
            nonzero_param = lambda val: cls.nonzero_param(val, shape)

            # Tau currents has to be re-usable
            Itau_mem = cls.leakage_current(mod.tau_mem, C_mem)
            Itau_syn = cls.leakage_current(mod.tau_syn, C_syn)

            # - Build a new neurons module to insert into the graph
            neurons = cls._factory(
                size_in=len(mod.input_nodes),
                size_out=len(mod.output_nodes),
                name=mod.name,
                computational_module=mod.computational_module,
                Ispkthr=cls.to_list_scale(mod.threshold, Iscale),
                Idc=cls.to_list_scale(mod.bias, Iscale),
                Itau_mem=Itau_mem,
                Itau_syn=Itau_syn,
                Itau_ahp=zero_param(),
                If_nmda=zero_param(),
                Igain_ahp=zero_param(),
                Igain_mem=cls.gain_current(r_gain_mem, Itau_mem),
                Igain_syn=cls.gain_current(r_gain_syn, Itau_syn),
                Ipulse_ahp=zero_param(),
                Ipulse=nonzero_param(cls.pulse_current(t_pulse, C_pulse)),
                Iref=nonzero_param(cls.pulse_current(t_ref, C_ref)),
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

    ### --- Utility Functions --- ###
    @staticmethod
    def get_equal_shape(*args) -> Tuple[int]:
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
    def zero_param(shape: Tuple[int]) -> List[float]:
        """
        zero_param creates full zero parameters

        :param shape: the desired shape
        :type shape: Tuple[int]
        :return: a zero array
        :rtype: List[float]
        """
        return np.zeros(shape, dtype=float).tolist()

    @staticmethod
    def nonzero_param(val: float, shape: Tuple[int]) -> List[float]:
        """
        nonzero_param creates a non-zero array filled with the same value

        :param val: the non-zero value
        :type val: float
        :param shape: the desired shape
        :type shape: Tuple[int]
        :return: a non-zero uniform array
        :rtype: List[float]
        """
        return np.full(shape, val, dtype=float).tolist()

    @staticmethod
    def to_list(v: FloatVector) -> List[float]:
        """
        to_list converts any FloatVector to a list

        :param v: the float vector of interest
        :type v: FloatVector
        :return: a float list
        :rtype: List[float]
        """
        return np.array(v, dtype=float).tolist()

    @staticmethod
    def to_list_scale(v: FloatVector, scale: float) -> List[float]:
        """
        to_list_scale converts any FloatVector to list and scale

        :param v: the float vector of interest
        :type v: FloatVector
        :param scale: the scaling factor
        :type scale: float
        :return: a float list
        :rtype: List[float]
        """
        return (np.array(v, dtype=float).flatten() * scale).tolist()

    @staticmethod
    def leakage_current(tc: FloatVector, C: float) -> FloatVector:
        """
        tau_current uses default layout configuration and converts a time constant to a leakage current using the conversion method defined in ``DynapSimTime`` module

        :param tc: the time constant in seconds
        :type tc: FloatVector
        :param C: the capacitance value in Farads
        :type C: float
        :return: the leakage current
        :rtype: FloatVector
        """
        kappa = (dlayout["kappa_n"] + dlayout["kappa_p"]) / 2.0
        Itau = DynapSimTime.tau_converter(
            np.array(tc).flatten(), dlayout["Ut"], kappa, C
        )
        return DynapseNeurons.to_list(Itau)

    @staticmethod
    def gain_current(r: float, Itau: FloatVector) -> FloatVector:
        """
        gain_current converts a gain ratio to a amplifier gain current using the leakage current provided

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
        return DynapseNeurons.to_list(Igain)

    @staticmethod
    def pulse_current(t_pw: FloatVector, C: float) -> FloatVector:
        """
        pulse_current uses default layout configuration and converts a pulse width to a pulse current using the conversion method defined in ``DynapSimTime`` module

        :param t_pw: the pulse width in seconds
        :type t_pw: FloatVector
        :param C: the capacitance value in Farads
        :type C: float
        :return: _description_
        :rtype: FloatVector
        """
        Ipw = DynapSimTime.pw_converter(np.array(t_pw), dlayout["Vth"], C)
        return DynapseNeurons.to_list(Ipw)

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
