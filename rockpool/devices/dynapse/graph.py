"""
Dynap-SE graph modules implementing conversion and translation methods

Note : Existing modules are reconstructed considering consistency with Xylo support.

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

15/09/2022
"""
from typing import Optional
from rockpool.graph import GenericNeurons

from dataclasses import dataclass, field

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

