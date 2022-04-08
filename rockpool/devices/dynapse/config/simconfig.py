"""
Dynap-SE Parameter classes to be used to configure DynapSEAdExpLIFJax simulation modules

renamed : dynapse1_simconfig.py -> simconfig.py @ 211208

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
24/08/2021
"""
from __future__ import annotations
from typing import Any, Dict, Generator, List, Optional, Tuple

import itertools
from dataclasses import dataclass

from jax import numpy as jnp
import numpy as np

from rockpool.devices.dynapse.infrastructure.router import Router
from rockpool.devices.dynapse.base import DynapSE, NeuronKey
from rockpool.devices.dynapse.config.layout import DynapSELayout, DynapSECapacitance
from rockpool.devices.dynapse.config.circuits import (
    WeightParameters,
    SynapseParameters,
    MembraneParameters,
    GABABParameters,
    GABAAParameters,
    NMDAParameters,
    AMPAParameters,
    AHPParameters,
)


_SAMNA_AVAILABLE = True

from rockpool.devices.dynapse.samna_alias.dynapse1 import Dynapse1Parameter

try:
    from samna.dynapse1 import (
        Dynapse1Configuration,
    )
except Exception as e:
    Dynapse1Configuration = Any

    print(
        e,
        "\nDynapSE1SimCore object cannot be factored from a samna config object!",
    )
    _SAMNA_AVAILABLE = False


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
    :param gaba_b: GABA_B synapse paramters (Isyn, Itau, Ith), defaults to None
    :type gaba_b: Optional[SynapseParameters], optional
    :param gaba_a: GABA_A (shunt) synapse paramters (Isyn, Itau, Ith), defaults to None
    :type gaba_a: Optional[SynapseParameters], optional
    :param nmda: NMDA synapse paramters (Isyn, Itau, Ith), defaults to None
    :type nmda: Optional[SynapseParameters], optional
    :param ampa: AMPA synapse paramters (Isyn, Itau, Ith), defaults to None
    :type ampa: Optional[SynapseParameters], optional
    :param ahp: Spike frequency adaptation block parameters (Isyn, Itau, Ith, Iw), defaults to None
    :type ahp: Optional[SynapseParameters], optional
    :param weights: Configurable connection base weight currents, defaults to None
    :type weights: Optional[WeightParameters], optional

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
    layout: Optional[DynapSELayout] = None
    capacitance: Optional[DynapSECapacitance] = None
    mem: Optional[MembraneParameters] = None
    gaba_b: Optional[SynapseParameters] = None
    gaba_a: Optional[SynapseParameters] = None
    nmda: Optional[SynapseParameters] = None
    ampa: Optional[SynapseParameters] = None
    ahp: Optional[SynapseParameters] = None
    weights: Optional[WeightParameters] = None

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
            self.layout = DynapSELayout()

        if self.capacitance is None:
            self.capacitance = DynapSECapacitance()

        if self.weights is None:
            self.weights = WeightParameters(layout=self.layout)

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
    def from_samna_parameters(
        cls,
        samna_parameters: Dict[str, Dynapse1Parameter],
        size: int,
        core_key: Tuple[np.uint8] = (0, 0),
        neuron_idx_map: Optional[Dict[np.uint8, np.uint16]] = None,
        fpulse_ahp: float = 0.1,
        Ispkthr: float = 1e-6,
        Ireset: Optional[float] = None,
        layout: Optional[DynapSELayout] = None,
        capacitance: Optional[DynapSECapacitance] = None,
    ) -> DynapSE1SimCore:
        """
        from_samna_parameters create a simulation configuration object a the samna config object.
        21/25 parameters used in the configuration of the simulator object.
        "IF_BUF_P", "IF_CASC_N", "R2R_P", and "IF_TAU2_N" has no effect on the simulator.
        The parameters which cannot be obtained from the parameter_group object should be defined explicitly

        :param samna_parameters: a parameter dictionary inside samna config object for setting the parameter group within one core
        :type samna_parameters: Dict[str, Dynapse1Parameter]
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
            layout = DynapSELayout()

        if capacitance is None:
            capacitance = DynapSECapacitance()

        mem = MembraneParameters.from_samna_parameters(
            samna_parameters,
            layout,
            C=capacitance.mem,
            Cref=capacitance.ref,
            Cpulse=capacitance.pulse,
            Ispkthr=Ispkthr,
            Ireset=Ireset,
        )

        # Slow inhibitory
        gaba_b = GABABParameters.from_samna_parameters(
            samna_parameters,
            layout,
            C=capacitance.gaba_b,
        )

        # Fast inhibitory (shunt)
        gaba_a = GABAAParameters.from_samna_parameters(
            samna_parameters,
            layout,
            C=capacitance.gaba_a,
        )

        # Slow Excitatory
        nmda = NMDAParameters.from_samna_parameters(
            samna_parameters,
            layout,
            C=capacitance.nmda,
        )

        # Fast Excitatory
        ampa = AMPAParameters.from_samna_parameters(
            samna_parameters,
            layout,
            C=capacitance.ampa,
        )

        ahp = AHPParameters.from_samna_parameters(
            samna_parameters,
            layout,
            C=capacitance.ahp,
        )

        weights = WeightParameters.from_samna_parameters(samna_parameters, layout)

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
            weights=weights,
        )
        return mod

    ## Property Utils ##

    def neuron_property(self, subcircuit: str, attr: str) -> jnp.DeviceArray:
        """
        neuron_property fetches an attribute from a neuron subcircuit and create a property array covering all the neurons allocated
            i.e. Itau_mem :jnp.DeviceArray = self.neuron_property("mem", "Itau")

        :param subcircuit: The subcircuit harboring the desired property like "mem", "ahp" or "layout"
        :type subcircuit: str
        :param attr: An attribute stored inside the given subcircuit like "Itau", "Ith" or etc.
        :type attr: str
        :return: 1D an array full of the value of the target attribute (`size`,)
        :rtype: jnp.DeviceArray
        """

        obj = self.__getattribute__(subcircuit)
        data = obj.__getattribute__(attr)
        array = jnp.full(self.size, data, dtype=jnp.float32)
        return array

    def syn_property(self, attr: str) -> jnp.DeviceArray:
        """
        syn_property fetches a attributes from the synaptic gate subcircuits in [GABA_B, GABA_A, NMDA, AMPA] order and create a property array covering all the neurons allocated

        :param attr: the target attribute
        :type attr: str
        :return: 2D an array full of the values of the target attributes of all 4 synapses (`size`, 4)
        :rtype: jnp.DeviceArray
        """

        def get_syn_vector() -> np.ndarray:
            """
            _get_syn_vector lists the synaptic parameters traversing the different object instances of the same class

            :return: An array of target parameter values obtained from the objects
            :rtype: np.ndarray
            """

            object_list = [self.gaba_b, self.gaba_a, self.nmda, self.ampa]

            param_list = [
                obj.__getattribute__(attr) if obj is not None else None
                for obj in object_list
            ]

            return np.array(param_list)

        data = get_syn_vector()
        array = jnp.full((self.size, 4), data, dtype=jnp.float32)
        return array

    ## -- Property Utils End -- ##

    @property
    def Imem(self) -> jnp.DeviceArray:
        """
        Imem is an array of membrane currents of the neurons with shape = (Nrec,)
        """
        return self.neuron_property("mem", "Imem")

    @property
    def Itau_mem(self) -> jnp.DeviceArray:
        """
        Itau_mem is an array of membrane leakage currents of the neurons with shape = (Nrec,)
        """
        return self.neuron_property("mem", "Itau")

    @property
    def Itau2_mem(self) -> jnp.DeviceArray:
        """
        Itau2_mem is an array of secondary membrane leakage currents of the neurons with shape = (Nrec,)
        """
        return self.neuron_property("mem", "Itau2")

    @property
    def f_gain_mem(self) -> jnp.DeviceArray:
        """
        f_gain_mem is an array of membrane gain parameter of the neurons with shape = (Nrec,)
        """
        return self.neuron_property("mem", "f_gain")

    @property
    def Isyn(self) -> jnp.DeviceArray:
        """
        Isyn is a 2D array of synapse currents of the neurons in the order of [GABA_B, GABA_A, NMDA, AMPA] with shape = (4,Nrec)
        """
        return self.syn_property("Isyn")

    @property
    def Itau_syn(self) -> jnp.DeviceArray:
        """
        Itau_syn is a 2D array of synapse leakage currents of the neurons in the order of [GABA_B, GABA_A, NMDA, AMPA] with shape = (4,Nrec)
        """
        return self.syn_property("Itau")

    @property
    def f_gain_syn(self) -> jnp.DeviceArray:
        """
        f_gain_syn is a 2D array of synapse gain parameters of the neurons in the order of [GABA_B, GABA_A, NMDA, AMPA] with shape = (4,Nrec)
        """
        return self.syn_property("f_gain")

    @property
    def Iahp(self) -> jnp.DeviceArray:
        """
        Iahp is a 1D array of AHP synapse currents of the neurons with shape = (Nrec,)
        """
        return self.neuron_property("ahp", "Isyn")

    @property
    def Itau_ahp(self) -> jnp.DeviceArray:
        """
        Itau_syn is a 1D array of AHP synapse leakage currents of the neurons with shape = (Nrec,)
        """
        return self.neuron_property("ahp", "Itau")

    @property
    def f_gain_ahp(self) -> jnp.DeviceArray:
        """
        f_gain_syn is a 1D array of AHP synapse gain parameters of the neurons with shape = (Nrec,)
        """
        return self.neuron_property("ahp", "f_gain")

    @property
    def Iw_ahp(self) -> jnp.DeviceArray:
        """
        Iw_ahp is 1D array of spike frequency adaptation currents of the neurons in Amperes with shape (Nrec,)
        """
        return self.neuron_property("ahp", "Iw")

    @property
    def Iw_0(self) -> jnp.DeviceArray:
        """
        Iw_0 is a 1D array of Iw bit 0 parameters of the neurons with shape = (Nrec,)
        """
        return self.neuron_property("weights", "Iw_0")

    @property
    def Iw_1(self) -> jnp.DeviceArray:
        """
        Iw_1 is a 1D array of Iw bit 1 parameters of the neurons with shape = (Nrec,)
        """
        return self.neuron_property("weights", "Iw_1")

    @property
    def Iw_2(self) -> jnp.DeviceArray:
        """
        Iw_2 is a 1D array of Iw bit 2 parameters of the neurons with shape = (Nrec,)
        """
        return self.neuron_property("weights", "Iw_2")

    @property
    def Iw_3(self) -> jnp.DeviceArray:
        """
        Iw_3 is a 1D array of Iw bit 0 parameters of the neurons with shape = (Nrec,)
        """
        return self.neuron_property("weights", "Iw_3")

    # [] TODO : Remove this
    @property
    def Iw_base(self) -> jnp.DeviceArray:
        """
        Iw_base is 1D array of connection base weight currents of the neurons in the order of [GABA_B, GABA_A, NMDA, AMPA] with shape = (4,)
        """
        return self.weights.get_vector()

    @property
    def kappa(self) -> jnp.DeviceArray:
        """
        kappa is the mean subthreshold slope factor of the transistors with shape (Nrec,)
        """
        return self.neuron_property("layout", "kappa")

    @property
    def Ut(self) -> jnp.DeviceArray:
        """
        Ut is the thermal voltage in Volts with shape (Nrec,)
        """
        return self.neuron_property("layout", "Ut")

    @property
    def Io(self) -> jnp.DeviceArray:
        """
        Io is the dark current in Amperes that flows through the transistors even at the idle state with shape (Nrec,)
        """
        return self.neuron_property("layout", "Io")

    @property
    def f_tau_mem(self) -> jnp.DeviceArray:
        """
        f_tau_mem is an array of tau factor for membrane circuit. :math:`f_{\\tau} = \\dfrac{U_T}{\\kappa \\cdot C}`, :math:`f_{\\tau} = I_{\\tau} \\cdot \\tau` with shape (Nrec,)
        """
        return self.neuron_property("mem", "f_tau")

    @property
    def f_tau_syn(self) -> jnp.DeviceArray:
        """
        f_tau_syn is a 2D array of tau factors in the following order: [GABA_B, GABA_A, NMDA, AMPA] with shape (4, Nrec)
        """
        return self.syn_property("f_tau")

    @property
    def f_tau_ahp(self) -> jnp.DeviceArray:
        """
        f_tau_ahp is is an array of tau factors for spike frequency adaptation circuit with shape (Nrec,)
        """
        return self.neuron_property("ahp", "f_tau")

    @property
    def f_ref(self) -> jnp.DeviceArray:
        """
        f_ref is an array of the factor of conversion from the refractory period current to the refractory period shape (Nrec,)
        """
        return self.neuron_property("mem", "f_ref")

    @property
    def f_pulse(self) -> jnp.DeviceArray:
        """
        f_pulse is an array of the factor of conversion from pulse width in seconds to pulse width bias current in Amperes with shape (Nrec,)
        """
        return self.neuron_property("mem", "f_pulse")

    @property
    def f_pulse_ahp(self) -> jnp.DeviceArray:
        """
        f_pulse_ahp is an array of ahp pulse width ratios of the neurons with shape = (Nrec,)
        """
        return jnp.full(self.size, self.fpulse_ahp, dtype=jnp.float32)

    @property
    def Iref(self) -> jnp.DeviceArray:
        """
        Iref is an array of the refractory period current in Amperes with shape  shape (Nrec,)
        """
        return self.neuron_property("mem", "Iref")

    @property
    def Ipulse(self) -> jnp.DeviceArray:
        """
        Ipulse is an array of the pulse width current in Amperes with shape  shape (Nrec,)
        """
        return self.neuron_property("mem", "Ipulse")

    @property
    def Ispkthr(self) -> jnp.DeviceArray:
        """
        Ispkthr is an array of the spiking threshold current in Amperes with shape  shape (Nrec,)
        """
        return self.neuron_property("mem", "Ispkthr")

    @property
    def Ireset(self) -> jnp.DeviceArray:
        """
        Ireset is an array of the reset current after spike generation with shape (Nrec,)
        """
        return self.neuron_property("mem", "Ireset")

    @property
    def Idc(self) -> jnp.DeviceArray:
        """
        Idc is an array of the constant DC current in Amperes, injected to membrane with shape (Nrec,)
        """
        return self.neuron_property("mem", "Idc")

    @property
    def If_nmda(self) -> jnp.DeviceArray:
        """
        If_nmda is an array of the The NMDA gate current in Amperes setting the NMDA gating voltage. If :math:`V_{mem} > V_{nmda}` : The :math:`I_{syn_{NMDA}}` current is added up to the input current, else it cannot with shape (Nrec,)
        """
        return self.neuron_property("mem", "If_nmda")


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
    bit_mask: np.ndarray = np.array([0b0001, 0b0010, 0b0100, 0b1000])

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

        self._attr_list = [
            "Imem",
            "Itau_mem",
            "Itau2_mem",
            "f_gain_mem",
            "Isyn",
            "Itau_syn",
            "f_gain_syn",
            "Iahp",
            "Itau_ahp",
            "f_gain_ahp",
            "Iw_ahp",
            "Iw_0",
            "Iw_1",
            "Iw_2",
            "Iw_3",
            "kappa",
            "Ut",
            "Io",
            "f_tau_mem",
            "f_tau_syn",
            "f_tau_ahp",
            "f_ref",
            "f_pulse",
            "f_pulse_ahp",
            "Iref",
            "Ipulse",
            "Ispkthr",
            "Ireset",
            "Idc",
            "If_nmda",
        ]

        for attr in self._attr_list:
            self.__setattr__(attr, self.merge_core_properties(attr))

    def __len__(self) -> int:
        """
        __len__ the size of the simulator board

        :return: number of active neruons allocated
        :rtype: int
        """
        size = 0
        for core in self.cores:
            size += len(core)
        return size

    def merge_core_properties(self, attr: str) -> jnp.DeviceArray:
        """
        merge_core_properties help merging property arrays of the cores given into one array.
        In this way, the simulator runs in the neruon level and it only knows the location of the
        neruon by looking at the index map. In this way, the simulator module can compute the evolution of
        the currents and spikes efficiently.

        :param attr: the target attribute to be merged
        :type attr: str
        :return: The parameter or state to be used in the simulation
        :rtype: jnp.DeviceArray
        """
        prop = jnp.concatenate([core.__getattribute__(attr) for core in self.cores])
        return prop

    # [] TODO :  Remove
    def weight_matrix(self, CAM: np.ndarray) -> jnp.DeviceArray:
        """
        weight_matrix uses the `DynapSE` weight matrix method to obtain a weight matrix from a CAM matrix.
        There can be multiple CAMs defined per connection, therefore, the funciton iterates over given CAM matrix and accumulates the weight
        currents indicated by the bit-masks.

        :param CAM: a CAM matrix, the number of CAMs defined per connection per neuron (3D, NinxNoutx4)
        :type CAM: np.ndarray
        :return: a Dynap-SE type weight matrix generated by collecting the base weights via bit mask dot product (N_pre,N_post,4) pre, post, syn_type
        :rtype: jnp.DeviceArray
        """
        shape = CAM.shape
        CAM = CAM.astype(int)
        cores = Router.core_matrix_from_idx_map(self.idx_map)
        w_rec = jnp.zeros(shape, dtype=jnp.float32)

        # There can be multiple CAMs defined
        while np.sum(CAM) > 0:
            # [] TODO : For now, w_recs are added up linearly. It has to be non-linear!
            bit_masks = CAM.astype(bool) * self.bit_mask
            w_rec += DynapSE.weight_matrix(self.Iw_base, cores, bit_masks)
            CAM = np.clip(CAM - 1, 0, None)

        return w_rec

    # TODO : Remove
    @property
    def Iw_base(self) -> Dict[Tuple[np.uint8], float]:
        """
        Iw_base creates a dictionary of base weight vectors of the different cores inside the board

        :return: base weight vectors of the simulation cores
        :rtype: Dict[Tuple[np.uint8], float]
        """
        Iw_base = {core.core_key: core.Iw_base for core in self.cores}
        return Iw_base

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

        if np.sum(nidx) * 2 != nidx[-1] * (nidx[-1] + 1):
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
        chip_core = np.unique(
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
            chip_core_id = np.array(np.tile(core_key, (len(neuron_idx), 1)))
            device_nid = np.array(list(neuron_idx.values())).reshape(-1, 1)
            values = np.hstack((chip_core_id, device_nid))
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
            _, idx_map = Router.CAM_rec_from_config(config, return_maps=True)
        core_dict = DynapSE1SimBoard.idx_map_to_core_dict(idx_map)

        # Gather `DynapSE1SimCore` objects
        for (chipID, coreID), neuron_map in core_dict.items():
            # Traverse the chip for core parameter group
            params = config.chips[chipID].cores[coreID].parameter_group.param_map
            sim_core = DynapSE1SimCore.from_samna_parameters(
                params, len(neuron_map), (chipID, coreID), neuron_map
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
