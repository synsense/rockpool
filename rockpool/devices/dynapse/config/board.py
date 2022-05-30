"""
Dynap-SE1/SE2 full board configuration classes and methods

renamed : dynapse1_simconfig.py -> simconfig.py @ 211208
renamed : simconfig.py -> board.py @ 220509

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
09/05/2022
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from dataclasses import dataclass, field

import numpy as np

from rockpool.devices.dynapse.base import CoreKey, NeuronKey
from rockpool.devices.dynapse.config.simcore import (
    DynapSimCore,
    DynapSimCurrents,
    DynapSimGain,
    DynapSimLayout,
    DynapSimTime,
    DynapSimWeightBits,
)

from rockpool.devices.dynapse.base import DynapSE
from rockpool.devices.dynapse.config.weights import WeightParameters
from rockpool.devices.dynapse.infrastructure.router import Router, Connector
from rockpool.nn.modules.jax.jax_module import JaxModule


Dynapse1Configuration = Any
Dynapse1Core = Any
Dynapse2Configuration = Any
Dynapse2Core = Any


@dataclass
class DynapSimConfig(DynapSimCore):
    """
    DynapSimConfig stores the simulation currents, layout parameters and weight matrices necessary to
    configure a DynapSE1/SE2 simulator

    :param shape: the network shape (n_input, n_hidden, n_output), defaults to None
    :type shape: Optional[Union[Tuple[int], int]], optional
    :param w_in: input weight matrix, defaults to None
    :type w_in: np.ndarray, optional
    :param w_rec: recurrent weight matrix, defaults to None
    :type w_rec: np.ndarray, optional
    :param w_out: output weight matrix, defaults to None
    :type w_out: np.ndarray, optional
    :param cores: dictionary of simulation cores, defaults to None
    :type cores: Optional[Dict[str, DynapSimCore]], optional
    :param router: the router object reading the memory content to create the weight masks, defaults to None
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
        Idc: float = None,
        If_nmda: float = None,
        r_gain_ahp: float = 4,  # 100
        r_gain_ampa: float = 4,  # 100
        r_gain_gaba: float = 4,  # 100
        r_gain_nmda: float = 4,  # 100
        r_gain_shunt: float = 4,  # 100
        r_gain_mem: float = 2,  # 4
        t_pulse_ahp: float = 1e-6,
        t_pulse: float = 10e-6,
        t_ref: float = 2e-3,
        Ispkthr: float = 1e-6,
        tau_ahp: float = 50e-3,
        tau_ampa: float = 10e-3,
        tau_gaba: float = 100e-3,
        tau_nmda: float = 100e-3,
        tau_shunt: float = 10e-3,
        tau_mem: float = 20e-3,
        Iw_0: float = 1e-7,
        Iw_1: float = 2e-7,
        Iw_2: float = 4e-7,
        Iw_3: float = 8e-7,
        Iw_ahp: float = 1e-7,
        C_ahp: float = 40e-12,
        C_ampa: float = 24.5e-12,
        C_gaba: float = 25e-12,
        C_nmda: float = 25e-12,
        C_pulse_ahp: float = 0.5e-12,
        C_pulse: float = 0.5e-12,
        C_ref: float = 1.5e-12,
        C_shunt: float = 24.5e-12,
        C_mem: float = 3e-12,
        Io: float = 5e-13,
        kappa_n: float = 0.75,
        kappa_p: float = 0.66,
        Ut: float = 25e-3,
        Vth: float = 7e-1,
    ) -> DynapSimConfig:
        """
        from_specification is a class factory method using the weight specifications and the current/layout parameters
        One can directly define w_in/w_rec/w_out or one can define the weight masks.
        If weight masks are defined, weight matrices are calculated using the simulation currents and weight masks together
        All the simulation currents and layout parameter can be passed by kwargs. For more info, please check `DynapSimCore.from_specification()`

        :param shape: the network shape (n_input, n_hidden, n_output). If shape is int, then it means that the input and output should not be considered.
        :type shape: Optional[Union[Tuple[int], int]]
        :param w_in: input weight matrix, defaults to None
        :type w_in: Optional[np.ndarray], optional
        :param w_rec: recurrent weight matrix, defaults to None
        :type w_rec: Optional[np.ndarray], optional
        :param w_in_mask: the weight mask to set the input weight matrix. Used if `w_in` is None, defaults to None
        :type w_in_mask: Optional[np.ndarray], optional
        :param w_rec_mask: the weight mask to set the recurrent weight matrix. Used if `w_rec` is None, defaults to None
        :type w_rec_mask: Optional[np.ndarray], optional
        :param w_out: the output weight mask (binary in general), defaults to None
        :type w_out: Optional[np.ndarray], optional
        :param Idc: Constant DC current injected to membrane in Amperes, defaults to None
        :type Idc: float, optional
        :param If_nmda: NMDA gate soft cut-off current setting the NMDA gating voltage in Amperes, defaults to None
        :type If_nmda: float, optional
        :param r_gain_ahp: spike frequency adaptation block gain ratio :math:`Igain_ahp/Itau_ahp`, defaults to 100
        :type r_gain_ahp: float, optional
        :param r_gain_ampa: excitatory AMPA synpse gain ratio :math:`Igain_ampa/Itau_ampa`, defaults to 100
        :type r_gain_ampa: float, optional
        :param r_gain_gaba: inhibitory GABA synpse gain ratio :math:`Igain_gaba/Itau_gaba `, defaults to 100
        :type r_gain_gaba: float, optional
        :param r_gain_nmda: excitatory NMDA synpse gain ratio :math:`Igain_nmda/Itau_nmda`, defaults to 100
        :type r_gain_nmda: float, optional
        :param r_gain_shunt: inhibitory SHUNT synpse gain ratio :math:`Igain_shunt/Itau_shunt`, defaults to 100
        :type r_gain_shunt: float, optional
        :param r_gain_mem: neuron membrane gain ratio :math:`Igain_mem/Itau_mem`, defaults to 2
        :type r_gain_mem: float, optional
        :param t_pulse_ahp: the spike pulse width for spike frequency adaptation circuit in seconds, defaults to 1e-6
        :type t_pulse_ahp: float, optional
        :param t_pulse: the spike pulse width for neuron membrane in seconds, defaults to 10e-6
        :type t_pulse: float, optional
        :param t_ref: refractory period of the neurons in seconds, defaults to 2e-2
        :type t_ref: float, optional
        :param Ispkthr: spiking threshold current, neuron spikes if :math:`Imem > Ispkthr` in Amperes, defaults to 1e-6
        :type Ispkthr: float, optional
        :param tau_ahp: Spike frequency leakage time constant in seconds, defaults to 50e-3
        :type tau_ahp: float, optional
        :param tau_ampa: AMPA synapse leakage time constant in seconds, defaults to 10e-3
        :type tau_ampa: float, optional
        :param tau_gaba: GABA synapse leakage time constant in seconds, defaults to 100e-3
        :type tau_gaba: float, optional
        :param tau_nmda: NMDA synapse leakage time constant in seconds, defaults to 100e-3
        :type tau_nmda: float, optional
        :param tau_shunt:SHUNT synapse leakage time constant in seconds, defaults to 10e-3
        :type tau_shunt: float, optional
        :param tau_mem: Neuron membrane leakage time constant in seconds, defaults to 20e-3
        :type tau_mem: float, optional
        :param Iw_0: weight bit 0 current of the neurons of the core in Amperes, defaults to 1e-6
        :type Iw_0: float
        :param Iw_1: weight bit 1 current of the neurons of the core in Amperes, defaults to 2e-6
        :type Iw_1: float
        :param Iw_2: weight bit 2 current of the neurons of the core in Amperes, defaults to 4e-6
        :type Iw_2: float
        :param Iw_3: weight bit 3 current of the neurons of the core in Amperes, defaults to 8e-6
        :type Iw_3: float
        :param Iw_ahp: spike frequency adaptation weight current of the neurons of the core in Amperes, defaults to 1e-6
        :type Iw_ahp: float
        :param C_ahp: AHP synapse capacitance in Farads, defaults to 40e-12
        :type C_ahp: float, optional
        :param C_ampa: AMPA synapse capacitance in Farads, defaults to 24.5e-12
        :type C_ampa: float, optional
        :param C_gaba: GABA synapse capacitance in Farads, defaults to 25e-12
        :type C_gaba: float, optional
        :param C_nmda: NMDA synapse capacitance in Farads, defaults to 25e-12
        :type C_nmda: float, optional
        :param C_pulse_ahp: spike frequency adaptation circuit pulse-width creation sub-circuit capacitance in Farads, defaults to 0.5e-12
        :type C_pulse_ahp: float, optional
        :param C_pulse: pulse-width creation sub-circuit capacitance in Farads, defaults to 0.5e-12
        :type C_pulse: float, optional
        :param C_ref: refractory period sub-circuit capacitance in Farads, defaults to 1.5e-12
        :type C_ref: float, optional
        :param C_shunt: SHUNT synapse capacitance in Farads, defaults to 24.5e-12
        :type C_shunt: float, optional
        :param C_mem: neuron membrane capacitance in Farads, defaults to 3e-12
        :type C_mem: float, optional
        :param Io: Dark current in Amperes that flows through the transistors even at the idle state, defaults to 5e-13
        :type Io: Union[float, np.ndarray], optional
        :param kappa_n: Subthreshold slope factor (n-type transistor), defaults to 0.75
        :type kappa_n: Union[float, np.ndarray], optional
        :param kappa_p: Subthreshold slope factor (p-type transistor), defaults to 0.66
        :type kappa_p: Union[float, np.ndarray], optional
        :param Ut: Thermal voltage in Volts, defaults to 25e-3
        :type Ut: Union[float, np.ndarray], optional
        :param Vth: The cut-off Vgs potential of the transistors in Volts (not type specific), defaults to 7e-1
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
            n_chips=(len(core_map) // DynapSE.NUM_CHIPS) + 1,
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

        h = lambda idx: idx // (DynapSE.NUM_CORES * DynapSE.NUM_NEURONS)
        c = lambda idx: idx // DynapSE.NUM_NEURONS
        n = lambda idx: idx - c(idx) * DynapSE.NUM_NEURONS
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


if __name__ == "__main__":
    # print(DynapSimLayout().get_full(10))
    simconfig = DynapSimConfig.from_specification(4, Iw_0=[1e-3, 2e-3, 4e-3, 5e-3])
    print(simconfig)
