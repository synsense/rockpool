"""
Dynap-SE1 simulator with bias parameter export facilities and samna configuration factory methods

renamed: dynapse1_simulator.py -> dynase1_jax.py @ 211208
renamed: dynapse1_jax.py -> se1_jax.py @ 220114

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
20/09/2021

TODOS:
[] TODO : Update Iw bias getters
"""
from __future__ import annotations
import json
import logging

from typing import Any, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np

from rockpool.nn.modules import TimedModuleWrapper
from rockpool.nn.combinators import Sequential

from rockpool.devices.dynapse.dynapsim import DynapSim
from rockpool.devices.dynapse.fpga_jax import DynapSEFPGA
from rockpool.devices.dynapse.config.simconfig import DynapSE1SimBoard
from rockpool.devices.dynapse.infrastructure.router import Router
from rockpool.devices.dynapse.infrastructure.biasgen import BiasGenSE1
from rockpool.devices.dynapse.base import NeuronKey
from rockpool.devices.dynapse.lookup import param_name

param_name_se1 = param_name.se1
param_name_table = param_name.table

from rockpool.devices.dynapse.samna_alias.dynapse1 import (
    Dynapse1ParameterGroup,
    Dynapse1Parameter,
    Dynapse1Configuration,
)


class DynapSE1Jax(DynapSim):
    """
    DynapSE1Jax is an extension to DynapSim module with device specific deployment utilities.
    The parameters and pipeline are explained in the superclass `DynapSim` doctring

    :Parameters:
    :param idx_map: a dictionary of the mapping between matrix indexes of the neurons and their global unique neuron keys. if index map is in proper format, a core dictionary can be inferred without an error
    :type idx_map: Dict[int, NeuronKey]

    :Instance Variables:
    :ivar core_dict: a dictionary from core keys (chipID, coreID) to an index map of neruons (neuron index : local neuronID) that the core allocates.
    :type core_dict: Dict[Tuple[int], Dict[int, int]]

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
    """

    __doc__ += DynapSim.__doc__

    biases = list(param_name_se1.keys())

    def __init__(
        self,
        shape: Optional[Tuple] = None,
        sim_config: Optional[DynapSE1SimBoard] = None,
        has_rec: bool = True,
        w_rec: Optional[jnp.DeviceArray] = None,
        idx_map: Optional[Dict[int, NeuronKey]] = None,
        dt: float = 1e-3,
        rng_key: Optional[Any] = None,
        spiking_input: bool = True,
        spiking_output: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """
        __init__ Initialize ``DynapSE1Jax`` module. Parameters are explained in the superclass docstring.
        """

        if sim_config is None:
            # Get a simulation board from an index map with default bias parameters
            if idx_map is not None:
                sim_config = DynapSE1SimBoard.from_idx_map(idx_map)
            else:
                sim_config = DynapSE1SimBoard(shape[-1])

        if idx_map is None:
            idx_map = sim_config.idx_map

        super(DynapSE1Jax, self).__init__(
            shape,
            sim_config,
            has_rec,
            w_rec,
            dt,
            rng_key,
            spiking_input,
            spiking_output,
            *args,
            **kwargs,
        )
        # Check if index map is in proper format, if so, a core dictionary can be inferred without an error.
        DynapSE1SimBoard.check_neuron_id_order(list(idx_map.keys()))
        self.core_dict = DynapSE1SimBoard.idx_map_to_core_dict(idx_map)
        self.idx_map = idx_map
        self.paramgen = BiasGenSE1()

    def samna_param_group(
        self, chipID: np.uint8, coreID: np.uint8
    ) -> Dynapse1ParameterGroup:
        """
        samna_param_group creates a samna Dynapse1ParameterGroup group object and configure the bias
        parameters by creating a dictionary. It does not check if the values are legal or not.

        :param chipID: Unique chip ID to of the parameter group
        :type chipID: np.uint8
        :param coreID: Non-unique core ID to of the parameter group
        :type coreID: np.uint8
        :return: samna config object to set a parameter group within one core
        :rtype: Dynapse1ParameterGroup
        """
        pg_json = json.dumps(self._param_group(chipID, coreID))
        pg_samna = Dynapse1ParameterGroup()
        pg_samna.from_json(pg_json)
        return pg_samna

    def _param_group(
        self, chipID: np.uint8, coreID: np.uint8
    ) -> Dict[
        str,
        Dict[str, Union[int, List[Dict[str, Union[str, Dict[str, Union[str, int]]]]]]],
    ]:
        """
        _param_group creates a samna type dictionary to configure the parameter group

        :param chipID: Unique chip ID
        :type chipID: np.uint8
        :param coreID: Non-unique core ID
        :type coreID: np.uint8
        :return: a dictonary of bias currents, and their coarse/fine values
        :rtype: Dict[str, Dict[str, Union[int, List[Dict[str, Union[str, Dict[str, Union[str, int]]]]]]]]
        """

        bias_getter = lambda bias: self.config_param(bias, chipID, coreID)

        def param_dict(param: Dynapse1Parameter) -> Dict[str, Union[str, int]]:
            serial = {
                "paramName": param.param_name,
                "coarseValue": param.coarse_value,
                "fineValue": param.fine_value,
                "type": param.type,
            }
            return serial

        param_map = [
            {"key": bias, "value": param_dict(bias_getter(bias))}
            for bias in self.biases
        ]

        group = {
            "value0": {
                "paramMap": param_map,
                "chipId": chipID,
                "coreId": coreID,
            }
        }

        return group

    @classmethod
    def from_config(
        cls,
        config: Dynapse1Configuration,
        sim_config: Optional[DynapSE1SimBoard] = None,
        default_bias: bool = True,
        *args,
        **kwargs,
    ) -> DynapSE1Jax:
        """
        from_config is a class factory method depending on a samna device configuration object. Using this,
        both the bias currents and the the neruon-neuron connections can be extracted easily.

        e.g. modSE1 = DynapSE1Jax.from_config(config)

        :param config: samna Dynapse1 configuration object used to configure a network on the chip
        :type config: Dynapse1Configuration
        :param sim_config: Dynap-SE1 bias currents and simulation configuration parameters, it can be provided explicitly, or created using default settings, or can be extracted from the config bias currents. defaults to None
        :type sim_config: Optional[DynapSE1SimBoard], optional
        :param default_bias: use default bias values or get the bias parameters from the config object, defaults to True
        :type default_bias: bool
        :return: `DynapSE1Jax` simulator object
        :rtype: DynapSE1Jax
        """
        CAM_rec, idx_map = Router.CAM_rec_from_config(config, return_maps=True)

        # CAM_shape: size_out, size_in // 4, 4
        CAM_shape = CAM_rec.shape  # N_pre, N_post, 4(syn_type)
        mod_shape = (CAM_shape[2] * CAM_shape[1], CAM_shape[0])
        has_rec = True

        if sim_config is None:
            if not default_bias:
                sim_config = DynapSE1SimBoard.from_config(config, idx_map)

            else:
                sim_config = DynapSE1SimBoard.from_idx_map(idx_map)

        w_rec = sim_config.weight_matrix(CAM_rec)
        mod = cls(mod_shape, sim_config, has_rec, w_rec, idx_map, *args, **kwargs)

        return mod

    @staticmethod
    def simulator_from_config(
        config: Dynapse1Configuration,
        sim_config: Optional[DynapSE1SimBoard] = None,
        default_bias: bool = True,
    ) -> TimedModuleWrapper:
        """
        simulator_from_config obtain a full DynapSE simulator from a samna config object.
        Returns a sequentially combined input layer and device layer.

        :param config: samna Dynapse1 configuration object used to configure a network on the chip
        :type config: Dynapse1Configuration
        :param sim_config: Dynap-SE1 bias currents and simulation configuration parameters, it can be provided explicitly, or created using default settings, or can be extracted from the config bias currents. defaults to None
        :type sim_config: Optional[DynapSE1SimBoard], optional
        :param default_bias: use default bias values or get the bias parameters from the samna config, defaults to True
        :type default_bias: bool
        :return: a `TimedModuleWrapper` sequentially combining the special FPGA input layer of the DyanpSE simulators.
        :rtype: TimedModuleWrapper

        [] TODO: Move this out and create a simulator class
        """
        CAM_in, idx_map_in = Router.CAM_in_from_config(config, return_maps=True)
        CAM_rec, idx_map_rec = Router.CAM_rec_from_config(config, return_maps=True)

        # CAM_shape: size_out, size_in // 4, 4
        in_shape = CAM_in.shape  # size_in, size_out // 4, 4
        rec_shape = CAM_rec.shape  # size_out, size_in // 4, 4

        # 2D module shapes
        fpga_shape = (in_shape[0], in_shape[1] * in_shape[2])
        se1_shape = (rec_shape[2] * rec_shape[1], rec_shape[0])
        has_rec = True

        if sim_config is None:
            if not default_bias:
                sim_config = DynapSE1SimBoard.from_config(config, idx_map_rec)

            else:
                sim_config = DynapSE1SimBoard.from_idx_map(idx_map_rec)

        w_in = sim_config.weight_matrix(CAM_in)
        w_rec = sim_config.weight_matrix(CAM_rec)

        # Obtain a TimedModuleWrapper and sequentially combine input layer with simulation layer
        fpga = DynapSEFPGA(fpga_shape, sim_config, w_in, idx_map_in)
        se1 = DynapSE1Jax(se1_shape, sim_config, has_rec, w_rec, idx_map_rec)
        simulator = TimedModuleWrapper(
            Sequential(fpga, se1),
            dt=se1.dt,
        )

        return simulator

    def get_bias(
        self,
        chipID: np.uint8,
        coreID: np.uint8,
        attribute: str,
        syn_type: Optional[str] = None,
    ) -> float:
        """
        get_bias obtains a bias current from the simulation currents to convert this
        into device bias with coarse and fine values

        :param chipID: Unique chip ID
        :type chipID: np.uint8
        :param coreID: Non-unique core ID
        :type coreID: np.uint8
        :param attribute: the name of the simulation attribute which corresponds to the desired bias
        :type attribute: str
        :param syn_type: the synase type of the attribute if the corresponding current is a synaptic current rather than membrane, defaults to None
        :type syn_type: Optional[str], optional
        :raises IndexError: Chip:{chipID},Core:{coreID} is not used! Please look at the core dictionary!
        :raises AttributeError: Bias: {attribute} is a synaptic current. Please define the syn_type!"
        :return: the bias current of a subset of neruons obtained from the core dictionary
        :rtype: float
        """
        I_bias = None
        if (chipID, coreID) not in self.core_dict:
            raise IndexError(
                f"Chip:{chipID},Core:{coreID} is not used! Please look at the core dictionary! {list(self.core_dict.keys())}"
            )
        else:
            # Obtain the neuron indices from the core dictionary
            idx = np.array(list(self.core_dict[(chipID, coreID)].keys()))
            I_base = self.__getattribute__(attribute)
            if syn_type is None:
                if len(I_base.shape) == 2:
                    raise AttributeError(
                        f"Bias: {attribute} is a synaptic current. Please define the syn_type!"
                    )
                I_bias = float(I_base[idx].mean())
            else:
                if len(I_base.shape) == 1:
                    logging.warning(
                        f"Bias: {attribute} is a a membrane current. Defined syn_type:{syn_type} has no effect!"
                    )
                    I_bias = float(I_base[idx].mean())
                else:
                    I_bias = float(I_base[idx, self.SYN[syn_type]].mean())

        return I_bias

    def bias_parameter(
        self,
        chipID: np.uint8,
        coreID: np.uint8,
        sim_name: str,
        syn_type: Optional[str] = None,
    ) -> Dynapse1Parameter:
        """
        bias_parameter wraps the bias current inference from the simulation currents and DynapSE1Parameter construction

        :param chipID: Unique chip ID
        :type chipID: np.uint8
        :param coreID: Non-unique core ID
        :type coreID: np.uint8
        :param sim_name: the name of the simulation attribute which corresponds to the desired bias
        :type sim_name: str
        :param syn_type: the synase type of the attribute if the corresponding current is a synaptic current rather than membrane, defaults to None
        :type syn_type: Optional[str], optional
        :raises ValueError: There is no device parameter in lookup table which relates to given `sim_name`
        :return: samna DynapSE1Parameter object
        :rtype: Dynapse1Parameter
        """
        I_bias = self.get_bias(chipID, coreID, sim_name, syn_type)

        # Convert the current to a coarse fine tuple
        coarse, fine = self.paramgen.bias_to_coarse_fine(I_bias)
        dev_name = param_name_table[sim_name][1]  # SE1
        if dev_name is None:
            raise ValueError(
                f"There is no device parameter in lookup table which relates to {sim_name}"
            )

        # Samna Object
        param = Dynapse1Parameter(dev_name, coarse, fine)
        return param

    def config_param(
        self, name: str, chipID: np.uint8, coreID: np.uint8
    ) -> Dynapse1Parameter:
        """
        config_param Obtains any samna configuration parameter which could be stored in DynapSE1ParameterGroup object.

        :param name: the name of the bias parameter in device
        :type name: str
        :param chipID: Unique chip ID
        :type chipID: np.uint8
        :param coreID: Non-unique core ID
        :type coreID: np.uint8
        :return: a DynapSE1 parameter object which is ready to configure a core parameter
        :rtype: Dynapse1Parameter
        """

        sim_name = param_name_se1[name]
        if sim_name is None:
            return self.__getattribute__(name)
        return self.bias_parameter(chipID, coreID, sim_name)

    @property
    def IF_BUF_P(self) -> Dynapse1Parameter:
        """
        IF_BUF_P has an effect on the membrane potential `Vm` readout, does not have a huge effect on simulation, use samna defaults.
        """
        return Dynapse1Parameter("IF_BUF_P", 4, 80)

    @property
    def IF_CASC_N(self) -> Dynapse1Parameter:
        """
        IF_CASC_N for spike frequency current (AHP) regularization, does not have a huge effect on simulation, use samna defaults.
        """
        return Dynapse1Parameter("IF_CASC_N", 0, 0)

    @property
    def R2R_P(self) -> Dynapse1Parameter:
        """
        R2R_P maybe something related to digital to analog converter(unsure), does not have a huge effect on simulation, use samna defaults.
        """
        return Dynapse1Parameter("R2R_P", 3, 85)
