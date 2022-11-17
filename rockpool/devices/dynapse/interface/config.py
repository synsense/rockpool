"""
Dynap-SE samna config helper functions
Handles the low-level hardware configuration under the hood and provide easy-to-use access to the user

Note : Existing modules are reconstructed considering consistency with Xylo support.

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

15/09/2022

[] TODO : available tag list
"""
from __future__ import annotations
import logging

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from rockpool.devices.dynapse.samna_alias.dynapse2 import (
    Dynapse2Configuration,
    Dynapse2Synapse,
    Dynapse2Destination,
    Dendrite,
)
from rockpool.devices.dynapse.default import CHIP_MAP, CHIP_POS

from rockpool.typehints import FloatVector, IntVector
from rockpool.devices.dynapse.config.simconfig import DynapSimCore

from dataclasses import dataclass
from rockpool.devices.dynapse.definitions import (
    DRCError,
    NUM_TAGS,
    NUM_SYNAPSES,
    NUM_DEST,
)
from rockpool.devices.dynapse.quantize.weight_handler import WeightHandler

# Try to import samna for device interfacing
SAMNA_AVAILABLE = True
try:
    import samna
except:
    samna = Any
    print(
        "Device interface requires `samna` package which is not installed on the system"
    )
    SAMNA_AVAILABLE = False

# - Configure exports
__all__ = ["config_from_specification", "save_config", "load_config"]


def __get_num_cores(config: Dynapse2Configuration) -> int:
    """
    __get_num_cores process a configuration object and returns the number of cores available

    :param config: samna dynapse2 configuration object
    :type config: Dynapse2Configuration
    :return: number of neural cores available
    :rtype: int
    """
    n_cores = 0
    for chip in config.chips:
        n_cores += len(chip.cores)
    return n_cores


# def config_from_specification(
#     config: Optional[Dynapse2Configuration] = None,
#     weights_in: Optional[np.ndarray] = None, # 4-bit
#     weights_rec: Optional[np.ndarray] = None, # 4-bit
#     weights_out: Optional[np.ndarray] = None, # 1-bit
#     # gain params
#     r_gain_ahp: Union[float, np.ndarray, None] = dgain["r_gain_ahp"],
#     r_gain_ampa: Union[float, np.ndarray, None] = dgain["r_gain_ampa"],
#     r_gain_gaba: Union[float, np.ndarray, None] = dgain["r_gain_gaba"],
#     r_gain_nmda: Union[float, np.ndarray, None] = dgain["r_gain_nmda"],
#     r_gain_shunt: Union[float, np.ndarray, None] = dgain["r_gain_shunt"],
#     r_gain_mem: Union[float, np.ndarray, None] = dgain["r_gain_mem"],
#     ## time params
#     t_pulse_ahp: Union[float, np.ndarray, None] = dtime["t_pulse_ahp"],
#     t_pulse: Union[float, np.ndarray, None] = dtime["t_pulse"],
#     t_ref: Union[float, np.ndarray, None] = dtime["t_ref"],
#     ## tau params
#     tau_ahp: Union[float, np.ndarray, None] = dtime["tau_ahp"],
#     tau_ampa: Union[float, np.ndarray, None] = dtime["tau_ampa"],
#     tau_gaba: Union[float, np.ndarray, None] = dtime["tau_gaba"],
#     tau_nmda: Union[float, np.ndarray, None] = dtime["tau_nmda"],
#     tau_shunt: Union[float, np.ndarray, None] = dtime["tau_shunt"],
#     tau_mem: Union[float, np.ndarray, None] = dtime["tau_mem"],
#     ## weight params
#     Iw_0: Union[float, np.ndarray, None] = dweight["Iw_0"],
#     Iw_1: Union[float, np.ndarray, None] = dweight["Iw_1"],
#     Iw_2: Union[float, np.ndarray, None] = dweight["Iw_2"],
#     Iw_3: Union[float, np.ndarray, None] = dweight["Iw_3"],
#     Iw_ahp: Union[float, np.ndarray, None] = dcurrents["Iw_ahp"],
#     Ispkthr: Union[float, np.ndarray, None] = dcurrents["Ispkthr"],
#     If_nmda: Union[float, np.ndarray, None] = dcurrents["If_nmda"],
#     Idc: Union[float, np.ndarray, None] = dcurrents["Idc"],
#     *args,
#     **kwargs
# ) -> Dynapse2Configuration:

#     if config is None:
#         config = samna.dynapse2.Dynapse2Configuration() if SAMNA_AVAILABLE else None
#         logging.warn(
#             "Fetch the samna object from the actual device and provide ``config = model.get_configuration()``!"
#         )

#     # get number of cores available
#     n_cores = __get_num_cores(config)


#     # Set Parameters

#     # Set Memory

#     return config


@dataclass
class WeightAllocator:
    weights_in: Optional[IntVector]
    weights_rec: Optional[IntVector]
    sign_in: Optional[IntVector]
    sign_rec: Optional[IntVector]
    core_map: Dict[int, int]
    chip_map = CHIP_MAP
    chip_pos = CHIP_POS
    tag_list: Optional[IntVector] = None

    def __post_init__(self) -> None:

        self.__shape_check()
        self.tag_list = (
            np.array(range(NUM_TAGS)) if self.tag_list is None else self.tag_list
        )
        if len(self.core_map) != self.n_neuron:
            raise DRCError("Core map does not match the number of neurons!")

        self.n_chip = len(set(self.chip_map.values()))
        self.w_in_bool = (
            WeightHandler.int2bit_mask(4, self.weights_in).T
            if self.weights_in is not None
            else None
        )  # neuron, connection, bits
        self.w_rec_bool = (
            WeightHandler.int2bit_mask(4, self.weights_rec).T
            if self.weights_rec is not None
            else None
        )  # neuron, connection, bits
        self.virtual_tags, self.recurrent_tags = self.__tag_selector()

    def __shape_check(self) -> None:
        if self.weights_in is None and self.weights_rec is None:
            raise DRCError("No weights given for allocation!")

        if self.weights_in is not None:
            self.weights_in = np.array(self.weights_in)
            self.sign_in = np.array(self.sign_in)
            self.n_in = self.weights_in.shape[0]
            self.n_neuron = self.weights_in.shape[1]
            if self.weights_in.shape != self.sign_in.shape:
                raise DRCError("Input sign shape does not match input weight shape!")

            if self.weights_rec is not None:
                self.weights_rec = np.array(self.weights_rec)
                self.sign_rec = np.array(self.sign_rec)
                if self.weights_rec.shape[1] != self.n_neuron:
                    raise DRCError(
                        "Number of neurons indicated by the input weights matrix does not match the number of neurons indicated by the recurrent weights!"
                    )
                if self.weights_rec.shape != self.sign_rec.shape:
                    raise DRCError(
                        "Recurrent sign shape does not match the recurrent weight shape!"
                    )

        elif self.weights_rec is not None:
            self.weights_rec = np.array(self.weights_rec)
            self.sign_rec = np.array(self.sign_rec)
            self.n_neuron = self.weights_rec.shape[1]
            self.n_in = self.weights_rec.shape[0]
            if self.weights_rec.shape != self.sign_rec.shape:
                raise DRCError(
                    "Recurrent sign shape does not match the recurrent weight shape!"
                )

        else:
            raise ValueError("Unexpected Error Occurded!")

    def __tag_selector(self) -> None:
        vtag_start = 0
        vtag_idx = (
            list(range(vtag_start, self.weights_in.shape[0]))
            if self.weights_in is not None
            else []
        )
        rtag_start = vtag_idx[-1] if vtag_idx else 0
        rtag_idx = (
            list(range(rtag_start, self.weights_rec.shape[0]))
            if self.weights_rec is not None
            else []
        )
        return self.tag_list[vtag_idx], self.tag_list[rtag_idx]

    def CAM_content(
        self, num_synapses: int = NUM_SYNAPSES
    ) -> Dict[int, List[Dynapse2Synapse]]:
        content = {
            nrec: [samna.dynapse2.Dynapse2Synapse() for _ in range(num_synapses)]
            for nrec in range(self.n_neuron)
        }
        syn_counter = [0 for _ in range(self.n_neuron)]

        if self.weights_in is not None:
            for nin in range(self.n_in):
                for nrec in range(self.n_neuron):
                    if self.weights_in[nin][nrec] > 0:
                        self.cam_set(
                            content[nrec][syn_counter[nrec]],
                            tag=self.virtual_tags[nin],
                            dendrite=None,
                            weight=self.w_in_bool[nrec][nin],
                        )
                        syn_counter[nrec] += 1

        if self.weights_rec is not None:
            for row in range(self.n_neuron):
                for nrec in range(self.n_neuron):
                    if self.weights_rec[row][nrec] > 0:
                        self.cam_set(
                            content[nrec][syn_counter[nrec]],
                            tag=self.recurrent_tags[row],
                            dendrite=None,
                            weight=self.w_rec_bool[nrec][row],
                        )
                        syn_counter[nrec] += 1

        return content

    def SRAM_content(
        self, num_dest: int = NUM_DEST
    ) -> Dict[int, List[Dynapse2Destination]]:

        content = {
            nrec: [Dynapse2Destination([0, 0, 0, 0], 0, 0, 0) for _ in range(num_dest)]
            for nrec in range(self.n_neuron)
        }
        dest_counter = [0 for _ in range(self.n_neuron)]
        dest_core_list = [[] for _ in range(self.n_neuron)]
        dest_list: List[Dict[int, List[bool]]] = [{} for _ in range(self.n_neuron)]

        if self.weights_rec is not None:
            for nrec in range(self.n_neuron):
                for col in range(self.n_neuron):
                    ## Destination cores
                    if self.weights_rec[nrec][col] > 0:
                        if self.core_map[col] not in dest_core_list[nrec]:
                            dest_core_list[nrec].append(self.core_map[col])
                            dest_counter[nrec] += 1

            for i, core_list in enumerate(dest_core_list):
                dest_list[i] = self.destinate(core_list) if core_list else {}

            for nrec in range(self.n_neuron):
                source_chip = self.chip_map[self.core_map[nrec]]
                for d, (destination_chip, core_map) in enumerate(
                    dest_list[nrec].items()
                ):
                    directions = np.array(self.chip_pos[destination_chip]) - np.array(
                        self.chip_pos[source_chip]
                    )
                    self.sram_set(
                        content[nrec][d],
                        tag=self.recurrent_tags[nrec],
                        core=core_map,
                        x_hop=directions[0],
                        y_hop=directions[0],
                    )


    @staticmethod
    def matrix_to_destination(
        matrix: np.ndarray,
        pre_core_map: Dict[int, int],
        post_core_map: Dict[int, int],
        chip_map: Dict[int, int],
        tag_list: List[int],
        num_dest: int = NUM_DEST,
    ) -> Dict[int, List[Dynapse2Destination]]:
        """
        matrix_to_destination interprets a given weight matrix and generates a list of destinations to be stored in SRAMs

        :param matrix: the weight matrix representing the connectivity structure
        :type matrix: np.ndarray
        :param pre_core_map: core map (neuron_id : core_id) for the pre-synaptic neurons (axis 0)
        :type pre_core_map: Dict[int, int]
        :param post_core_map: core map (neuron_id : core_id) for the post-synaptic neurons (axis 1)
        :type post_core_map: Dict[int, int]
        :param chip_map: global chip map (core_id : chip_id)
        :type chip_map: Dict[int, int]
        :param tag_list: neuron-tag mapping (neuron_id : tag) which maps the neurons to their virtual addresses
        :type tag_list: List[int]
        :param num_dest: maximum number of destinations, defaults to NUM_DEST
        :type num_dest: int, optional
        :raises DRCError: DRCError: Maximum destination limit reached!
        :return: a dictionary of SRAM entries
        :rtype: Dict[int, List[Dynapse2Destination]]
        """

        n_pre, n_post = matrix.shape

        # pre neurons send events to several post locations
        content = {pre: [] for pre in range(n_pre)}
        dest_cores: List[List[int]] = [[] for _ in range(n_pre)]
        dest_chips: List[Dict[int, List[bool]]] = [{} for _ in range(n_pre)]

        # - Get a list of destination cores of the pre-neurons
        for pre in range(n_pre):
            for post in range(n_post):
                ## Identify destination cores
                if matrix[pre][post] > 0:
                    if post_core_map[post] not in dest_cores[pre]:
                        dest_cores[pre].append(post_core_map[post])

        # - Convert a list of destination cores to chip and core mask pairs
        for i, core_list in enumerate(dest_cores):
            dest_chips[i] = (
                WeightAllocator.mask_cores(core_list, chip_map) if core_list else {}
            )

        # - Find the number of hops between source and destination chips and fill the mem content
        for pre, dest_dict in enumerate(dest_chips):
            if dest_dict:
                source_chip = chip_map[pre_core_map[pre]]
                for dest_chip, core_mask in dest_dict.items():
                    x_hop, y_hop = WeightAllocator.manhattan(dest_chip, source_chip)
                if len(content[pre]) < num_dest:
                    content[pre].append(
                        WeightAllocator.sram_entry(
                            core_mask, x_hop, y_hop, tag_list[pre]
                        )
                    )
                else:
                    raise DRCError("Maximum destination limit reached!")

        return content

    @staticmethod
    def mask_cores(
        core_list: List[int], chip_map: Dict[int, int]
    ) -> Dict[int, List[bool]]:
        """
        mask_cores gets a core list and converts it into a chip id & coremask representaion

        :param core_list: the list of global core ids
        :type core_list: List[int]
        :param chip_map: global chip map (core_id : chip_id)
        :type chip_map: Dict[int, int]
        :return: dictionary of destination chips and the coremask to be applied to them
        :rtype: Dict[int, List[bool]]
        """

        target_cores: Dict[int, List[bool]] = {}
        reverse_chip_map: Dict[int, List[int]] = {v: [] for v in set(chip_map.values())}
        for k, v in chip_map.items():
            reverse_chip_map[v].append(k)

        # - Get a chip & core list
        for core in core_list:
            chip = chip_map[core]
            if chip not in target_cores:
                target_cores[chip] = [core]
            elif core not in target_cores[chip]:
                target_cores[chip].append(core)

        # Fill the core mask
        for key in target_cores:
            mask = [cid in target_cores[key] for cid in reverse_chip_map[key]]
            target_cores[key] = mask

        return target_cores

    @staticmethod
    def manhattan(
        dest_chip: int,
        source_chip: int,
        chip_pos: Optional[Dict[int, Tuple[int]]] = None,
    ) -> Tuple[int]:
        """
        manhattan calculates the manhattan distance between two chips installed on the system

        :param dest_chip: destination chip ID
        :type dest_chip: int
        :param source_chip: source chip ID
        :type source_chip: int
        :param chip_pos: chip position dictionary
        :type chip_pos: Dict[int, Tuple[int]]
        :return: x_hop, y_hop
            :x_hop: number of chip hops on x axis
            :y_hop: number of chip hops on y axis
        :rtype: Tuple[int]
        """
        if dest_chip == source_chip:
            return (0, 0)
        elif chip_pos is None:
            raise ValueError("More than one chip! Provide position dictionary!")
        else:
            dest_position = np.array(chip_pos[dest_chip])
            source_position = np.array(chip_pos[source_chip])
            distance = dest_position - source_position
            return (distance[0], distance[1])

    @staticmethod
    def sram_entry(
        core: Optional[List[bool]] = None,
        x_hop: Optional[int] = None,
        y_hop: Optional[int] = None,
        tag: Optional[int] = None,
        use_samna: bool = False,
    ) -> Dynapse2Destination:
        """
        sram_entry constructs a ``Dynapse2Destinaton`` object and updates its data segment if the parameters are provided

        :param core: the core mask used while sending the events, defaults to None
            [1,1,1,1] means all 4 cores are on the target
            [0,0,1,0] means the event will arrive at core 2 only
        :type core: Optional[List[bool]], optional
        :param x_hop: number of chip hops on x axis, defaults to None
        :type x_hop: Optional[int], optional
        :param y_hop: number of chip hops on y axis, defaults to None
        :type y_hop: Optional[int], optional
        :param tag: globally multiplexed locally unique event tag which is used to identify the connection between two neurons, defaults to None
        :type tag: Optional[int], optional
        :param use_samna: use original samna package or the alias version, defaults to True
        :type use_samna: bool, optional
        :return: a configured ``Dynapse2Destination`` object
        :rtype: Dynapse2Destination
        """
        if use_samna:
            dest = samna.dynapse2.Dynapse2Destination()
            if core is not None:
                dest.core = core
            if x_hop is not None:
                dest.x_hop = x_hop
            if y_hop is not None:
                dest.y_hop = y_hop
            if tag is not None:
                dest.tag = tag
        else:
            dest = Dynapse2Destination(core, x_hop, y_hop, tag)
        return dest

    @staticmethod
    def cam_entry(
        dendrite: Optional[Dendrite] = None,
        weight: Optional[List[bool]] = None,
        tag: Optional[int] = None,
        use_samna: bool = False,
    ) -> Dynapse2Synapse:
        """
        cam_entry constructs a ``Dynapse2Synapse`` object and updates its data segment if the parameters are provided

        :param dendrite: the type of the dendrite, AMPA, GABA, NMDA, SHUNT and NONE are options, defaults to None
        :type dendrite: Optional[Dendrite], optional
        :param weight: 4 bit weight mask chosing the base weight parameters, defaults to None
        :type weight: Optional[List[bool]], optional
        :param tag: the virtual address, defaults to None
        :type tag: Optional[int], optional
        :param use_samna: use original samna package or the alias version, defaults to True
        :type use_samna: bool, optional
        :return: a configured ``Dynapse2Synapse`` samna object
        :rtype: Dynapse2Synapse
        """

        if use_samna:
            syn = samna.dynapse2.Dynapse2Synapse()
            if dendrite is not None:
                syn.dendrite = dendrite
            if weight is not None:
                syn.weight = weight
            if tag is not None:
                syn.tag = tag
        else:
            syn = Dynapse2Synapse(
                dendrite=dendrite,
                stp=False,
                weight=weight,
                precise_delay=False,
                mismatched_delay=False,
                tag=tag,
            )
        return syn


def config_from_specification(
    weights_in: Optional[IntVector],
    weights_rec: Optional[IntVector],
    sign_in: Optional[IntVector],
    sign_rec: Optional[IntVector],
    If_nmda: FloatVector,
    Itau_syn: FloatVector,
    Ispkthr: FloatVector,
    Ipulse: FloatVector,
    Igain_ahp: FloatVector,
    Itau_mem: FloatVector,
    Itau_ahp: FloatVector,
    Ipulse_ahp: FloatVector,
    Igain_mem: FloatVector,
    Igain_syn: FloatVector,
    Idc: FloatVector,
    Iw_ahp: FloatVector,
    Iref: FloatVector,
    Iw: FloatVector,
    chip_map: Dict[int, int],
    chip_pos: Dict[int, Tuple[int]],
    *args,
    **kwargs,
) -> Dynapse2Configuration:

    core = DynapSimCore(
        Idc=Idc[0],
        If_nmda=If_nmda[0],
        Igain_ahp=Igain_ahp[0],
        Igain_ampa=Igain_syn[0],
        Igain_gaba=Igain_syn[0],
        Igain_nmda=Igain_syn[0],
        Igain_shunt=Igain_syn[0],
        Igain_mem=Igain_mem[0],
        Ipulse_ahp=Ipulse_ahp[0],
        Ipulse=Ipulse[0],
        Iref=Iref[0],
        Ispkthr=Ispkthr[0],
        Itau_ahp=Itau_ahp[0],
        Itau_ampa=Itau_syn[0],
        Itau_gaba=Itau_syn[0],
        Itau_nmda=Itau_syn[0],
        Itau_shunt=Itau_syn[0],
        Itau_mem=Itau_mem[0],
        Iw_ahp=Iw_ahp[0],
        Iw_0=Iw[0],
        Iw_1=Iw[1],
        Iw_2=Iw[2],
        Iw_3=Iw[3],
    )

    n_neurons = np.array(weights_rec).shape[1]
    core_map = {n: 0 for n in range(n_neurons)}

    __handler = WeightAllocator(
        weights_in=weights_in,
        weights_rec=weights_rec,
        sign_in=sign_in,
        sign_rec=sign_rec,
        core_map=core_map,
        chip_map=chip_map,
        chip_pos=chip_pos,
    )

    return core.export_Dynapse2Parameters()
