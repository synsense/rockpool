# ----
# tools.py - A few useful funcitons that can be run in cortexcontrol.
# Author: Felix Bauer, SynSense AG, felix.bauer@synsense.ai
# ----

import copy
import os
from typing import Optional, Union, List, Tuple

from rpyc.core.netref import BaseNetref
import CtxDynapse
from rockpool.devices.dynapse.NeuronNeuronConnector import DynapseConnector
from rockpool.devices.dynapse.params import (
    FPGA_ISI_LIMIT,
    NUM_NEURONS_CORE,
    NUM_CORES_CHIP,
    NUM_CHIPS,
    NUM_NEURONS_CHIP,
    CAMTYPES,
    DEF_CAM_STR,
)

__all__ = [
    "local_arguments",
    "extract_event_data",
    "generate_fpga_event_list",
    "generate_buffered_filter",
    "load_biases",
    "save_biases",
    "copy_biases",
    "get_all_neurons",
    "init_chips",
    "reset_connections",
    "remove_all_connections_to",
    "remove_all_connections_from",
    "set_connections",
    "get_connection_info",
    "silence_neurons",
    "reset_silencing",
]

# - Base for converting core mask to core IDs
COREMASK_BASE = tuple(2**i for i in range(NUM_CORES_CHIP))
# - Map cam types to integers
CAMTYPE_DICT = {
    getattr(CtxDynapse.DynapseCamType, camtype): i for i, camtype in enumerate(CAMTYPES)
}
# - Default cam types that neurons are reset to
DEF_CAM_TYPE = getattr(CtxDynapse.DynapseCamType, DEF_CAM_STR)
# - Dict that can be used to store variables in cortexcontrol. They will persist even if
#   RPyC connection breaks down.
storage = dict()


def store_var(name: str, value):
    storage[name] = copy.copy(value)


def _auto_insert_dummies(
    discrete_isi_list: list, neuron_ids: list, fpga_isi_limit: int = FPGA_ISI_LIMIT
) -> (list, list):
    """
    auto_insert_dummies - Insert dummy events where ISI limit is exceeded
    :param discrete_isi_list:  list  Inter-spike intervals of events
    :param neuron_ids:     list  IDs of neurons corresponding to the ISIs

    :return:
        corrected_isi_list     list  ISIs that do not exceed limit, including dummies
        corrected_id_list      list  Neuron IDs corresponding to corrected ISIs. Dummy events have None.
        isdummy_list           list  Boolean indicating which events are dummy events
    """
    # - List of lists with corrected entries
    corrected: List[List] = [
        _replace_too_large_value(isi, fpga_isi_limit) for isi in discrete_isi_list
    ]
    # - Number of new entries for each old entry
    new_event_counts = [len(l) for l in corrected]
    # - List of lists with neuron IDs corresponding to ISIs. Dummy events have ID None
    id_lists: List[List] = [
        [id_neur, *(None for _ in range(length - 1))]
        for id_neur, length in zip(neuron_ids, new_event_counts)
    ]
    # - Flatten out lists
    corrected_isi_list = [isi for l in corrected for isi in l]
    corrected_id_list = [id_neur for l in id_lists for id_neur in l]
    # - Count number of added dummy events (each one has None as ID)
    num_dummies = len(tuple(filter(lambda x: x is None, corrected_id_list)))
    if num_dummies > 0:
        print("dynapse_control: Inserted {} dummy events.".format(num_dummies))

    return corrected_isi_list, corrected_id_list


def _replace_too_large_value(value, limit: int = FPGA_ISI_LIMIT):
    """
    replace_too_large_entry - Return a list of integers <= limit, that sum up to value
    :param value:    int  Value to be replaced
    :param limit:  int  Maximum allowed value
    :return:
        lnReplace   list  Values to replace value
    """
    if value > limit:
        reps = (value - 1) // limit
        # - Return reps times limit, then the remainder
        #   For modulus shift value to avoid replacing with 0 if value==limit
        return [*(limit for _ in range(reps)), (value - 1) % limit + 1]
    else:
        # - If clause in particular for case where value <= 0
        return [value]


def local_arguments(func):
    def local_func(*args, **kwargs):
        for i, argument in enumerate(args):
            newargs = list(args)
            if isinstance(argument, BaseNetref):
                newargs[i] = copy.copy(argument)
        for key, val in kwargs.items():
            if isinstance(key, BaseNetref):
                del kwargs[key]
                kwargs[copy.copy(key)] = copy.copy(val)
            elif isinstance(val, BaseNetref):
                kwargs[key] = copy.copy(val)

        return func(*newargs, **kwargs)

    return local_func


def extract_event_data(events) -> (tuple, tuple):
    """
    extract_event_data - Extract timestamps and neuron IDs from list of recorded events.
    :param events:     list  SpikeEvent objects from BufferedEventFilter
    :return:
        lTimeStamps     list  Timestamps of events
        neuron_ids      list  Neuron IDs of events
    """
    # Extract event timestamps and neuron IDs. Skip events with neuron None.
    event_tuples: List[Tuple] = [
        (event.timestamp, event.neuron.get_id())
        for event in events
        if isinstance(event.neuron, CtxDynapse.DynapseNeuron)
    ]
    try:
        timestamps, neuron_ids = zip(*event_tuples)
    except ValueError as e:
        # - Handle emptly event lists
        if len(event_tuples) == 0:
            timestamps = ()
            neuron_ids = ()
        else:
            raise e
    return timestamps, neuron_ids


def generate_fpga_event_list(
    discrete_isi_list: list,
    neuron_ids: list,
    targetcore_mask: int,
    targetchip_id: int,
    fpga_isi_limit: int = FPGA_ISI_LIMIT,
    correct_isi: bool = True,
) -> list:
    """
    generate_fpga_event_list - Generate a list of FpgaSpikeEvent objects
    :param discrete_isi_list:  array-like  Inter-spike intervalls in Fpga time base
    :param neuron_ids:     array-like  IDs of neurons corresponding to events
    :param neuron_id:       int ID of source neuron
    :param targetcore_mask: int Coremask to determine target cores
    :fpga_isi_limit:         int Maximum ISI size (in time steps)
    :correct_isi:           bool Correct too large ISIs in discrete_isi_list

    :return:
        event  list of generated FpgaSpikeEvent objects.
    """
    # - Make sure objects live on required side of RPyC connection
    targetcore_mask = int(targetcore_mask)
    targetchip_id = int(targetchip_id)
    neuron_ids = copy.copy(neuron_ids)
    discrete_isi_list = copy.copy(discrete_isi_list)

    if correct_isi:
        discrete_isi_list, neuron_ids = _auto_insert_dummies(
            discrete_isi_list, neuron_ids, fpga_isi_limit
        )

    def generate_fpga_event(neuron_id: int, isi: int) -> CtxDynapse.FpgaSpikeEvent:
        """
        generate_fpga_event - Generate a single FpgaSpikeEvent objects.
        :param neuron_id:       int ID of source neuron
        :param isi:            int Timesteps after previous event before
                                    this event will be sent
        :return:
            event  CtxDynapse.FpgaSpikeEvent
        """
        event = CtxDynapse.FpgaSpikeEvent()
        event.target_chip = targetchip_id
        event.core_mask = 0 if neuron_id is None else targetcore_mask
        event.neuron_id = 0 if neuron_id is None else neuron_id
        event.isi = isi
        return event

    # - Generate events
    print("dynapse_control: Generating event list")
    events = [
        generate_fpga_event(neuron_id, isi)
        for neuron_id, isi in zip(neuron_ids, discrete_isi_list)
    ]
    return events


def generate_buffered_filter(model: CtxDynapse.Model, record_neuron_ids: list):
    """
    generate_buffered_filter - Generate and return a BufferedEventFilter object that
                               records from neurons specified in record_neuron_ids.
    :param model:               CtxDynapse model
    :param record_neuron_ids:    list  IDs of neurons to be recorded.
    """
    return CtxDynapse.BufferedEventFilter(model, record_neuron_ids)


def load_biases(filename: str, core_ids: Optional[Union[list, int]] = None):
    """
    load_biases - Load biases from python file under path filename
    :param filename:  str  Path to file where biases are stored.
    :param core_ids:    list, int or None  IDs of cores for which biases
                                            should be loaded. Load all if
                                            None.
    """

    def use_line(codeline: str):
        """
        use_line - Determine whether codeline should be executed considering
                   the IDs of cores in core_ids
        :param codeline:  str  Line of code to be analyzed.
        :return:  bool  Whether line should be executed or not.
        """
        try:
            core_id = int(codeline.split("get_bias_groups()[")[1][0])
        except IndexError:
            # - Line is not specific to core ID
            return True
        else:
            # - Return true if addressed core is in core_ids
            return core_id in core_ids

    if core_ids is None:
        core_ids = list(range(NUM_CHIPS * NUM_CORES_CHIP))
        load_all_cores = True
    else:
        # - Handle integer arguments
        if isinstance(core_ids, int):
            core_ids = [core_ids]
        load_all_cores = False

    with open(os.path.abspath(filename)) as file:
        # list of lines of code of the file. Skip import statement.
        codeline_list = file.readlines()[1:]
        # Iterate over lines of file to apply biases
        for command in codeline_list:
            if load_all_cores or use_line(command):
                exec(command)

    print(
        "dynapse_control: Biases have been loaded from {}.".format(
            os.path.abspath(filename)
        )
    )


def save_biases(filename: str, core_ids: Optional[Union[list, int]] = None):
    """
    save_biases - Save biases in python file under path filename
    :param filename:  str  Path to file where biases should be saved.
    :param core_ids:    list, int or None  ID(s) of cores whose biases
                                            should be saved. If None,
                                            save all cores.
    """

    if core_ids is None:
        core_ids = list(range(NUM_CHIPS * NUM_CORES_CHIP))
    else:
        # - Handle integer arguments
        if isinstance(core_ids, int):
            core_ids = [core_ids]
        # - Include cores in filename, consider possible file endings
        filename_parts: List[str] = filename.split(".")
        # Information to be inserted in filename
        insertstring = "_cores_" + "_".join(str(core_id) for core_id in core_ids)
        try:
            filename_parts[-2] += insertstring
        except IndexError:
            # filename does not contain file ending
            filename_parts[0] += insertstring
            filename_parts.append("py")
        filename = ".".join(filename_parts)

    biasgroup_list = CtxDynapse.model.get_bias_groups()
    # - Only save specified cores
    biasgroup_list = [biasgroup_list[i] for i in core_ids]
    with open(filename, "w") as file:
        file.write("import CtxDynapse as CtxDynapse\n")
        file.write("save_file_model_ = CtxDynapse.model\n")
        for core_id, bias_group in zip(core_ids, biasgroup_list):
            biases = bias_group.get_biases()
            for bias in biases:
                file.write(
                    'save_file_model_.get_bias_groups()[{0}].set_bias("{1}", {2}, {3})\n'.format(
                        core_id,
                        bias.get_bias_name(),
                        bias.get_fine_value(),
                        bias.get_coarse_value(),
                    )
                )
    print(
        "dynapse_control: Biases have been saved under {}.".format(
            os.path.abspath(filename)
        )
    )


def copy_biases(sourcecore_id: int = 0, targetcore_ids: Optional[List[int]] = None):
    """
    copy_biases - Copy biases from one core to one or more other cores.
    :param sourcecore_id:   int  ID of core from which biases are copied
    :param targetcore_ids: int or array-like ID(s) of core(s) to which biases are copied
                            If None, will copy to all other neurons
    """

    targetcore_ids = copy.copy(targetcore_ids)
    if targetcore_ids is None:
        # - Copy biases to all other cores except the source core
        targetcore_ids = list(range(16))
        targetcore_ids.remove(sourcecore_id)
    elif isinstance(targetcore_ids, int):
        targetcore_ids = [targetcore_ids]

    # - List of bias groups from all cores
    biasgroup_list = CtxDynapse.model.get_bias_groups()
    sourcebiases = biasgroup_list[sourcecore_id].get_biases()

    # - Set biases for target cores
    for tgtcore_id in targetcore_ids:
        for bias in sourcebiases:
            biasgroup_list[tgtcore_id].set_bias(
                bias.get_bias_name(), bias.get_fine_value(), bias.get_coarse_value()
            )

    print(
        "dynapse_control: Biases have been copied from core {} to core(s) {}".format(
            sourcecore_id, targetcore_ids
        )
    )


def get_all_neurons(
    model: CtxDynapse.Model, virtual_model: CtxDynapse.VirtualModel
) -> (List, List, List):
    """
    get_all_neurons - Get hardware, virtual and shadow state neurons
                      from model and virtual_model and return them
                      in arrays.
    :param model:  CtxDynapse.Model
    :param virtual_model: CtxDynapse.VirtualModel
    :return:
        List  Hardware neurons
        List  Virtual neurons
        List  Shadow state neurons
    """
    hw_neurons: List = model.get_neurons()
    virtual_neurons: List = virtual_model.get_neurons()
    shadow_neurons: List = model.get_shadow_state_neurons()
    print("dynapse_control: Fetched all neurons from models.")
    return hw_neurons, virtual_neurons, shadow_neurons


def init_chips(chip_ids: Optional[list] = None):
    """
    init_chips - Clear the physical CAM and SRAM cells of the chips defined
                       in chip_ids.
                       This is necessary when CtxControl is loaded (and only then)
                       to make sure that the configuration of the model neurons
                       matches the hardware.

    :param chip_ids:   list  IDs of chips to be cleared.
    """

    # - Make sure chip_ids is a list
    if chip_ids is None:
        return

    if isinstance(chip_ids, int):
        chip_ids = [chip_ids]

    # - Make sure that chip_ids is on correct side of RPyC connection
    chip_ids = copy.copy(chip_ids)

    for nchip in chip_ids:
        print("dynapse_control: Clearing chip {}.".format(nchip))

        # - Clear CAMs
        CtxDynapse.dynapse.clear_cam(int(nchip))
        print("\t CAMs cleared.")

        # - Clear SRAMs
        CtxDynapse.dynapse.clear_sram(int(nchip))
        print("\t SRAMs cleared.")

    # - Make sure that neuron models are cleared as well
    core_ids = [
        core
        for chip in chip_ids
        for core in range(chip * NUM_CORES_CHIP, (chip + 1) * NUM_CORES_CHIP)
    ]
    # - Reset ctxctl neuron model to match hardware state
    _reset_ctxctl_model(core_ids)

    reset_connections(core_ids, True, True, True)

    print("dynapse_control: {} chip(s) cleared.".format(len(chip_ids)))


def _reset_ctxctl_model(core_ids: List[int]):
    """
    _reset_ctxctl_model - Reset ctxctl neuron model to match hardware state
    :param core_ids:  List with IDs of cores whose neuron need to be reset
    """
    # - Neuron IDs corresponding to provided cores
    neuron_ids = [
        n_id
        for c_id in core_ids
        for n_id in range(c_id * NUM_NEURONS_CORE, (c_id + 1) * NUM_NEURONS_CORE)
    ]
    # - Get neuron models from cortexcontrol
    all_neurons = CtxDynapse.model.get_neurons()
    # - Select only relevant neurons
    neurons = [all_neurons[n_id] for n_id in neuron_ids]
    # - Reset neuron states
    for neuron in neurons:
        # - SRAMs
        for sram in neuron.get_srams()[1:]:
            sram.set_target_chip_id(0)
            sram.set_virtual_core_id(0)
            sram.set_used(False)
            sram.set_core_mask(0)
        # - CAMs
        for cam in neuron.get_cams():
            cam.set_pre_neuron_id(0)
            cam.set_pre_neuron_core_id(0)
            cam.set_type(DEF_CAM_TYPE)


def reset_connections(
    core_ids: Union[int, List[int], None] = None,
    presynaptic: bool = True,
    postsynaptic: bool = True,
    apply_diff=True,
):
    """
    reset_connections -   Remove pre- and/or postsynaptic connections of all nerons
                          of cores defined in core_ids.
    :param core_ids:      IDs of cores to be reset (between 0 and 15)
    :param presynaptic:   Remove presynaptic connections to neurons on specified cores.
    :param postsynaptic:  Remove postsynaptic connections of neurons on specified cores.
    :param apply_diff:    Apply changes to hardware. Setting False is useful
                          if new connections will be set afterwards.
    """
    # - Make sure core_ids is a list
    if core_ids is None:
        return

    if isinstance(core_ids, int):
        core_ids = [core_ids]

    # - Make sure that core_ids is on correct side of RPyC connection
    core_ids = copy.copy(core_ids)

    # - Get shadow state neurons
    shadow_neurons = CtxDynapse.model.get_shadow_state_neurons()

    for core_id in core_ids:
        print("dynapse_control: Clearing connections of core {}.".format(core_id))

        # - Reset neuron states
        for neuron in shadow_neurons[
            core_id * NUM_NEURONS_CORE : (core_id + 1) * NUM_NEURONS_CORE
        ]:
            if postsynaptic:
                # - Reset SRAMs for this neuron
                for sram in neuron.get_srams()[1:]:
                    sram.set_target_chip_id(0)
                    sram.set_virtual_core_id(0)
                    sram.set_used(False)
                    sram.set_core_mask(0)

            if presynaptic:
                # - Reset CAMs for this neuron
                for cam in neuron.get_cams():
                    cam.set_pre_neuron_id(0)
                    cam.set_pre_neuron_core_id(0)
                    cam.set_type(DEF_CAM_TYPE)
        print("\t Model neuron weights have been reset.")
    print("dynapse_control: {} core(s) cleared.".format(len(core_ids)))

    if apply_diff:
        # - Apply changes to the connections on chip
        CtxDynapse.model.apply_diff_state()
        print("dynapse_control: New state has been applied to the hardware")


def remove_all_connections_to(
    neuron_ids: List, model: CtxDynapse.Model, apply_diff: bool = True
):
    """
    remove_all_connections_to - Remove all presynaptic connections
                                to neurons defined in neuron_ids
    :param neuron_ids:      list  IDs of neurons whose presynaptic
                                      connections should be removed
    :param model:          CtxDynapse.model
    :param apply_diff:      bool If False do not apply the changes to
                                 chip but only to shadow states of the
                                 neurons. Useful if new connections are
                                 going to be added to the given neurons.
    """
    # - Make sure that neuron_ids is on correct side of RPyC connection
    neuron_ids = copy.copy(neuron_ids)

    # - Get shadow state neurons
    shadow_neurons = CtxDynapse.model.get_shadow_state_neurons()

    # - Neurons that whose cams are to be cleared
    clear_neurons = [shadow_neurons[i] for i in neuron_ids]

    # - Reset neuron weights in model
    for neuron in clear_neurons:
        # - Reset CAMs
        for cam in neuron.get_cams():
            cam.set_pre_neuron_id(0)
            cam.set_pre_neuron_core_id(0)
            cam.set_type(DEF_CAM_TYPE)

    print("dynapse_control: Shadow state neuron weights have been reset")

    if apply_diff:
        # - Apply changes to the connections on chip
        model.apply_diff_state()
        print("dynapse_control: New state has been applied to the hardware")


def remove_all_connections_from(
    neuron_ids: List, model: CtxDynapse.Model, apply_diff: bool = True
):
    """
    remove_all_connections_to - Remove all postsynaptic connections
                                from neurons defined in neuron_ids
    :param neuron_ids:      list  IDs of neurons whose postsynaptic
                                  connections should be removed
    :param model:          CtxDynapse.model
    :param apply_diff:      bool If False do not apply the changes to
                                 chip but only to shadow states of the
                                 neurons. Useful if new connections are
                                 going to be added to the given neurons.
    """
    # - Make sure that neuron_ids is on correct side of RPyC connection
    neuron_ids = copy.copy(neuron_ids)

    # - Get shadow state neurons
    shadow_neurons = CtxDynapse.model.get_shadow_state_neurons()

    # - Neurons that whose cams are to be cleared
    clear_neurons = [shadow_neurons[i] for i in neuron_ids]

    # - Reset neuron weights in model
    for neuron in clear_neurons:
        # - Reset SRAMs
        for sram in neuron.get_srams():
            sram.set_target_chip_id(0)
            sram.set_virtual_core_id(0)
            sram.set_core_mask(0)
            sram.set_used(False)

    print("dynapse_control: Shadow state neuron weights have been reset")

    if apply_diff:
        # - Apply changes to the connections on chip
        model.apply_diff_state()
        print("dynapse_control: New state has been applied to the hardware")


def set_connections(
    preneuron_ids: list,
    postneuron_ids: list,
    syntypes: list,
    shadow_neurons: list,
    virtual_neurons: Optional[list],
    connector: "DynapseConnector",
):
    """
    set_connections - Set connections between pre- and post synaptic neurons from lists.
    :param preneuron_ids:       list  N Presynaptic neurons
    :param postneuron_ids:      list  N Postsynaptic neurons
    :param syntypes:            list  N or 1 Synapse type(s)
    :param shadow_neurons:      list  Shadow neurons that the indices correspond to.
    :param virtual_neurons:     list  If None, presynaptic neurons are shadow neurons,
                                      otherwise virtual neurons from this list.
    :param connector:   DynapseConnector
    """
    preneuron_ids = copy.copy(preneuron_ids)
    postneuron_ids = copy.copy(postneuron_ids)
    syntypes = copy.copy(syntypes)
    presyn_isvirtual = virtual_neurons is not None
    presyn_neuron_population: List = (
        virtual_neurons if presyn_isvirtual else shadow_neurons
    )

    # - Neurons to be connected
    presyn_neurons = [presyn_neuron_population[i] for i in preneuron_ids]
    postsyn_neurons = [shadow_neurons[i] for i in postneuron_ids]

    initialized_neurons = storage.get("initialized_neurons", [])

    if not presyn_isvirtual:
        # - Logical IDs of pre neurons
        logical_pre_ids = [neuron.get_id() for neuron in presyn_neurons]
        # - Make sure that pre neurons are on initialized chips
        if not set(logical_pre_ids).issubset(initialized_neurons):
            raise ValueError(
                "dynapse_control: Some of the presynaptic neurons are on chips that have not"
                + " been cleared since starting cortexcontrol. This may result in unexpected"
                + " behavior. Use the `init_chips` method to clear those chips first."
            )
    # - Logical IDs of post neurons
    logical_post_ids = [neuron.get_id() for neuron in postsyn_neurons]
    # - Make sure that post neurons are on initialized chips
    if not set(logical_post_ids).issubset(initialized_neurons):
        raise ValueError(
            "dynapse_control: Some of the postsynaptic neurons are on chips that have not"
            + " been cleared since starting cortexcontrol. This may result in unexpected"
            + " behavior. Use the `init_chips` method to clear those chips first."
        )

    # - Set connections
    connector.add_connection_from_list(presyn_neurons, postsyn_neurons, syntypes)

    print(
        "dynapse_control: {} {}connections have been set.".format(
            len(preneuron_ids), "virtual " * presyn_isvirtual
        )
    )


def get_connection_info(
    consider_chips: Optional[List[int]] = None,
) -> Tuple[List[int], List[List[int]], List[List[int]], List[List[int]]]:
    consider_chips = (
        list(range(NUM_CHIPS)) if consider_chips is None else consider_chips
    )

    if len(consider_chips) == 0:
        return [], [], [], []

    shadowneurons = CtxDynapse.model.get_shadow_state_neurons()
    # - IDs of neurons that are considered
    neuron_ids = [
        i
        for chip_id in consider_chips
        for i in range(chip_id * NUM_NEURONS_CHIP, (chip_id + 1) * NUM_NEURONS_CHIP)
    ]
    # - Neurons that are considered
    neuron_list = [shadowneurons[i] for i in neuron_ids]
    # - List of lists of all targeted cores for each neuron
    targetcore_lists = [
        [
            targetcore
            for sram in neuron.get_srams()[1:]
            if sram.is_used()
            for targetcore in read_sram_targetcores(sram)
        ]
        for neuron in neuron_list
    ]
    # - List of input IDs to all neurons
    #   (list over neurons, contains list over cams, contains tuples with pre_neuron id and camtype)
    cam_info: List[List[Tuple[int, int]]] = [
        [
            (
                NUM_NEURONS_CORE * cam.get_pre_neuron_core_id()
                + cam.get_pre_neuron_id(),
                CAMTYPE_DICT[cam.get_type()],
            )
            for cam in neuron.get_cams()
        ]
        for neuron in neuron_list
    ]
    # - Separate input ids from cam types
    cam_info_neuronwise = (zip(*neuron) for neuron in cam_info)
    inputid_lists, camtype_lists = zip(*cam_info_neuronwise)

    return neuron_ids, targetcore_lists, inputid_lists, camtype_lists


def read_sram_targetcores(sram: CtxDynapse.DynapseSram) -> List[int]:
    # - Offset of core ID depending on ID of corresponding chip
    core_offset = sram.get_target_chip_id() * NUM_CORES_CHIP
    core_mask = sram.get_core_mask()
    # - Convert core mask into list of core IDs
    return [
        n_core + core_offset for n_core, b in enumerate(COREMASK_BASE) if b & core_mask
    ]


@local_arguments
def silence_neurons(neuron_ids: Union[int, Tuple[int], List[int]]):
    """
    silence_neurons - Assign time contant tau2 to neurons definedin neuron_ids
                      to make them silent.
    :param neuron_ids:  list  IDs of neurons to be silenced
    """
    if isinstance(neuron_ids, int):
        neuron_ids = (neuron_ids,)
    for id_neur in neuron_ids:
        CtxDynapse.dynapse.set_tau_2(
            id_neur // NUM_NEURONS_CHIP,  # Chip ID
            id_neur % NUM_NEURONS_CHIP,  # Neuron ID on chip
        )
    print("dynapse_control: Set {} neurons to tau 2.".format(len(neuron_ids)))


@local_arguments
def reset_silencing(core_ids: Union[int, Tuple[int], List[int]]):
    """
    reset_silencing - Assign time constant tau1 to all neurons on cores defined
                      in core_ids. Convenience function that does the same as
                      global _reset_silencing but also updates self._is_silenced.
    :param core_ids:   list  IDs of cores to be reset
    """
    if isinstance(core_ids, int):
        core_ids = (core_ids,)
    for id_neur in core_ids:
        CtxDynapse.dynapse.reset_tau_1(
            id_neur // NUM_CORES_CHIP,  # Chip ID
            id_neur % NUM_CORES_CHIP,  # Core ID on chip
        )
    print("dynapse_control: Set neurons of cores {} to tau 1.".format(core_ids))
