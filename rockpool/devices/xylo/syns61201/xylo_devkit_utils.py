"""
Utilities for working with the Xylo HDK SYNS61201

Ideally you should not need to use these utility functions. You should try using :py:class:`.XyloSamna` and :py:class:`.XyloSim` for high-level interfaces to Xylo.

See Also:
    The tutorials in :ref:`/devices/xylo-overview.ipynb` and :ref:`/devices/torch-training-spiking-for-xylo.ipynb`.

"""

from rockpool.devices.xylo.syns61201.xa2_devkit_utils import (
    to_hex,
    write_memory,
    read_memory,
    read_output_events,
    read_register,
    XyloState,
)
from rockpool.devices.xylo.syns61300.xylo_devkit_utils import num_buffer_neurons

# - `samna` imports
import samna

from samna.xyloCore2.configuration import XyloConfiguration

# - Other imports
from warnings import warn
import time
import numpy as np
from pathlib import Path
from os import makedirs
import json

# - Typing and useful proxy types
from typing import Any, List, Iterable, Optional, NamedTuple, Union, Tuple

XyloA2HDK = Any
Xylo2ReadBuffer = samna.BasicSinkNode_xylo_core2_event_output_event
Xylo2WriteBuffer = samna.BasicSourceNode_xylo_core2_event_input_event
Xylo2NeuronStateBuffer = samna.xyloCore2.NeuronStateSinkNode


class XyloAllRam(NamedTuple):
    """
    ``NamedTuple`` that encapsulates a recorded Xylo HDK state
    """

    # - state Ram
    Nin: int
    """ int: The number of input-layer neurons """

    Nhidden: int
    """ int: The number of hidden-layer neurons """

    Nout: int
    """ int: The number of output layer neurons """

    V_mem_hid: np.ndarray
    """ np.ndarray: Membrane potential of hidden neurons ``(Nhidden,)``"""

    I_syn_hid: np.ndarray
    """ np.ndarray: Synaptic current 1 of hidden neurons ``(Nhidden,)``"""

    V_mem_out: np.ndarray
    """ np.ndarray: Membrane potential of output neurons ``(Nhidden,)``"""

    I_syn_out: np.ndarray
    """ np.ndarray: Synaptic current of output neurons ``(Nout,)``"""

    I_syn2_hid: np.ndarray
    """ np.ndarray: Synaptic current 2 of hidden neurons ``(Nhidden,)``"""

    Spikes_hid: np.ndarray
    """ np.ndarray: Spikes from hidden layer neurons ``(Nhidden,)``"""

    Spikes_out: np.ndarray
    """ np.ndarray: Spikes from output layer neurons ``(Nout,)``"""

    # - config RAM
    IWTRAM_state: np.ndarray
    """ np.ndarray: Contents of IWTRAM """

    IWT2RAM_state: np.ndarray
    """ np.ndarray: Contents of IWT2RAM """

    NDSRAM_state: np.ndarray
    """ np.ndarray: Contents of NDSRAM """

    RDS2RAM_state: np.ndarray
    """ np.ndarray: Contents of RDS2RAM """

    NDMRAM_state: np.ndarray
    """ np.ndarray: Contents of NMDRAM """

    NTHRAM_state: np.ndarray
    """ np.ndarray: Contents of NTHRAM """

    RCRAM_state: np.ndarray
    """ np.ndarray: Contents of RCRAM """

    RARAM_state: np.ndarray
    """ np.ndarray: Contents of RARAM """

    REFOCRAM_state: np.ndarray
    """ np.ndarray: Contents of REFOCRAM """

    RFORAM_state: np.ndarray
    """ np.ndarray: Contents of RFORAM """

    RWTRAM_state: np.ndarray
    """ np.ndarray: Contents of RWTRAM """

    RWT2RAM_state: np.ndarray
    """ np.ndarray: Contents of RWT2RAM """

    OWTRAM_state: np.ndarray
    """ np.ndarray: Contents of OWTRAM """


from ..syns61201.xa2_devkit_utils import find_xylo_a2_boards


def find_xylo_boards() -> List[XyloA2HDK]:
    """
    Search for and return a list of Xylo HDKs

    Iterate over devices and search for Xylo HDKs. Return a list of available Xylo HDKs, or an empty list if none are found.

    Returns:
        List[XyloHDK]: A (possibly empty) list of Xylo HDK hdks.
    """
    # - Get a list of devices
    device_list = samna.device.get_all_devices()

    # - Search for a xylo dev kit
    xylo_hdk_list = [
        samna.device.open_device(d)
        for d in device_list
        if d.device_type_name == "XyloDevKit" or d.device_type_name == "XyloTestBoard"
    ]

    return xylo_hdk_list


def zero_memory(
    write_buffer: Xylo2WriteBuffer,
) -> None:
    """
    Clear all Xylo memory

    This function writes zeros to all memory banks on a Xylo HDK.

    Args:
        write_buffer (XyloWriteBuffer): A write buffer connected to the desired Xylo HDK
    """
    # - Define the memory banks
    memory_table = {
        "iwtram": (0x0100, 16000),
        "iwt2ram": (0x3F80, 16000),
        "nscram": (0x7E00, 1008),
        "rsc2ram": (0x81F0, 1000),
        "nmpram": (0x85D8, 1008),
        "ndsram": (0x89C8, 1008),
        "rds2ram": (0x8DB8, 1000),
        "ndmram": (0x91A0, 1008),
        "nthram": (0x9590, 1008),
        "rcram": (0x9980, 1000),
        "raram": (0x9D68, 1000),
        "rspkram": (0xA150, 1000),
        "refocram": (0xA538, 1000),
        "rforam": (0xA920, 32000),
        "rwtram": (0x12620, 32000),
        "rwt2ram": (0x1A320, 32000),
        "owtram": (0x22020, 8000),
    }

    # - Zero each bank in turn
    for bank in memory_table.values():
        write_memory(write_buffer, *bank)


def read_allram_state(
    read_buffer: Xylo2ReadBuffer,
    write_buffer: Xylo2WriteBuffer,
    Nin: int = 16,
    Nhidden: int = 1000,
    Nout: int = 8,
) -> XyloAllRam:
    """
    Read and return the all ram in each step as a state

    Args:
        read_buffer (XyloReadBuffer): A read buffer connected to the Xylo HDK
        write_buffer (XyloWriteBuffer): A write buffer connected to the Xylo HDK

    Returns:
        :py:class:`.XyloState`: The recorded state as a ``NamedTuple``. Contains keys ``V_mem_hid``,  ``V_mem_out``, ``I_syn_hid``, ``I_syn_out``, ``I_syn2_hid``, ``Nhidden``, ``Nout``. This state has **no time axis**; the first axis is the neuron ID.

    """
    # - Define the memory bank addresses
    memory_table = {
        "nscram": 0x7E00,
        "rsc2ram": 0x81F0,
        "nmpram": 0x85D8,
        "rspkram": 0xA150,
        "IWTRAM": 0x00100,
        "IWT2RAM": 0x03F80,
        "NDSRAM": 0x089C8,
        "RDS2RAM": 0x08DB8,
        "NDMRAM": 0x091A0,
        "NTHRAM": 0x09590,
        "RCRAM": 0x09980,
        "RARAM": 0x09D68,
        "REFOCRAM": 0x0A538,
        "RFORAM": 0x0A920,
        "RWTRAM": 0x12620,
        "RWT2RAM": 0x1A320,
        "OWTRAM": 0x22020,
    }

    # - Read synaptic currents
    Isyn = read_memory(
        read_buffer,
        write_buffer,
        memory_table["nscram"],
        Nhidden + Nout + num_buffer_neurons(Nhidden),
    )

    # - Read synaptic currents 2
    Isyn2 = read_memory(read_buffer, write_buffer, memory_table["rsc2ram"], Nhidden)

    # - Read membrane potential
    Vmem = read_memory(
        read_buffer,
        write_buffer,
        memory_table["nmpram"],
        Nhidden + Nout + num_buffer_neurons(Nhidden),
    )

    # - Read reservoir spikes
    Spikes = read_memory(read_buffer, write_buffer, memory_table["rspkram"], Nhidden)

    # - Read config RAM including buffer neuron(s)
    input_weight_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["IWTRAM"],
        Nin * (Nhidden + num_buffer_neurons(Nhidden)),
    )

    input_weight_2ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["IWT2RAM"],
        Nin * (Nhidden + num_buffer_neurons(Nhidden)),
    )

    neuron_dash_syn_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["NDSRAM"],
        Nhidden + Nout + num_buffer_neurons(Nhidden),
    )

    reservoir_dash_syn_2ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["RDS2RAM"],
        Nhidden + num_buffer_neurons(Nhidden),
    )

    neuron_dash_mem_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["NDMRAM"],
        Nhidden + Nout + num_buffer_neurons(Nhidden),
    )

    neuron_threshold_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["NTHRAM"],
        Nhidden + Nout + num_buffer_neurons(Nhidden),
    )

    reservoir_config_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["RCRAM"],
        Nhidden + num_buffer_neurons(Nhidden),
    )

    reservoir_aliasing_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["RARAM"],
        Nhidden + num_buffer_neurons(Nhidden),
    )

    reservoir_effective_fanout_count_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["REFOCRAM"],
        # Nhidden + num_buffer_neurons(Nhidden), --> dummy neuron
        Nhidden,
    )

    recurrent_fanout_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["RFORAM"],
        np.sum(np.array(reservoir_effective_fanout_count_ram, "int16")),
    )

    recurrent_weight_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["RWTRAM"],
        np.sum(np.array(reservoir_effective_fanout_count_ram, "int16")),
    )

    recurrent_weight_2ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["RWT2RAM"],
        np.sum(np.array(reservoir_effective_fanout_count_ram, "int16")),
    )

    output_weight_ram = read_memory(
        read_buffer,
        write_buffer,
        memory_table["OWTRAM"],
        Nout * (Nhidden + num_buffer_neurons(Nhidden)),
    )

    # - Return the all ram state
    return XyloAllRam(
        Nin,
        Nhidden,
        Nout,
        # - state RAM
        np.array(Vmem[:Nhidden], "int16"),
        np.array(Isyn[:Nhidden], "int16"),
        np.array(Vmem[-Nout:], "int16"),
        np.array(Isyn[-Nout:], "int16"),
        np.array(Isyn2, "int16"),
        np.array(Spikes, "int16"),
        read_output_events(read_buffer, write_buffer)[:Nout],
        # - config RAM
        np.array(input_weight_ram, "int16"),
        np.array(input_weight_2ram, "int16"),
        np.array(neuron_dash_syn_ram, "int16"),
        np.array(reservoir_dash_syn_2ram, "int16"),
        np.array(neuron_dash_mem_ram, "int16"),
        np.array(neuron_threshold_ram, "int16"),
        np.array(reservoir_config_ram, "int16"),
        np.array(reservoir_aliasing_ram, "int16"),
        np.array(reservoir_effective_fanout_count_ram, "int16"),
        np.array(recurrent_fanout_ram, "int16"),
        np.array(recurrent_weight_ram, "int16"),
        np.array(recurrent_weight_2ram, "int16"),
        np.array(output_weight_ram, "int16"),
    )


def print_debug_ram(
    read_buffer: Xylo2ReadBuffer,
    write_buffer: Xylo2WriteBuffer,
    Nin: int = 10,
    Nhidden: int = 10,
    Nout: int = 2,
) -> None:
    """
    Print memory contents for debugging purposes

    Args:
        read_buffer (XyloReadBuffer): A connected Xylo read buffer to use when reading memory
        write_buffer (XyloWriteBuffer): A connected Xylo write buffer to use
        Nin (int): Number of input neurons to display. Default: ``10``.
        Nhidden (int): Number of hidden neurons to display. Default: ``10``.
    """
    print(
        "iwtram",
        read_memory(
            read_buffer,
            write_buffer,
            0x100,
            Nin * (Nhidden + num_buffer_neurons(Nhidden)),
        ),
    )
    print(
        "iwt2ram",
        read_memory(
            read_buffer,
            write_buffer,
            0x3F80,
            Nin * (Nhidden + num_buffer_neurons(Nhidden)),
        ),
    )

    print(
        "nscram",
        read_memory(
            read_buffer,
            write_buffer,
            0x7E00,
            Nhidden + Nout + num_buffer_neurons(Nhidden),
        ),
    )
    print(
        "rsc2ram",
        read_memory(
            read_buffer, write_buffer, 0x81F0, Nhidden + num_buffer_neurons(Nhidden)
        ),
    )
    print(
        "nmpram",
        read_memory(
            read_buffer,
            write_buffer,
            0x85D8,
            Nhidden + Nout + num_buffer_neurons(Nhidden),
        ),
    )

    print(
        "ndsram",
        read_memory(
            read_buffer,
            write_buffer,
            0x89C8,
            Nhidden + Nout + num_buffer_neurons(Nhidden),
        ),
    )
    print(
        "rds2ram",
        read_memory(
            read_buffer, write_buffer, 0x8DB8, Nhidden + num_buffer_neurons(Nhidden)
        ),
    )
    print(
        "ndmram",
        read_memory(
            read_buffer,
            write_buffer,
            0x91A0,
            Nhidden + Nout + num_buffer_neurons(Nhidden),
        ),
    )

    print(
        "nthram",
        read_memory(
            read_buffer,
            write_buffer,
            0x9590,
            Nhidden + Nout + num_buffer_neurons(Nhidden),
        ),
    )

    print(
        "rcram",
        read_memory(
            read_buffer, write_buffer, 0x9980, Nhidden + num_buffer_neurons(Nhidden)
        ),
    )
    print(
        "raram",
        read_memory(
            read_buffer, write_buffer, 0x9D68, Nhidden + num_buffer_neurons(Nhidden)
        ),
    )

    print(
        "rspkram",
        read_memory(
            read_buffer, write_buffer, 0xA150, Nhidden + num_buffer_neurons(Nhidden)
        ),
    )

    print(
        "refocram",
        read_memory(
            read_buffer, write_buffer, 0xA538, Nhidden + num_buffer_neurons(Nhidden)
        ),
    )
    print(
        "rforam",
        read_memory(
            read_buffer, write_buffer, 0xA920, Nhidden + num_buffer_neurons(Nhidden)
        ),
    )

    print(
        "rwtram",
        read_memory(
            read_buffer, write_buffer, 0x12620, Nhidden + num_buffer_neurons(Nhidden)
        ),
    )
    print(
        "rwt2ram",
        read_memory(
            read_buffer, write_buffer, 0x1A320, Nhidden + num_buffer_neurons(Nhidden)
        ),
    )

    print(
        "owtram",
        read_memory(
            read_buffer,
            write_buffer,
            0x22020,
            (Nhidden + num_buffer_neurons(Nhidden) * Nout),
        ),
    )


def export_registers(
    read_buffer: Xylo2ReadBuffer,
    write_buffer: Xylo2WriteBuffer,
    file,
) -> None:
    """
    Print register contents for debugging purposes

    Args:
        read_buffer (XyloReadBuffer): A connected Xylo read buffer to use in reading registers
        write_buffer (XyloWriteBuffer): A write buffer connected to a Xylo HDK
        file: a file to save the registers
    """

    with open(file, "w+") as f:
        f.write("ctrl1 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x1)[0]))
        f.write("\n")

        f.write("ctrl2 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x2)[0]))
        f.write("\n")

        f.write("ctrl3 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x3)[0]))
        f.write("\n")

        f.write("pwrctrl1 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x04)[0]))
        f.write("\n")

        f.write("pwrctrl2 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x05)[0]))
        f.write("\n")

        f.write("pwrctrl3 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x06)[0]))
        f.write("\n")

        f.write("pwrctrl4 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x07)[0]))
        f.write("\n")

        f.write("ie ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x08)[0]))
        f.write("\n")

        f.write("ctrl4 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x09)[0]))
        f.write("\n")

        f.write("baddr ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x0A)[0]))
        f.write("\n")

        f.write("blen ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x0B)[0]))
        f.write("\n")

        f.write("ispkreg00 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x0C)[0]))
        f.write("\n")

        f.write("ispkreg01 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x0D)[0]))
        f.write("\n")

        f.write("ispkreg10 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x0E)[0]))
        f.write("\n")

        f.write("ispkreg11 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x0F)[0]))
        f.write("\n")

        f.write("stat ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x10)[0]))
        f.write("\n")

        f.write("int ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x11)[0]))
        f.write("\n")

        f.write("omp_stat0 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x12)[0]))
        f.write("\n")

        f.write("omp_stat1 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x13)[0]))
        f.write("\n")

        f.write("omp_stat2 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x14)[0]))
        f.write("\n")

        f.write("omp_stat3 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x15)[0]))
        f.write("\n")

        f.write("monsel0 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x16)[0]))
        f.write("\n")

        f.write("monsel1 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x17)[0]))
        f.write("\n")

        f.write("dbg_ctrl1 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x18)[0]))
        f.write("\n")

        f.write("dbg_stat1 ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x19)[0]))
        f.write("\n")

        f.write("tr_cntr_stat ")
        f.write(hex(read_register(read_buffer, write_buffer, 0x1A)[0]))
        f.write("\n")


def print_debug_registers(
    read_buffer: Xylo2ReadBuffer,
    write_buffer: Xylo2WriteBuffer,
) -> None:
    """
    Print register contents of a Xylo HDK for debugging purposes

    Args:
        write_buffer (XyloWriteBuffer): A connected write buffer to a Xylo HDK
        read_buffer (XyloReadBuffer): A connected Xylo read buffer to use when reading registers
    """
    print("ctrl1", hex(read_register(read_buffer, write_buffer, 0x1)[0]))
    print("ctrl2", hex(read_register(read_buffer, write_buffer, 0x2)[0]))
    print("ctrl3", hex(read_register(read_buffer, write_buffer, 0x3)[0]))
    print("pwrctrl1", hex(read_register(read_buffer, write_buffer, 0x04)[0]))
    print("pwrctrl2", hex(read_register(read_buffer, write_buffer, 0x05)[0]))
    print("pwrctrl3", hex(read_register(read_buffer, write_buffer, 0x06)[0]))
    print("pwrctrl4", hex(read_register(read_buffer, write_buffer, 0x07)[0]))
    print("ie", hex(read_register(read_buffer, write_buffer, 0x08)[0]))
    print("ctrl4", hex(read_register(read_buffer, write_buffer, 0x09)[0]))
    print("baddr", hex(read_register(read_buffer, write_buffer, 0x0A)[0]))
    print("blen", hex(read_register(read_buffer, write_buffer, 0x0B)[0]))
    print("ispkreg00", hex(read_register(read_buffer, write_buffer, 0x0C)[0]))
    print("ispkreg01", hex(read_register(read_buffer, write_buffer, 0x0D)[0]))
    print("ispkreg10", hex(read_register(read_buffer, write_buffer, 0x0E)[0]))
    print("ispkreg11", hex(read_register(read_buffer, write_buffer, 0x0F)[0]))
    print("stat", hex(read_register(read_buffer, write_buffer, 0x10)[0]))
    print("int", hex(read_register(read_buffer, write_buffer, 0x11)[0]))
    print("omp_stat0", hex(read_register(read_buffer, write_buffer, 0x12)[0]))
    print("omp_stat1", hex(read_register(read_buffer, write_buffer, 0x13)[0]))
    print("omp_stat2", hex(read_register(read_buffer, write_buffer, 0x14)[0]))
    print("omp_stat3", hex(read_register(read_buffer, write_buffer, 0x15)[0]))
    print("monsel0", hex(read_register(read_buffer, write_buffer, 0x16)[0]))
    print("monsel1", hex(read_register(read_buffer, write_buffer, 0x17)[0]))
    print("dbg_ctrl1", hex(read_register(read_buffer, write_buffer, 0x18)[0]))
    print("dbg_stat1", hex(read_register(read_buffer, write_buffer, 0x19)[0]))
    print("tr_cntr_stat", hex(read_register(read_buffer, write_buffer, 0x1A)[0]))


def export_config(
    path: Union[str, Path],
    config: XyloConfiguration,
    dt: float,
) -> None:
    """
    Export a network configuration to a set of text files, for debugging purposes

    This function produces a large number of text files under the directory ``path``, which will be created if necessary. These files contain detailed memory contents and configuration options specified by an HDK configuration object ``config``.

    Args:
        path (Union[str, path]): Directory to write data
        config (XyloConfiguration): A Xylo configuraiton to export
        dt (float): The time step of the simulation
    """
    # - Check base path
    path = Path(path)
    if not path.exists():
        makedirs(path)

    # - Generate a XyloSim module from the config
    from rockpool.devices.xylo.syns61201 import XyloSim

    sim = XyloSim.from_config(config, dt=dt)
    model = sim._xylo_layer

    inp_size = len(model.synapses_in)
    num_neurons = len(model.synapses_rec)

    # transfer input synapses to matrix
    num_targets = num_neurons
    mat = np.zeros((2, inp_size, num_targets), dtype=int)
    for pre, syns in enumerate(model.synapses_in):
        for syn in syns:
            mat[syn.target_synapse_id, pre, syn.target_neuron_id] = syn.weight

    # iwtram and iwt2ram (input neurons of synapse IDs 0 and 1)
    for ram, mat_syn in zip(("iwt", "iwt2"), mat):
        # save to file
        print(f"Writing {ram}ram.ini", end="\r")
        with open(path / f"{ram}ram.ini", "w+") as f:
            for pre, line in enumerate(mat_syn):
                f.write(f"// {ram} for IN{pre} \n")
                for post, weight in enumerate(line):
                    f.write(to_hex(weight, 2))
                    f.write("\n")

    # create matrix for recurrent weights (slightly different convention than for input)
    mat = np.zeros((num_neurons, num_neurons, 2), dtype=int)
    for pre, syns in enumerate(model.synapses_rec):
        for syn in syns:
            mat[pre, syn.target_neuron_id, syn.target_synapse_id] = syn.weight

    # rwtram (recurrent neurons of synapse IDs 0)
    print("Writing rwtram.ini", end="\r")
    with open(path / "rwtram.ini", "w+") as f:
        for pre, line in enumerate(mat):
            f.write(f"// rwt of RSN{pre} \n")
            for syns in line:
                if np.any(syns != 0):
                    weight = syns[0]
                    f.write(to_hex(weight, 2))
                    f.write("\n")

    # rwtram2 (recurrent neurons of synapse IDs 1)
    print("Writing rwt2ram.ini", end="\r")
    with open(path / "rwt2ram.ini", "w+") as f:
        for pre, line in enumerate(mat):
            f.write(f"// rwt2 of RSN{pre} \n")
            for syns in line:
                if np.any(syns != 0):
                    weight = syns[1]
                    f.write(to_hex(weight, 2))
                    f.write("\n")

    # rforam (recurrent fanout, or target ids)
    print("Writing rforam.ini", end="\r")
    with open(path / "rforam.ini", "w+") as f:
        for pre, line in enumerate(mat):
            f.write(f"// rfo of RSN{pre} \n")
            for post, syns in enumerate(line):
                if np.any(syns != 0):
                    f.write(to_hex(post, 3))
                    f.write("\n")

    # refocram (recurrent effective fanout, or number of targets)
    print("Writing refocram.ini", end="\r")
    with open(path / "refocram.ini", "w+") as f:
        for pre, line in enumerate(mat):
            count = 0
            for post, syns in enumerate(line):
                if np.any(syns != 0):
                    count += 1
            f.write(to_hex(count, 2))
            f.write("\n")

    # owtram (output weights)
    # transfer output synapses to matrix
    post_ids_out = [
        [syn.target_neuron_id for syn in l_syn if syn.target_synapse_id == 0]
        for l_syn in model.synapses_out
    ]
    weights_out = [
        [syn.weight for syn in l_syn if syn.target_synapse_id == 0]
        for l_syn in model.synapses_out
    ]
    try:
        readout_size = max(max(post_ids_out)) + 1
    except:
        readout_size = 0

    size_total = readout_size + num_neurons
    mat = np.zeros((num_neurons, readout_size), dtype=int)
    for pre, (post, weight) in enumerate(zip(post_ids_out, weights_out)):
        mat[pre, post] = weight

    # save to file
    print("Writing owtram.ini", end="\r")
    with open(path / "owtram.ini", "w+") as f:
        for pre, line in enumerate(mat):
            f.write(f"// owt for RSN{pre} \n")
            for post, weight in enumerate(line):
                f.write(to_hex(weight, 2))
                f.write("\n")

    # ndmram (membrane time constants)
    mat = np.zeros(size_total, dtype=int)
    mat[:num_neurons] = [n.v_mem_decay for n in config.reservoir.neurons]
    mat[num_neurons:size_total] = [n.v_mem_decay for n in config.readout.neurons]

    # save to file
    print("Writing ndmram.ini", end="\r")
    with open(path / "ndmram.ini", "w+") as f:
        for pre, dash in enumerate(mat):
            f.write(to_hex(dash, 1))
            f.write("\n")

    # ndsram (synaptic time constants, ID=0)
    mat = np.zeros(size_total, dtype=int)
    mat[:num_neurons] = [n.i_syn_decay for n in config.reservoir.neurons]
    mat[num_neurons:size_total] = [n.i_syn_decay for n in config.readout.neurons]

    # save to file
    print("Writing ndsram.ini", end="\r")
    with open(path / "ndsram.ini", "w+") as f:
        for pre, dash in enumerate(mat):
            f.write(to_hex(dash, 1))
            f.write("\n")

    # nds2ram (synaptic time constants, ID=1) --> rds2ram
    if config.synapse2_enable:
        mat = [n.i_syn2_decay for n in config.reservoir.neurons]
    else:
        mat = np.zeros(num_neurons, int)

    # save to file
    print("Writing rds2ram.ini", end="\r")
    with open(path / "rds2ram.ini", "w+") as f:
        for pre, dash in enumerate(mat):
            f.write(to_hex(dash, 1))
            f.write("\n")

    # nthram (thresholds)
    thresholds = [n.threshold for n in config.reservoir.neurons] + [
        n.threshold for n in config.readout.neurons
    ]

    # save to file
    print("Writing nthram.ini", end="\r")
    with open(path / "nthram.ini", "w+") as f:
        for pre, th in enumerate(thresholds):
            f.write(to_hex(th, 4))
            f.write("\n")

    # raram and rcram (aliases)
    mat = np.zeros(num_neurons, dtype=int) - 1
    is_source = np.zeros(num_neurons, dtype=int)
    is_target = np.zeros(num_neurons, dtype=int)
    num_sources = np.zeros(num_neurons, dtype=int)
    for i, aliases in enumerate(model.aliases):
        if len(aliases) > 0:
            mat[i] = aliases[0]
            is_source[i] = 1
            is_target[aliases[0]] += 1

    # save to file
    print("Writing raram.ini", end="\r")
    with open(path / "raram.ini", "w+") as f:
        for pre, alias in enumerate(mat):
            f.write(to_hex(alias, 3))
            f.write("\n")

    # save to file
    print("Writing rcram.ini", end="\r")
    with open(path / "rcram.ini", "w+") as f:
        for pre, issource in enumerate(is_source):
            # print(
            #     pre,
            #     "->",
            #     mat[pre],
            #     ":",
            #     is_target[mat[pre]],
            #     issource,
            #     is_target[pre],
            #     ((is_target[mat[pre]] > 1) << 2)
            #     + (issource << 1)
            #     + (is_target[pre] > 0),
            # )
            f.write(
                to_hex(
                    ((is_target[mat[pre]] > 1) << 2)
                    + (issource << 1)
                    + (is_target[pre] > 0),
                    1,
                )
            )
            f.write("\n")

    # basic config
    print("Writing basic_config.json", end="\r")
    with open(path / "basic_config.json", "w+") as f:
        conf = {}

        # number of neurons
        conf["IN"] = len(model.synapses_in)
        conf["RSN"] = len(model.synapses_rec)

        # determine output size by getting the largest target neuron id
        syns = np.hstack(model.synapses_out)
        conf["ON"] = int(np.max([s.target_neuron_id for s in syns]) + 1)

        # bit shift values
        conf["IWBS"] = model.weight_shift_inp
        conf["RWBS"] = model.weight_shift_rec
        conf["OWBS"] = model.weight_shift_out

        # expansion neurons
        # if num_expansion is not None:
        #    conf["IEN"] = num_expansion

        # dt
        conf["time_resolution_wrap"] = config.time_resolution_wrap
        conf["DT"] = sim.dt

        # number of synapses
        n_syns = 1
        syns_in = np.hstack(model.synapses_in)
        if np.any(np.array([s.target_synapse_id for s in syns_in]) == 1):
            n_syns = 2
        syns_rec = np.hstack(model.synapses_rec)
        if np.any(np.array([s.target_synapse_id for s in syns_rec]) == 1):
            n_syns = 2

        conf["N_SYNS"] = n_syns

        # aliasing
        if max([len(a) for a in model.aliases]) > 0:
            conf["RA"] = True
        else:
            conf["RA"] = False

        json.dump(conf, f)


def export_frozen_state(
    path: Union[str, Path], config: XyloConfiguration, state: XyloState
) -> None:
    """
    Export a single time-step frozen state of a Xylo network

    This function will produce a series of RAM initialisation files containing a Xylo state, written to the directory ``path``, which will be created if necessary.

    Args:
        path (Path): The directory to export the state to
        config (XyloConfiguration): The configuration of the Xylo network
        state (XyloState): A single time-step state of a Xylo network to export
    """
    # - Make `path` a path
    path = Path(path)
    if not path.exists():
        makedirs(path)

    # - Check that we have a single time point
    for k, v in zip(state._fields, state):
        if k in ["Nhidden", "Nout"]:
            continue

        assert (v.shape[0] == 1) or (
            np.ndim(v) == 1
        ), "`state` must define a single time point"

    # - Determine network size
    inp_size = np.shape(config.input.weights)[0]
    num_neurons = np.shape(config.reservoir.weights)[1]
    readout_size = np.shape(config.readout.weights)[1]
    size_total = num_neurons + readout_size

    T = 1

    # rspkram
    mat = np.zeros((T, num_neurons), dtype=int)
    spks = np.array(np.atleast_2d(state.Spikes_hid)).astype(int)

    if len(spks) > 0:
        mat[:, : spks.shape[1]] = spks

        print("Writing rspkram.ini", end="\r")
        for t, spks in enumerate(mat):
            with open(path / f"rspkram.ini", "w+") as f:
                for val in spks:
                    f.write(to_hex(val, 2))
                    f.write("\n")

    # ospkram
    mat = np.zeros((T, readout_size), dtype=int)
    spks = np.array(np.atleast_2d(state.Spikes_out)).astype(int)

    if len(spks) > 0:
        mat[:, : spks.shape[1]] = spks
        print("Writing ospkram.ini", end="\r")
        for t, spks in enumerate(mat):
            with open(path / f"ospkram.ini", "w+") as f:
                for val in spks:
                    f.write(to_hex(val, 2))
                    f.write("\n")

    # nscram
    mat = np.zeros((T, size_total), dtype=int)
    isyns = np.array(np.atleast_2d(state.I_syn_hid)).astype(int)
    mat[:, : isyns.shape[1]] = isyns
    isyns_out = np.array(np.atleast_2d(state.I_syn_out)).astype(int)
    mat[:, num_neurons : num_neurons + isyns_out.shape[1]] = isyns_out

    print("Writing nscram.ini", end="\r")
    for t, vals in enumerate(mat):
        with open(path / f"nscram.ini", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 4))
                f.write("\n")

    # nsc2ram --> rsc2ram
    if not hasattr(state, "I_syn2_hid"):
        mat = np.zeros((0, num_neurons), int)
    else:
        mat = np.zeros((T, num_neurons), dtype=int)
        isyns2 = np.array(np.atleast_2d(state.I_syn2_hid)).astype(int)
        mat[:, : isyns2.shape[1]] = isyns2

    print("Writing rsc2ram.ini", end="\r")
    for t, vals in enumerate(mat):
        with open(path / f"rsc2ram.ini", "w+") as f:
            for val in vals:
                f.write(to_hex(val, 4))
                f.write("\n")

    # nmpram
    mat = np.zeros((T, size_total), dtype=int)
    vmems = np.array(np.atleast_2d(state.V_mem_hid)).astype(int)
    mat[:, : vmems.shape[1]] = vmems
    vmems_out = np.array(np.atleast_2d(state.V_mem_out)).astype(int)
    mat[:, num_neurons : num_neurons + vmems_out.shape[1]] = vmems_out
    print("Writing nmpram.ini", end="\r")
    for t, vals in enumerate(mat):
        with open(path / f"nmpram.ini", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 4))
                f.write("\n")


def export_temporal_state(
    path: Union[Path, str],
    config: XyloConfiguration,
    inp_spks: np.ndarray,
    state: XyloState,
) -> None:
    """
    Export the state of a Xylo network over time, for debugging purposes

    This function will produce a series of RAM files, per time-step, containing the recorded state evolution of a Xylo network. The files will be save to the directory ``path``, which will be created if necessary.

    Args:
        path (Path): The directory to export the state to
        config (XyloConfiguration): The configuration of the Xylo network
        inp_spks (np.ndarray): The input spikes for this simulation
        state (XyloState): A temporal state of a Xylo network to export
    """
    # - Make `path` a path
    path = Path(path)

    # - Determine network size
    inp_size = np.shape(config.input.weights)[0]
    num_neurons = np.shape(config.reservoir.weights)[1]
    readout_size = np.shape(config.readout.weights)[1]
    size_total = num_neurons + readout_size

    # rspkram
    mat = np.zeros((np.shape(state.Spikes_hid)[0], num_neurons), dtype=int)
    spks = np.array(state.Spikes_hid).astype(int)

    if len(spks) > 0:
        mat[:, : spks.shape[1]] = spks

        path_spkr = path / "spk_res"
        if not path_spkr.exists():
            makedirs(path_spkr)

        print("Writing rspkram files in spk_res", end="\r")
        for t, spks in enumerate(mat):
            with open(path_spkr / f"rspkram_{t}.txt", "w+") as f:
                for val in spks:
                    f.write(to_hex(val, 2))
                    f.write("\n")

    # ospkram
    mat = np.zeros((np.shape(state.Spikes_out)[0], readout_size), dtype=int)
    spks = np.array(state.Spikes_out).astype(int)

    if len(spks) > 0:
        mat[:, : spks.shape[1]] = spks

        path_spko = path / "spk_out"
        if not path_spko.exists():
            makedirs(path_spko)

        print("Writing ospkram files in spk_out", end="\r")
        for t, spks in enumerate(mat):
            with open(path_spko / f"ospkram_{t}.txt", "w+") as f:
                for val in spks:
                    f.write(to_hex(val, 2))
                    f.write("\n")

    # nscram
    mat = np.zeros((np.shape(state.I_syn_hid)[0], size_total), dtype=int)
    isyns = np.array(state.I_syn_hid).astype(int)
    mat[:, : isyns.shape[1]] = isyns
    isyns_out = np.array(state.I_syn_out).astype(int)
    mat[:, num_neurons : num_neurons + isyns_out.shape[1]] = isyns_out

    path_isyn = path / "isyn"
    if not path_isyn.exists():
        makedirs(path_isyn)

    print("Writing nscram files in isyn", end="\r")
    for t, vals in enumerate(mat):
        with open(path_isyn / f"nscram_{t}.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 4))
                f.write("\n")

    # nsc2ram --> rsc2ram
    if not hasattr(state, "I_syn2_hid"):
        mat = np.zeros((0, num_neurons), int)
    else:
        mat = np.zeros((np.shape(state.I_syn2_hid)[0], num_neurons), dtype=int)
        isyns2 = np.array(state.I_syn2_hid).astype(int)
        mat[:, : isyns2.shape[1]] = isyns2

    path_isyn2 = path / "isyn2"
    if not path_isyn2.exists():
        makedirs(path_isyn2)

    print("Writing rsc2ram files in isyn2", end="\r")
    for t, vals in enumerate(mat):
        with open(path_isyn2 / f"rsc2ram_{t}.txt", "w+") as f:
            for val in vals:
                f.write(to_hex(val, 4))
                f.write("\n")

    # nmpram
    mat = np.zeros((np.shape(state.V_mem_hid)[0], size_total), dtype=int)
    vmems = np.array(state.V_mem_hid).astype(int)
    mat[:, : vmems.shape[1]] = vmems
    vmems_out = np.array(state.V_mem_out).astype(int)
    mat[:, num_neurons : num_neurons + vmems_out.shape[1]] = vmems_out

    path_vmem = path / "vmem"
    if not path_vmem.exists():
        makedirs(path_vmem)

    print("Writing nmpram files in vmem", end="\r")
    for t, vals in enumerate(mat):
        with open(path_vmem / f"nmpram_{t}.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 4))
                f.write("\n")

    if inp_spks is not None:
        # input spikes
        path_spki = path / "spk_in"
        if not path_spki.exists():
            makedirs(path_spki)

        print("Writing inp_spks.txt", end="\r")
        with open(path_spki / "inp_spks.txt", "w+") as f:
            idle = -1
            for t, chans in enumerate(inp_spks):
                idle += 1
                if not np.all(chans == 0):
                    f.write(f"// time step {t}\n")
                    if idle > 0:
                        f.write(f"idle {idle}\n")
                    idle = 0
                    for chan, num_spikes in enumerate(chans):
                        for _ in range(num_spikes):
                            f.write(f"wr IN{to_hex(chan, 1)}\n")


def export_allram_state(
    path: Union[Path, str],
    config: XyloConfiguration,
    inp_spks: np.ndarray,
    state: XyloState,
) -> None:
    """
    Export the all RAM state of a Xylo network over time, for debugging purposes

    This function will produce a series of RAM files, per time-step, containing the recorded state evolution of a Xylo network. The files will be written to a directory ``path``, which will be created if necessary.

    Args:
        path (Path): The directory to export the state to
        config (XyloConfiguration): The configuration of the Xylo network
        inp_spks (np.ndarray): The input spikes for this simulation
        state (XyloState): A temporal state of a Xylo network to export
    """
    # - Make `path` a path
    path = Path(path)

    # - Determine network size
    inp_size = np.shape(config.input.weights)[0]
    num_neurons = np.shape(config.reservoir.weights)[1]
    readout_size = np.shape(config.readout.weights)[1]
    size_total = num_neurons + readout_size

    Nin = inp_size
    Nhidden = num_neurons
    Nout = readout_size

    # - rspkram
    mat = np.zeros((np.shape(state.Spikes_hid)[0], num_neurons), dtype=int)
    spks = np.array(state.Spikes_hid).astype(int)

    if len(spks) > 0:
        mat[:, : spks.shape[1]] = spks

        path_spkr = path / "spk_res"
        if not path_spkr.exists():
            makedirs(path_spkr)

        print("Writing rspkram files in spk_res", end="\r")
        for t, spks in enumerate(mat):
            with open(
                path_spkr / f"rspkram_{t-1}.txt", "w+"
            ) as f:  # t-1 because add 1 before evolve
                for val in spks:
                    f.write(to_hex(val, 2))
                    f.write("\n")

    # - ospkram
    mat = np.zeros((np.shape(state.Spikes_out)[0], readout_size), dtype=int)
    spks = np.array(state.Spikes_out).astype(int)

    if len(spks) > 0:
        mat[:, : spks.shape[1]] = spks[:, 0:Nout]  # [:, 0:Nout] for the last step ???

        path_spko = path / "spk_out"
        if not path_spko.exists():
            makedirs(path_spko)

        print("Writing ospkram files in spk_out", end="\r")
        for t, spks in enumerate(mat):
            with open(path_spko / f"ospkram_{t-1}.txt", "w+") as f:
                for val in spks:
                    f.write(to_hex(val, 2))
                    f.write("\n")

    # - nscram
    mat = np.zeros((np.shape(state.I_syn_hid)[0], size_total), dtype=int)
    isyns = np.array(state.I_syn_hid).astype(int)
    mat[:, : isyns.shape[1]] = isyns
    isyns_out = np.array(state.I_syn_out).astype(int)
    mat[:, num_neurons : num_neurons + isyns_out.shape[1]] = isyns_out

    path_isyn = path / "isyn"
    if not path_isyn.exists():
        makedirs(path_isyn)

    print("Writing nscram files in isyn", end="\r")
    for t, vals in enumerate(mat):
        with open(path_isyn / f"nscram_{t-1}.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 4))
                f.write("\n")

    # - nsc2ram renamed as rsc2ram (correct name in the Xylo datasheet)
    if not hasattr(state, "I_syn2_hid"):
        mat = np.zeros((0, num_neurons), int)
    else:
        mat = np.zeros((np.shape(state.I_syn2_hid)[0], num_neurons), dtype=int)
        isyns2 = np.array(state.I_syn2_hid).astype(int)
        mat[:, : isyns2.shape[1]] = isyns2

    path_isyn2 = path / "isyn2"
    if not path_isyn2.exists():
        makedirs(path_isyn2)

    print("Writing rsc2ram files in isyn2", end="\r")
    for t, vals in enumerate(mat):
        with open(path_isyn2 / f"rsc2ram_{t-1}.txt", "w+") as f:
            for val in vals:
                f.write(to_hex(val, 4))
                f.write("\n")

    # - nmpram
    mat = np.zeros((np.shape(state.V_mem_hid)[0], size_total), dtype=int)
    vmems = np.array(state.V_mem_hid).astype(int)
    mat[:, : vmems.shape[1]] = vmems
    vmems_out = np.array(state.V_mem_out).astype(int)
    mat[:, num_neurons : num_neurons + vmems_out.shape[1]] = vmems_out

    path_vmem = path / "vmem"
    if not path_vmem.exists():
        makedirs(path_vmem)

    print("Writing nmpram files in vmem", end="\r")
    for t, vals in enumerate(mat):
        with open(path_vmem / f"nmpram_{t-1}.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 4))
                f.write("\n")

    if inp_spks is not None:
        # - input spikes
        path_spki = path / "spk_in"
        if not path_spki.exists():
            makedirs(path_spki)

        print("Writing inp_spks.txt", end="\r")
        with open(path_spki / "inp_spks.txt", "w+") as f:
            idle = -1
            for t, chans in enumerate(inp_spks):
                idle += 1
                if not np.all(chans == 0):
                    f.write(f"// time step {t}\n")
                    if idle > 0:
                        f.write(f"idle {idle}\n")
                    idle = 0
                    for chan, num_spikes in enumerate(chans):
                        for _ in range(num_spikes):
                            f.write(f"wr IN{to_hex(chan, 1)}\n")

    # - Save config RAM, not export dummy neuron
    # IWTRAM_state: input_weight_ram_ts
    mat = np.zeros((np.shape(state.IWTRAM_state)[0], Nin * Nhidden), dtype=int)
    input_weight = np.array(state.IWTRAM_state).astype(int)
    mat[:, : input_weight.shape[1]] = input_weight[:, 0 : Nin * Nhidden]

    path_IWTRAM_state = path / "IWTRAM_state"
    if not path_IWTRAM_state.exists():
        makedirs(path_IWTRAM_state)

    print("Writing IWTRAM files in input_weight", end="\r")
    for t, vals in enumerate(mat):
        with open(path_IWTRAM_state / f"IWTRAM_{t-1}.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                if i_neur % Nhidden == 0:
                    f.write(f"// iwt for IN{i_neur//Nhidden} \n")
                f.write(to_hex(val, 2))
                f.write("\n")

    # - IWT2RAM_state: input_weight_2ram_ts
    mat = np.zeros((np.shape(state.IWT2RAM_state)[0], Nin * Nhidden), dtype=int)
    input_weight_2 = np.array(state.IWT2RAM_state).astype(int)
    mat[:, : input_weight_2.shape[1]] = input_weight_2[:, 0 : Nin * Nhidden]

    path_IWT2RAM_state = path / "IWT2RAM_state"
    if not path_IWT2RAM_state.exists():
        makedirs(path_IWT2RAM_state)

    print("Writing IWT2RAM files in input_weight_2", end="\r")
    for t, vals in enumerate(mat):
        with open(path_IWT2RAM_state / f"IWT2RAM_{t-1}.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                if i_neur % Nhidden == 0:
                    f.write(f"// iwt2 for IN{i_neur//Nhidden} \n")
                f.write(to_hex(val, 2))
                f.write("\n")

    # - NDSRAM_state: neuron_dash_syn_ram_ts
    mat = np.zeros(
        (np.shape(state.NDSRAM_state)[0], Nhidden + num_buffer_neurons(Nhidden) + Nout),
        dtype=int,
    )
    neuron_dash_syn = np.array(state.NDSRAM_state).astype(int)
    mat[:, : neuron_dash_syn.shape[1]] = neuron_dash_syn[
        :, 0 : Nhidden + num_buffer_neurons(Nhidden) + Nout
    ]

    path_NDSRAM_state = path / "NDSRAM_state"
    if not path_NDSRAM_state.exists():
        makedirs(path_NDSRAM_state)

    print("Writing NDSRAM files in input_weight", end="\r")
    for t, vals in enumerate(mat):
        with open(path_NDSRAM_state / f"NDSRAM_{t-1}.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 1))
                f.write("\n")

    # - RDS2RAM_state: reservoir_dash_syn_2ram_ts
    mat = np.zeros(
        (np.shape(state.RDS2RAM_state)[0], Nhidden + num_buffer_neurons(Nhidden)),
        dtype=int,
    )
    reservoir_dash_syn_2 = np.array(state.RDS2RAM_state).astype(int)
    mat[:, : reservoir_dash_syn_2.shape[1]] = reservoir_dash_syn_2[
        :, 0 : Nhidden + num_buffer_neurons(Nhidden)
    ]
    path_RDS2RAM_state = path / "RDS2RAM_state"
    if not path_RDS2RAM_state.exists():
        makedirs(path_RDS2RAM_state)

    print("Writing RDS2RAM files in reservoir_dash_syn_2", end="\r")
    for t, vals in enumerate(mat):
        with open(path_RDS2RAM_state / f"RDS2RAM_{t-1}.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 1))
                f.write("\n")

    # - NDMRAM_state: neuron_dash_mem_ram_ts
    mat = np.zeros(
        (np.shape(state.NDMRAM_state)[0], Nhidden + num_buffer_neurons(Nhidden)),
        dtype=int,
    )
    neuron_dash_mem = np.array(state.NDMRAM_state).astype(int)
    mat[:, : neuron_dash_mem.shape[1]] = neuron_dash_mem[
        :, 0 : Nhidden + num_buffer_neurons(Nhidden)
    ]

    path_NDMRAM_state = path / "NDMRAM_state"
    if not path_NDMRAM_state.exists():
        makedirs(path_NDMRAM_state)

    print("Writing NDMRAM files in neuron_dash_mem", end="\r")
    for t, vals in enumerate(mat):
        with open(path_NDMRAM_state / f"NDMRAM_{t-1}.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 1))
                f.write("\n")

    # - NTHRAM_state: neuron_threshold_ram_ts
    mat = np.zeros(
        (np.shape(state.NTHRAM_state)[0], Nhidden + num_buffer_neurons(Nhidden) + Nout),
        dtype=int,
    )
    neuron_threshold = np.array(state.NTHRAM_state).astype(int)
    mat[:, : neuron_threshold.shape[1]] = neuron_threshold[
        :, 0 : Nhidden + num_buffer_neurons(Nhidden) + Nout
    ]

    path_NTHRAM_state = path / "NTHRAM_state"
    if not path_NTHRAM_state.exists():
        makedirs(path_NTHRAM_state)

    print("Writing NTHRAM files in neuron_threshold", end="\r")
    for t, vals in enumerate(mat):
        with open(path_NTHRAM_state / f"NTHRAM_{t-1}.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 4))
                f.write("\n")

    # - RCRAM_state: reservoir_config_ram_ts
    mat = np.zeros(
        (np.shape(state.RCRAM_state)[0], Nhidden + num_buffer_neurons(Nhidden)),
        dtype=int,
    )
    reservoir_config = np.array(state.RCRAM_state).astype(int)
    mat[:, : reservoir_config.shape[1]] = reservoir_config[
        :, 0 : Nhidden + num_buffer_neurons(Nhidden)
    ]
    path_RCRAM_state = path / "RCRAM_state"
    if not path_RCRAM_state.exists():
        makedirs(path_RCRAM_state)

    print("Writing RCRAM files in input_weight", end="\r")
    for t, vals in enumerate(mat):
        with open(path_RCRAM_state / f"RCRAM_{t-1}.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 1))
                f.write("\n")

    # - RARAM_state: reservoir_aliasing_ram_ts
    mat = np.zeros(
        (np.shape(state.RARAM_state)[0], Nhidden + num_buffer_neurons(Nhidden)),
        dtype=int,
    )
    reservoir_aliasing = np.array(state.RARAM_state).astype(int)
    mat[:, : reservoir_aliasing.shape[1]] = reservoir_aliasing[
        :, 0 : Nhidden + num_buffer_neurons(Nhidden)
    ]
    path_RARAM_state = path / "RARAM_state"
    if not path_RARAM_state.exists():
        makedirs(path_RARAM_state)

    print("Writing RARAM files in reservoir_aliasing", end="\r")
    for t, vals in enumerate(mat):
        with open(path_RARAM_state / f"RARAM_{t-1}.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 3))
                f.write("\n")

    # - REFOCRAM_state: reservoir_effective_fanout_count_ram_ts
    mat = np.zeros(
        (np.shape(state.REFOCRAM_state)[0], Nhidden + num_buffer_neurons(Nhidden)),
        dtype=int,
    )
    reservoir_effective_fanout_count = np.array(state.REFOCRAM_state).astype(int)
    mat[
        :, : reservoir_effective_fanout_count.shape[1]
    ] = reservoir_effective_fanout_count[:, 0 : Nhidden + num_buffer_neurons(Nhidden)]
    path_REFOCRAM_state = path / "REFOCRAM_state"
    if not path_REFOCRAM_state.exists():
        makedirs(path_REFOCRAM_state)

    print("Writing REFOCRAM files in reservoir_effective_fanout_count", end="\r")
    for t, vals in enumerate(mat):
        with open(path_REFOCRAM_state / f"REFOCRAM_{t-1}.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 2))
                f.write("\n")

    # - RFORAM_state: recurrent_fanout_ram_ts
    mat = np.zeros(
        (np.shape(state.RFORAM_state)[0], np.shape(state.RFORAM_state)[1]), dtype=int
    )  # 32000
    recurrent_fanout = np.array(state.RFORAM_state).astype(int)
    mat[:, : recurrent_fanout.shape[1]] = recurrent_fanout

    path_RFORAM_state = path / "RFORAM_state"
    if not path_RFORAM_state.exists():
        makedirs(path_RFORAM_state)

    print("Writing RFORAM files in recurrent_fanout", end="\r")
    for t, rforam in enumerate(mat):
        with open(path_RFORAM_state / f"RFORAM_{t-1}.txt", "w+") as f:
            reservoir_fanout_total = 0
            for res_neur_index, fanout_count in enumerate(
                reservoir_effective_fanout_count[t]
            ):
                f.write(f"// rfo of RSN{res_neur_index} \n")
                for fanout_index in range(fanout_count):
                    f.write(to_hex(rforam[reservoir_fanout_total], 3))
                    f.write("\n")
                    reservoir_fanout_total += 1

    # - RWTRAM_state: recurrent_weight_ram_ts
    mat = np.zeros(
        (np.shape(state.RWTRAM_state)[0], np.shape(state.RWTRAM_state)[1]), dtype=int
    )  # 32000
    recurrent_weight = np.array(state.RWTRAM_state).astype(int)
    mat[:, : recurrent_weight.shape[1]] = recurrent_weight

    path_RWTRAM_state = path / "RWTRAM_state"
    if not path_RWTRAM_state.exists():
        makedirs(path_RWTRAM_state)

    print("Writing RWTRAM files in recurrent_weight", end="\r")
    for t, rforam in enumerate(mat):
        with open(path_RWTRAM_state / f"RWTRAM_{t-1}.txt", "w+") as f:
            reservoir_fanout_total = 0
            for res_neur_index, fanout_count in enumerate(
                reservoir_effective_fanout_count[t]
            ):
                f.write(f"// rwt of RSN{res_neur_index} \n")
                for fanout_index in range(fanout_count):
                    f.write(to_hex(rforam[reservoir_fanout_total], 2))
                    f.write("\n")
                    reservoir_fanout_total += 1

    # - RWT2RAM_state: recurrent_weight_2ram_ts
    mat = np.zeros(
        (np.shape(state.RWT2RAM_state)[0], np.shape(state.RWT2RAM_state)[1]), dtype=int
    )  # 32000
    recurrent_weight_2 = np.array(state.RWT2RAM_state).astype(int)
    mat[:, : recurrent_weight_2.shape[1]] = recurrent_weight_2

    path_RWT2RAM_state = path / "RWT2RAM_state"
    if not path_RWT2RAM_state.exists():
        makedirs(path_RWT2RAM_state)

    print("Writing RWT2RAM files in recurrent_weight_2", end="\r")
    for t, rforam in enumerate(mat):
        with open(path_RWT2RAM_state / f"RWT2RAM_{t-1}.txt", "w+") as f:
            reservoir_fanout_total = 0
            for res_neur_index, fanout_count in enumerate(
                reservoir_effective_fanout_count[t]
            ):
                f.write(f"// rwt2 of RSN{res_neur_index} \n")
                for fanout_index in range(fanout_count):
                    f.write(to_hex(rforam[reservoir_fanout_total], 2))
                    f.write("\n")
                    reservoir_fanout_total += 1

    # - OWTRAM_state: output_weight_ram_ts
    mat = np.zeros(
        (
            np.shape(state.OWTRAM_state)[0],
            Nout * (Nhidden + num_buffer_neurons(Nhidden)),
        ),
        dtype=int,
    )
    output_weight = np.array(state.OWTRAM_state).astype(int)
    mat[:, : output_weight.shape[1]] = output_weight[
        :, 0 : Nout * (Nhidden + num_buffer_neurons(Nhidden))
    ]

    path_OWTRAM_state = path / "OWTRAM_state"
    if not path_OWTRAM_state.exists():
        makedirs(path_OWTRAM_state)

    print("Writing OWTRAM files in output_weight", end="\r")
    for t, vals in enumerate(mat):
        with open(path_OWTRAM_state / f"OWTRAM_{t-1}.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                if i_neur % Nout == 0:
                    f.write(f"// owt for RSN{i_neur//Nout} \n")
                f.write(to_hex(val, 2))
                f.write("\n")


def export_last_state(
    path: Union[Path, str],
    config: XyloConfiguration,
    inp_spks: np.ndarray,
    state: XyloState,
) -> None:
    """
    Export the final state of a Xylo network, evolved over an input

    This function will produce a series of RAM files, per time-step, containing the recorded state evolution of a Xylo network. The files will be written to the directory ``path``, which will be created if necessary.

    Args:
        path (Path): The directory to export the state to
        config (XyloConfiguration): The configuration of the Xylo network
        inp_spks (np.ndarray): The input spikes for this simulation
        state (XyloState): A temporal state of a Xylo network to export
    """
    # - Make `path` a path
    path = Path(path)

    # - Determine network size
    inp_size = np.shape(config.input.weights)[0]
    num_neurons = np.shape(config.reservoir.weights)[1]
    readout_size = np.shape(config.readout.weights)[1]
    size_total = num_neurons + readout_size

    Nin = inp_size
    Nhidden = num_neurons
    Nout = readout_size

    # rspkram
    mat = np.zeros((np.shape(state.Spikes_hid)[0], num_neurons), dtype=int)
    spks = np.array(state.Spikes_hid).astype(int)

    if len(spks) > 0:
        mat[:, : spks.shape[1]] = spks

        path_spkr = path / "spk_res"
        if not path_spkr.exists():
            makedirs(path_spkr)

        print("Writing rspkram files in spk_res", end="\r")
        for t, spks in enumerate(mat):
            with open(
                path_spkr / "rspkram_last.txt", "w+"
            ) as f:  # t-1 because add 1 before evolve
                for val in spks:
                    f.write(to_hex(val, 2))
                    f.write("\n")

    # ospkram
    mat = np.zeros((np.shape(state.Spikes_out)[0], readout_size), dtype=int)
    spks = np.array(state.Spikes_out).astype(int)

    if len(spks) > 0:
        mat[:, : spks.shape[1]] = spks[:, 0:Nout]  # [:, 0:Nout] for the last step ???

        path_spko = path / "spk_out"
        if not path_spko.exists():
            makedirs(path_spko)

        print("Writing ospkram files in spk_out", end="\r")
        for t, spks in enumerate(mat):
            with open(path_spko / "ospkram_last.txt", "w+") as f:
                for val in spks:
                    f.write(to_hex(val, 2))
                    f.write("\n")

    # nscram
    mat = np.zeros((np.shape(state.I_syn_hid)[0], size_total), dtype=int)
    isyns = np.array(state.I_syn_hid).astype(int)
    mat[:, : isyns.shape[1]] = isyns
    isyns_out = np.array(state.I_syn_out).astype(int)
    mat[:, num_neurons : num_neurons + isyns_out.shape[1]] = isyns_out

    path_isyn = path / "isyn"
    if not path_isyn.exists():
        makedirs(path_isyn)

    print("Writing nscram files in isyn", end="\r")
    for t, vals in enumerate(mat):
        with open(path_isyn / "nscram_last.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 4))
                f.write("\n")

    # nsc2ram --> rsc2ram
    if not hasattr(state, "I_syn2_hid"):
        mat = np.zeros((0, num_neurons), int)
    else:
        mat = np.zeros((np.shape(state.I_syn2_hid)[0], num_neurons), dtype=int)
        isyns2 = np.array(state.I_syn2_hid).astype(int)
        mat[:, : isyns2.shape[1]] = isyns2

    path_isyn2 = path / "isyn2"
    if not path_isyn2.exists():
        makedirs(path_isyn2)

    print("Writing rsc2ram files in isyn2", end="\r")
    for t, vals in enumerate(mat):
        with open(path_isyn2 / "rsc2ram_last.txt", "w+") as f:
            for val in vals:
                f.write(to_hex(val, 4))
                f.write("\n")

    # nmpram
    mat = np.zeros((np.shape(state.V_mem_hid)[0], size_total), dtype=int)
    vmems = np.array(state.V_mem_hid).astype(int)
    mat[:, : vmems.shape[1]] = vmems
    vmems_out = np.array(state.V_mem_out).astype(int)
    mat[:, num_neurons : num_neurons + vmems_out.shape[1]] = vmems_out

    path_vmem = path / "vmem"
    if not path_vmem.exists():
        makedirs(path_vmem)

    print("Writing nmpram files in vmem", end="\r")
    for t, vals in enumerate(mat):
        with open(path_vmem / "nmpram_last.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 4))
                f.write("\n")

    if inp_spks is not None:
        # input spikes
        path_spki = path / "spk_in"
        if not path_spki.exists():
            makedirs(path_spki)

        print("Writing inp_spks.txt", end="\r")
        with open(path_spki / "inp_spks.txt", "w+") as f:
            idle = -1
            for t, chans in enumerate(inp_spks):
                idle += 1
                if not np.all(chans == 0):
                    f.write(f"// time step {t}\n")
                    if idle > 0:
                        f.write(f"idle {idle}\n")
                    idle = 0
                    for chan, num_spikes in enumerate(chans):
                        for _ in range(num_spikes):
                            f.write(f"wr IN{to_hex(chan, 1)}\n")

    # - Save config RAM, not export dummy neuron
    # IWTRAM_state: input_weight_ram_ts
    mat = np.zeros((np.shape(state.IWTRAM_state)[0], Nin * Nhidden), dtype=int)
    input_weight = np.array(state.IWTRAM_state).astype(int)
    mat[:, : input_weight.shape[1]] = input_weight[:, 0 : Nin * Nhidden]

    path_IWTRAM_state = path / "IWTRAM_state"
    if not path_IWTRAM_state.exists():
        makedirs(path_IWTRAM_state)

    print("Writing IWTRAM files in input_weight", end="\r")
    for t, vals in enumerate(mat):
        with open(path_IWTRAM_state / "IWTRAM_last.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                if i_neur % Nhidden == 0:
                    f.write(f"// iwt for IN{i_neur//Nhidden} \n")
                f.write(to_hex(val, 2))
                f.write("\n")

    # IWT2RAM_state input_weight_2ram_ts
    mat = np.zeros((np.shape(state.IWT2RAM_state)[0], Nin * Nhidden), dtype=int)
    input_weight_2 = np.array(state.IWT2RAM_state).astype(int)
    mat[:, : input_weight_2.shape[1]] = input_weight_2[:, 0 : Nin * Nhidden]

    path_IWT2RAM_state = path / "IWT2RAM_state"
    if not path_IWT2RAM_state.exists():
        makedirs(path_IWT2RAM_state)

    print("Writing IWT2RAM files in input_weight_2", end="\r")
    for t, vals in enumerate(mat):
        with open(path_IWT2RAM_state / "IWT2RAM_last.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                if i_neur % Nhidden == 0:
                    f.write(f"// iwt2 for IN{i_neur//Nhidden} \n")
                f.write(to_hex(val, 2))
                f.write("\n")

    # NDSRAM_state:neuron_dash_syn_ram_ts
    mat = np.zeros(
        (np.shape(state.NDSRAM_state)[0], Nhidden + num_buffer_neurons(Nhidden) + Nout),
        dtype=int,
    )
    neuron_dash_syn = np.array(state.NDSRAM_state).astype(int)
    mat[:, : neuron_dash_syn.shape[1]] = neuron_dash_syn[
        :, 0 : Nhidden + num_buffer_neurons(Nhidden) + Nout
    ]

    path_NDSRAM_state = path / "NDSRAM_state"
    if not path_NDSRAM_state.exists():
        makedirs(path_NDSRAM_state)

    print("Writing NDSRAM files in input_weight", end="\r")
    for t, vals in enumerate(mat):
        with open(path_NDSRAM_state / "NDSRAM_last.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 1))
                f.write("\n")

    # RDS2RAM_state: reservoir_dash_syn_2ram_ts
    mat = np.zeros(
        (np.shape(state.RDS2RAM_state)[0], Nhidden + num_buffer_neurons(Nhidden)),
        dtype=int,
    )
    reservoir_dash_syn_2 = np.array(state.RDS2RAM_state).astype(int)
    mat[:, : reservoir_dash_syn_2.shape[1]] = reservoir_dash_syn_2[
        :, 0 : Nhidden + num_buffer_neurons(Nhidden)
    ]
    path_RDS2RAM_state = path / "RDS2RAM_state"
    if not path_RDS2RAM_state.exists():
        makedirs(path_RDS2RAM_state)

    print("Writing RDS2RAM files in reservoir_dash_syn_2", end="\r")
    for t, vals in enumerate(mat):
        with open(path_RDS2RAM_state / "RDS2RAM_last.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 1))
                f.write("\n")

    # NDMRAM_state: neuron_dash_mem_ram_ts
    mat = np.zeros(
        (np.shape(state.NDMRAM_state)[0], Nhidden + num_buffer_neurons(Nhidden)),
        dtype=int,
    )
    neuron_dash_mem = np.array(state.NDMRAM_state).astype(int)
    mat[:, : neuron_dash_mem.shape[1]] = neuron_dash_mem[
        :, 0 : Nhidden + num_buffer_neurons(Nhidden)
    ]

    path_NDMRAM_state = path / "NDMRAM_state"
    if not path_NDMRAM_state.exists():
        makedirs(path_NDMRAM_state)

    print("Writing NDMRAM files in neuron_dash_mem", end="\r")
    for t, vals in enumerate(mat):
        with open(path_NDMRAM_state / "NDMRAM_last.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 1))
                f.write("\n")

    # NTHRAM_state: neuron_threshold_ram_ts
    mat = np.zeros(
        (np.shape(state.NTHRAM_state)[0], Nhidden + num_buffer_neurons(Nhidden) + Nout),
        dtype=int,
    )
    neuron_threshold = np.array(state.NTHRAM_state).astype(int)
    mat[:, : neuron_threshold.shape[1]] = neuron_threshold[
        :, 0 : Nhidden + num_buffer_neurons(Nhidden) + Nout
    ]

    path_NTHRAM_state = path / "NTHRAM_state"
    if not path_NTHRAM_state.exists():
        makedirs(path_NTHRAM_state)

    print("Writing NTHRAM files in neuron_threshold", end="\r")
    for t, vals in enumerate(mat):
        with open(path_NTHRAM_state / "NTHRAM_last.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 4))
                f.write("\n")

    # RCRAM_state: reservoir_config_ram_ts
    mat = np.zeros(
        (np.shape(state.RCRAM_state)[0], Nhidden + num_buffer_neurons(Nhidden)),
        dtype=int,
    )
    reservoir_config = np.array(state.RCRAM_state).astype(int)
    mat[:, : reservoir_config.shape[1]] = reservoir_config[
        :, 0 : Nhidden + num_buffer_neurons(Nhidden)
    ]
    path_RCRAM_state = path / "RCRAM_state"
    if not path_RCRAM_state.exists():
        makedirs(path_RCRAM_state)

    print("Writing RCRAM files in input_weight", end="\r")
    for t, vals in enumerate(mat):
        with open(path_RCRAM_state / "RCRAM_last.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 1))
                f.write("\n")

    # RARAM_state: reservoir_aliasing_ram_ts
    mat = np.zeros(
        (np.shape(state.RARAM_state)[0], Nhidden + num_buffer_neurons(Nhidden)),
        dtype=int,
    )
    reservoir_aliasing = np.array(state.RARAM_state).astype(int)
    mat[:, : reservoir_aliasing.shape[1]] = reservoir_aliasing[
        :, 0 : Nhidden + num_buffer_neurons(Nhidden)
    ]
    path_RARAM_state = path / "RARAM_state"
    if not path_RARAM_state.exists():
        makedirs(path_RARAM_state)

    print("Writing RARAM files in reservoir_aliasing", end="\r")
    for t, vals in enumerate(mat):
        with open(path_RARAM_state / "RARAM_last.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 3))
                f.write("\n")

    # REFOCRAM_state: reservoir_effective_fanout_count_ram_ts
    mat = np.zeros(
        (np.shape(state.REFOCRAM_state)[0], Nhidden + num_buffer_neurons(Nhidden)),
        dtype=int,
    )
    reservoir_effective_fanout_count = np.array(state.REFOCRAM_state).astype(int)
    mat[
        :, : reservoir_effective_fanout_count.shape[1]
    ] = reservoir_effective_fanout_count[:, 0 : Nhidden + num_buffer_neurons(Nhidden)]
    path_REFOCRAM_state = path / "REFOCRAM_state"
    if not path_REFOCRAM_state.exists():
        makedirs(path_REFOCRAM_state)

    print("Writing REFOCRAM files in reservoir_effective_fanout_count", end="\r")
    for t, vals in enumerate(mat):
        with open(path_REFOCRAM_state / "REFOCRAM_last.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                f.write(to_hex(val, 2))
                f.write("\n")

    # RFORAM_state: recurrent_fanout_ram_ts
    mat = np.zeros(
        (np.shape(state.RFORAM_state)[0], np.shape(state.RFORAM_state)[1]), dtype=int
    )  # 32000
    recurrent_fanout = np.array(state.RFORAM_state).astype(int)
    mat[:, : recurrent_fanout.shape[1]] = recurrent_fanout

    path_RFORAM_state = path / "RFORAM_state"
    if not path_RFORAM_state.exists():
        makedirs(path_RFORAM_state)

    print("Writing RFORAM files in recurrent_fanout", end="\r")
    for t, rforam in enumerate(mat):
        with open(path_RFORAM_state / "RFORAM_last.txt", "w+") as f:
            reservoir_fanout_total = 0
            for res_neur_index, fanout_count in enumerate(
                reservoir_effective_fanout_count[t]
            ):
                f.write(f"// rfo of RSN{res_neur_index} \n")
                for fanout_index in range(fanout_count):
                    f.write(to_hex(rforam[reservoir_fanout_total], 3))
                    f.write("\n")
                    reservoir_fanout_total += 1

    # RWTRAM_state: recurrent_weight_ram_ts
    mat = np.zeros(
        (np.shape(state.RWTRAM_state)[0], np.shape(state.RWTRAM_state)[1]), dtype=int
    )  # 32000
    recurrent_weight = np.array(state.RWTRAM_state).astype(int)
    mat[:, : recurrent_weight.shape[1]] = recurrent_weight

    path_RWTRAM_state = path / "RWTRAM_state"
    if not path_RWTRAM_state.exists():
        makedirs(path_RWTRAM_state)

    print("Writing RWTRAM files in recurrent_weight", end="\r")
    for t, rforam in enumerate(mat):
        with open(path_RWTRAM_state / "RWTRAM_last.txt", "w+") as f:
            reservoir_fanout_total = 0
            for res_neur_index, fanout_count in enumerate(
                reservoir_effective_fanout_count[t]
            ):
                f.write(f"// rwt of RSN{res_neur_index} \n")
                for fanout_index in range(fanout_count):
                    f.write(to_hex(rforam[reservoir_fanout_total], 2))
                    f.write("\n")
                    reservoir_fanout_total += 1

    # RWT2RAM_state: recurrent_weight_2ram_ts
    mat = np.zeros(
        (np.shape(state.RWT2RAM_state)[0], np.shape(state.RWT2RAM_state)[1]), dtype=int
    )  # 32000
    recurrent_weight_2 = np.array(state.RWT2RAM_state).astype(int)
    mat[:, : recurrent_weight_2.shape[1]] = recurrent_weight_2

    path_RWT2RAM_state = path / "RWT2RAM_state"
    if not path_RWT2RAM_state.exists():
        makedirs(path_RWT2RAM_state)

    print("Writing RWT2RAM files in recurrent_weight_2", end="\r")
    for t, rforam in enumerate(mat):
        with open(path_RWT2RAM_state / "RWT2RAM_last.txt", "w+") as f:
            reservoir_fanout_total = 0
            for res_neur_index, fanout_count in enumerate(
                reservoir_effective_fanout_count[t]
            ):
                f.write(f"// rwt2 of RSN{res_neur_index} \n")
                for fanout_index in range(fanout_count):
                    f.write(to_hex(rforam[reservoir_fanout_total], 2))
                    f.write("\n")
                    reservoir_fanout_total += 1

    # OWTRAM_state: output_weight_ram_ts
    mat = np.zeros(
        (
            np.shape(state.OWTRAM_state)[0],
            Nout * (Nhidden + num_buffer_neurons(Nhidden)),
        ),
        dtype=int,
    )
    output_weight = np.array(state.OWTRAM_state).astype(int)
    mat[:, : output_weight.shape[1]] = output_weight[
        :, 0 : Nout * (Nhidden + num_buffer_neurons(Nhidden))
    ]

    path_OWTRAM_state = path / "OWTRAM_state"
    if not path_OWTRAM_state.exists():
        makedirs(path_OWTRAM_state)

    print("Writing OWTRAM files in output_weight", end="\r")
    for t, vals in enumerate(mat):
        with open(path_OWTRAM_state / "OWTRAM_last.txt", "w+") as f:
            for i_neur, val in enumerate(vals):
                if i_neur % Nout == 0:
                    f.write(f"// owt for RSN{i_neur//Nout} \n")
                f.write(to_hex(val, 2))
                f.write("\n")
