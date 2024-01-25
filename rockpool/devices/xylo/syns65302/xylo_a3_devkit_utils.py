"""
Low-level device kit utilities for the Xylo Audio 3 HDK
"""

import samna
from samna.xyloAudio3.configuration import XyloConfiguration

# - Other imports
import time
import numpy as np

# - Useful constants for XA3
from . import ram, reg, constants


# - Typing and useful proxy types
from typing import List, Optional, NamedTuple, Tuple, Union

XyloAudio3ReadBuffer = samna.BasicSinkNode_xylo_audio3_event_output_event
XyloAudio3WriteBuffer = samna.BasicSourceNode_xylo_audio3_event_input_event
ReadoutEvent = samna.xyloAudio3.event.Readout
XyloAudio3HDK = samna.xyloAudio3Boards.XyloAudio3TestBoard


class XyloState(NamedTuple):
    """
    `.NamedTuple` that encapsulates a recorded Xylo HDK state
    """

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


def find_xylo_a3_boards() -> List[XyloAudio3HDK]:
    """
    Search for and return a list of Xylo A3 HDKs

    Iterate over devices and search for Xylo A3 HDK nodes. Return a list of available A3 HDKs, or an empty list if none are found.

    Returns:
        List[XyloAudio3HDK]: A (possibly empty) list of Xylo A3 HDK nodes.
    """

    # - Get a list of devices
    device_list = samna.device.get_all_devices()

    # - Search for a xylo dev kit
    imu_hdk_list = [
        samna.device.open_device(d)
        for d in device_list
        if d.device_type_name == "XyloAudio3TestBoard"
    ]

    return imu_hdk_list


def new_xylo_read_buffer(
    hdk: XyloAudio3HDK,
) -> XyloAudio3ReadBuffer:
    """
    Create and connect a new buffer to read from a Xylo HDK

    Args:
        hdk (XyloAudio3HDK): A Xylo HDK to create a new buffer for

    Returns:
        XyloAudio3ReadBuffer: A connected event read buffer
    """
    return samna.graph.sink_from(hdk.get_model_source_node())


def new_xylo_write_buffer(hdk: XyloAudio3HDK) -> XyloAudio3WriteBuffer:
    """
    Create a new buffer for writing events to a Xylo HDK

    Args:
        hdk (XyloAudio3HDK): A Xylo HDK to create a new buffer for

    Returns:
        XyloAudio3WriteBuffer: A connected event write buffer
    """
    return samna.graph.source_to(hdk.get_model_sink_node())


def config_pdm_h_regs(write_buffer):
    # copied from dig_full_path_test_pdm_external_rise_sample_with_nn/dfe_full_path_test.sysint.stim
    write_register(write_buffer, reg.pdm_h_reg0, 0x0229_FF8D)
    write_register(write_buffer, reg.pdm_h_reg1, 0x0F4C_F856)
    write_register(write_buffer, reg.pdm_h_reg2, 0x0CF2_4E17)
    write_register(write_buffer, reg.pdm_h_reg3, 0x0215_F8E2)
    write_register(write_buffer, reg.pdm_h_reg4, 0x0000_0000)
    write_register(write_buffer, reg.pdm_h_reg5, 0xF8E2_0215)
    write_register(write_buffer, reg.pdm_h_reg6, 0x4E17_0CF2)
    write_register(write_buffer, reg.pdm_h_reg7, 0xF856_0F4C)
    write_register(write_buffer, reg.pdm_h_reg8, 0xFF8D_0229)
    write_register(write_buffer, reg.pdm_h_reg9, 0x01FE_0000)
    write_register(write_buffer, reg.pdm_h_reg10, 0x0AAF_F973)
    write_register(write_buffer, reg.pdm_h_reg11, 0x11BA_4DE7)
    write_register(write_buffer, reg.pdm_h_reg12, 0x023A_F7D0)
    write_register(write_buffer, reg.pdm_h_reg13, 0x0000_FF95)
    write_register(write_buffer, reg.pdm_h_reg14, 0xFA06_01E4)
    write_register(write_buffer, reg.pdm_h_reg15, 0x4D86_0885)
    write_register(write_buffer, reg.pdm_h_reg16, 0xF750_143C)
    write_register(write_buffer, reg.pdm_h_reg17, 0xFF9E_0247)
    write_register(write_buffer, reg.pdm_h_reg18, 0x01C9_0000)
    write_register(write_buffer, reg.pdm_h_reg19, 0x0674_FA9A)
    write_register(write_buffer, reg.pdm_h_reg20, 0x16CE_4CF5)
    write_register(write_buffer, reg.pdm_h_reg21, 0x0250_F6DA)
    write_register(write_buffer, reg.pdm_h_reg22, 0x0000_FFA6)
    write_register(write_buffer, reg.pdm_h_reg23, 0xFB2F_01AD)
    write_register(write_buffer, reg.pdm_h_reg24, 0x4C35_047F)
    write_register(write_buffer, reg.pdm_h_reg25, 0xF66D_196F)
    write_register(write_buffer, reg.pdm_h_reg26, 0xFFB0_0254)
    write_register(write_buffer, reg.pdm_h_reg27, 0x018F_0000)
    write_register(write_buffer, reg.pdm_h_reg28, 0x02A7_FBC2)
    write_register(write_buffer, reg.pdm_h_reg29, 0x1C1B_4B46)
    write_register(write_buffer, reg.pdm_h_reg30, 0x0253_F60D)
    write_register(write_buffer, reg.pdm_h_reg31, 0x0000_FFBA)
    write_register(write_buffer, reg.pdm_h_reg32, 0xFC53_0171)
    write_register(write_buffer, reg.pdm_h_reg33, 0x4A2B_00EC)
    write_register(write_buffer, reg.pdm_h_reg34, 0xF5BA_1ED1)
    write_register(write_buffer, reg.pdm_h_reg35, 0xFFC5_024C)
    write_register(write_buffer, reg.pdm_h_reg36, 0x0152_0000)
    write_register(write_buffer, reg.pdm_h_reg37, 0xFF50_FCE1)
    write_register(write_buffer, reg.pdm_h_reg38, 0x218D_48E3)
    write_register(write_buffer, reg.pdm_h_reg39, 0x023F_F577)
    write_register(write_buffer, reg.pdm_h_reg40, 0x0000_FFD1)
    write_register(write_buffer, reg.pdm_h_reg41, 0xFD6A_0134)
    write_register(write_buffer, reg.pdm_h_reg42, 0x4772_FDD1)
    write_register(write_buffer, reg.pdm_h_reg43, 0xF544_244D)
    write_register(write_buffer, reg.pdm_h_reg44, 0xFFDE_022C)
    write_register(write_buffer, reg.pdm_h_reg45, 0x0115_0000)
    write_register(write_buffer, reg.pdm_h_reg46, 0xFC72_FDEF)
    write_register(write_buffer, reg.pdm_h_reg47, 0x270E_45D9)
    write_register(write_buffer, reg.pdm_h_reg48, 0x0211_F524)
    write_register(write_buffer, reg.pdm_h_reg49, 0x0000_FFEC)
    write_register(write_buffer, reg.pdm_h_reg50, 0xFE6E_00F8)
    write_register(write_buffer, reg.pdm_h_reg51, 0x4419_FB33)
    write_register(write_buffer, reg.pdm_h_reg52, 0xF518_29CC)
    write_register(write_buffer, reg.pdm_h_reg53, 0xFFFC_01EF)
    write_register(write_buffer, reg.pdm_h_reg54, 0x00DA_0000)
    write_register(write_buffer, reg.pdm_h_reg55, 0xFA13_FEE6)
    write_register(write_buffer, reg.pdm_h_reg56, 0x2C86_4235)
    write_register(write_buffer, reg.pdm_h_reg57, 0x01C6_F522)
    write_register(write_buffer, reg.pdm_h_reg58, 0x0000_000B)
    write_register(write_buffer, reg.pdm_h_reg59, 0xFF58_00BE)
    write_register(write_buffer, reg.pdm_h_reg60, 0x4030_F911)
    write_register(write_buffer, reg.pdm_h_reg61, 0xF542_2F37)
    write_register(write_buffer, reg.pdm_h_reg62, 0x001D_0195)
    write_register(write_buffer, reg.pdm_h_reg63, 0x00A3_0000)
    write_register(write_buffer, reg.pdm_h_reg64, 0xF82F_FFC2)
    write_register(write_buffer, reg.pdm_h_reg65, 0x31DD_3E0B)
    write_register(write_buffer, reg.pdm_h_reg66, 0x015B_F57B)
    write_register(write_buffer, reg.pdm_h_reg67, 0x0000_0030)
    write_register(write_buffer, reg.pdm_h_reg68, 0x0023_008A)
    write_register(write_buffer, reg.pdm_h_reg69, 0x3BC9_F76B)
    write_register(write_buffer, reg.pdm_h_reg70, 0xF5CE_3475)
    write_register(write_buffer, reg.pdm_h_reg71, 0x0044_011A)
    write_register(write_buffer, reg.pdm_h_reg72, 0x0071_0000)
    write_register(write_buffer, reg.pdm_h_reg73, 0xF6C5_007E)
    write_register(write_buffer, reg.pdm_h_reg74, 0x36FC_396E)
    write_register(write_buffer, reg.pdm_h_reg75, 0x00D0_F63C)
    write_register(write_buffer, reg.pdm_h_reg76, 0x0000_005A)
    write_register(write_buffer, reg.pdm_h_reg77, 0x00D0_005A)
    write_register(write_buffer, reg.pdm_h_reg78, 0x36FC_F63C)
    write_register(write_buffer, reg.pdm_h_reg79, 0xF6C5_396E)
    write_register(write_buffer, reg.pdm_h_reg80, 0x0071_007E)
    write_register(write_buffer, reg.pdm_h_reg81, 0x0044_0000)
    write_register(write_buffer, reg.pdm_h_reg82, 0xF5CE_011A)
    write_register(write_buffer, reg.pdm_h_reg83, 0x3BC9_3475)
    write_register(write_buffer, reg.pdm_h_reg84, 0x0023_F76B)
    write_register(write_buffer, reg.pdm_h_reg85, 0x0000_008A)
    write_register(write_buffer, reg.pdm_h_reg86, 0x015B_0030)
    write_register(write_buffer, reg.pdm_h_reg87, 0x31DD_F57B)
    write_register(write_buffer, reg.pdm_h_reg88, 0xF82F_3E0B)
    write_register(write_buffer, reg.pdm_h_reg89, 0x00A3_FFC2)
    write_register(write_buffer, reg.pdm_h_reg90, 0x001D_0000)
    write_register(write_buffer, reg.pdm_h_reg91, 0xF542_0195)
    write_register(write_buffer, reg.pdm_h_reg92, 0x4030_2F37)
    write_register(write_buffer, reg.pdm_h_reg93, 0xFF58_F911)
    write_register(write_buffer, reg.pdm_h_reg94, 0x0000_00BE)
    write_register(write_buffer, reg.pdm_h_reg95, 0x01C6_000B)
    write_register(write_buffer, reg.pdm_h_reg96, 0x2C86_F522)
    write_register(write_buffer, reg.pdm_h_reg97, 0xFA13_4235)
    write_register(write_buffer, reg.pdm_h_reg98, 0x00DA_FEE6)
    write_register(write_buffer, reg.pdm_h_reg99, 0xFFFC_0000)
    write_register(write_buffer, reg.pdm_h_reg100, 0xF518_01EF)
    write_register(write_buffer, reg.pdm_h_reg101, 0x4419_29CC)
    write_register(write_buffer, reg.pdm_h_reg102, 0xFE6E_FB33)
    write_register(write_buffer, reg.pdm_h_reg103, 0x0000_00F8)
    write_register(write_buffer, reg.pdm_h_reg104, 0x0211_FFEC)
    write_register(write_buffer, reg.pdm_h_reg105, 0x270E_F524)
    write_register(write_buffer, reg.pdm_h_reg106, 0xFC72_45D9)
    write_register(write_buffer, reg.pdm_h_reg107, 0x0115_FDEF)
    write_register(write_buffer, reg.pdm_h_reg108, 0xFFDE_0000)
    write_register(write_buffer, reg.pdm_h_reg109, 0xF544_022C)
    write_register(write_buffer, reg.pdm_h_reg110, 0x4772_244D)
    write_register(write_buffer, reg.pdm_h_reg111, 0xFD6A_FDD1)
    write_register(write_buffer, reg.pdm_h_reg112, 0x0000_0134)
    write_register(write_buffer, reg.pdm_h_reg113, 0x023F_FFD1)
    write_register(write_buffer, reg.pdm_h_reg114, 0x218D_F577)
    write_register(write_buffer, reg.pdm_h_reg115, 0xFF50_48E3)
    write_register(write_buffer, reg.pdm_h_reg116, 0x0152_FCE1)
    write_register(write_buffer, reg.pdm_h_reg117, 0xFFC5_0000)
    write_register(write_buffer, reg.pdm_h_reg118, 0xF5BA_024C)
    write_register(write_buffer, reg.pdm_h_reg119, 0x4A2B_1ED1)
    write_register(write_buffer, reg.pdm_h_reg120, 0xFC53_00EC)
    write_register(write_buffer, reg.pdm_h_reg121, 0x0000_0171)
    write_register(write_buffer, reg.pdm_h_reg122, 0x0253_FFBA)
    write_register(write_buffer, reg.pdm_h_reg123, 0x1C1B_F60D)
    write_register(write_buffer, reg.pdm_h_reg124, 0x02A7_4B46)
    write_register(write_buffer, reg.pdm_h_reg125, 0x018F_FBC2)
    write_register(write_buffer, reg.pdm_h_reg126, 0xFFB0_0000)
    write_register(write_buffer, reg.pdm_h_reg127, 0xF66D_0254)
    write_register(write_buffer, reg.pdm_h_reg128, 0x4C35_196F)
    write_register(write_buffer, reg.pdm_h_reg129, 0xFB2F_047F)
    write_register(write_buffer, reg.pdm_h_reg130, 0x0000_01AD)
    write_register(write_buffer, reg.pdm_h_reg131, 0x0250_FFA6)
    write_register(write_buffer, reg.pdm_h_reg132, 0x16CE_F6DA)
    write_register(write_buffer, reg.pdm_h_reg133, 0x0674_4CF5)
    write_register(write_buffer, reg.pdm_h_reg134, 0x01C9_FA9A)
    write_register(write_buffer, reg.pdm_h_reg135, 0xFF9E_0000)
    write_register(write_buffer, reg.pdm_h_reg136, 0xF750_0247)
    write_register(write_buffer, reg.pdm_h_reg137, 0x4D86_143C)
    write_register(write_buffer, reg.pdm_h_reg138, 0xFA06_0885)
    write_register(write_buffer, reg.pdm_h_reg139, 0x0000_01E4)
    write_register(write_buffer, reg.pdm_h_reg140, 0x023A_FF95)
    write_register(write_buffer, reg.pdm_h_reg141, 0x11BA_F7D0)
    write_register(write_buffer, reg.pdm_h_reg142, 0x0AAF_4DE7)
    write_register(write_buffer, reg.pdm_h_reg143, 0x01FE_F973)


def update_register_field(read_buffer, write_buffer, addr, lsb_pos, msb_pos, val):
    data = read_register(read_buffer, write_buffer, addr)[0]
    data_h = data >> (msb_pos + 1)
    data_l = data & (2**lsb_pos - 1)
    data = (data_h << (msb_pos + 1)) + (val << lsb_pos) + data_l
    write_register(write_buffer, addr, data)


def config_pdm_clk(read_buffer, write_buffer, clk_div=1, debug=0):
    update_register_field(
        read_buffer,
        write_buffer,
        reg.clk_div,
        reg.clk_div__sdm__pos_lsb,
        reg.clk_div__sdm__pos_msb,
        max((clk_div >> 1) - 1, 0),
    )
    print(
        "clk_div:  0x" + format(read_register(reg.clk_div), "_X")
    ) if debug >= 1 else None
    update_register_field(
        read_buffer,
        write_buffer,
        reg.clk_ctrl,
        reg.clk_ctrl__sdm__pos,
        reg.clk_ctrl__sdm__pos,
        1,
    )
    print(
        "clk_ctrl: 0x" + format(read_register(reg.clk_ctrl), "_X")
    ) if debug >= 1 else None


def initialise_xylo_hdk(
    hdk: XyloAudio3HDK,
    read_buffer: XyloAudio3ReadBuffer,
    write_buffer: XyloAudio3WriteBuffer,
    sleep_time: float = 50e-3,
    pdm_clk_dir: int = 0,
    pdm_clk_edge: int = 1,
) -> None:
    """
    Initialise the Xylo IMU HDK

    Args:
        write_buffer (XyloAudio3WriteBuffer): A write buffer connected to a Xylo HDK to initialise
    """
    ioc = hdk.get_io_control_module()
    io = hdk.get_io_module()

    ioc.write_config(0, 0x0)  # power off
    time.sleep(sleep_time)
    io.write_config(0x0008, 0)  # main clock disable
    time.sleep(sleep_time)
    ioc.write_config(0, 0x1B)  # power on
    time.sleep(sleep_time)
    io.write_config(0x0008, 1)  # main clock enable
    time.sleep(sleep_time)


def enable_pdm_input(
    hdk: XyloAudio3HDK,
    read_buffer: XyloAudio3ReadBuffer,
    write_buffer: XyloAudio3WriteBuffer,
) -> None:
    raise NotImplementedError


def enable_saer_i(
    hdk: XyloAudio3HDK,
    read_buffer: XyloAudio3ReadBuffer,
    write_buffer: XyloAudio3WriteBuffer,
) -> None:
    io = hdk.get_io_module()

    # set SAER clock
    io.write_config(0x0020, 0)  # saer clock msw
    io.write_config(0x0021, 3)  # saer clock lsw
    io.write_config(0x0022, 1)  # saer clock enable
    io.write_config(0x0026, 0)  # stif_select: saer

    # FPGA drive PDM_DATA pin (for SAER input)
    io.write_config(0x0012, 1)
    # pdm port write enable
    io.write_config(0x0013, 1)

    # setup SAER_I pads, PAD_CTRL.CTRL[012] = 2
    # write_register(reg.pad_ctrl, 0x0000_0222)
    update_register_field(
        read_buffer,
        write_buffer,
        reg.pad_ctrl,
        reg.pad_ctrl__ctrl0__pos_lsb,
        reg.pad_ctrl__ctrl0__pos_msb,
        2,
    )
    update_register_field(
        read_buffer,
        write_buffer,
        reg.pad_ctrl,
        reg.pad_ctrl__ctrl1__pos_lsb,
        reg.pad_ctrl__ctrl1__pos_msb,
        2,
    )
    update_register_field(
        read_buffer,
        write_buffer,
        reg.pad_ctrl,
        reg.pad_ctrl__ctrl2__pos_lsb,
        reg.pad_ctrl__ctrl2__pos_msb,
        2,
    )
    # DBG_CTRL1.SPK_SRC_SEL = 1, DBG_CTRL1.STIF_SEL = 0
    # write_register(reg.dbg_ctrl1, 0x0004_0000)
    update_register_field(
        read_buffer,
        write_buffer,
        reg.dbg_ctrl1,
        reg.dbg_ctrl1__spk_src_sel__pos,
        reg.dbg_ctrl1__spk_src_sel__pos,
        1,
    )
    update_register_field(
        read_buffer,
        write_buffer,
        reg.dbg_ctrl1,
        reg.dbg_ctrl1__stif_sel__pos,
        reg.dbg_ctrl1__stif_sel__pos,
        0,
    )


def advance_time_step(write_buffer: XyloAudio3WriteBuffer) -> None:
    """
    Take a single manual time-step on a Xylo HDK

    Args:
        write_buffer (XyloAudio3WriteBuffer): A write buffer connected to the Xylo HDK
    """
    write_buffer.write([samna.xyloAudio3.event.TriggerProcessing()])


def is_xylo_ready(
    read_buffer: XyloAudio3ReadBuffer, write_buffer: XyloAudio3WriteBuffer
) -> bool:
    stat2 = read_register(read_buffer, write_buffer, reg.stat2)[0]
    return stat2 & (1 << reg.stat2__pd__pos)


def set_power_measure(
    hdk: XyloAudio3HDK,
    frequency: Optional[float] = 5.0,
) -> Tuple[
    samna.BasicSinkNode_unifirm_modules_events_measurement,
    samna.boards.common.power.PowerMonitor,
]:
    raise NotImplementedError
    """
    Initialize power consumption measure on a hdk

    Args:
        hdk (XyloAudio3HDK): The Xylo HDK to be measured
        frequency (float): The frequency of power measurement. Default: 5.0

    Returns:
        power_buf: Event buffer to read power monitoring events from
        power_monitor: The power monitoring object
    """
    power_monitor = hdk.get_power_monitor()
    power_source = power_monitor.get_source_node()
    power_buf = samna.graph.sink_from(power_source)
    power_monitor.start_auto_power_measurement(frequency)
    return power_buf, power_monitor


def apply_configuration(
    hdk: XyloAudio3HDK,
    config: XyloConfiguration,
    *_,
    **__,
) -> None:
    """
    Apply a configuration to the Xylo HDK

    Args:
        hdk (XyloAudio3HDK): The Xylo HDK to write the configuration to
        config (XyloConfiguration): A configuration for Xylo
    """
    # - Ideal -- just write the configuration using samna
    hdk.get_model().apply_configuration(config)


def configure_single_step_time_mode(
    config: XyloConfiguration,
) -> XyloConfiguration:
    """
    Switch on single-step mode on a Xylo Audio 3 HDK

    Args:
        config (XyloConfiguration): The desired Xylo configuration to use
    """
    # - Write the configuration
    config.operation_mode = samna.xyloAudio3.OperationMode.Manual
    return config


def reset_input_spikes(write_buffer: XyloAudio3WriteBuffer) -> None:
    """
    Reset the input spike registers on a Xylo A3 HDK

    Args:
        write_buffer (XyloAudio3WriteBuffer): A write buffer connected to the Xylo HDK to access
    """
    for register in [reg.ispkreg0l, reg.ispkreg0h, reg.ispkreg1l, reg.ispkreg1h]:
        write_register(write_buffer, register)


def send_immediate_input_spikes(
    write_buffer: XyloAudio3WriteBuffer, neurons: List[int]
) -> None:
    """
    Send a list of immediate input events to a Xylo A3 HDK in manual mode

    Args:
        write_buffer (XyloAudio3WriteBuffer): A write buffer connected to the Xylo HDK to access
        neurons (List[int]): A list of neurons to send spike events to, one per entry in the list
    """
    events = [samna.xyloAudio3.event.Spike(n) for n in neurons]
    write_buffer.write(events)


def read_output_events(
    read_buffer: XyloAudio3ReadBuffer,
    write_buffer: XyloAudio3WriteBuffer,
) -> np.ndarray:
    """
    Read the spike flags from the output neurons on a Xylo HDK

    Args:
        read_buffer (XyloReadBuffer): A read buffer to use
        write_buffer (XyloWriteBuffer): A write buffer to use

    Returns:
        np.ndarray: A boolean array of output event flags
    """
    # - Read the status register
    status = read_register(read_buffer, write_buffer, reg.stat1)
    status.extend(read_register(read_buffer, write_buffer, reg.stat2))

    # - Convert to neuron events and return
    string = format(int(status[0]), "0>32b")[-8:] + format(int(status[1]), "0>32b")[-8:]
    return np.array([bool(int(e)) for e in string[::-1]], "bool")


def configure_accel_time_mode(
    hdk: XyloAudio3HDK,
    config: XyloConfiguration,
    Nout: int = 0,
    monitor_Nhidden: int = 0,
    monitor_Noutput: int = 0,
    readout="Spike",
    record=False,
) -> Tuple[XyloConfiguration]:
    raise NotImplementedError
    """
    Switch on accelerated-time mode on a Xylo hdk, and configure network monitoring

    Notes:
        Use :py:func:`new_xylo_state_monitor_buffer` to generate a buffer to monitor neuron and synapse state.

    Args:
        config (XyloConfiguration): The desired Xylo configuration to use
        Nout (int): Number of output neurons in total
        monitor_Nhidden (Optional[int]): The number of hidden neurons for which to monitor state during evolution. Default: ``0``, don't monitor any hidden neurons.
        monitor_Noutput (Optional[int]): The number of output neurons for which to monitor state during evolution. Default: ``0``, don't monitor any output neurons.
        readout: The readout out mode for which to output neuron states. Default: ``Spike''. Must be one of ``['Vmem', 'Spike', 'Isyn']``.
        record (bool): Iff ``True``, record state during evolution. Default: ``False``, do not record state.

    Returns:
        XyloConfiguration: `config`
    """

    # Set imu sensor enable to open the port for geting spikes from imu sensor
    hdk.enable_manual_input_acceleration(True)

    assert readout in ["Vmem", "Spike", "Isyn"], f"{readout} is not supported."

    # - Select accelerated time mode, and general configuration
    config.operation_mode = samna.xyloAudio3.OperationMode.AcceleratedTime
    # config.imu_if_input_enable = False
    config.debug.always_update_omp_stat = True

    # Configurations set for state memory reading
    config.debug.isyn_clock_enable = True
    config.debug.ra_clock_enable = True
    config.debug.bias_clock_enable = True
    config.debug.hm_clock_enable = True
    config.debug.ram_power_enable = True

    config.debug.monitor_neuron_spike = None
    config.debug.monitor_neuron_v_mem = None

    if record:
        config.debug.monitor_neuron_spike = samna.xyloAudio3.configuration.NeuronRange(
            0, monitor_Nhidden
        )
        config.debug.monitor_neuron_v_mem = samna.xyloAudio3.configuration.NeuronRange(
            0, monitor_Nhidden + monitor_Noutput
        )
        config.debug.monitor_neuron_i_syn = samna.xyloAudio3.configuration.NeuronRange(
            0, monitor_Nhidden + monitor_Noutput
        )

    else:
        if readout == "Isyn":
            config.debug.monitor_neuron_i_syn = (
                samna.xyloAudio3.configuration.NeuronRange(
                    monitor_Nhidden, monitor_Nhidden + Nout
                )
            )
        elif readout == "Vmem":
            config.debug.monitor_neuron_v_mem = (
                samna.xyloAudio3.configuration.NeuronRange(
                    monitor_Nhidden, monitor_Nhidden + Nout
                )
            )

    # - Return the configuration and buffer
    return config


def blocking_read(
    read_buffer: XyloAudio3ReadBuffer,
    target_timestep: Optional[int] = None,
    count: Optional[int] = None,
    timeout: Optional[float] = None,
) -> Tuple[List, bool]:
    """
    Perform a blocking read on a buffer, optionally waiting for a certain count, a target timestep, or imposing a timeout

    You should not provide `count` and `target_timestep` together.

    Args:
        read_buffer (XyloAudio3ReadBuffer): A buffer to read from
        target_timestep (Optional[int]): The desired final timestep. Read until this timestep is returned in an event. Default: ``None``, don't wait until a particular timestep is read.
        count (Optional[int]): The count of required events. Default: ``None``, just wait for any data.
        timeout (Optional[float]): The time in seconds to wait for a result. Default: ``None``, no timeout: block until a read is made.

    Returns:
        (List, bool): `event_list`, `is_timeout`
        `event_list` is a list of events read from the HDK. `is_timeout` is a boolean flag indicating that the read resulted in a timeout
    """
    all_events = []

    # - Read at least a certain number of events
    continue_read = True
    is_timeout = False
    start_time = time.time()
    while continue_read:
        # - Perform a read and save events
        events = read_buffer.get_events()
        all_events.extend(events)

        # - Check if we reached the desired timestep
        if target_timestep:
            timesteps = [
                e.timestep
                for e in events
                if hasattr(e, "timestep") and e.timestep is not None
            ]

            if timesteps:
                reached_timestep = timesteps[-1] >= target_timestep
                continue_read &= ~reached_timestep

        # - Check timeout
        if timeout:
            is_timeout = (time.time() - start_time) > timeout
            continue_read &= not is_timeout

        # - Check number of events read
        if count:
            continue_read &= len(all_events) < count

        # - Continue reading if no events have been read
        if not target_timestep and not count:
            continue_read &= len(all_events) == 0

    # - Perform one final read for good measure
    all_events.extend(read_buffer.get_events())

    # - Return read events
    return all_events, is_timeout


def decode_accel_mode_data(
    readout_events: List[ReadoutEvent],
    Nin: int,
    Nhidden_monitor: int,
    Nout_monitor: int,
    Nout: int,
    T_start: int,
    T_end: int,
) -> XyloState:
    raise NotImplementedError
    """
    Read accelerated simulation mode data from a Xylo HDK

    Args:
        readout_events (List[ReadoutEvent]): A list of `ReadoutEvent`s recorded from Xylo IMU
        Nin (int): The number of input channels for the configured network
        Nhidden_monitor (int): The number of hidden neurons to monitor
        Nout_monitor (int): The number of output neurons to monitor
        Nout (int): The number of output neurons in total
        T_start (int): Initial timestep
        T_end (int): Final timestep

    Returns:
        XyloState: The encapsulated state read from the Xylo device
    """
    # - Initialise lists for recording state
    T_count = T_end - T_start + 1
    vmem_ts = np.zeros((T_count, Nhidden_monitor), np.int16)
    isyn_ts = np.zeros((T_count, Nhidden_monitor), np.int16)
    vmem_out_ts = np.zeros((T_count, Nout), np.int16)
    isyn_out_ts = np.zeros((T_count, Nout_monitor), np.int16)
    spikes_ts = np.zeros((T_count, Nhidden_monitor), np.int8)
    output_ts = np.zeros((T_count, Nout), np.int8)

    # print(f"decode_accel_mode_data: T_start {T_start} T_end {T_end}; T_count {T_count}")

    # - Loop over time steps
    for ev in readout_events:
        if type(ev) is ReadoutEvent:
            timestep = ev.timestep - T_start
            vmems = ev.neuron_v_mems
            vmem_ts[timestep, 0:Nhidden_monitor] = vmems[0:Nhidden_monitor]
            vmem_out_ts[timestep, 0:Nout] = ev.output_v_mems

            isyns = ev.neuron_i_syns
            isyn_ts[timestep, 0:Nhidden_monitor] = isyns[0:Nhidden_monitor]
            isyn_out_ts[timestep, 0:Nout] = isyns[
                Nhidden_monitor : Nhidden_monitor + Nout_monitor
            ]

            spikes_ts[timestep] = ev.hidden_spikes
            output_ts[timestep] = ev.output_spikes

    # - Return as a XyloState object
    return XyloState(
        Nin=Nin,
        Nhidden=Nhidden_monitor,
        Nout=Nout,
        V_mem_hid=vmem_ts,
        I_syn_hid=isyn_ts,
        V_mem_out=vmem_out_ts,
        I_syn_out=isyn_out_ts,
        Spikes_hid=spikes_ts,
        Spikes_out=output_ts,
    )


def decode_realtime_mode_data(
    readout_events: List[ReadoutEvent],
    Nout: int,
    T_start: int,
    T_end: int,
) -> Tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError
    """
    Read realtime simulation mode data from a Xylo HDK

    Args:
        readout_events (List[ReadoutEvent]): A list of `ReadoutEvent`s recorded from Xylo IMU
        Nout (int): The number of output neurons to monitor
        T_start (int): Initial timestep
        T_end (int): Final timestep

    Returns:
        Tuple[np.ndarray, np.ndarray]: (`vmem_out_ts`, `output_ts`) The membrane potential and output event trains from Xylo
    """
    # - Initialise lists for recording state
    T_count = T_end - T_start + 1
    vmem_out_ts = np.zeros((T_count, Nout), np.int16)
    output_ts = np.zeros((T_count, Nout), np.int8)

    # - Loop over time steps
    for ev in readout_events:
        if type(ev) is ReadoutEvent:
            timestep = ev.timestep - T_start
            if timestep >= 0:
                vmem_out_ts[timestep, 0:Nout] = ev.output_v_mems
                output_ts[timestep] = ev.output_spikes

    # - Return Vmem and spikes
    return vmem_out_ts, output_ts


def gen_clear_input_registers_events() -> List:
    raise NotImplementedError
    """
    Create events to clear the input event registers
    """
    events = []
    for addr in [0x47, 0x48, 0x49, 0x4A]:
        event = samna.xyloAudio3.event.WriteRegisterValue()
        event.address = addr
        events.append(event)

    return events


def config_hibernation_mode(
    config: XyloConfiguration, hibernation_mode: bool
) -> XyloConfiguration:
    raise NotImplementedError

    """
    Switch on hibernaton mode on a Xylo hdk

    Args:
        config (XyloConfiguration): The desired Xylo configuration to use
    """
    config.enable_hibernation_mode = hibernation_mode
    return config


def get_current_timestep(
    read_buffer: XyloAudio3ReadBuffer,
    write_buffer: XyloAudio3WriteBuffer,
    timeout: float = 3.0,
) -> int:
    raise NotImplementedError

    """
    Retrieve the current timestep on a Xylo HDK

    Args:
        read_buffer (XyloAudio3ReadBuffer): A connected read buffer for the xylo HDK
        write_buffer (XyloAudio3WriteBuffer): A connected write buffer for the Xylo HDK
        timeout (float): A timeout for reading

    Returns:
        int: The current timestep on the Xylo HDK
    """

    # - Clear read buffer
    read_buffer.get_events()

    # - Trigger a readout event on Xylo
    e = samna.xyloAudio3.event.TriggerReadout()
    write_buffer.write([e])

    # - Wait for the readout event to be sent back, and extract the timestep
    timestep = None
    continue_read = True
    start_t = time.time()
    while continue_read:
        readout_events = read_buffer.get_events()
        ev_filt = [
            e for e in readout_events if isinstance(e, samna.xyloAudio3.event.Readout)
        ]
        if ev_filt:
            timestep = ev_filt[0].timestep
            continue_read = False
        else:
            # - Check timeout
            continue_read &= (time.time() - start_t) < timeout

    if timestep is None:
        raise TimeoutError(f"Timeout after {timeout}s when reading current timestep.")

    # - Return the timestep
    return timestep


def config_realtime_mode(
    config: XyloConfiguration,
    dt: float,
    main_clk_rate: int,
) -> XyloConfiguration:
    raise NotImplementedError

    """
    Set the Xylo HDK to real-time mode

    Args:
        config (XyloConfiguration): A configuration for Xylo IMU
        dt (float): The simulation time-step to use for this Module
        main_clk_rate (int): The main clock rate of Xylo in Hz

    Return:
        updated Xylo configuration
    """
    # - Select real-time operation mode
    config.operation_mode = samna.xyloAudio3.OperationMode.RealTime

    config.debug.always_update_omp_stat = True
    config.imu_if_input_enable = True
    config.debug.imu_if_clk_enable = True

    # - Configure Xylo IMU clock rate
    config.time_resolution_wrap = int(dt * main_clk_rate)
    IMU_IF_clk_rate = 50_000  # IMU IF clock must be 50 kHz
    config.debug.imu_if_clock_freq_div = int(main_clk_rate / IMU_IF_clk_rate - 1)

    # - Set configuration timeout
    config.input_interface.configuration_timeout = 20_000

    # - No monitoring of internal state in realtime mode
    config.debug.monitor_neuron_v_mem = None
    config.debug.monitor_neuron_i_syn = None
    config.debug.monitor_neuron_spike = None

    return config


def read_register(
    read_buffer: XyloAudio3ReadBuffer,
    write_buffer: XyloAudio3WriteBuffer,
    address: int,
    timeout: float = 2.0,
) -> List[int]:
    """
    Read the contents of a register

    Args:
        read_buffer (XyloAudio3ReadBuffer): A connected read buffer to the XYlo HDK
        write_buffer (XyloAudio3WriteBuffer): A connected write buffer to the Xylo HDK
        address (int): The register address to read
        timeout (float): A timeout in seconds

    Returns:
        List[int]: A list of events returned from the read
    """
    # - Set up a register read
    write_buffer.write([samna.xyloAudio3.event.ReadRegisterValue(address=address)])

    # - Wait for data and read it
    start_t = time.time()
    continue_read = True
    while continue_read:
        # - Read from the buffer
        events = read_buffer.get_events()

        # - Filter returned events for the desired address
        ev_filt = [e for e in events if hasattr(e, "address") and e.address == address]

        # - Should we continue the read?
        continue_read &= len(ev_filt) == 0
        continue_read &= (time.time() - start_t) < timeout

    # - If we didn't get the required register read, raise an error
    if len(ev_filt) == 0:
        raise TimeoutError(f"Timeout after {timeout}s when reading register {address}.")

    # - Return data
    return [e.data for e in ev_filt]


def write_register(
    write_buffer: XyloAudio3WriteBuffer, register: int, data: int = 0
) -> None:
    """
    Write data to a register on a Xylo A3 HDK

    Args:
        write_buffer (XyloAudio3WriteBuffer): A connected write buffer to the destination Xylo A3 HDK
        register (int): The address of the register to write to
        data (int): The data to write. Default: 0x0
    """
    write_buffer.write(
        [samna.xyloAudio3.event.WriteRegisterValue(address=register, data=data)]
    )


def write_memory(
    write_buffer: XyloAudio3WriteBuffer,
    start_address: int,
    data: Union[List[int], int] = 0,
) -> None:
    """
    Write data to memory on a Xylo A3 HDK

    This function will write a list of data to sequential locations in the Xylo A3 memory

    Args:
        write_buffer (XyloAudio3WriteBuffer): A connected write buffer to the target Xylo A3 HDK
        start_address (int): The start memory address to write to
        data (Union[List[int], int]): A list of integers, or single integer, to write to memory
    """
    # - Convert a single integer to a list
    if isinstance(data, int):
        data = [data]

    # - Generate write events
    write_events = [
        samna.xyloAudio3.event.WriteMemoryValue(addr, d)
        for (addr, d) in zip(range(start_address, start_address + len(data)), data)
    ]

    # - Send the memory write events
    write_buffer.write(write_events)


def read_memory(
    read_buffer: XyloAudio3ReadBuffer,
    write_buffer: XyloAudio3WriteBuffer,
    start_address: int,
    length: int = 1,
    read_timeout: float = 2.0,
) -> List[int]:
    # - Generate read events
    read_events = [
        samna.xyloAudio3.event.ReadMemoryValue(addr)
        for addr in range(start_address, start_address + length)
    ]

    # - Clear buffer
    read_buffer.get_events()

    # - Request read
    write_buffer.write(read_events)

    # - Read data
    events, is_timeout = blocking_read(read_buffer, count=length, timeout=read_timeout)
    if is_timeout:
        raise TimeoutError(
            f"Memory read timed out after {read_timeout} s. Reading @{start_address}+{length}."
        )

    # - Filter returned events for the desired addresses
    return [
        e.data
        for e in events[1:]
        if hasattr(e, "address")
        and e.address >= start_address
        and e.address < start_address + length
    ]


def set_xylo_core_clock_freq(device: XyloAudio3HDK, desired_freq_MHz: float) -> float:
    raise NotImplementedError
    """
    Set the inference core clock frequency used by Xylo

    Args:
        device (XyloAudio3HDK): A Xylo device to configure
        desired_freq_MHz (float): The desired Xylo core clock frequency in MHz

    Returns:
        (float): Actual frequency obtained, in MHz
    """
    # - Determine wait period and actual obtained clock frequency
    wait_period = int(np.ceil(round(100 / desired_freq_MHz) / 2 - 1))
    actual_freq = 100 / (2 * (wait_period + 1))

    # - Configure device
    device.get_io_module().set_main_clk_rate(int(actual_freq * 1e6))

    return actual_freq


def enable_ram_access(device: XyloAudio3HDK, enabled: bool) -> None:
    if enabled:
        device.get_model().open_ram_access()
    else:
        device.get_model().close_ram_access()
