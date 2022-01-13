"""
Utility functions for simulation vs. DynapSE parameters comparison

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
11/11/2021
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from rockpool.devices.dynapse.dynapse1_jax import DynapSE1Jax

_PANDAS_AVAILABLE = True

try:
    import pandas as pd

except ModuleNotFoundError as e:
    pd = Any
    print(
        e, "\nDevice vs. Simulation comparison dataframes cannot be generated!",
    )
    _PANDAS_AVAILABLE = False


def bias_current_table(
    mod: DynapSE1Jax,
    chipID: np.uint8,
    coreID: np.uint8,
    reciprocals: List[Tuple[str]],
    default_mod: Optional[DynapSE1Jax] = None,
) -> Dict[str, List[Union[str, Tuple[int], float]]]:
    """
    bias_current_table creates a table in the form of a dictionary. It includes the bias parameters, coarse and fine values,
    corresponding simulation currents, ampere values and nominal simulation values for comparison purposes.
    The dicionary can easily be converted to a pandas dataframe if desired.

    :param mod: the module to be investigated
    :type mod: DynapSE1Jax
    :param chipID: Unique chip ID to of the simulation module to examine
    :type chipID: np.uint8
    :param coreID: Non-unique core ID to of the simulation module to examine
    :type coreID: np.uint8
    :param reciprocals: a mapping from bias parameters to simualtion currents. Looks like [("PS_WEIGHT_INH_S_N", "Iw_gaba_b")]
    :type reciprocals: List[Tuple[str]]
    :param default_mod: a default simulation module to extract nominal simulation values, defaults to None
    :type default_mod: Optional[DynapSE1Jax], optional
    :return: a dictionary encapsulating all the bias current simulation current information extracted depending on reciprocal list

    e.g. (The dictionary converted to a dataframe `pd.DataFrame(data)`)
                        Bias Coarse,Fine    Current  Amperes  Nominal(A)
        0  PS_WEIGHT_INH_S_N      (0, 0)  Iw_gaba_b 0.00e+00    1.00e-07
        1  PS_WEIGHT_INH_F_N    (7, 255)  Iw_gaba_a 2.40e-05    1.00e-07
        2  PS_WEIGHT_EXC_S_N     (6, 82)    Iw_nmda 1.03e-06    1.00e-07
        3  PS_WEIGHT_EXC_F_N    (7, 219)    Iw_ampa 2.06e-05    1.00e-07
        4           IF_AHW_P      (0, 0)     Iw_ahp 0.00e+00    1.00e-07
        5            IF_DC_P    (1, 254)        Idc 1.05e-10    5.00e-13
        6          IF_NMDA_N      (0, 0)    If_nmda 0.00e+00    5.00e-13

    :rtype: Dict[str, List[Union[str, Tuple[int], float]]]
    """
    if default_mod is None:
        default_mod = DynapSE1Jax((4, 1))

    # Construct the dictionary
    keys = ["Bias", "Coarse,Fine", "Current", "Amperes", "Nominal(A)"]
    data = dict(zip(keys, [list() for i in keys]))
    data["Bias"], data["Current"] = map(list, zip(*reciprocals))

    # Fill bias coarse and fine values
    for bias in data["Bias"]:
        param = getattr(mod, bias)(chipID, coreID)
        data["Coarse,Fine"].append((param.coarse_value, param.fine_value))

    # Fill current values in amperes and also the nominal values
    for sim_current in data["Current"]:
        data["Amperes"].append(mod.get_bias(chipID, coreID, sim_current))
        data["Nominal(A)"].append(default_mod.get_bias(0, 0, sim_current))

    return data


def high_level_parameter_table(
    mod: DynapSE1Jax,
    chipID: np.uint8,
    coreID: np.uint8,
    param_list: List[str],
    default_mod: Optional[DynapSE1Jax] = None,
) -> Dict[str, List[Union[str, Tuple[int], float]]]:
    """
    high_level_parameter_table creates a table in the form of a dictionary. It includes high level parameters, their values
    and also the nominal simulation values for comparison purposes.
    The dicionary can easily be converted to a pandas dataframe if desired.

    :param mod: the module to be investigated
    :type mod: DynapSE1Jax
    :param chipID: Unique chip ID to of the simulation module to examine
    :type chipID: np.uint8
    :param coreID: Non-unique core ID to of the simulation module to examine
    :type coreID: np.uint8
    :param param_list: a list of high level parameters to be obtained from the module and stored in a table-compatible dictionary. e.g ["tau_mem", "tau_ampa"]
    :type param_list: List[str]
    :param default_mod: a default simulation module to extract nominal simulation values, defaults to None
    :type default_mod: Optional[DynapSE1Jax], optional
    :return: a dictionary encapsulating all the high level parameter information extracted depending on param list

    e.g. (The dictionary converted to a dataframe `pd.DataFrame(data)`)
            Parameter      Value  Nominal
        0     tau_mem   1.26e-06 2.00e-02
        1  tau_gaba_b   4.05e-03 1.00e-01
        2  tau_gaba_a   3.62e-08 1.00e-02
        3    tau_nmda   1.27e-02 1.00e-01
        4    tau_ampa   5.40e-03 1.00e-02
        5     tau_ahp   9.04e-05 5.00e-02
        6       t_ref   3.55e-06 1.00e-02
        7     t_pulse   2.96e-06 1.00e-05

    :rtype: Dict[str, List[Union[str, Tuple[int], float]]]
    """

    if default_mod is None:
        default_mod = DynapSE1Jax((4, 1))

    # Construct the dictionary
    keys = ["Parameter", "Value", "Nominal"]
    data = dict(zip(keys, [list() for i in keys]))

    data["Parameter"] = param_list

    for param in data["Parameter"]:
        data["Value"].append(mod.get_bias(chipID, coreID, param))
        data["Nominal"].append(default_mod.get_bias(0, 0, param))

    return data


def merge_biases_high_level(
    mod: DynapSE1Jax,
    chipID: np.uint8,
    coreID: np.uint8,
    reciprocals: List[Tuple[str]],
    default_mod: Optional[DynapSE1Jax] = None,
) -> Dict[str, List[Union[str, Tuple[int], float]]]:
    """
    merge_biases_high_level merge the bias current&simulation currents table data and high level parameter data.
    It can be considered as a wrapper for using  `bias_current_table()` and `high_level_parameter_table()` together.
    The tuples in the reciprocal list should be like `("IF_TAU1_N", "Itau_mem", "tau_mem")`. First the bias param,
    then corresponding simulation current, then the related high-level parameter.

    :param mod: the module to be investigated
    :type mod: DynapSE1Jax
    :param chipID: Unique chip ID to of the simulation module to examine
    :type chipID: np.uint8
    :param coreID: Non-unique core ID to of the simulation module to examine
    :type coreID: np.uint8
    :param reciprocals: a mapping from bias parameters to simualtion currents. Looks like [("PS_WEIGHT_INH_S_N", "Iw_gaba_b")]
    :type reciprocals: List[Tuple[str]]
    :param default_mod: a default simulation module to extract nominal simulation values, defaults to None
    :type default_mod: Optional[DynapSE1Jax], optional
    :return: a dictionary encapsulating all the bias current simulation current information, and their related higher level parameter information extracted depending on reciprocal list
    :rtype: Dict[str, List[Union[str, Tuple[int], float]]]
    """

    bias_current, parameter = map(
        list, zip(*list(map(lambda t: ((t[0], t[1]), t[2]), reciprocals)))
    )

    # Obtain bias + high-level parameters table
    data = bias_current_table(mod, chipID, coreID, bias_current, default_mod)
    taus = high_level_parameter_table(mod, chipID, coreID, parameter, default_mod)

    data.update(taus)

    return data


def time_const_table(
    mod: DynapSE1Jax,
    chipID: np.uint8,
    coreID: np.uint8,
    default_mod: Optional[DynapSE1Jax] = None,
    float_format: str = "{:,.2e}".format,
) -> pd.DataFrame:
    """
    time_const_table creates a pandas dataframe to investigate the consequences of
    the bias currents which have a corresponding time constant used in the simulator.
    ['IF_TAU1_N', 'NPDPII_TAU_S_P', 'NPDPII_TAU_F_P', 'NPDPIE_TAU_S_P', 'NPDPIE_TAU_F_P', 'IF_AHTAU_N', 'IF_RFR_N', 'PULSE_PWLK_P']

    :param mod: the module to be investigated
    :type mod: DynapSE1Jax
    :param chipID: Unique chip ID to of the simulation module to examine
    :type chipID: np.uint8
    :param coreID: Non-unique core ID to of the simulation module to examine
    :type coreID: np.uint8
    :param default_mod: a default simulation module to extract nominal simulation values, defaults to None
    :type default_mod: Optional[DynapSE1Jax], optional
    :param float_format: the float printing format used printing the dataframe, defaults to "{:,.2e}".format
    :type float_format: str, optional
    :return: a table for examining the bias currents which sets some time constants on the device.

    e.g.
                    Bias Coarse,Fine      Current  Amperes  Nominal(A)   Parameter     Value  Nominal
        0       IF_TAU1_N     (5, 90)     Itau_mem 1.41e-07    8.87e-12     tau_mem  1.26e-06 2.00e-02
        1  NPDPII_TAU_S_P     (2, 68)  Itau_gaba_b 2.19e-10    8.87e-12  tau_gaba_b  4.05e-03 1.00e-01
        2  NPDPII_TAU_F_P    (7, 255)  Itau_gaba_a 2.40e-05    8.69e-11  tau_gaba_a  3.62e-08 1.00e-02
        3  NPDPIE_TAU_S_P    (1, 169)    Itau_nmda 6.96e-11    8.87e-12    tau_nmda  1.27e-02 1.00e-01
        4  NPDPIE_TAU_F_P     (2, 50)    Itau_ampa 1.61e-10    8.69e-11    tau_ampa  5.40e-03 1.00e-02
        5      IF_AHTAU_N     (4, 80)     Itau_ahp 1.57e-08    2.84e-11     tau_ahp  9.04e-05 5.00e-02
        6        IF_RFR_N    (3, 196)         Iref 5.00e-09    1.77e-12       t_ref  3.55e-06 1.00e-02
        7    PULSE_PWLK_P    (3, 235)       Ipulse 5.99e-09    1.77e-09     t_pulse  2.96e-06 1.00e-05

    :rtype: pd.DataFrame
    """

    if default_mod is None:
        default_mod = DynapSE1Jax((4, 1))

    # Hard-coded reciprocal list
    reciprocals = [
        ("IF_TAU1_N", "Itau_mem", "tau_mem"),
        ("NPDPII_TAU_S_P", "Itau_gaba_b", "tau_gaba_b"),
        ("NPDPII_TAU_F_P", "Itau_gaba_a", "tau_gaba_a"),
        ("NPDPIE_TAU_S_P", "Itau_nmda", "tau_nmda"),
        ("NPDPIE_TAU_F_P", "Itau_ampa", "tau_ampa"),
        ("IF_AHTAU_N", "Itau_ahp", "tau_ahp"),
        ("IF_RFR_N", "Iref", "t_ref"),
        ("PULSE_PWLK_P", "Ipulse", "t_pulse"),
    ]

    data = merge_biases_high_level(mod, chipID, coreID, reciprocals, default_mod)

    pd.options.display.float_format = float_format
    return pd.DataFrame(data)


def gain_table(
    mod: DynapSE1Jax,
    chipID: np.uint8,
    coreID: np.uint8,
    default_mod: Optional[DynapSE1Jax] = None,
    float_format: str = "{:,.2e}".format,
) -> pd.DataFrame:
    """
    gain_table creates a pandas dataframe to investigate the consequences of
    the bias currents which have a corresponding gain factor used in the simulator.
    ['IF_THR_N', 'NPDPII_THR_S_P', 'NPDPII_THR_F_P', 'NPDPIE_THR_S_P', 'NPDPIE_THR_F_P', 'IF_AHTHR_N']

    :param mod: the module to be investigated
    :type mod: DynapSE1Jax
    :param chipID: Unique chip ID to of the simulation module to examine
    :type chipID: np.uint8
    :param coreID: Non-unique core ID to of the simulation module to examine
    :type coreID: np.uint8
    :param default_mod: a default simulation module to extract nominal simulation values, defaults to None
    :type default_mod: Optional[DynapSE1Jax], optional
    :param float_format: the float printing format used printing the dataframe, defaults to "{:,.2e}".format
    :type float_format: str, optional
    :return: a table for examining the bias currents which sets the gain factors of the DPI circuits on the device.

    e.g.
                     Bias Coarse,Fine     Current  Amperes  Nominal(A)    Parameter     Value  Nominal   f_gain  Nominal(1)
        0        IF_THR_N    (2, 254)     Ith_mem 8.17e-10    3.55e-11     Itau_mem  1.04e-08 8.87e-12 7.86e-02    4.00e+00
        1  NPDPII_THR_S_P    (7, 255)  Ith_gaba_b 2.40e-05    3.55e-11  Itau_gaba_b  2.40e-05 8.87e-12 1.00e+00    4.00e+00
        2  NPDPII_THR_F_P     (2, 44)  Ith_gaba_a 1.41e-10    3.48e-10  Itau_gaba_a  7.00e-11 8.69e-11 2.02e+00    4.00e+00
        3  NPDPIE_THR_S_P     (3, 38)    Ith_nmda 9.69e-10    3.55e-11    Itau_nmda  4.76e-10 8.87e-12 2.04e+00    4.00e+00
        4  NPDPIE_THR_F_P     (5, 46)    Ith_ampa 7.22e-08    3.48e-10    Itau_ampa  3.63e-08 8.69e-11 1.99e+00    4.00e+00
        5      IF_AHTHR_N    (4, 161)     Ith_ahp 3.16e-08    1.13e-10     Itau_ahp  1.57e-08 2.84e-11 2.01e+00    4.00e+00

    :rtype: pd.DataFrame
    """

    if default_mod is None:
        default_mod = DynapSE1Jax((4, 1))

    # Hard-coded reciprocal list
    reciprocals = [
        ("IF_THR_N", "Ith_mem", "Itau_mem"),
        ("NPDPII_THR_S_P", "Ith_gaba_b", "Itau_gaba_b"),
        ("NPDPII_THR_F_P", "Ith_gaba_a", "Itau_gaba_a"),
        ("NPDPIE_THR_S_P", "Ith_nmda", "Itau_nmda"),
        ("NPDPIE_THR_F_P", "Ith_ampa", "Itau_ampa"),
        ("IF_AHTHR_N", "Ith_ahp", "Itau_ahp"),
    ]

    data = merge_biases_high_level(mod, chipID, coreID, reciprocals, default_mod)

    # Add gain rows
    keys = ["f_gain", "Nominal(1)"]
    data.update(dict(zip(keys, [list() for i in keys])))

    # Iterate over existing data and calculate
    for Igain, Ileak, Igain_nom, Ileak_nom in zip(
        data["Amperes"], data["Value"], data["Nominal(A)"], data["Nominal"]
    ):
        data["f_gain"].append(Igain / Ileak)
        data["Nominal(1)"].append(Igain_nom / Ileak_nom)

    pd.options.display.float_format = float_format
    return pd.DataFrame(data)


def synapse_table(
    mod: DynapSE1Jax,
    chipID: np.uint8,
    coreID: np.uint8,
    default_mod: Optional[DynapSE1Jax] = None,
    float_format: str = "{:,.2e}".format,
) -> pd.DataFrame:
    """
    synapse_table creates a pandas dataframe to investigate the consequences of the bias currents which have a
    corresponding synapse weight current or a current affecting the synapse weight indirectly and used in the simulator.
    ['PS_WEIGHT_INH_S_N', 'PS_WEIGHT_INH_F_N', 'PS_WEIGHT_EXC_S_N', 'PS_WEIGHT_EXC_F_N', 'IF_AHW_P', 'IF_DC_P', 'IF_NMDA_N']

    :param mod: the module to be investigated
    :type mod: DynapSE1Jax
    :param chipID: Unique chip ID to of the simulation module to examine
    :type chipID: np.uint8
    :param coreID: Non-unique core ID to of the simulation module to examine
    :type coreID: np.uint8
    :param default_mod: a default simulation module to extract nominal simulation values, defaults to None
    :type default_mod: Optional[DynapSE1Jax], optional
    :param float_format: the float printing format used printing the dataframe, defaults to "{:,.2e}".format
    :type float_format: str, optional
    :return: a table for examining the bias currents which affects the synaptic weights of the DPI circuits on the device.

    e.g.

                        Bias Coarse,Fine    Current  Amperes  Nominal(A)
        0  PS_WEIGHT_INH_S_N      (0, 0)  Iw_gaba_b 0.00e+00    1.00e-07
        1  PS_WEIGHT_INH_F_N    (7, 255)  Iw_gaba_a 2.40e-05    1.00e-07
        2  PS_WEIGHT_EXC_S_N     (6, 82)    Iw_nmda 1.03e-06    1.00e-07
        3  PS_WEIGHT_EXC_F_N    (7, 219)    Iw_ampa 2.06e-05    1.00e-07
        4           IF_AHW_P      (0, 0)     Iw_ahp 0.00e+00    1.00e-07
        5            IF_DC_P    (1, 254)        Idc 1.05e-10    5.00e-13
        6          IF_NMDA_N      (0, 0)    If_nmda 0.00e+00    5.00e-13


    :rtype: pd.DataFrame
    """

    if default_mod is None:
        default_mod = DynapSE1Jax((4, 1))

    # Hard-coded reciprocal list
    reciprocals = [
        ("PS_WEIGHT_INH_S_N", "Iw_gaba_b"),
        ("PS_WEIGHT_INH_F_N", "Iw_gaba_a"),
        ("PS_WEIGHT_EXC_S_N", "Iw_nmda"),
        ("PS_WEIGHT_EXC_F_N", "Iw_ampa"),
        ("IF_AHW_P", "Iw_ahp"),
        ("IF_DC_P", "Idc"),
        ("IF_NMDA_N", "If_nmda"),
    ]

    data = bias_current_table(mod, chipID, coreID, reciprocals, default_mod)

    pd.options.display.float_format = float_format
    return pd.DataFrame(data)


def bias_table(
    mod: DynapSE1Jax,
    chipID: np.uint8,
    coreID: np.uint8,
    default_mod: Optional[DynapSE1Jax] = None,
    float_format: str = "{:,.2e}".format,
) -> pd.DataFrame:
    """
    bias_table merges `time_const_table`, `gain_table`, and `synapse_table` together in one table by
    providing proper keys for each one and represent all the simulated biases within one core in categories.

    :param mod: the module to be investigated
    :type mod: DynapSE1Jax
    :param chipID: Unique chip ID to of the simulation module to examine
    :type chipID: np.uint8
    :param coreID: Non-unique core ID to of the simulation module to examine
    :type coreID: np.uint8
    :param default_mod: a default simulation module to extract nominal simulation values, defaults to None
    :type default_mod: Optional[DynapSE1Jax], optional
    :param float_format: the float printing format used printing the dataframe, defaults to "{:,.2e}".format
    :type float_format: str, optional
    :return: a table for examining all the bias currents configuring the network on the device.
    :rtype: pd.DataFrame
    """

    if default_mod is None:
        default_mod = DynapSE1Jax((4, 1))

    # Generate Tables
    syn_tab = synapse_table(mod, chipID, coreID, default_mod, float_format)
    time_tab = time_const_table(mod, chipID, coreID, default_mod, float_format)
    gain_tab = gain_table(mod, chipID, coreID, default_mod, float_format)

    bias_tab = pd.concat(
        [time_tab, gain_tab, syn_tab], keys=["Time Const.", "Gain", "Synapses"]
    )
    return bias_tab


def device_vs_simulation(
    mod: DynapSE1Jax,
    default_mod: Optional[DynapSE1Jax] = None,
    float_format: str = "{:,.2e}".format,
) -> pd.DataFrame:
    """
    device_vs_simulation merges `bias_table`s for each active core together in one table by
    providing proper keys for each one and represent all the simulated biases in categories.

    :param mod: the module to be investigated
    :type mod: DynapSE1Jax
    :param default_mod: a default simulation module to extract nominal simulation values, defaults to None
    :type default_mod: Optional[DynapSE1Jax], optional
    :param float_format: the float printing format used printing the dataframe, defaults to "{:,.2e}".format
    :type float_format: str, optional
    :return: a merged table across different cores for examining all the bias currents configuring the network on the device.
    :rtype: pd.DataFrame
    """

    if default_mod is None:
        default_mod = DynapSE1Jax((4, 1))

    tables = []
    keys = []

    # Iterate through the active cores
    for chipID, coreID in list(mod.core_dict.keys()):
        tables.append(bias_table(mod, chipID, coreID, default_mod, float_format))
        keys.append(f"C{chipID}c{coreID}")

    comp_tab = pd.concat(tables, keys=keys)

    return comp_tab
