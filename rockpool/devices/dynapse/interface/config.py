"""
Dynap-SE samna config helper functions
Handles the low-level hardware configuration under the hood and provide easy-to-use access to the user

Note : Existing modules are reconstructed considering consistency with Xylo support.

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

15/09/2022
"""
from __future__ import annotations
import logging

from typing import Any, Optional, Union
import numpy as np
from rockpool.devices.dynapse.samna_alias.dynapse2 import Dynapse2Configuration
from rockpool.devices.dynapse.default import dcurrents, dgain, dlayout, dtime, dweight

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


def config_from_specification(
    config: Optional[Dynapse2Configuration] = None,
    weights_in: Optional[np.ndarray] = None, # 4-bit
    weights_rec: Optional[np.ndarray] = None, # 4-bit
    weights_out: Optional[np.ndarray] = None, # 1-bit
    # gain params
    r_gain_ahp: Union[float, np.ndarray, None] = dgain["r_gain_ahp"],
    r_gain_ampa: Union[float, np.ndarray, None] = dgain["r_gain_ampa"],
    r_gain_gaba: Union[float, np.ndarray, None] = dgain["r_gain_gaba"],
    r_gain_nmda: Union[float, np.ndarray, None] = dgain["r_gain_nmda"],
    r_gain_shunt: Union[float, np.ndarray, None] = dgain["r_gain_shunt"],
    r_gain_mem: Union[float, np.ndarray, None] = dgain["r_gain_mem"],
    ## time params
    t_pulse_ahp: Union[float, np.ndarray, None] = dtime["t_pulse_ahp"],
    t_pulse: Union[float, np.ndarray, None] = dtime["t_pulse"],
    t_ref: Union[float, np.ndarray, None] = dtime["t_ref"],
    ## tau params
    tau_ahp: Union[float, np.ndarray, None] = dtime["tau_ahp"],
    tau_ampa: Union[float, np.ndarray, None] = dtime["tau_ampa"],
    tau_gaba: Union[float, np.ndarray, None] = dtime["tau_gaba"],
    tau_nmda: Union[float, np.ndarray, None] = dtime["tau_nmda"],
    tau_shunt: Union[float, np.ndarray, None] = dtime["tau_shunt"],
    tau_mem: Union[float, np.ndarray, None] = dtime["tau_mem"],
    ## weight params
    Iw_0: Union[float, np.ndarray, None] = dweight["Iw_0"],
    Iw_1: Union[float, np.ndarray, None] = dweight["Iw_1"],
    Iw_2: Union[float, np.ndarray, None] = dweight["Iw_2"],
    Iw_3: Union[float, np.ndarray, None] = dweight["Iw_3"],
    Iw_ahp: Union[float, np.ndarray, None] = dcurrents["Iw_ahp"],
    Ispkthr: Union[float, np.ndarray, None] = dcurrents["Ispkthr"],
    If_nmda: Union[float, np.ndarray, None] = dcurrents["If_nmda"],
    Idc: Union[float, np.ndarray, None] = dcurrents["Idc"],
    *args,
    **kwargs
) -> Dynapse2Configuration:

    if config is None:
        config = samna.dynapse2.Dynapse2Configuration() if SAMNA_AVAILABLE else None
        logging.warn(
            "Fetch the samna object from the actual device and provide ``config = model.get_configuration()``!"
        )

    # get number of cores available
    n_cores = __get_num_cores(config)



    # Set Parameters

    # Set Memory

    return config


def save_config(*args, **kwargs) -> Any:
    None


def load_config(*args, **kwargs) -> Any:
    None
