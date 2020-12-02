import multiprocessing
from importlib import util

import numpy as np

from typing import Optional, Union, List, Dict, Tuple
from warnings import warn

from rockpool.nn.modules.timed_module import astimedmodule 
from rockpool.nn.layers.iaf_nest import FFIAFNest as FFIAFNestV1 
from rockpool.nn.layers.iaf_nest import RecIAFSpkInNest as RecIAFSpkInNestV1 
from rockpool.nn.layers.aeif_nest import RecAEIFSpkInNest as RecAEIFSpkInNestV1

FFIAFNest = astimedmodule(
    parameters=[
        "weights",
        "bias",
        "tau_mem",
        "capacity",
        "v_thresh",
        "v_rest",
        "v_rest",
        "refractory",
        "capacity",
    ],
    simulation_parameters=["dt", "num_cores", "record"],
    states=["state"],
)(FFIAFNestV1)

RecIAFSpkInNest = astimedmodule(
parameters=[
        "weights_in",
        "weights_rec",
        "delay_in",
        "delay_rec",
        "bias",
        "tau_mem",
        "tau_syn_exc",
        "tau_syn_inh",
        "capacity",
        "v_thresh",
        "v_reset",
        "v_rest",
        "refractory",
    ],
    simulation_parameters=["dt", "record", "num_cores"],
    states=["state"],
)(RecIAFSpkInNestV1)

RecAEIFSpkInNest = astimedmodule(
parameters=[
        "weights_in",
        "weights_rec",
        "delay_in",
        "delay_rec",
        "bias",
        "tau_mem",
        "tau_syn_exc",
        "tau_syn_inh",
        "capacity",
        "v_thresh",
        "v_reset",
        "v_rest",
        "refractory",
        "conductance",
        "subthresh_adapt",
        "spike_adapt",
        "delta_t",
        "tau_adapt",
    ],
    simulation_parameters=["dt", "record", "num_cores"],
    states=["state"],
)(RecAEIFSpkInNestV1)

