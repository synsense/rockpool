"""
Torch loss functions and regularizers useful for training networks using Torch Modules.
"""

from importlib import util

if util.find_spec("torch") is None:
if util.find_spec("torch") is None:
    raise ModuleNotFoundError(
        "'Torch' backend not found. Modules that rely on Torch will not be available."
    )

import torch
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np

import rockpool.parameters as rp
from rockpool.typehints import FloatVector, P_float, P_tensor

from typing import Union, List, Tuple, Optional, Tuple, Any

class ParameterBoundaryRegularizer():

    def __init__(lower_bound=None, upper_bound=None):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(input):
        if self.upper_bound:
            upper_loss = torch.sum(torch.clamp(torch.exp(input - self.upper_bound), min=1))
        else:
            upper_loss = 0
        if self.lower_bound:
            lower_loss = torch.sum(torch.clamp(torch.exp(self.lower_bound - input), min=1))
        else:
            lower_loss = 0
        return lower_loss + upper_loss
