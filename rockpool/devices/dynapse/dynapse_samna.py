"""
Dynap-SE samna backend bridge
Handles the low-level hardware configuration under the hood and provide easy-to-use access to the user

Note : Existing modules are reconstructed considering consistency with Xylo support.

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

15/09/2022
"""

from __future__ import annotations

from typing import Any


# - Rockpool imports
from rockpool.nn.modules.module import Module

# - Configure exports
__all__ = ["config_from_specification", "save_config", "load_config", "DynapseSamna"]


def config_from_specification(*args, **kwargs) -> Any:
    None


def save_config(*args, **kwargs) -> Any:
    None


def load_config(*args, **kwargs) -> Any:
    None


class DynapseSamna(Module):
    def __init__(self):
        None
