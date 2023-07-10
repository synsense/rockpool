from typing import Optional, Union

import rockpool
import torch
import nir
from nirtorch import extract_nir_graph

__all__ = ["to_nir"]

def _extract_rockpool_module(module) -> Optional[nir.NIRNode]:
    if type(module) == rockpool.nn.modules.ExpSynTorch:
        return nir.LI(
            tau=module.tau.detach(),
            v_leak=torch.zeros_like(module.tau).detach(),
            r=(module.tau * torch.exp(-module.dt/module.tau)/module.dt).detach(),
        )

    if type(module) == rockpool.nn.modules.LIFTorch:
        return nir.CubaLIF(
            tau_syn=module.tau_syn.detach(),
            tau_mem=module.tau_mem.detach(),
            r=(module.tau_mem * torch.exp(-module.dt / module.tau_mem) / module.dt).detach(),
            v_leak=torch.zeros_like(module.tau_syn).detach(),
            v_threshold=module.threshold,
        )

    elif isinstance(module, rockpool.nn.modules.LinearTorch):
        if module.bias is None:  # Add zero bias if none is present
            return nir.Linear(module.weight.detach())
        else:
            return nir.Affine(module.weight.detach(), module.bias.detach())

    return None


def to_nir(
    module: torch.nn.Module, sample_data: torch.Tensor, model_name: str = "rockpool"
) -> nir.NIR:
    return extract_nir_graph(
        module, _extract_rockpool_module, sample_data, model_name=model_name
    )
