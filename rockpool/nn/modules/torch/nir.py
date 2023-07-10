from typing import Optional

import rockpool
import torch
import nir
from nirtorch import extract_nir_graph

from rockpool.nn.modules import LIFTorch, LinearTorch, ExpSynTorch

def _extract_rockpool_module(module: rockpool.nn.modules) -> Optional[nir.NIRNode]:
    if type(module) == ExpSynTorch:
        return nir.LI(
            tau=module.tau.detach(),
            v_leak=torch.zeros_like(module.tau).detach(),
            r=(module.tau * torch.exp(-module.dt/module.tau)/module.dt).detach(),
        )

    if type(module) == LIFTorch:
        return nir.CubaLIF(
            tau_syn=module.tau_syn.detach(),
            tau_mem=module.tau_mem.detach(),
            v_leak=torch.zeros_like(module.tau_syn).detach(),
            r=(module.tau * torch.exp(-module.dt / module.tau_mem) / module.dt).detach(),
        )

    elif isinstance(module, LinearTorch):
        if module.bias is None:  # Add zero bias if none is present
            return nir.Linear(
                module.weight.detach(), torch.zeros(*module.weight.shape[:-1])
            )
        else:
            return nir.Linear(module.weight.detach(), module.bias.detach())

    return None


def to_nir(
    module: torch.nn.Module, sample_data: torch.Tensor, model_name: str = "rockpool"
) -> nir.NIR:
    return extract_nir_graph(
        module, _extract_rockpool_module, sample_data, model_name=model_name
    )
