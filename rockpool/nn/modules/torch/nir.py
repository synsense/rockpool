from numbers import Number
from os import PathLike
from typing import Optional, Union
from rockpool.nn.modules.torch import LIFTorch, ExpSynTorch, LinearTorch
import torch
import nir
from nirtorch import extract_nir_graph, load

__all__ = ["to_nir", "from_nir"]

def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    if isinstance(x, tuple):
        return tuple(_to_tensor(y) for y in x)
    if isinstance(x, Number):
        return x
    parsed = torch.from_numpy(x)
    if parsed.numel() == 1:
        return parsed.item()
    return parsed

def _to_numpy(x):
    return x.detach().numpy()

class RecurrentCubaLIF(torch.nn.Module):
    def __init__(self, lif, lin):
        super().__init__()
        self.lif = lif
        self.lin = lin

    def forward(self, x, state = None):
        if state is None:
            state = torch.zeros_like(x)
        x, _ = self.lif(x + state)
        return x, self.lin(x)

def _convert_nir_to_rockpool(node: nir.NIRNode) -> Optional[torch.nn.Module]:
    if isinstance(node, nir.Input) or isinstance(node, nir.Output):
        return None
    if isinstance(node, nir.NIRGraph):
        # Currently, just parse a recurrent recurrent Cuba LIF graph
        types = {type(v): v for v in node.nodes.values()}
        if len(node.nodes) == 4 and nir.CubaLIF in types and nir.Affine in types:
            layer_lif = _convert_nir_to_rockpool(types[nir.CubaLIF])
            layer_affine = _convert_nir_to_rockpool(types[nir.Affine])
            return RecurrentCubaLIF(layer_lif, layer_affine)
    if isinstance(node, nir.LI):
        return ExpSynTorch(
            shape=int(node.input_type["input"]),
            tau=_to_tensor(node.tau).int(),
            dt=_to_tensor(node.tau / (1+node.r)),
        )
    if isinstance(node, nir.CubaLIF):
        return LIFTorch(
            shape=_to_tensor(node.input_type["input"]),
            tau_mem=_to_tensor(node.tau_mem),
            tau_syn=_to_tensor(node.tau_syn),
            threshold=_to_tensor(node.v_threshold),
            dt=_to_tensor(node.tau_mem / (1+node.r)),
        )
    if isinstance(node, nir.Linear):
        return LinearTorch(
            shape=(int(node.weight.shape[1]), int(node.weight.shape[0])),
            weight=_to_tensor(node.weight.T),
        )
    if isinstance(node, nir.Affine):
        return LinearTorch(
            shape=(int(node.weight.shape[1]), int(node.weight.shape[0])),
            weight=_to_tensor(node.weight.T),
            bias=_to_tensor(node.bias),
        )

    raise NotImplementedError(f"Cannot convert {type(node)} to rockpool module")

def from_nir(source: Union[PathLike, nir.NIRNode]):
    """Generates a rockpool model from a NIR representation.

    Parameters:
        source: If string or Path will try to load from file. Otherwise should be NIR object
    """
    if isinstance(source, PathLike):
        source = nir.load(source)
    
    return load(source, _convert_nir_to_rockpool)

def _extract_rockpool_module(module) -> Optional[nir.NIRNode]:
    if type(module) == ExpSynTorch:
        return nir.LI(
            tau=module.tau,
            v_leak=torch.zeros_like(module.tau).detach(),
            r=(module.tau * torch.exp(-module.dt/module.tau)/module.dt).detach(),
        )

    if type(module) == LIFTorch:
        return nir.CubaLIF(
            tau_syn=_to_numpy(module.tau_syn.squeeze()), # TODO: Necessary to squeeze?
            tau_mem=_to_numpy(module.tau_mem.squeeze()), # TODO: Necessary to squeeze?
            r=_to_numpy(module.tau_mem * torch.exp(-module.dt / module.tau_mem) / module.dt),
            v_leak=_to_numpy(torch.zeros_like(module.tau_syn).squeeze()), # TODO: Necessary to squeeze?
            v_threshold=_to_numpy(module.threshold),
        )

    elif isinstance(module, LinearTorch):
        if module.bias is None:
            return nir.Linear(module.weight.detach().T)
        else:
            return nir.Affine(module.weight.detach().T, module.bias.detach())

    return None


def to_nir(
    module: torch.nn.Module, sample_data: torch.Tensor, model_name: str = "rockpool"
) -> nir.NIRNode:
    return extract_nir_graph(
        module, _extract_rockpool_module, sample_data, model_name=model_name
    )
