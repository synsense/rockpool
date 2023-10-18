from numbers import Number
from os import PathLike
from typing import Optional, Union
from rockpool.nn.modules.torch import LIFTorch, ExpSynTorch, LinearTorch, TorchModule
import torch
import nir
from nirtorch import extract_nir_graph, load
from nirtorch.from_nir import GraphExecutor
import warnings
import numpy as np
import rockpool.graph as rg
from types import MethodType

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

def _convert_nir_to_rockpool(node: nir.NIRNode) -> Optional[TorchModule]:
    if isinstance(node, nir.Input) or isinstance(node, nir.Output):
        return None
    
    if isinstance(node, nir.NIRGraph):
        # Currently, just parse a recurrent recurrent Cuba LIF graph
        types = {type(v): v for v in node.nodes.values()}
        if len(node.nodes) == 4 and nir.CubaLIF in types and nir.Affine in types:
            lif_node = types[nir.CubaLIF]
            affine_node = types[nir.Affine]
            layer_lif = LIFTorch(
                shape=_to_tensor(lif_node.input_type["input"]),
                tau_mem=_to_tensor(lif_node.tau_mem),
                tau_syn=_to_tensor(lif_node.tau_syn),
                threshold=_to_tensor(lif_node.v_threshold),
                dt=torch.min(torch.tensor(_to_tensor(lif_node.tau_mem / (1+lif_node.r)))).item(),
                has_rec=True,
                w_rec=_to_tensor(affine_node.weight.T),
                bias=_to_tensor(lif_node.v_leak),
            )

            if not torch.allclose(_to_tensor(affine_node.bias), torch.zeros_like(_to_tensor(affine_node.bias))):
                warnings.warn('Affine biases are not supported in recurrent LIF modules, on import from NIR.')

            return layer_lif

    if isinstance(node, nir.LI):
        return ExpSynTorch(
            shape=_to_tensor(node.input_type["input"]),
            tau=_to_tensor(node.tau).int(),
            dt=torch.min(torch.tensor(_to_tensor(node.tau / (1+node.r)))).item(),
        )
    
    if isinstance(node, nir.CubaLIF):
        return LIFTorch(
            shape=_to_tensor(node.input_type["input"]),
            tau_mem=_to_tensor(node.tau_mem),
            tau_syn=_to_tensor(node.tau_syn),
            threshold=_to_tensor(node.v_threshold),
            dt=torch.min(torch.tensor(_to_tensor(node.tau_mem / (1+node.r)))).item(),
            bias=_to_tensor(node.v_leak),
        )
    
    if isinstance(node, nir.Linear):
        return LinearTorch(
            shape=(int(node.weight.shape[1]), int(node.weight.shape[0])),
            weight=_to_tensor(node.weight.T),
            has_bias=False,
        )
    
    if isinstance(node, nir.Affine):
        return LinearTorch(
            shape=(int(node.weight.shape[1]), int(node.weight.shape[0])),
            weight=_to_tensor(node.weight.T),
            bias=_to_tensor(node.bias),
            has_bias=True,
        )

    raise NotImplementedError(f"Cannot convert {type(node)} to rockpool module")

def _convert_nir_to_rockpool_torch(node: nir.NIRNode) -> Optional[torch.nn.Module]:
    mod = _convert_nir_to_rockpool(node)
    if mod is not None:
        mod.to_torch()
    return mod

def from_nir(source: Union[PathLike, nir.NIRNode]):
    """Generates a rockpool model from a NIR representation.

    Parameters:
        source: If string or Path will try to load from file. Otherwise should be NIR object
    """
    if isinstance(source, PathLike):
        source = nir.load(source)
    
    ge = load(source, _convert_nir_to_rockpool_torch)

    # - Add in an `as_graph` method
    def as_graph(self: GraphExecutor) -> rg.GraphModuleBase:
        mod_graphs = []

        for node in self.graph.node_list:
            if node.name not in ['input', 'output']:
                mod = self.get_submodule(node.name)
                mod_graphs.append(mod.as_graph())
                node._index = len(mod_graphs)-1

        for node in self.graph.node_list:
            for parent in self.graph.find_source_nodes_of(node):
                rg.connect_modules(mod_graphs[parent._index], mod_graphs[node._index])

        # - Find input and output nodes
        for index in range(len(self.execution_order)):
            if self.execution_order[index].name in ['input', 'output']:
                pass
            else:
                mod_input = mod_graphs[self.execution_order[index]._index]
                break

        for index in range(len(self.execution_order)-1, 0, -1):
            if self.execution_order[index].name in ['input', 'output']:
                pass
            else:
                mod_output = mod_graphs[self.execution_order[index]._index]
                break

        return rg.GraphHolder(
            mod_input.input_nodes,
            mod_output.output_nodes,
            f"{type(self).__name__}_{id(self)}",
            self,
        )

    ge.as_graph = MethodType(as_graph, ge)

    return ge


def _extract_rockpool_module(module) -> Optional[nir.NIRNode]:
    if type(module) == ExpSynTorch:
        return nir.LI(
            tau=module.tau,
            v_leak=torch.zeros_like(module.tau).detach(),
            r=(module.tau * torch.exp(-module.dt/module.tau)/module.dt).detach(),
        )

    if type(module) == LIFTorch:
        if module.size_out != module.size_in:
            raise NotImplementedError('Multiple synaptc states are not yet supported for export')
        
        return nir.CubaLIF(
            tau_syn=np.broadcast_to(_to_numpy(module.tau_syn.squeeze()), module.size_out), # TODO: Necessary to squeeze?
            tau_mem=np.broadcast_to(_to_numpy(module.tau_mem.squeeze()), module.size_out), # TODO: Necessary to squeeze?
            r=np.broadcast_to(_to_numpy(module.tau_mem * torch.exp(-module.dt / module.tau_mem) / module.dt), module.size_out),
            v_leak=np.broadcast_to(_to_numpy(torch.zeros_like(module.tau_syn).squeeze()), module.size_out), # TODO: Necessary to squeeze?
            v_threshold=np.broadcast_to(_to_numpy(module.threshold), module.size_out),
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
