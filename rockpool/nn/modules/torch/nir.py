"""
Utilities for import and export of Rockpool ``torch`` networks to NIR.
For more information see https://neuroir.org/docs/

Manuscript: https://arxiv.org/abs/2311.14641

Github: https://github.com/neuromorphs/NIR

Defines the :py:func:`.to_nir` and :py:func:`.from_nir` helper functions.

For more information see :ref:`/advanced/nir_export_import.ipynb`.
"""

from numbers import Number
from os import PathLike
from typing import Optional, Union
from rockpool.nn.modules.torch import (
    LIFTorch,
    ExpSynTorch,
    LinearTorch,
    TorchModule,
    LIFNeuronTorch,
    TorchModule,
)
import torch
import nir
import nirtorch
from nirtorch import extract_nir_graph, load
from nirtorch.from_nir import GraphExecutor
import warnings
import copy
import numpy as np
import rockpool.graph as rg
from types import MethodType

from rockpool.typehints import Tensor

__all__ = ["to_nir", "from_nir"]


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    if isinstance(x, tuple):
        return tuple(_to_tensor(y) for y in x)
    if isinstance(x, Number):
        return x
    parsed = torch.from_numpy(copy.copy(x))
    if parsed.numel() == 1:
        return parsed.item()
    return parsed


def _to_numpy(x):
    return x.detach().numpy()


def _convert_nir_to_rockpool(node: nir.NIRNode) -> Optional[TorchModule]:
    """
    Helper function for mapping NIR node instances to equivalent Rockpool modules

    Args:
        node (nir.NIRNode): An NIR node instance to convert

    Returns:
        TorchModule: A Rockpool :py:class:`.Torchmodule` instance generated from ``node``

    """
    if isinstance(node, nir.ir.Input) or isinstance(node, nir.ir.Output):
        return None

    if isinstance(node, nir.ir.NIRGraph):
        # Currently, just parse a recurrent recurrent Cuba LIF graph
        types = {type(v): v for v in node.nodes.values()}
        if (
            len(node.nodes) == 4
            and nir.ir.CubaLIF in types
            and (nir.ir.Affine in types or nir.ir.Linear in types)
        ):
            lif_node = types[nir.ir.CubaLIF]
            affine_node = (
                types[nir.ir.Affine] if nir.ir.Affine in types else types[nir.ir.Linear]
            )

            layer_lif = LIFTorch(
                shape=_to_tensor(lif_node.input_type["input"]),
                tau_mem=_to_tensor(lif_node.tau_mem),
                tau_syn=_to_tensor(lif_node.tau_syn),
                threshold=_to_tensor(lif_node.v_threshold),
                dt=torch.min(
                    torch.as_tensor(_to_tensor(lif_node.tau_mem / (1 + lif_node.r)))
                ).item(),
                has_rec=True,
                w_rec=_to_tensor(affine_node.weight.T),
                bias=_to_tensor(lif_node.v_leak),
            )

            if nir.ir.Affine in types and not torch.allclose(
                _to_tensor(affine_node.bias),
                torch.zeros_like(_to_tensor(affine_node.bias)),
            ):
                warnings.warn(
                    "Affine biases are not supported in recurrent LIF modules, on import from NIR."
                )

            return layer_lif

    if isinstance(node, nir.ir.LI):
        return ExpSynTorch(
            shape=_to_tensor(node.input_type["input"]),
            tau=_to_tensor(node.tau).int(),
            dt=torch.min(torch.as_tensor(_to_tensor(node.tau / (1 + node.r)))).item(),
        )

    if isinstance(node, nir.ir.LIF):
        return LIFNeuronTorch(
            shape=_to_tensor(node.input_type["input"]),
            tau_mem=_to_tensor(node.tau_mem),
            threshold=_to_tensor(node.v_threshold),
            bias=_to_tensor(node.v_leak),
            dt=torch.min(
                torch.as_tensor(_to_tensor(node.tau_mem / (1 + node.r)))
            ).item(),
        )

    if isinstance(node, nir.ir.CubaLIF):
        return LIFTorch(
            shape=_to_tensor(node.input_type["input"]),
            tau_mem=_to_tensor(node.tau_mem),
            tau_syn=_to_tensor(node.tau_syn),
            threshold=_to_tensor(node.v_threshold),
            dt=torch.min(
                torch.as_tensor(_to_tensor(node.tau_mem / (1 + node.r)))
            ).item(),
            bias=_to_tensor(node.v_leak),
        )

    if isinstance(node, nir.ir.Linear):
        return LinearTorch(
            shape=(int(node.weight.shape[1]), int(node.weight.shape[0])),
            weight=_to_tensor(node.weight.T),
            has_bias=False,
        )

    if isinstance(node, nir.ir.Affine):
        return LinearTorch(
            shape=(int(node.weight.shape[1]), int(node.weight.shape[0])),
            weight=_to_tensor(node.weight.T),
            bias=_to_tensor(node.bias),
            has_bias=True,
        )

    raise NotImplementedError(f"Cannot convert {type(node)} to rockpool module")


def _convert_nir_to_rockpool_torch(node: nir.NIRNode) -> Optional[torch.nn.Module]:
    """
    A helper function that converts NIR graphs to Rockpool nets, returning ``torch``-compatible modules

    Args:
        node (nir.NIRNode): A NIR node to convert

    Returns:
        torch.nn.Module: A ``torch``-compatible Rockpool module converted from ``node``
    """
    mod = _convert_nir_to_rockpool(node)
    if mod is not None:
        mod.to_torch()
    return mod


def from_nir(source: Union[PathLike, nir.NIRNode]) -> torch.nn.Module:
    """
    Generate a rockpool model from a NIR representation

    Args:
        source (Union[PathLike, nir.NIRNode]): Either a filename containing a NIR graph (produced with ``nir.write()``), or an already-loaded NIR graph (e.g. loaded with ``nir.read()``)

    Returns:
        torch.nn.Module: A ``torch``-compatible Rockpool module converted from ``source``
    """
    # - Load from file
    if isinstance(source, PathLike):
        source = nir.read(source)

    # - Use nirtorch to convert the NIR graph
    ge = nirtorch.load(source, _convert_nir_to_rockpool_torch)

    # - Add in an `as_graph` method for Rockpool mapping support
    def as_graph(self: GraphExecutor) -> rg.GraphModuleBase:
        mod_graphs = []

        for node in self.graph.node_list:
            if node.name not in ["input", "output"]:
                mod = self.get_submodule(node.name)
                mod_graphs.append(mod.as_graph())
                node._index = len(mod_graphs) - 1

        for node in self.graph.node_list:
            if node.name not in ["input", "output"]:
                for parent in self.graph.find_source_nodes_of(node):
                    if parent.name not in ["input", "output"]:
                        rg.connect_modules(
                            mod_graphs[parent._index], mod_graphs[node._index]
                        )

        # - Find input and output nodes
        for index in range(len(self.execution_order)):
            if self.execution_order[index].name in ["input", "output"]:
                pass
            else:
                mod_input = mod_graphs[self.execution_order[index]._index]
                break

        for index in range(len(self.execution_order) - 1, 0, -1):
            if self.execution_order[index].name in ["input", "output"]:
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

    # - Patch the instance with the `.as_graph()` method and return
    ge.as_graph = MethodType(as_graph, ge)

    return ge


def _extract_rockpool_module(module: TorchModule) -> Optional[nir.NIRNode]:
    """
    A helper function mapping Rockpool modules to NIR classes

    Args:
        module (TorchModule): A Rockpool :py:class:`.TorchModule` to convert

    Returns:
        Optional[nir.NIRNode]: An NIR node entity equivalent to ``module``. If no mapping is possible, ``None`` is returned
    """
    if type(module) == ExpSynTorch:
        return nir.LI(
            tau=module.tau,
            v_leak=torch.zeros_like(module.tau).detach(),
            r=(module.tau * torch.exp(-module.dt / module.tau) / module.dt).detach(),
        )

    if type(module) == LIFTorch:
        if module.size_out != module.size_in:
            raise NotImplementedError(
                "Multiple synaptc states are not yet supported for export"
            )

        return nir.CubaLIF(
            tau_syn=np.broadcast_to(
                _to_numpy(module.tau_syn.squeeze()), module.size_out
            ),
            tau_mem=np.broadcast_to(
                _to_numpy(module.tau_mem.squeeze()), module.size_out
            ),
            r=np.broadcast_to(
                _to_numpy(
                    module.tau_mem * torch.exp(-module.dt / module.tau_mem) / module.dt
                ),
                module.size_out,
            ),
            v_leak=np.broadcast_to(
                _to_numpy(torch.zeros_like(module.bias).squeeze()), module.size_out
            ),
            v_threshold=np.broadcast_to(_to_numpy(module.threshold), module.size_out),
        )

    elif isinstance(module, LinearTorch):
        if module.bias is None:
            return nir.Linear(module.weight.detach().T)
        else:
            return nir.Affine(module.weight.detach().T, module.bias.detach())

    elif isinstance(module, LIFNeuronTorch):
        return nir.LIF(
            tau=np.broadcast_to(_to_numpy(module.tau_mem.squeeze()), module.size_out),
            r=np.broadcast_to(
                _to_numpy(
                    module.tau_mem * torch.exp(-module.dt / module.tau_mem) / module.dt
                ),
                module.size_out,
            ),
            v_leak=np.broadcast_to(
                _to_numpy(torch.zeros_like(module.bias).squeeze()), module.size_out
            ),
            v_threshold=np.broadcast_to(_to_numpy(module.threshold), module.size_out),
        )

    return None


def to_nir(
    module: TorchModule,
    sample_data: Optional[Tensor] = None,
    model_name: str = "rockpool",
) -> nir.NIRNode:
    """
    Convert a Rockpool module into a NIR graph for export

    Args:
        module (TorchModule): A Rockpool :py:class:`.TorchModule` to export
        sample_data (Optional[Tensor]): If needed, a ``Tensor`` containg dummy data of the shape expected by ``module``. Default: Will be generated automatically from ``module.size_in``.
        model_name (str): An optional string naming the model. Default: ``"rockpool"``.

    Returns:
        nir.NIRNode: A NIR graph for export
    """

    # - Generate sample data if not provided
    sample_data = (
        torch.zeros((1, module.size_in)) if sample_data is None else sample_data
    )

    # - Extract and return NIR graph
    return extract_nir_graph(
        module, _extract_rockpool_module, sample_data, model_name=model_name
    )
