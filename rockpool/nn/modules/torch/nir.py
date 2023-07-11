from os import PathLike
from typing import Optional, Union
from rockpool.nn.modules.torch import LIFTorch, ExpSynTorch, LinearTorch
from rockpool.nn.combinators import Sequential
import torch
import nir
from nirtorch import extract_nir_graph

__all__ = ["to_nir", "from_nir"]

def expsyntorch_from_nir(
        node: nir.LI, shape: Union[tuple, int]=None,
):
    # if node.v_leak != 0:
    #     raise ValueError("`v_leak` must be 0")

    return ExpSynTorch(
        shape=shape,
        tau=node.tau,
        dt=node.tau / (1+node.r),
    )

def cubalif_from_nir(
    node: nir.CubaLIF, shape: Union[tuple, int]=None,
):
    # if torch.sum(node.v_leak) != 0:
    #     raise ValueError("`v_leak` must be 0")

    return LIFTorch(
        shape=shape,
        tau_mem=node.tau_mem,
        tau_syn=node.tau_syn,
        threshold=node.v_threshold,
        dt=node.tau_mem / (1+node.r),
    )


def linear_from_nir(
    node: nir.Linear,
):
    return LinearTorch(
        shape=(node.weight.shape[0], node.weight.shape[1]),
        weight=node.weight,
    )

def affline_from_nir(
    node: nir.Affine,
):

    return LinearTorch(
        shape=(node.weight.shape[0], node.weight.shape[1]),
        weight=node.weight,
        bias=node.bias,
    )

node_conversion_functions = {
    nir.LI: expsyntorch_from_nir,
    nir.CubaLIF: cubalif_from_nir,
    nir.Linear: linear_from_nir,
    nir.Affine: affline_from_nir,
}

def from_nir(
    source: Union[PathLike, nir.NIR], shape: Union[tuple, int]=None,
):
    """Generates a rockpool model from a NIR representation.

    Parameters:
        source: If string or Path will try to load from file. Otherwise should be NIR object
    """

    # Convert nodes to rockpool layers
    # layers = [
    #     node_conversion_functions[type(node)](node)
    #     for node in source.nodes
    # ]
    layers = []
    for node in source.nodes:

        if type(node) in [nir.LI, nir.CubaLIF]:
            num_neuron = store_node.weight.shape[1]
            layer = node_conversion_functions[type(node)](node, num_neuron)
        elif type(node) in [nir.Linear, nir.Affine]:
            layer = node_conversion_functions[type(node)](node)
        store_node = node
        layers.append(layer)

    edge_array = torch.tensor(source.edges)
    # subtract source edges and check if all edges are sequential
    edge_array = edge_array - edge_array[:, :1]
    if edge_array[:, 0].sum() == 0 and edge_array[:, 1].sum() == edge_array.shape[0]:
        return Sequential(*layers)
    else:
        raise NotImplementedError("Only sequential models are supported at the moment")


def _extract_rockpool_module(module) -> Optional[nir.NIRNode]:
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
            r=(module.tau_mem * torch.exp(-module.dt / module.tau_mem) / module.dt).detach(),
            v_leak=torch.zeros_like(module.tau_syn).detach(),
            v_threshold=module.threshold,
        )

    elif isinstance(module, LinearTorch):
        if module.bias is None:
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
