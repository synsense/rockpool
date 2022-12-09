"""
DynapSE convenience method to analyse a network and return a parameter subtree on which it makese sense to apply mismatch

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
09/12/2022
"""

from typing import Any, Dict
from jax.tree_util import PyTreeDef

from rockpool.nn.modules.jax.linear_jax import LinearJax
from rockpool.nn.modules.jax import JaxModule
from rockpool.nn.combinators import JaxSequential
from rockpool.transform.mismatch import module_registery

from jax.tree_util import tree_flatten, tree_unflatten

__all__ = ["static_mismatch_prototype", "dynamic_mismatch_prototype"]

# static_mismatch_prototype
def static_mismatch_prototype(mod: JaxModule) -> PyTreeDef:

    ref_tree = module_registery(mod)
    set_tree = mod.simulation_parameters()

    ref_tree = __set_leaves_bool(ref_tree, False)
    set_tree = __set_leaves_bool(set_tree, True)

    prototype = __update_nested_leaves(ref_tree, set_tree)
    return prototype


def dynamic_mismatch_prototype(mod: JaxModule) -> PyTreeDef:

    ref_tree = module_registery(mod)
    set_tree = mod.parameters()

    ref_tree = __set_leaves_bool(ref_tree, False)
    set_tree = __set_leaves_bool(set_tree, True)

    prototype = __update_nested_leaves(ref_tree, set_tree)
    return prototype


def __set_leaves_bool(tree: Dict[str, Any], val: bool) -> Dict[str, bool]:
    tree_flat, treedef = tree_flatten(tree)
    leaves = [val for _ in tree_flat]
    prototype = tree_unflatten(treedef, leaves)
    return prototype


def __update_nested_leaves(
    params_ref: Dict[str, Any], params_set: Dict[str, Any]
) -> Dict[str, Any]:
    for key, val in params_ref.items():
        if not isinstance(params_ref[key], dict):
            if key in params_set:
                params_ref[key] = params_set[key]
        elif key in params_set:
            __update_nested_leaves(params_ref[key], params_set[key])

    return params_ref

