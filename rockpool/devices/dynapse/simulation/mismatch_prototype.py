"""
DynapSE convenience method to analyse a network and return a parameter subtree on which it makese sense to apply mismatch

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
09/12/2022
"""

from typing import Any, Dict
from jax.tree_util import tree_flatten, tree_unflatten

from rockpool.nn.modules.jax import JaxModule
from rockpool.transform.mismatch import module_registery

__all__ = ["frozen_mismatch_prototype", "dynamic_mismatch_prototype"]


def frozen_mismatch_prototype(mod: JaxModule) -> Dict[str, bool]:
    """
    frozen_mismatch_prototype process the module parameter tree and returns a frozen mismatch prototype which indicates the values to be deviated.
    Frozen means that the parameters indicated here should be deviated at compile time. Prior to simulating the circuitry.
    One can use this mismatch prototype in trainining pipeliene as well, but the parameter deviations might result going off the operational limits.

    :param mod: DynapSim network module which subject to mismatch deviations
    :type mod: JaxModule
    :return: the prototype tree leading the mismatch generation process
    :rtype: Dict[str, bool]
    """

    ref_tree = module_registery(mod)
    set_tree = mod.simulation_parameters()

    ref_tree = __set_leaves_bool(ref_tree, False)
    set_tree = __set_leaves_bool(set_tree, True)

    prototype = __update_nested_leaves(ref_tree, set_tree)
    return prototype


def dynamic_mismatch_prototype(mod: JaxModule) -> Dict[str, bool]:
    """
    dynamic_mismatch_prototype process the module parameter tree and returns a dynamical mismatch prototype which indicates the values to be deviated at run-time.
    The resulting prototype can be used during training pipeline. The method targets only the trainables.

    :param mod: DynapSim network module which subject to mismatch deviations
    :type mod: JaxModule
    :return: the prototype tree leading the mismatch generation process
    :rtype: Dict[str, bool]
    """

    ref_tree = module_registery(mod)
    set_tree = mod.parameters()

    ref_tree = __set_leaves_bool(ref_tree, False)
    set_tree = __set_leaves_bool(set_tree, True)

    prototype = __update_nested_leaves(ref_tree, set_tree)
    return prototype


def __set_leaves_bool(tree: Dict[str, Any], val: bool) -> Dict[str, bool]:
    """
    __set_leaves_bool flattens a parameter tree and creates a conditional prototype tree setting all the leaves to a boolean value

    :param tree: the target parameter tree
    :type tree: Dict[str, Any]
    :param val: the boolean value that all the leaves will be set
    :type val: bool
    :return: a prototype tree leading a condititonal process
    :rtype: Dict[str, bool]
    """
    tree_flat, treedef = tree_flatten(tree)
    leaves = [val for _ in tree_flat]
    prototype = tree_unflatten(treedef, leaves)
    return prototype


def __update_nested_leaves(
    params_ref: Dict[str, Any], params_set: Dict[str, Any]
) -> Dict[str, Any]:
    """
    __update_nested_leaves is a merging utility which takes a bigger nested tree as a reference and updates its leaves by a secondary sub-set tree

    :param params_ref: the bigger reference tree whose leaves are subject to update
    :type params_ref: Dict[str, Any]
    :param params_set: the tree which updates the leaves of the reference tree
    :type params_set: Dict[str, Any]
    :return: an updated version of the reference tree
    :rtype: Dict[str, Any]
    """
    for key in params_ref:
        if not isinstance(params_ref[key], dict):
            if key in params_set:
                params_ref[key] = params_set[key]
        elif key in params_set:
            __update_nested_leaves(params_ref[key], params_set[key])

    return params_ref
