"""
DynapSE convenience method to analyse a network and return a parameter subtree on which it makese sense to apply mismatch
"""

from typing import Any, Dict, List
from jax.tree_util import tree_flatten, tree_unflatten

from rockpool.nn.modules.jax import JaxModule
from rockpool.transform.mismatch import module_registery

__all__ = ["frozen_mismatch_prototype", "dynamic_mismatch_prototype"]


def frozen_mismatch_prototype(mod: JaxModule) -> Dict[str, bool]:
    """
    frozen_mismatch_prototype process the module attributes tree and returns a frozen mismatch prototype which indicates the values to be deviated.
    Frozen means that the parameters indicated here should be deviated at compile time. Prior to simulating the circuitry.
    One can use this mismatch prototype in trainining pipeliene as well, but the parameter deviations might result going off the operational limits.

    :param mod: DynapSim network module which subject to mismatch deviations
    :type mod: JaxModule
    :return: the prototype tree leading the mismatch generation process
    :rtype: Dict[str, bool]
    """

    ref_tree = __ref_tree(mod)
    prototype = __update_nested_leaves(ref_tree, __frozen_set_tree_dynapsim(mod))
    prototype = __update_nested_leaves(ref_tree, __frozen_set_tree_linear(mod))

    return prototype


def dynamic_mismatch_prototype(mod: JaxModule) -> Dict[str, bool]:
    """
    dynamic_mismatch_prototype process the module attributes tree and returns a dynamical mismatch prototype which indicates the values to be deviated at run-time.
    The resulting prototype can be used during training pipeline. The method targets only the trainables.

    :param mod: DynapSim network module which subject to mismatch deviations
    :type mod: JaxModule
    :return: the prototype tree leading the mismatch generation process
    :rtype: Dict[str, bool]
    """

    ref_tree = __ref_tree(mod)
    prototype = __update_nested_leaves(ref_tree, __dynamic_set_tree_dynapsim(mod))
    prototype = __update_nested_leaves(ref_tree, __dynamic_set_tree_linear(mod))

    return prototype


### --- Private Section --- ###


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


def __set_tree(
    module: JaxModule, target_mod_name: str, *args: List[str]
) -> Dict[str, bool]:
    """
    __set_tree traces all the nested module and registered parameters of the `JaxModule` base given and returns a tree,
    whose leaves includes only the parameters listed in the arguments list

    :param module: a self-standing simulation module or a combinator object
    :type module: JaxModule
    :param target_mod_name: the function considers only the parameters of this module.
    :type target_mod_name: str
    :return: a nested parameter tree whose leaves are all True, indicating the parameters to process
    :rtype: Dict[str, bool]
    """
    __attrs, __modules = module._get_attribute_registry()
    __dict = {
        k: True
        for k, v in __attrs.items()
        if ((k in args) and (target_mod_name == module.class_name))
    }

    # - Append sub-module attributes as nested dictionaries
    __sub_attrs = {}
    for k, m in __modules.items():
        # [0] -> module , [1] -> name
        __sub_attrs[k] = __set_tree(m[0], target_mod_name, *args)

    __dict.update(__sub_attrs)
    return __dict


def __frozen_set_tree_dynapsim(module: JaxModule) -> Dict[str, bool]:
    """
    __frozen_set_tree_dynapsim the set-tree for the parameters of `DynapSim` to apply frozen mismatch

    :param module: a self-standing simulation module or a combinator object
    :type module: JaxModule
    :return: a set-tree, whose leaves are all True, listing the parameters to process
    :rtype: Dict[str, bool]
    """
    return __set_tree(
        module,
        "DynapSim",
        "Idc",
        "If_nmda",
        "Igain_ahp",
        "Igain_mem",
        "Igain_syn",
        "Ipulse_ahp",
        "Ipulse",
        "Iref",
        "Ispkthr",
        "Itau_ahp",
        "Itau_mem",
        "Itau_syn",
        "Iw_ahp",
        "C_ahp",
        "C_syn",
        "C_pulse_ahp",
        "C_pulse",
        "C_ref",
        "C_mem",
        "Io",
        "kappa_n",
        "kappa_p",
        "Ut",
        "Vth",
    )


def __frozen_set_tree_linear(module: JaxModule) -> Dict[str, bool]:
    """
    __frozen_set_tree_linear the set-tree for the parameters of `LinearJax` to apply frozen mismatch

    :param module: a self-standing simulation module or a combinator object
    :type module: JaxModule
    :return: a set-tree, whose leaves are all True, listing the parameters to process
    :rtype: Dict[str, bool]
    """
    return __set_tree(module, "LinearJax")


def __dynamic_set_tree_dynapsim(module: JaxModule) -> Dict[str, bool]:
    """
    __dynamic_set_tree_dynapsim the set-tree for the parameters of `DynapSim` to apply dynamic mismatch

    :param module: a self-standing simulation module or a combinator object
    :type module: JaxModule
    :return: a set-tree, whose leaves are all True, listing the parameters to process
    :rtype: Dict[str, bool]
    """
    return __set_tree(module, "DynapSim", "w_rec")


def __dynamic_set_tree_linear(module: JaxModule) -> Dict[str, bool]:
    """
    __dynamic_set_tree_linear the set-tree for the parameters of `LinearJax` to apply dynamic mismatch

    :param module: a self-standing simulation module or a combinator object
    :type module: JaxModule
    :return: a set-tree, whose leaves are all True, listing the parameters to process
    :rtype: Dict[str, bool]
    """
    return __set_tree(module, "LinearJax", "weight")


def __ref_tree(module: JaxModule) -> Dict[str, bool]:
    """
    __ref_tree returns a reference tree, listing all possible `SimulationParameters` and `Parameters` of a `JaxModule`

    :param module: a self-standing simulation module or a combinator object
    :type module: JaxModule
    :return: a set-tree, whose leaves are all False, listing the options to process
    :rtype: Dict[str, bool]
    """
    ref_tree = module_registery(module)
    ref_tree = __set_leaves_bool(ref_tree, False)
    return ref_tree
