import copy

from rockpool.typehints import Tree, Leaf, Node, TreeDef
from typing import Tuple, Any, Dict, Callable, Union, List


def tree_map(tree: Tree, f: Callable) -> Tree:
    # - Initialise a new root
    root = {}

    # - Loop over nodes
    for k, v in tree.items():
        # - Is this a nested dict?
        if isinstance(v, dict):
            # - Recurse
            root[k] = tree_map(v, f)

        else:
            # - Map function over this value
            root[k] = f(v)

    return root


def tree_flatten(tree: Tree, leaves: Union[List, None] = None) -> Tuple[Tuple, TreeDef]:
    # - Initialise leaves if starting from the root
    leaves = [] if leaves is None else leaves

    # - Initialise a new treedef root
    treedef = {}

    # - Loop over nodes
    for k, v in tree.items():
        # - Is this a nested dict?
        if isinstance(v, dict):
            # - Recurse and build the treedef
            _, treedef[k] = tree_flatten(v, leaves)

        else:
            # - Record this leaf and build the treedef
            leaves.append(v)
            treedef[k] = None

    return leaves, treedef


def tree_unflatten(treedef: TreeDef, leaves: List, leaves_tail: List = None) -> Tree:
    tree = copy.deepcopy(treedef)
    leaves_tail = copy.deepcopy(leaves) if leaves_tail is None else leaves_tail

    # - Loop over nodes
    for k, v in tree.items():
        # - Is this a nested dict?
        if isinstance(v, dict):
            # - Recurse
            tree[k] = tree_unflatten(treedef[k], leaves, leaves_tail)

        else:
            tree[k] = leaves_tail.pop(0)

    return tree


def tree_update(target: Tree, additional: Tree) -> None:
    for k, v in additional.items():
        if isinstance(v, Tree) and k in target:
            tree_update(target[k], v)
        else:
            target.update({k: v})
