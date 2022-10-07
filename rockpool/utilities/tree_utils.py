"""
Tree manipulation utilities with no external dependencies

This module provides methods for building and manipulating trees. 

A `Tree` is a nested dictionary. A `Leaf` is any other object. A `Node` is a non-leaf `Tree` node. A `TreeDef` is a nested dictionary with no data, only structure.
"""

import copy

from rockpool.typehints import Tree, Leaf, Node, TreeDef
from typing import Tuple, Any, Dict, Callable, Union, List, Optional

__all__ = ["tree_map", "tree_flatten", "tree_unflatten"]


def tree_map(tree: Tree, f: Callable) -> Tree:
    """
    Map a function over the leaves of a tree

    This function performs a recurdive depth-first traversal of the tree.

    Args:
        tree (Tree): A tree to traverse
        f (Callable): A function which is called on each leaf of the tree. Must have the signature ``Callable[Leaf] -> Any``

    Returns:
        Tree: A tree with the same structure as ``tree``, with leaf nodes replaced with the result of calling ``f`` on each leaf.

    """
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


def tree_flatten(
    tree: Tree, leaves: Union[List[Any], None] = None
) -> Tuple[List[Any], TreeDef]:
    """
    Flatten a tree into a linear list of leaf nodes and a tree description

    This function operates similar to ``jax.tree_utils.tree_flatten``, but is *not* directly compatible.

    A `Tree` ``tree`` will be serialised into a simple list of leaf nodes, which can then be conveniently manipulated. A `TreeDef` will also be returned, which is a nested dictionary with the same structure as ``tree``.

    The function :py:func:`.tree_unflatten` performs the reverse operation.

    Args:
        tree (Tree): A tree to flatten
        leaves (Optional[List[Any]]): Used recursively. Should be left as ``None`` by the user.

    Returns:
        Tuple[List[Any], TreeDef]: A list of leaf nodes from the flattened tree, and a tree definition.
    """

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


def tree_unflatten(
    treedef: TreeDef, leaves: List, leaves_tail: Optional[List[Any]] = None
) -> Tree:
    """
    Build a tree from a flat list of leaves, plus a tree definition

    This function takes a flattened tree representation, as built by :py:func:`.tree_flatten`, and reconstructs a matching `Tree` structure.

    Args:
        treedef (TreeDef): A tree definition as returned by :py:func:`.tree_flatten`
        leaves (List[Any]): A list of leaf nodes to use in constructing the tree
        leaves_tail (Optional[List[Any]]): Used recursively. Should be left as ``None`` by the end user

    Returns:
        Tree: The reconstructed tree, with leaves taken from ``leaves``
    """
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
