"""
Tree manipulation utilities with no external dependencies

This module provides methods for building and manipulating trees. 

A `Tree` is a nested dictionary. A `Leaf` is any other object. A `Node` is a non-leaf `Tree` node. A `TreeDef` is a nested dictionary with no data, only structure.
"""

import copy

from warnings import warn

from rockpool.typehints import Tree, Leaf, Node, TreeDef
from typing import Tuple, Any, Dict, Callable, Union, List, Optional, Generator

__all__ = [
    "branches",
    "get_nested",
    "set_nested",
    "set_matching",
    "set_matching_select",
    "make_prototype_tree",
    "tree_map",
    "tree_flatten",
    "tree_unflatten",
    "tree_update",
    "tree_find",
]


def branches(tree: Tree, prefix: list = None) -> Generator[Tuple, None, None]:
    """
    Generate all branches (paths from root node to any leaf) in a tree

    Args:
        tree (Tree): A nested dictionary tree structure
        prefix (list): The current branch prefix. Default: `None` (this is the root)

    Yields:
        Tuple[str]:
    """
    # - Start from the root node
    if prefix is None:
        prefix = []

    # - Loop over nodes
    for k, v in tree.items():
        # - Is this a nested dict?
        if isinstance(v, dict):
            # - Get branches from subtree
            yield from branches(v, prefix + [k])

        else:
            # - Yield this branch
            yield tuple(prefix + [k])


def get_nested(tree: Tree, branch: Tuple) -> None:
    """
    Get a value from a tree branch, specifying a branch

    Args:
        tree (Tree): A nested dictionary tree structure
        branch (Tuple[str]): A branch: a tuple of indices to walk through the tree
    """
    # - Start from the root node
    node = tree

    # - Iterate along the branch
    for key in branch[:-1]:
        node = node[key]

    # - Get the leaf value
    return node[branch[-1]]


def set_nested(tree: Tree, branch: Tuple, value: Any, inplace: bool = False) -> Tree:
    """
    Set a value in a tree branch, specifying a branch

    The leaf node must already exist in the tree.

    Args:
        tree (Tree): A nested dictionary tree structure
        branch (Tuple[str]): A branch: a tuple of indices to walk through the tree
        value (Any): The value to set at the tree leaf
        inplace (bool): If ``False`` (default), a copy of the tree will be returned. If ``True``, the operation will be performed in place, and the original tree will be returned

    Returns:
        Tree: The modified tree
    """
    if not inplace:
        tree = copy.deepcopy(tree)

    # - Start from the root node
    node = tree

    # - Iterate along the branch
    for key in branch[:-1]:
        node = node[key]

    # - Set the leaf value
    node[branch[-1]] = value

    return tree


def set_matching(
    full_tree: Tree, target_tree: Tree, value: Any, inplace: bool = False
) -> Tree:
    """
    Set the values in a full tree, for branches that match a target tree

    Args:
        full_tree (Tree): A tree to search over. The values in this tree will be replaced with ``value``
        target_tree (Tree): A tree that defines the target branches to set in ``full_tree``. Matching branches in ``full_tree`` will have their values replaced with ``value``
        value (Any): The value to set in ``full_tree``.
        inplace (bool): If ``False`` (default), a copy of the tree will be returned. If ``True``, the operation will be performed in place, and the original tree will be returned

    Returns:
        Tree: The modified tree
    """
    if not inplace:
        full_tree = copy.deepcopy(full_tree)

    for branch in branches(target_tree):
        set_nested(full_tree, branch, value, inplace=True)

    return full_tree


def set_matching_select(
    full_tree: Tree, target_tree: Tree, value: Any, inplace: bool = False
) -> Tree:
    """
    Set the values in a full tree, for branches that match a target tree, if the target tree leaf nodes evaluate to ``True``

    Args:
        full_tree (Tree): A tree to search over. The values in this tree will be replaced with ``value``
        target_tree (Tree): A tree that defines the target branches to set in ``full_tree``. Matching branches in ``full_tree`` will have their values replaced with ``value``, if the leaf node in ``target_tree` evaluates to ``True``
        value (Any): The value to set in ``full_tree``.
        inplace (bool): If ``False`` (default), a copy of the tree will be returned. If ``True``, the operation will be performed in place, and the original tree will be returned

    Returns:
        Tree: The modified tree
    """
    if not inplace:
        full_tree = copy.deepcopy(full_tree)

    for branch in branches(target_tree):
        if get_nested(target_tree, branch):
            set_nested(full_tree, branch, value, inplace=True)

    return full_tree


def make_prototype_tree(full_tree: Tree, target_tree: Tree) -> Tree:
    """
    Construct a tree with boolean leaves, for nodes that match a target tree

    Make a prototype tree, indicating which nodes in a large tree should be selected for analysis or processing. This is done on the basis of a smaller "target" tree, which contains only the leaf nodes of interest.

    Examples:
        >>> target_tree = {'a': 0, 'b': {'b2': 0}}
        >>> full_tree = {'a': 1, 'b': {'b1': 2, 'b2': 3}, 'c': 4, 'd': 5}
        >>> make_prototype_tree(full_tree, target_tree)
        {'a': True, 'b': {'b1': False, 'b2': True}, 'c': False, 'd': False}

    Args:
        full_tree (Tree): A large tree to search through.
        target_tree (Tree): A tree with only few leaf nodes. These nodes will be identifed within the full tree.

    Returns:
        Tree: A nested tree with the same tree structure as `full_tree`, but with ``bool`` leaf nodes. Leaf nodes will be ``True`` for branches matching those specified in `target_tree`, and ``False`` otherwise.
    """
    # - Make a copy of the input tree
    prototype = copy.deepcopy(full_tree)

    # - Get a list of target and full branches
    targets = list(branches(target_tree))
    full_branches = list(branches(full_tree))

    # - Sanity check the trees
    if len(full_branches) < len(targets):
        warn(
            SyntaxWarning(
                "make_prototype_tree: The `target` tree has more nodes than the `full` tree. Please check the order of arguments."
            )
        )

    # - Loop over all leaf branches in full tree
    for branch in full_branches:
        # - Is this a target branch?
        if branch in targets:
            # - Assign `True` in the prototype tree
            set_nested(prototype, branch, True, inplace=True)
        else:
            # - Assign `False` in the prototype tree
            set_nested(prototype, branch, False, inplace=True)

    # - Return the prototype tree
    return prototype


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


def tree_update(target: Tree, additional: Tree, inplace: bool = False) -> Tree:
    """
    Perform a recursive update of a tree to insert or replace nodes from a second tree

    Requires a ``target`` `Tree` and a source `Tree` ``additional``, which will provide the source data to update in ``target``.

    Both ``target`` and ``additional`` will be traversed depth-first simultaneously. `Leaf` nodes that exist in ``target`` but not in ``additional`` will not be modified. `Leaf` nodes that exist in ``additional`` but not in ``target`` will be inserted into ``target`` at the corresponding location. `Leaf` nodes that exist in both trees will have their data updated from ``additional`` to ``target``, using the python :py:func:`update` function.

    Args:
        target (Tree): The tree to update.
        additional (Tree): The source tree to insert / replace nodes from, into ``target``. Will not be modified.
        inplace (bool): If ``False`` (default), a copy of the tree will be returned. If ``True``, the operation will be performed in place, and the original tree will be returned.

    Returns:
        Tree: The modified target tree
    """
    if not inplace:
        target = copy.deepcopy(target)

    for k, v in additional.items():
        if isinstance(v, dict) and k in target:
            tree_update(target[k], v, inplace=True)
        else:
            target.update({k: v})

    return target


def tree_find(tree: Tree) -> Generator[Tuple, None, None]:
    """
    Generate the tree branches to tree nodes that evaluate to ``True``

    Args:
        tree (Tree): A tree to examine

    Returns:
        list: A list of all tree branches, for which the corresponding tree leaf evaluate to ``True``
    """
    # - Loop over tree branches
    for branch in branches(tree):
        # - Yield branches to leaves that evaluate to `True`
        if get_nested(tree, branch):
            yield branch
