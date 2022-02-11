"""
Utility functions for working with trees.
"""

from warnings import warn

from typing import Tuple, Generator, Any, Callable, List

import copy

import jax
import jax.tree_util as tu

import functools

# - Set up some useful types
from rockpool.typehints import Tree, Leaf, Value, JaxRNGKey


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


def set_nested(tree: Tree, branch: Tuple, value: Any) -> None:
    """
    Set a value in a tree branch, specifying a branch

    The leaf node must already exist in the tree.

    Args:
        tree (Tree): A nested dictionary tree structure
        branch (Tuple[str]): A branch: a tuple of indices to walk through the tree
        value (Any): The value to set at the tree leaf
    """
    # - Start from the root node
    node = tree

    # - Iterate along the branch
    for key in branch[:-1]:
        node = node[key]

    # - Set the leaf value
    node[branch[-1]] = value


def set_matching(full_tree: Tree, target_tree: Tree, value: Any) -> None:
    """
    Set the values in a full tree in-place, for branches that match a target tree

    Args:
        full_tree (Tree): A tree to search over. The values in this tree will be replaced with ``value``
        target_tree (Tree): A tree that defines the target branches to set in ``full_tree``. Matching branches in ``full_tree`` will have their values replaced with ``value``
        value (Any): The value to set in ``full_tree``.
    """
    for branch in branches(target_tree):
        set_nested(full_tree, branch, value)


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
            set_nested(prototype, branch, True)
        else:
            # - Assign `False` in the prototype tree
            set_nested(prototype, branch, False)

    # - Return the prototype tree
    return prototype


def tree_map_reduce_select(
    tree: Tree,
    protoype_tree: Tree,
    map_fun: Callable[[Leaf], Value],
    reduce_fun: Callable[[Value, Value], Value],
    null: Any = jax.numpy.nan,
) -> Value:
    """
    Perform a map-reduce operation over a tree, but only matching selected nodes

    `map_fun()` is a function that will be mapped over the leaves of a tree. It will only be applied to leaf nodes in `tree` when the corresponding leaf node in `prototype_tree` is ``True``.

    `map_fun()` can return whatever you like; however, you must ensure that the `null` value has the same shape as the value returned by `map_fun()` applied to a leaf node.

    `reduce_fun()` is a reduction function. This is called using `functools.reduce` over the mapped outputs of `map_fun()`.

    `reduce_fun(intermediate_value, map_value)` accepts two mapped leaf values, and should combine them into a single result. The first value will be retained as the intermediate computaiton, and passed to the next call to `reduce()`.

    Examples:
        >>> map_fun = np.nanmax
        >>>
        >>> def reduce_fun(x, y):
        >>>     return np.nanmax(np.array([x, y]))
        >>>
        >>> tree_map_reduce_select({'a': 1}, {'a': 1}, map_fun, reduce_fun)
        1

        >>> def map_fun(leaf):
        >>>     return {'mean': np.mean(leaf),
        >>>             'std': np.std(leaf),
        >>>             }
        >>>
        >>> def reduce_fun(x, y):
        >>>     return {'mean': np.mean(np.array([x['mean'], y['mean']])),
        >>>             'std': x['std'] + y['std'],
        >>>             }
        >>>
        >>> null = map_fun(np.nan)
        >>>
        >>> tree_map_reduce_select({'a': 1}, {'a': 1}, map_fun, reduce_fun, null)
        {'mean': 1, 'std': NaN}

    Args:
        tree (Tree): A tree over which to operate
        prototype_tree (Tree): A prototype tree, with structure matching that of `tree`, but with ``bool`` leaves. Only leaf nodes in `tree` with ``True`` in the corresponding prototype tree node will be modified.
        map_fun (Callable[[Leaf], Value]): A function to perform on selected nodes in `tree`. `map_fun` has the signature `map_fun(leaf_node) -> value`
        reduce_fun (Callable[[Value, Value], Value]): A function that collects two values and returns the combination of the two values, to reduce over the mapped function. `reduce_fun` has the signature `reduce_fun(value, value) -> value`.
        null (Any): The "null" value to return from the map operation, if the leaf node is not selected in `prototype_tree`. Default: ``jax.numpy.nan``

    Returns:
        Value: The result of the map-reduce operation over the tree
    """

    tree_flat, _ = tu.tree_flatten(tree)
    proto_flat, _ = tu.tree_flatten(protoype_tree)

    def map_or_null(leaf: Any, select: bool) -> Any:
        return jax.lax.cond(
            select,
            lambda _: map_fun(leaf),
            lambda _: null,
            0,
        )

    # - Map function over leaves
    mapped = [map_or_null(*xs) for xs in zip(tree_flat, proto_flat)]

    # - Reduce function over leaves
    return functools.reduce(reduce_fun, mapped)


def tree_map_select(
    tree: Tree, prototype_tree: Tree, map_fun: Callable[[Leaf], Value]
) -> Tree:
    """
    Map a scalar function over a tree, but only matching selected nodes

    Notes:
        `map_fun` must be a scalar function. This means that if the input is of shape ``(N, M)``, the output must also be of shape ``(N, M)``. Otherwise you will get an error.

    Args:
        tree (Tree): A tree over which to operate
        prototype_tree (Tree): A prototype tree, with structure matching that of `tree`, but with ``bool`` leaves. Only leaf nodes in `tree` with ``True`` in the corresponding prototype tree node will be modified.
        map_fun (Callable[[Leaf], Value]): A scalar function to perform on selected nodes in `tree`

    Returns:
        Tree: A tree with the same structure as `tree`, with leaf nodes replaced with the output of `map_fun()` for each leaf.
    """
    # - Flatten both trees
    tree_flat, treedef = tu.tree_flatten(tree)
    proto_flat, _ = tu.tree_flatten(prototype_tree)

    # - A function that conditionally maps over the tree leaves
    def map_or_original(leaf: Any, select: bool) -> Any:
        return jax.lax.cond(
            select,
            lambda _: map_fun(leaf),
            lambda _: leaf,
            0,
        )

    # - Map function over leaves
    mapped = [map_or_original(*xs) for xs in zip(tree_flat, proto_flat)]

    # - Return tree
    return tu.tree_unflatten(treedef, mapped)


def tree_map_select_with_rng(
    tree: Tree,
    prototype_tree: Tree,
    map_fun: Callable[[Leaf, JaxRNGKey], Value],
    rng_key: JaxRNGKey,
) -> Tree:
    """
    Map a scalar function over a tree, but only matching selected nodes. Includes jax-compatible random state

    Notes:
        `map_fun()` must be a scalar function. This means that if the input is of shape ``(N, M)``, the output must also be of shape ``(N, M)``. Otherwise you will get an error.
        The signature of `map_fun()` is `map_fun(leaf, rng_key) -> value`.

    Args:
        tree (PyTree): A tree over which to operate
        prototype_tree (PyTree): A prototype tree, with structure matching that of `tree`, but with ``bool`` leaves. Only leaf nodes in `tree` with ``True`` in the corresponding prototype tree node will be modified.
        map_fun (Callable[[Leaf, JaxRNGKey], Value]): A scalar function to perform on selected nodes in `tree`. The second argument is a jax pRNG key to use when generating random state.
    """
    # - Flatten both trees
    tree_flat, treedef = tu.tree_flatten(tree)
    proto_flat, _ = tu.tree_flatten(prototype_tree)

    # - A function that conditionally maps over the tree leaves
    def map_or_original(leaf: Any, select: bool, rng_key: Any) -> Any:
        return jax.lax.cond(
            select,
            lambda _: map_fun(leaf, rng_key),
            lambda _: leaf,
            0,
        )

    # - Map function over leaves
    _, *subkeys = jax.random.split(rng_key, len(tree_flat) + 1)
    mapped = [map_or_original(*xs) for xs in zip(tree_flat, proto_flat, subkeys)]

    # - Return tree
    return tu.tree_unflatten(treedef, mapped)


def tree_multimap_with_rng(
    tree: Tree,
    map_fun: Callable[[Value, JaxRNGKey, Any], Value],
    rng_key: JaxRNGKey,
    *rest: Any,
) -> Tree:
    """
    Perform a multimap over a tree, splitting and inserting an RNG key for each leaf

    This utility maps a function over the leaves of a tree, when the function requires an RNG key to operate. The utility will automatically split the RNG key to generate a new key for each leaf. Then `map_fun` will be called for each leaf, with the signature ``map_fun(leaf_value, rng_key, *rest)``.

    `rest` is an optional further series of arguments to map over the tree, such that each additional argument must have the same tree structure as `tree`. See the documentation for `jax.tree_util.tree_multimap` for more information.

    Args:
        tree (Tree): A tree to work over
        map_fun (Callable[[Value, JaxRNGKey, Any], Value]): A function to map over the tree. The function must have the signature ``map_fun(leaf_value, rng_key, *rest)``
        rng_key (JaxRNGKey): An initial RNG key to split
        *rest: A tuple of additional `tree`-shaped arguments that will be collectively mapped over `tree` when calling `map_fun`.

    Returns:
        Tree: The `tree`-shaped result of mapping `map_fun` over `tree`.
    """
    # - Flatten the input tree
    tree_flat, treedef = tu.tree_flatten(tree)

    # - Split RNG keys for each tree leaf
    _, *subkeys = jax.random.split(rng_key, len(tree_flat) + 1)
    subkeys_tree = tu.tree_unflatten(treedef, subkeys)

    # - Map function over the tree and return
    return tu.tree_multimap(map_fun, tree, subkeys_tree, *rest)


def tree_find(tree: Tree) -> List:
    """
    Return the tree branches to tree nodes that evaluate to ``True``

    Args:
        tree (Tree): A tree to examine

    Returns:
        list: A list of all tree branches, for which the corresponding tree leaf evaluate to ``True``
    """
    # - Get a list of all tree branches
    all_branches = list(branches(tree))

    # - Return a list of branches to leaves that evaluate to `True`
    return [branch for branch in all_branches if get_nested(tree, branch)]
