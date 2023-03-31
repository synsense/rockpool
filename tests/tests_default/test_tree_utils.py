def test_imports():
    import rockpool.utilities.tree_utils as tu


def test_branches():
    import rockpool.utilities.tree_utils as tu

    test_tree = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": {
            "e": 5,
            "f": 6,
        },
    }

    branches_known = [("a",), ("b",), ("c",), ("d", "e"), ("d", "f")]

    for b, kb in zip(tu.branches(test_tree), branches_known):
        assert b == kb, f"Found non-matching branch: {b}, expected {kb}."


def test_get_nested():
    import rockpool.utilities.tree_utils as tu

    test_tree = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": {
            "e": 5,
            "f": 6,
        },
    }

    assert tu.get_nested(test_tree, ("a")) == 1
    assert tu.get_nested(test_tree, ("b")) == 2
    assert tu.get_nested(test_tree, ("c")) == 3
    assert tu.get_nested(test_tree, ("d", "e")) == 5
    assert tu.get_nested(test_tree, ("d", "f")) == 6


def test_set_nested_copy():
    import rockpool.utilities.tree_utils as tu

    test_tree = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": {
            "e": 5,
            "f": 6,
        },
    }

    t2 = tu.set_nested(test_tree, ("d", "e"), 101, inplace=False)
    assert t2["d"]["e"] == 101
    assert test_tree["d"]["e"] == 5


def test_set_nested_inplace():
    import rockpool.utilities.tree_utils as tu

    test_tree = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": {
            "e": 5,
            "f": 6,
        },
    }

    tu.set_nested(test_tree, ("d", "f"), 0, inplace=True)
    assert test_tree["d"]["f"] == 0


def test_set_matching_copy():
    import rockpool.utilities.tree_utils as tu

    test_tree = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": {
            "e": 5,
            "f": 6,
        },
    }

    sub_tree = {"a": True, "d": {"e": True}}

    modified_tree = tu.set_matching(test_tree, sub_tree, 0, inplace=False)

    assert modified_tree["a"] == 0
    assert test_tree["a"] == 1
    assert modified_tree["b"] == 2
    assert modified_tree["d"]["e"] == 0
    assert test_tree["d"]["e"] == 5


def test_set_matching_inplace():
    import rockpool.utilities.tree_utils as tu

    test_tree = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": {
            "e": 5,
            "f": 6,
        },
    }

    sub_tree = {"a": True, "d": {"e": True}}

    tu.set_matching(test_tree, sub_tree, 0, inplace=True)
    assert test_tree["a"] == 0
    assert test_tree["d"]["e"] == 0


def test_set_matching_select_copy():
    import rockpool.utilities.tree_utils as tu

    test_tree = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": {
            "e": 5,
            "f": 6,
        },
    }

    sub_tree = {"a": True, "d": {"e": True, "f": False}}

    modified_tree = tu.set_matching_select(test_tree, sub_tree, 0, inplace=False)
    assert modified_tree["a"] == 0
    assert test_tree["a"] == 1
    assert modified_tree["b"] == 2
    assert modified_tree["d"]["e"] == 0
    assert test_tree["d"]["e"] == 5
    assert modified_tree["d"]["f"] == 6


def test_set_matching_select_inplace():
    import rockpool.utilities.tree_utils as tu

    test_tree = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": {
            "e": 5,
            "f": 6,
        },
    }

    sub_tree = {"a": True, "d": {"e": True, "f": False}}

    tu.set_matching_select(test_tree, sub_tree, 0, inplace=True)
    assert test_tree["a"] == 0
    assert test_tree["d"]["e"] == 0
    assert test_tree["d"]["f"] == 6


def test_make_prototype_tree():
    import rockpool.utilities.tree_utils as tu

    test_tree = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": {
            "e": 5,
            "f": 6,
        },
    }

    sub_tree = {"a": 1, "d": {"e": 5}}

    proto_tree = tu.make_prototype_tree(test_tree, sub_tree)

    branches_known = [("a",), ("b",), ("c",), ("d", "e"), ("d", "f")]

    for b, kb in zip(tu.branches(proto_tree), branches_known):
        assert b == kb, f"Found non-matching branch: {b}, expected {kb}."

    assert proto_tree["a"] is True
    assert proto_tree["b"] is False
    assert proto_tree["c"] is False
    assert proto_tree["d"]["e"] is True
    assert proto_tree["d"]["f"] is False


def test_tree_map():
    import rockpool.utilities.tree_utils as tu

    test_tree = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": {
            "e": 5,
            "f": 6,
        },
    }

    def map_fun(leaf):
        return leaf + 1

    mapped_tree = tu.tree_map(test_tree, map_fun)
    assert mapped_tree["a"] == 2
    assert mapped_tree["b"] == 3
    assert mapped_tree["c"] == 4
    assert mapped_tree["d"]["e"] == 6
    assert mapped_tree["d"]["f"] == 7


def test_tree_flatten():
    import rockpool.utilities.tree_utils as tu

    test_tree = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": {
            "e": 5,
            "f": 6,
        },
    }

    leaves, treedef = tu.tree_flatten(test_tree)
    known_leaves = [1, 2, 3, 5, 6]

    for f, k in zip(leaves, known_leaves):
        assert f == k, f"Found unexpected value {f}, expected {k}."

    known_branches = [("a",), ("b",), ("c",), ("d", "e"), ("d", "f")]

    for b, kb in zip(tu.branches(treedef), known_branches):
        assert b == kb, f"Found unexpected branch {b}, expected {kb}."


def test_tree_unflatten():
    import rockpool.utilities.tree_utils as tu

    treedef = {
        "a": None,
        "b": None,
        "c": None,
        "d": {
            "e": None,
            "f": None,
        },
    }

    leaves = [1, 2, 3, 5, 6]

    tree = tu.tree_unflatten(treedef, leaves)

    known_branches = [("a",), ("b",), ("c",), ("d", "e"), ("d", "f")]

    for b, kb in zip(tu.branches(tree), known_branches):
        assert b == kb, f"Found unexpected branch {b}, expected {kb}."

    assert tree["a"] == 1
    assert tree["b"] == 2
    assert tree["c"] == 3
    assert tree["d"]["e"] == 5
    assert tree["d"]["f"] == 6


def test_tree_update_copy():
    import rockpool.utilities.tree_utils as tu

    test_tree = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": {
            "e": 5,
            "f": 6,
        },
    }

    additional_tree = {"g": {"h": 7, "i": 8}}

    modified_tree = tu.tree_update(test_tree, additional_tree, inplace=False)

    assert modified_tree["a"] == 1
    assert "g" in modified_tree
    assert modified_tree["g"]["h"] == 7
    assert modified_tree["g"]["i"] == 8
    assert "g" not in test_tree


def test_tree_update_inplace():
    import rockpool.utilities.tree_utils as tu

    test_tree = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": {
            "e": 5,
            "f": 6,
        },
    }

    additional_tree = {"g": {"h": 7, "i": 8}}

    tu.tree_update(test_tree, additional_tree, inplace=True)
    assert "g" in test_tree
    assert test_tree["g"]["h"] == 7
    assert test_tree["g"]["i"] == 8


def test_tree_find():
    import rockpool.utilities.tree_utils as tu

    test_tree = {
        "a": True,
        "b": False,
        "c": False,
        "d": {
            "e": True,
            "f": False,
        },
    }

    known_branches = [("a",), ("d", "e")]

    for b, kb in zip(tu.tree_find(test_tree), known_branches):
        assert b == kb, f"Found unexpected branch {b}, expected {kb}."
