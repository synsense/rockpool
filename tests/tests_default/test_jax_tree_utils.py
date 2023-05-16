def test_imports():
    import pytest

    pytest.importorskip("jax")

    import rockpool.utilities.jax_tree_utils as jtu
    from jax.config import config


def test_tree_map_reduce_select():
    import pytest

    pytest.importorskip("jax")

    import rockpool.utilities.jax_tree_utils as jtu
    from jax.config import config
    import jax.numpy as jnp

    config.update("jax_debug_nans", False)

    test_tree = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": {
            "e": 5,
            "f": 6,
        },
    }

    sub_tree = {
        "a": None,
        "d": {
            "e": None,
        },
    }

    proto = jtu.make_prototype_tree(test_tree, sub_tree)

    def map_fun(leaf):
        return jnp.nanmax(jnp.array(leaf, float))

    def reduce_fun(x, y):
        return jnp.nanmax(jnp.array([x, y]))

    assert (
        jtu.tree_map_reduce_select(
            test_tree, proto, map_fun, reduce_fun, jnp.array(jnp.nan)
        )
        == 5.0
    ), "Got the incorrect value for map reduce"


def test_tree_map_select():
    import pytest

    pytest.importorskip("jax")

    import rockpool.utilities.jax_tree_utils as jtu

    test_tree = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": {
            "e": 5,
            "f": 6,
        },
    }

    sub_tree = {
        "a": None,
        "d": {
            "e": None,
        },
    }

    proto = jtu.make_prototype_tree(test_tree, sub_tree)

    def map_fun(leaf):
        return leaf - 1

    mapped_tree = jtu.tree_map_select(test_tree, proto, map_fun)

    assert mapped_tree["a"] == 0
    assert mapped_tree["d"]["e"] == 4


def test_tree_map_select_with_rng():
    import pytest

    pytest.importorskip("jax")

    import rockpool.utilities.jax_tree_utils as jtu
    import jax

    test_tree = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": {
            "e": 5,
            "f": 6,
        },
    }

    sub_tree = {
        "a": None,
        "d": {
            "e": None,
        },
    }

    proto = jtu.make_prototype_tree(test_tree, sub_tree)

    def map_fun(leaf, rng):
        return leaf - 1

    mapped_tree = jtu.tree_map_select_with_rng(
        test_tree, proto, map_fun, jax.random.PRNGKey(0)
    )

    assert mapped_tree["a"] == 0
    assert mapped_tree["d"]["e"] == 4


def test_tree_map_with_rng():
    import pytest

    pytest.importorskip("jax")

    import rockpool.utilities.jax_tree_utils as jtu
    import jax

    test_tree = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": {
            "e": 5,
            "f": 6,
        },
    }

    def map_fun(leaf, rng):
        return leaf - 1

    mapped_tree = jtu.tree_map_with_rng(test_tree, map_fun, jax.random.PRNGKey(0))

    assert mapped_tree["a"] == 0
    assert mapped_tree["b"] == 1
    assert mapped_tree["c"] == 2
    assert mapped_tree["d"]["e"] == 4
    assert mapped_tree["d"]["f"] == 5
