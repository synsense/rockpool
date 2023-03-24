import pytest

pytest.importorskip("jax")


def test_imports():
    import rockpool.utilities.jax_tree_utils as jtu


def test_tree_map_reduce_select():
    import rockpool.utilities.jax_tree_utils as jtu
    import jax.numpy as jnp

    test_tree = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": {
            "e": 5,
            "f": 6,
        },
    }

    proto = jtu.make_prototype_tree(test_tree, test_tree)

    def map_fun(leaf):
        return jnp.nanmax(jnp.array(leaf, float))

    def reduce_fun(x, y):
        return jnp.nanmax(jnp.array([x, y]))

    assert (
        jtu.tree_map_reduce_select(
            test_tree, proto, map_fun, reduce_fun, jnp.array(jnp.nan)
        )
        == 6.0
    ), "Got the incorrect value for map reduce"


def test_tree_map_select():
    import rockpool.utilities.jax_tree_utils as jtu
    import jax.numpy as jnp

    test_tree = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": {
            "e": 5,
            "f": 6,
        },
    }

    proto = jtu.make_prototype_tree(test_tree, test_tree)

    def map_fun(leaf):
        return leaf - 1

    mapped_tree = jtu.tree_map_select(test_tree, proto, map_fun)

    assert mapped_tree["a"] == 0
    assert mapped_tree["b"] == 1
    assert mapped_tree["c"] == 2
    assert mapped_tree["d"]["e"] == 4
    assert mapped_tree["d"]["f"] == 5


def test_tree_map_select_with_rng():
    import rockpool.utilities.jax_tree_utils as jtu
    import jax.numpy as jnp
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

    proto = jtu.make_prototype_tree(test_tree, test_tree)

    def map_fun(leaf, rng):
        return leaf - 1

    mapped_tree = jtu.tree_map_select_with_rng(
        test_tree, proto, map_fun, jax.random.PRNGKey(0)
    )

    assert mapped_tree["a"] == 0
    assert mapped_tree["b"] == 1
    assert mapped_tree["c"] == 2
    assert mapped_tree["d"]["e"] == 4
    assert mapped_tree["d"]["f"] == 5


def test_tree_map_with_rng():
    import rockpool.utilities.jax_tree_utils as jtu
    import jax.numpy as jnp
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
