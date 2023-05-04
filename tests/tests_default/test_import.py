"""
Test library integrity
"""
import pytest


def test_import():
    """
    Test the import of top level package
    """
    import rockpool


def test_submodule_import():
    """
    Test the import of submodules
    """
    import rockpool.nn.modules
    import rockpool.training
    import rockpool.utilities
    import rockpool.parameters
    import rockpool.nn.combinators


def test_base_attributes():
    import rockpool

    assert rockpool.__version__ == rockpool.__version__
    assert rockpool.list_backends() == rockpool.list_backends()
