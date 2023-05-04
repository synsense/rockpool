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
    print("Rockpool version", rockpool.__version__)

if __name__ == '__main__':
    test_base_attributes()