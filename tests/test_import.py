"""
Test library integrity
"""

import sys


def test_import():
    """
    Test the import of top level package
    """
    import NetworksPython


def test_submodule_import():
    """
    Test the import of submodules
    """
    from NetworksPython import layers

    # from NetworksPython.layers import recurrent
    # from NetworksPython.layers import internal
