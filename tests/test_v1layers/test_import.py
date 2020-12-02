"""
Test library integrity
"""

import sys


def test_import():
    """
    Test the import of top level package
    """
    import rockpool


def test_submodule_import():
    """
    Test the import of submodules
    """
    from nn import layers

    # from rockpool.layers import recurrent
    # from rockpool.layers import internal
