"""
Test library integrity
"""

import sys


def test_import():
    """
    Test the import of top level package
    """
    import Rockpool


def test_submodule_import():
    """
    Test the import of submodules
    """
    from Rockpool import layers

    # from Rockpool.layers import recurrent
    # from Rockpool.layers import internal
