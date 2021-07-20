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
    import rockpool.nn.modules
    import rockpool.training
    import rockpool.utilities
    import rockpool.parameters

    # from rockpool.layers import recurrent
    # from rockpool.layers import internal
