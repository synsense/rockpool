"""
Test backend availablity

NOTE : `samna` requires an extra step of control. For details check the ``test_samna_availability`` function below
"""


def test_samna_availability():
    """
    test_samna_availability checks if the ``backend_available()`` utility can correctly identifies the samna availability or not.
    In the case that one installed samna and then uninstalled via pip
    * `pip install samna` then `pip uninstall samna`,
    samna leaves a trace and `import samna` does not raise an error even though the package is not available.

    NOTE : unistall samna and try again.
    """
    from rockpool.utilities import backend_available

    try:
        import samna

        try:
            print(samna.__version__)
            assert backend_available("samna") == True
        except:
            assert backend_available("samna") == False
    except:
        assert backend_available("samna") == False
