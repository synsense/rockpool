"""
Optional interfacing tests making sure that Dynap-SE2 HDK and computer connection is healthy.

NOTE : The tests here requires a Dynap-SE2 HDK connected to the computer.
"""

import pytest

def test_time_stamp():
    """
    test_time_stamp tries to connect to the HDK and read the time stamp reported by the FPGA.
    It compares the FPGA time with the clock time
    """
    from numpy.testing import assert_allclose
    from rockpool.devices.dynapse import find_dynapse_boards, DynapseSamna
    import samna
    import time

    # - Connect to device
    try:
        se2_devices = find_dynapse_boards()
    
    except ConnectionError:
        pytest.skip("This test requires a connected Dynap-SE2 Stack Board HDK.")

    # - Let's say some samna devices are connected like DynapCNN, but no SE2.
    if not len(se2_devices) > 0:
        pytest.skip("This test requires a connected Dynap-SE2 Stack Board HDK.")

    else:
        se2 = DynapseSamna(se2_devices[0], samna.dynapse2.Dynapse2Configuration(), {})

    for sleep_interval in [0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]:

        # - Initial time reading from the CPU and HDK connected
        tic = toc = time.time()
        t_0 = se2.current_timestamp()

        # - Some idle time
        time.sleep(sleep_interval)

        # - Final time reading
        toc = time.time()
        t_1 = se2.current_timestamp()

        # - make sure that the time readings from the FPGA makes sense
        assert_allclose((t_1 - t_0), (toc - tic), atol=1e-2)
