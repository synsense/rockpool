def test_sample_and_hold():
    import numpy as np
    from rockpool.devices.xylo.imu.preprocessing import SampleAndHold, Quantizer
    from rockpool.nn.combinators import Sequential

    # sample and hold of a signal of size larger than period
    for length in np.arange(2, 100):
        sig_in = np.random.randn(1, length)
        mod = Sequential(
            Quantizer(scale=0.99 / np.max(np.abs(sig_in)), num_bits=30),
            SampleAndHold(sampling_period=10),
        )
        out, _, _ = mod(sig_in)
        out_diff = np.diff(out.flatten())
        assert np.sum(np.abs(out_diff) > 0) == (length - 1) // mod[1].sampling_period
