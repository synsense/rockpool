import pytest


def test_imports():
    from rockpool.devices.xylo.xylo_a3 import DeltaSigma, PDM_Microphone


def test_deltasigma():
    """
    this module tests the deltasigma module implemented for converting an analog signal into a pulse-density-modulated {+1,-1} bitstream.
    """
    import numpy as np
    from rockpool.devices.xylo.xylo_a3 import DeltaSigma

    #* produce a sinusoid signal
    freq = 1000
    num_periods = 4
    duration = num_periods/freq

    # deltasigma params
    bandwidth = 5 * freq
    fs_min = 2*bandwidth
    
    oversampling = 32
    fs = oversampling * fs_min
    amplitude = 2.3

    # check all deltasigma orders
    order_list = [1, 2, 3, 4, 5]

    for order in order_list:

        # build the deltasigma module
        ds = DeltaSigma(
            amplitude=amplitude,
            bandwidth=bandwidth,
            order=order,
            fs=fs
        )

        #* an oversampled input signal
        time_vec = np.arange(0, duration, step=1/fs)
        safe_amplitude = 0.9 * amplitude

        sig_in = safe_amplitude * np.sin(2*np.pi*freq*time_vec)

        
        sig_out_Q, sig_out, sig_in_resampled, recording = ds.evolve(
            sig_in=sig_in,
            sample_rate=fs,
            record=True
        )

        # check the dimensions
        assert len(sig_out_Q) == len(sig_in)
        assert len(sig_out_Q) == len(sig_out)
        assert len(sig_in_resampled) == len(sig_in)

        assert len(np.unique(sig_out_Q)) <= 2

        # check the recording dimension
        assert recording.shape == (len(sig_in), order)

        # check that the outputs are all close to the target amplitude
        assert np.allclose(np.abs(sig_out_Q), amplitude*np.ones_like(sig_out_Q))


        #* a signal sampled with a lower sampling rate
        time_vec = np.arange(0, duration, step=1/fs_min)
        safe_amplitude = 0.9 * amplitude

        sig_in = safe_amplitude * np.sin(2*np.pi*freq*time_vec)

        sig_out_Q, sig_out, sig_in_resampled, recording = ds.evolve(
            sig_in=sig_in,
            sample_rate=fs_min,
            record=True
        )

        # check the dimensions
        assert len(sig_out_Q) == len(sig_in_resampled)
        assert len(sig_out_Q) == len(sig_out)
        assert len(sig_in_resampled) >= len(sig_in)

        assert len(np.unique(sig_out_Q)) <= 2

        # check the recording dimension
        assert recording.shape == (len(sig_in_resampled), order)

        # check that all the amplitudes are close to the target amplitude
        assert np.allclose(np.abs(sig_out_Q), amplitude*np.ones_like(sig_out_Q))


def test_pdm_mic():
    """ this module verifies the default PDM microphone setting in Xylo-A3. """
    from rockpool.devices.xylo.xylo_a3 import PDM_Microphone
    import numpy as np
    
    mic = PDM_Microphone()

    # produce a sinusoid signal and pass it from the microphone
    freq = 1_000
    num_periods = 2
    duartion = num_periods/freq

    pdm_rate = mic.fs

    time_vec = np.arange(0, duartion, step=1/pdm_rate)
    safe_amplitude = 0.9
    sig_in = safe_amplitude * np.sin(2*np.pi*freq*time_vec)

    pdm_bits, state, recording = mic.evolve(
        audio=sig_in,
        sample_rate=pdm_rate,
        record=True,
    )

    # check the dimensions
    assert len(np.unique(pdm_bits)) == 2
    assert np.allclose(np.abs(pdm_bits), np.ones_like(pdm_bits))

    assert len(pdm_bits) == len(sig_in)

    # check the recordings
    deltasigma_signal_pre_Q = recording['deltasigma_signal_pre_Q']
    assert np.all(np.sign(deltasigma_signal_pre_Q) * pdm_bits >= 0)

    deltasigma_filter_states = recording['deltasigma_filter_states']
    assert deltasigma_filter_states.shape == (len(pdm_bits), mic.sdm_order)


if __name__ == '__main__':
    #test_deltasigma()
    test_pdm_mic()