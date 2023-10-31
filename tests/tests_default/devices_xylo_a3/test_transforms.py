def test_imports():
    from rockpool.devices.xylo.syns65302.transform import (
        AudioQuantizer,
        AmplitudeNormalizer,
    )

    assert AudioQuantizer is not None
    assert AmplitudeNormalizer is not None


def test_normalizer():
    """
    this module tests AmplitudeNormalizer transform which used to normalize the input signal's amplitude into
    [-1.0, 1.0].
    """
    from rockpool.devices.xylo.syns65302.transform import AmplitudeNormalizer
    import numpy as np

    # signal info
    fs = 1000
    time_sec = 1.0
    freq = 10
    time_vec = np.arange(0, time_sec, step=1 / fs)
    max_amp = 2.0
    sig_in = max_amp * np.sin(2 * np.pi * freq * time_vec)

    # add outlier
    sample_index = 10
    sig_in[sample_index] += 10

    # init normalizer
    normalizer = AmplitudeNormalizer(p=0.002)

    sig_out, _, _ = normalizer(sig_in)
    assert sig_out.shape[1] == time_vec.shape[0]
    assert sig_out.max() <= 1.0 and sig_out.min() >= -1.0


def test_quantizer():
    """
    this module tests Audio Amplitude Quantizer which used to quantize the digital audio signal to specific number of
    bits integer data.
    """
    from rockpool.devices.xylo.syns65302.transform import (
        AudioQuantizer,
        AmplitudeNormalizer,
    )
    import numpy as np

    # signal info
    fs = 1000
    time_sec = 1.0
    freq = 10
    time_vec = np.arange(0, time_sec, step=1 / fs)
    max_amp = 2.0
    sig_in = max_amp * np.sin(2 * np.pi * freq * time_vec)

    # add outlier
    sample_index = 10
    sig_in[sample_index] += 10

    # init normalizer quantizer
    normalizer = AmplitudeNormalizer(p=0.002)
    quantizer = AudioQuantizer()
    num_bits = quantizer.num_bits

    sig_out, _, _ = normalizer(sig_in)
    quant_out, _, _ = quantizer(sig_out)

    assert quant_out.max() <= 2 ** (num_bits - 1)
    assert quant_out.min() >= -(2 ** (num_bits - 1))
