""" 
Getting a deployable configuration file starting from a simulated network requires a couple of conversion steps
The tests here makes sure that intermediate steps works as expected.
To test the pipeline, two weight matrices (w_in and w_rec) are loaded
Every test sequentially combines a `LinearJax` and a `DynapSim` layer whose weight matrices are loaded externally.
"""
import pytest

pytest.importorskip("samna")

from numpy.testing import assert_equal, assert_allclose, assert_array_equal


def test_network_building_first_step():
    """
    test_network_building_first_step checks if the weight matrices after the network construction is identically the same
    """
    ### --- Preliminaries --- ###
    import os
    import numpy as np
    from rockpool.nn.modules import LinearJax
    from rockpool.nn.combinators import Sequential
    from rockpool.devices.dynapse import DynapSim

    # - Path building
    __dirname__ = os.path.dirname(os.path.abspath(__file__))
    __datapath = os.path.join(__dirname__, "data")

    # - Read data
    with open(os.path.join(__datapath, "w_in_optimized.npy"), "rb") as f:
        w_in_opt = np.load(f)

    with open(os.path.join(__datapath, "w_rec_optimized.npy"), "rb") as f:
        w_rec_opt = np.load(f)

    # - Build the network
    Nin, Nrec = w_in_opt.shape
    net = Sequential(
        LinearJax(shape=(Nin, Nrec), has_bias=False, weight=w_in_opt),
        DynapSim(Nrec, has_rec=True, percent_mismatch=0.05, w_rec=w_rec_opt),
    )
    ### --- ###

    # - Test starts here - #
    assert_equal(net[0].weight, w_in_opt)
    assert_equal(net[1].w_rec, w_rec_opt)


def test_net_from_spec():
    """
    test_net_from_spec checks if the network constructed after mapping is the same network before the mapping
    """
    ### --- Preliminaries --- ###
    import os
    import numpy as np
    from rockpool.nn.modules import LinearJax
    from rockpool.nn.combinators import Sequential
    from rockpool.devices.dynapse import DynapSim

    # - Path building
    __dirname__ = os.path.dirname(os.path.abspath(__file__))
    __datapath = os.path.join(__dirname__, "data")

    # - Read data
    with open(os.path.join(__datapath, "w_in_optimized.npy"), "rb") as f:
        w_in_opt = np.load(f)

    with open(os.path.join(__datapath, "w_rec_optimized.npy"), "rb") as f:
        w_rec_opt = np.load(f)

    # - Build the network
    Nin, Nrec = w_in_opt.shape
    net = Sequential(
        LinearJax(shape=(Nin, Nrec), has_bias=False, weight=w_in_opt),
        DynapSim(Nrec, has_rec=True, w_rec=w_rec_opt),
    )
    ### --- ###

    # - Test starts here - #
    from rockpool.devices.dynapse import mapper, dynapsim_net_from_spec

    # - Map the network
    spec = mapper(net.as_graph())
    net_from_spec = dynapsim_net_from_spec(**spec)

    # - Weights and scaling should be identically the same
    assert_equal(net_from_spec[0].weight, w_in_opt)
    assert_equal(net_from_spec[1].w_rec, w_rec_opt)
    assert_equal(net_from_spec[1].Iscale, net[1].Iscale)

    # - Check the parameters
    for key in spec["unclustered"]:
        assert_array_equal(getattr(net[1], key), getattr(net_from_spec[1], key))


def test_net_from_spec_mismatch():
    """
    test_net_from_spec_mismatch is a similar test to `test_net_from_spec`.
    The difference is it uses frozen mismatch in network generation and the resulting current values deviates a bit from the theoretical values
    It checks if the deviation is within the expected limits.
    """

    ### --- Preliminaries --- ###
    import os
    import numpy as np
    from rockpool.nn.modules import LinearJax
    from rockpool.nn.combinators import Sequential
    from rockpool.devices.dynapse import DynapSim

    # - Path building
    __dirname__ = os.path.dirname(os.path.abspath(__file__))
    __datapath = os.path.join(__dirname__, "data")

    # - Read data
    with open(os.path.join(__datapath, "w_in_optimized.npy"), "rb") as f:
        w_in_opt = np.load(f)

    with open(os.path.join(__datapath, "w_rec_optimized.npy"), "rb") as f:
        w_rec_opt = np.load(f)

    # - Build the network
    Nin, Nrec = w_in_opt.shape
    net = Sequential(
        LinearJax(shape=(Nin, Nrec), has_bias=False, weight=w_in_opt),
        DynapSim(Nrec, has_rec=True, percent_mismatch=0.05, w_rec=w_rec_opt),
    )
    ### --- ###

    # - Test starts here - #
    from rockpool.devices.dynapse import mapper, dynapsim_net_from_spec

    # - Map the network
    spec = mapper(net.as_graph())
    net_from_spec = dynapsim_net_from_spec(**spec)

    # - Weights and scaling should be identically the same
    assert_equal(net_from_spec[0].weight, w_in_opt)
    assert_equal(net_from_spec[1].w_rec, w_rec_opt)
    assert_equal(net_from_spec[1].Iscale, net[1].Iscale)

    # Relative tolerance is mismatch
    for key in spec["unclustered"]:
        assert_allclose(getattr(net[1], key), getattr(net_from_spec[1], key), rtol=0.05)


def test_quantization():
    """
    test_quantization checks if the quantized and reconstructed weights are close enough to the original weights
    """

    ### --- Preliminaries --- ###
    import os
    import numpy as np
    from rockpool.nn.modules import LinearJax
    from rockpool.nn.combinators import Sequential
    from rockpool.devices.dynapse import DynapSim

    # - Path building
    __dirname__ = os.path.dirname(os.path.abspath(__file__))
    __datapath = os.path.join(__dirname__, "data")

    # - Read data
    with open(os.path.join(__datapath, "w_in_optimized.npy"), "rb") as f:
        w_in_opt = np.load(f)

    with open(os.path.join(__datapath, "w_rec_optimized.npy"), "rb") as f:
        w_rec_opt = np.load(f)

    # - Build the network
    Nin, Nrec = w_in_opt.shape
    net = Sequential(
        LinearJax(shape=(Nin, Nrec), has_bias=False, weight=w_in_opt),
        DynapSim(Nrec, has_rec=True, w_rec=w_rec_opt),
    )
    ### --- ###

    # - Test starts here - #
    from rockpool.devices.dynapse.quantization.autoencoder.weight_handler import (
        WeightHandler,
    )
    from rockpool.devices.dynapse import mapper, autoencoder_quantization

    # - Map and quantize the network
    spec = mapper(net.as_graph())
    spec.update(autoencoder_quantization(**spec))

    ## Reconstruct the weight matrix indicated by the hardware specification manually
    code = [spec["Iw_0"][0], spec["Iw_1"][0], spec["Iw_2"][0], spec["Iw_3"][0]]

    ## - Input weights
    bits_trans_in = WeightHandler.int2bit_mask(
        n_bits=4, int_mask=spec["weights_in"][0]
    ).T
    w_in = np.sum(bits_trans_in * code, axis=-1).T * spec["sign_in"] / spec["Iscale"]

    ## - Recurrent weights
    bits_trans_rec = WeightHandler.int2bit_mask(
        n_bits=4, int_mask=spec["weights_rec"][0]
    ).T
    w_rec = np.sum(bits_trans_rec * code, axis=-1).T * spec["sign_rec"] / spec["Iscale"]

    # Check if the quantized weight matrix and the original one is close enough
    assert_allclose(w_in[0], w_in_opt, atol=5)
    assert_allclose(w_rec[0], w_rec_opt, atol=3)


def test_network_from_config():
    """
    test_network_from_config goes through all steps of the deployment and reconstructs a network from the configuration object
    It checks if the weight matrices and the parameters are close enough
    The weight parameters are deviated a lot because of quantization
    The network parameters will be deviated less due to bias parameter selection
    """

    ### --- Preliminaries --- ###
    import os
    import numpy as np
    from rockpool.nn.modules import LinearJax
    from rockpool.nn.combinators import Sequential
    from rockpool.devices.dynapse import DynapSim

    # - Path building
    __dirname__ = os.path.dirname(os.path.abspath(__file__))
    __datapath = os.path.join(__dirname__, "data")

    # - Read data
    with open(os.path.join(__datapath, "w_in_optimized.npy"), "rb") as f:
        w_in_opt = np.load(f)

    with open(os.path.join(__datapath, "w_rec_optimized.npy"), "rb") as f:
        w_rec_opt = np.load(f)

    # - Build the network
    Nin, Nrec = w_in_opt.shape
    net = Sequential(
        LinearJax(shape=(Nin, Nrec), has_bias=False, weight=w_in_opt),
        DynapSim(Nrec, has_rec=True, w_rec=w_rec_opt),
    )

    ### --- ###

    # - Test starts here - #
    # - Go through all deployment steps
    # .. seealso ::
    #   :ref:`/devices/DynapSE/post-training.ipynb`

    from rockpool.devices.dynapse import (
        mapper,
        autoencoder_quantization,
        config_from_specification,
        dynapsim_net_from_config,
    )

    # - Deployment Steps
    spec = mapper(net.as_graph())
    spec.update(autoencoder_quantization(**spec))
    config = config_from_specification(**spec)
    net_from_config = dynapsim_net_from_config(**config)

    # - Weight Matrices
    assert_allclose(net_from_config[0].weight, w_in_opt, atol=5)
    assert_allclose(net_from_config[1].w_rec, w_rec_opt, atol=3)

    # - Bias Parameters
    for key in spec["unclustered"]:
        assert_allclose(
            getattr(net[1], key), getattr(net_from_config[1], key), rtol=0.2
        )
