import warnings
warnings.filterwarnings('ignore')
import numpy as np
import numpy.random as npr
from rockpool.networks import NetworkADS
import os
import tempfile
import traceback
import sys

def test_it():

    def test_network_ads(nInSize, nResSize, nOutSize, activation_func):

        w_in = 2*npr.rand(nInSize, nResSize)-1
        w_rec_fast = npr.randn(nResSize, nResSize) / np.sqrt(nResSize)
        w_rec_slow = npr.randn(nResSize, nResSize) / np.sqrt(nResSize)
        w_out = 2*npr.rand(nResSize, nOutSize)-1
        tDt = 0.0001
        noise_std = 0.01
        Nb = nResSize
        M = np.random.randn(Nb,nResSize)
        theta = 0.1*np.random.randn(Nb)
        eta = 0.001
        k=0.1
        v_thresh = np.random.randn(nResSize)
        v_reset = np.random.randn(nResSize)
        v_rest = np.random.randn(nResSize)
        tau_mem = 0.01
        tau_syn_r_fast = 0.02
        tau_syn_r_slow = 0.03
        tau_syn_out = 0.04

        net = NetworkADS.SpecifyNetwork(N=nResSize,
                                Nc=nInSize,
                                Nb=Nb,
                                weights_in=w_in,
                                weights_out= w_out,
                                weights_fast= w_rec_fast,
                                weights_slow=w_rec_slow,
                                M=M,
                                theta=theta,
                                eta=eta,
                                k=k,
                                noise_std=noise_std,
                                dt=tDt,
                                v_thresh=v_thresh,
                                v_reset=v_reset,
                                v_rest=v_rest,
                                tau_mem=tau_mem,
                                tau_syn_r_fast=tau_syn_r_fast,
                                tau_syn_r_slow=tau_syn_r_slow,
                                tau_syn_out=tau_syn_out,
                                phi=activation_func,
                                record=True)

        # Store the network in a temporary file
        with tempfile.TemporaryDirectory() as directory:
            path_to_save = os.path.join(os.path.join(os.getcwd(), directory), "network_ads.json")
            net.save(path_to_save) 

            # Load the network
            network_loaded = NetworkADS.load(path_to_save)

            # Assert that all parameters are the same
            assert(np.array_equal(net.lyrRes.weights_in, network_loaded.lyrRes.weights_in)), "weights_in are not the same"
            assert(np.array_equal(net.lyrRes.weights_out, network_loaded.lyrRes.weights_out)), "weights_out are not the same"
            assert(np.array_equal(net.lyrRes.weights_fast, network_loaded.lyrRes.weights_fast)), "weights_fast are not the same"
            assert(np.array_equal(net.lyrRes.weights_slow, network_loaded.lyrRes.weights_slow)), "weights_slow are not the same"
            assert(np.array_equal(net.lyrRes.M, network_loaded.lyrRes.M)), "M matrix is not the same"
            assert(np.array_equal(net.lyrRes.theta, network_loaded.lyrRes.theta)), "theta is not the same"
            assert(np.array_equal(net.lyrRes.eta, network_loaded.lyrRes.eta)), "eta not the same"
            assert(np.array_equal(net.lyrRes.k, network_loaded.lyrRes.k)), "k not the same"
            assert(np.array_equal(net.lyrRes.noise_std, network_loaded.lyrRes.noise_std)), "noise_std are not the same"
            assert(np.array_equal(net.lyrRes.dt, network_loaded.lyrRes.dt)), "dt not the same"
            assert(np.array_equal(net.lyrRes.v_thresh, network_loaded.lyrRes.v_thresh)), "v_thresh not the same"
            assert(np.array_equal(net.lyrRes.v_reset, network_loaded.lyrRes.v_reset)), "v_reset are not the same"
            assert(np.array_equal(net.lyrRes.v_rest, network_loaded.lyrRes.v_rest)), "v_rest are not the same"
            assert(np.array_equal(net.lyrRes.tau_mem, network_loaded.lyrRes.tau_mem)), "tau_mem are not the same"
            assert(np.array_equal(net.lyrRes.tau_syn_r_fast, network_loaded.lyrRes.tau_syn_r_fast)), "tau_syn_r_fast are not the same"
            assert(np.array_equal(net.lyrRes.tau_syn_r_slow, network_loaded.lyrRes.tau_syn_r_slow)), "tau_syn_r_slow are not the same"
            assert(np.array_equal(net.lyrRes.phi_name, network_loaded.lyrRes.phi_name)), "phi are not the same"

            assert(np.array_equal(net.output_layer.tau_syn, network_loaded.output_layer.tau_syn)), "tau_syn_out of output does not match"

    test_network_ads(2,100,2,"tanh")
    

test_it()
