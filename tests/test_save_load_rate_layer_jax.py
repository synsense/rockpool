"""
Test that creates layer of type RecRateEulerJax and ForceRateEulerJax with random connections
saves it as a JSON file under os.getcwd()/tmp_rec_rate_jax{tmp_force_rate_jax}.json
loads it and checks if all elements are the same.
It will also check the activation function used in both cases.
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import numpy.random as npr
from rockpool.layers import RecRateEulerJax, ForceRateEulerJax, H_ReLU, H_tanh
import os
import tempfile
import traceback
import sys

def test_it():

    def test_rate_euler_jax_save_and_load(nInSize, nResSize, nOutSize, activation_func, cls):

        w_in = 2*npr.rand(nInSize, nResSize)-1
        w_rec = npr.randn(nResSize, nResSize) / np.sqrt(nResSize)
        w_out = 2*npr.rand(nResSize, nOutSize)-1
        tau = np.random.randn(nResSize)
        bias = np.random.randn(nResSize)
        bias = 0
        tDt = 0.0001
        noise_std = 0.01

        # Create layer
        if(cls == RecRateEulerJax):
            lyr_rate_jax = cls(w_in, w_rec, w_out, tau, bias,
                                    dt = tDt, noise_std = noise_std,
                                    activation_func = activation_func)
        else:
            lyr_rate_jax = cls(w_in, w_out, tau, bias,
                                    dt = tDt, noise_std = noise_std,
                                    activation_func = activation_func)

        # Create tmp folder
        with tempfile.TemporaryDirectory() as directory:
            path_to_save = os.path.join(os.path.join(os.getcwd(), directory), "tmp_rate_jax.json")
            lyr_rate_jax.save_layer(path_to_save)
            

            # Load the layer
            lyr_rate_jax_loaded = cls.load_from_file(path_to_save)

            assert(np.array_equal(lyr_rate_jax_loaded.w_in, lyr_rate_jax.w_in)), "Stored W_in does not match saved W_in"
            if(cls == RecRateEulerJax):
                assert(np.array_equal(lyr_rate_jax_loaded.w_recurrent, lyr_rate_jax.w_recurrent)), "Stored W_recurrent does not match saved W_recurrent"
            assert(np.array_equal(lyr_rate_jax_loaded.w_out, lyr_rate_jax.w_out)), "Stored W_out does not match saved W_out"
            assert(np.array_equal(lyr_rate_jax_loaded.tau, lyr_rate_jax.tau)), "Stored tau does not match saved tau"
            assert(np.array_equal(lyr_rate_jax_loaded.bias, lyr_rate_jax.bias)), "Stored bias does not match saved bias"
            assert(np.array_equal(lyr_rate_jax_loaded.noise_std, lyr_rate_jax.noise_std)), "Stored noise_std does not match saved noise_std"
            assert(np.array_equal(lyr_rate_jax_loaded.dt, lyr_rate_jax.dt)), "Stored dt does not match saved dt"
            assert(lyr_rate_jax_loaded._H == lyr_rate_jax._H), "Stored activation_func does not match saved activation_func"


    test_rate_euler_jax_save_and_load(2,100,2,H_tanh,RecRateEulerJax)
    test_rate_euler_jax_save_and_load(2,100,2,H_tanh,ForceRateEulerJax)


test_it()