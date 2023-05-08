# -----------------------------------------------------------
# This module tests the Sample-and-Hold mdoule with different set of parameters.
# 
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 28.08.2022
# -----------------------------------------------------------

# required packages
import numpy as np
from imu_preprocessing.sample_and_hold import SampleAndHold
from imu_preprocessing.quantizer import Quantizer


# varios test modules
def test_sample_and_hold():
    print('\nTesting the sample-and-hold module: random sampling is used!')
    # sample-and-hold module
    sh = SampleAndHold(
        sampling_period=10
    )
    
    # quantizer
    Q = Quantizer()
    
    
    # sample and hold of a signal of size larger than period
    for length in np.arange(2,100):
        sig_in = np.random.randn(length)
        sig_in_q = Q.quantize(
            sig_in,
            scale=0.99/np.max(np.abs(sig_in)),
            num_bits=30,
        )
        
        sig_out_q = sh.evolve(sig_in_q)
        sig_out_q_diff = np.diff(sig_out_q)
        
        
        assert sum(np.abs(sig_out_q_diff)>0) == (length-1)//sh.sampling_period
        
    print('all test were passed successfully!')
    
    

# main module
def main():
    test_sample_and_hold()
    
    
if __name__ == '__main__':
    main()
    print('end of test for sample-and-hold module!')
    
    