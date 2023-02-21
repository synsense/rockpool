# -----------------------------------------------------------
# This module provides some test cases for the PDM ADC implemented as a two part module:
# (i)  PDM microphone
# (ii) Low-pass filter + decimation
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 19.01.2023
# -----------------------------------------------------------
from cgi import test
import numpy as np
from xylo_a3_sim.pdm_adc import PDM_ADC


def test_pdm_adc():
    # set all default parameters
    pdm_adc = PDM_ADC()

    print(pdm_adc)


def main():
    test_pdm_adc()


if __name__ == "__main__":
    main()
