# ----
# params.py - Constant hardware parameters
# Author: Felix Bauer, aiCTX AG, felix.bauer@ai-ctx.com
# ----

CORE_DIMENSIONS = (16, 16)  # Numbers of neurons in core (rows, columns)
NUM_NEURONS_CORE = (
    CORE_DIMENSIONS[0] * CORE_DIMENSIONS[1]
)  # Number of neurons on one core
NUM_CORES_CHIP = 4  # Number of cores per chip
NUM_NEURONS_CHIP = NUM_NEURONS_CORE * NUM_CORES_CHIP  # Number of neurons per chip
NUM_CHIPS = 4  # Number of available chips
NUM_CAMS_NEURON = 64  # Fan-in per neuron
NUM_SRAMS_NEURON = 3  # Fan-out chips per neuron
NUM_NEURONS = NUM_CHIPS * NUM_NEURONS_CHIP  # Total numeber of neurons on DynapSE board
# (This can be changed to 4 but then activities of the corresponding neuron cannot be recoreded anymore)
# SYNAPSE_TYPES = ("fast_exc", "slow_exc", "fast_inh", "slow_inh")  # Available synapse types. Currently not supported by VirtualDynapse
BIT_RESOLUTION_WEIGHTS = 1  # Resolution of individual weights in bits
# Standard deviation of the distribution for drawing mismatched parameters (relative to mean)
STDDEV_MISMATCH = 0.2
