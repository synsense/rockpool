# ----
# params.py - Constant hardware parameters
# Author: Felix Bauer, SynSense AG, felix.bauer@synsense
# ----

SRAM_EVENT_LIMIT = int(2**19 - 1)  # Max. number of events that can be loaded to SRAM
FPGA_EVENT_LIMIT = int(2**16 - 1)  # Max. number of events that can be sent to FPGA
FPGA_ISI_LIMIT = int(
    2**16 - 1
)  # Max. number of timesteps for single inter-spike interval between FPGA events, as well
# as max. value for the isi_multiplier
FPGA_TIMESTEP = 1.0 / 9.0 * 1e-7  # Internal clock of FPGA, 11.111...ns
CORE_DIMENSIONS = (16, 16)  # Numbers of neurons in core (rows, columns)
NUM_NEURONS_CORE = (
    CORE_DIMENSIONS[0] * CORE_DIMENSIONS[1]
)  # Number of neurons on one core
NUM_CORES_CHIP = 4  # Number of cores per chip
NUM_NEURONS_CHIP = NUM_NEURONS_CORE * NUM_CORES_CHIP  # Number of neurons per chip
NUM_CHIPS = 4  # Number of available chips
CAMTYPES = [
    "FAST_EXC",
    "SLOW_EXC",
    "FAST_INH",
    "SLOW_INH",
]  # Names of available synapse types
DEF_CAM_STR = (
    "SLOW_INH"  # Probably. In ctxctl it is `SLOW_EXC` but that seems to be wrong
)

NUM_CAMS_NEURON = 64  # Fan-in per neuron
NUM_SRAMS_NEURON = 3  # Fan-out chips per neuron
NUM_NEURONS = NUM_CHIPS * NUM_NEURONS_CHIP  # Total numeber of neurons on DynapSE board
# (This can be changed to 4 but then activities of the corresponding neuron cannot be recoreded anymore)
# SYNAPSE_TYPES = ("fast_exc", "slow_exc", "fast_inh", "slow_inh")  # Available synapse types. Currently not supported by VirtualDynapse
BIT_RESOLUTION_WEIGHTS = 1  # Resolution of individual weights in bits
# Standard deviation of the distribution for drawing mismatched parameters (relative to mean)
STDDEV_MISMATCH = 0.2
