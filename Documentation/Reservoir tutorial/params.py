# ----
# params.py - Constant hardware parameters
# Author: Felix Bauer, aiCTX AG, felix.bauer@ai-ctx.com
# ----

SRAM_EVENT_LIMIT = int(2 ** 19 - 1)  # Max. number of events that can be loaded to SRAM
FPGA_EVENT_LIMIT = int(2 ** 16 - 1)  # Max. number of events that can be sent to FPGA
FPGA_ISI_LIMIT = int(
    2 ** 16 - 1
)  # Max. number of timesteps for single inter-spike interval between FPGA events
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
    "SLOW_INH"
)  # Probably. In ctxctl it is `SLOW_EXC` but that seems to be wrong
