# ----------------------------------------------------------------------------------------------------------------------
# This module contains the design specs for Xylo-A3 and the AGC module.
#
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np


# ===========================================================================
# *    some constants defined according to Xylo-A3 specifications
# ===========================================================================
# NOTE: We refer to the following documentation file for further details on the design
# https://spinystellate.office.synsense.ai/saeid.haghighatshoar/agc-for-xylo-v3/blob/master/README.md


# ================================================
# *     power supply voltage specifications
# ================================================
XYLO_VCC = 1.1
XYLO_MAX_AMP_UNIPOLAR = XYLO_VCC / 2.0
XYLO_MAX_AMP = XYLO_VCC  # this is due to the differential design of the amplifier


# ================================================
# *           clock rates on the chip
# ================================================
# highest clock rate on the chip
SYSTEM_CLOCK_RATE = 50_000_000  # 50 MHz

# audio sampling rate obtained after dividing the original clock
AUDIO_SAMPLING_RATE = SYSTEM_CLOCK_RATE / (64 * 16)


# ================================================
# *               ADC parameters
# ================================================
NUM_BITS_AGC_ADC = 10


# ================================================
# *               PGA parameters
# ================================================
# * the settling time (due to working point variation) in PGA
# NOTE: this is at the moment less than half clock period. So no waiting is needed when the gain changes
SETTLING_TIME_PGA = 10e-6

# NOTE: we implement the following simple protocol to take the settling time into account
#   - if the data is still unstable and invalid the amplifier generated None rather than a valid number.
#   - the module receiving amplifier output is ADC:
#       - when it receives a valid data, it converts it into quantized version.
#       - when it receives None, it repeats its previously generated sample.

# * high-pass corner due to AC coupling
HIGH_PASS_CORNER = 50

# * low-pass corner due to frequency response or low-pass filtering (e.g., anti-aliasing low-pass filter)
LOW_PASS_CORNER = 20_000
if LOW_PASS_CORNER > AUDIO_SAMPLING_RATE / 2.0:
    raise ValueError(
        "low-pass corner should be smaller than half the sampling rate of the audio!"
    )

# * maximum target gain for PGA
MAX_PGA_GAIN = 32


# * number of bits assigned to command from envelope controller (EC) to programmable-gain amplifier (PGA)
NUM_BITS_COMMAND = 4

# * gain vector used in the design of AGC
# NOTE: we use an exponential gain pattern but other gain patterns are also possible
EXP_PGA_GAIN_VEC = np.asarray(
    [
        MAX_PGA_GAIN ** (i / (2**NUM_BITS_COMMAND - 1))
        for i in range(2**NUM_BITS_COMMAND)
    ]
)


# * we have a fixed-gain mode for PGA where PGA ignores the gain-change commands it receives from EC module.
# NOTE: in this mode, we set the maximum gain for PGA as default.
DEFAULT_PGA_COMMAND_IN_FIXED_GAIN_FOR_PGA_MODE = 2**NUM_BITS_COMMAND - 1


# ================================================
# *                Envelope Controller
# ================================================

# * some default rise and fall time-constants used for estimating the envelope
RISE_TIME_CONSTANT = 0.1e-3
FALL_TIME_CONSTANT = 300e-3

# * what is the saturation level boundary
# NOTE: we should set the saturation level a little bit lower for two reasons
#       (i)  when it is low, the system is more cautious and, when the signal becomes strong suddenly, goes outside saturation very fast.
#       (ii) at the moment, we are using an oversampled ADC with decimation filter where as a result of processing, the quantized signal may not have full rail-to-rail dynamics
#            due to some inner attenuation. If we use a very large saturation level, the weak signal after attenuation may indeed be in saturation but not get detected by EC.
SATURATION_LEVEL = int(2 ** (NUM_BITS_AGC_ADC - 1) * 0.7)

# * waiting times used for waiting before any gain switch
# NOTE (1): in each region between thresholds we have a waiting time
# so for a 4-bit command, we have 16 + 1 = 17 regions
# However since we never WAIT in the saturation region, we have only 16 waiting times.
# NOTE (2): we design a square-root pattern for waiting times such that waiting times are larger for larger amplitude levels
# The rationale is that at those cases the signal is large enough and we should be patient and cautious in amplifying the signal
# since we may push it into saturation region.
WAITING_TIME_VEC = FALL_TIME_CONSTANT * np.sqrt(np.arange(1, 2**NUM_BITS_COMMAND + 1))

# * maximum waiting time before change gain
# NOTE: This parameter makes sure that the gain change happens with at least some interval independent of how much it is delayed!
MAX_WAITING_TIME_BEFORE_GAIN_CHANGE = np.max(WAITING_TIME_VEC)


# * reliable hysteresis in detecting the maximum
# NOTE: to make sure that waiting times are working well, we need to extend waiting times when the signal amplitude increase is significant
# we measue this by a hysteresis parameter which should be typically around 2 ~ 10 for an ADC with 10 bits
RELIABLE_MAX_HYSTERESIS = max([2, int(2 ** (NUM_BITS_AGC_ADC - 1) / 100)])


# * dynamics of the gain-change at each amplitude level
# NOTE (1): for the generality of the design, we allow the gain index variation at various amplitude levels to be flexible.
# since there are 16 amplitude thresholds, we have 16 + 1 = 17 regions where for each of which we need
# to assign pga_gain_index variation parameter.
# NOTE (2): In brief, this simply means how much the pga_gain should change when the signal after PGA amplifications has a specific amplitude level.
# At the moment, we use the default setting where the pga_gain drops by
#       - [-1] in saturation mode to push the signal outside saturation.
#       - [0 ] in the next two lower amplitude levels to stop over amplification when the signal amplitude changes very fast.
#       - [+1] in the lower amplitude levels to keep amplifying the signal.
# NOTE (3): These numbers are opportunistic in the sense that a requested gain change may be fulfilled if there is resources in PGA.
# More specifically, if the signal is very weak and PGA uses the maximum gain to amplify it but the signal is still very weak, EC may request increasing gain
# but since already all the gain is used up, this will be ignored by PGA.
# NOTE (4): This design guarantees a good AGC performance when the first/fixed amplifier is designed properly such that it does not force the signal into saturation level or
# weak below-noise level.

# * default gain variation
# from weak to saturated mode
#          --->
# [+1, +1, ...., 0, 0, -1]
PGA_GAIN_INDEX_VARIATION = np.ones(2**NUM_BITS_COMMAND + 1, dtype=np.int64)

# saturation: always decrease
PGA_GAIN_INDEX_VARIATION[len(PGA_GAIN_INDEX_VARIATION) - 1] = -1

# two levels below saturation: no change, stay there
# this design seems to work very well for audio signal with spiky nature such as baby-crying.
# For stationary audio such as constant-level music, perhaps, we can set value to 0, +1 to allow some more amplification.
PGA_GAIN_INDEX_VARIATION[len(PGA_GAIN_INDEX_VARIATION) - 2] = 0
PGA_GAIN_INDEX_VARIATION[len(PGA_GAIN_INDEX_VARIATION) - 3] = 0


# * threshold amplitude levels
#! NOTE (1): here we are working with the exponential amplitude levels that match the exponential pattern of the gains sequence in PGA.
#! In general, there should be a matching between the PGA gain sequence and the amplitude thresholds in EC for a good performance.
#! Otherwise, we may push the signal to the saturation mode very fast or we may stop increasing the gain when the signal is weak.

# NOTE (2): we are always using an ascending sequence for PGA gain and also amplitude thresholds
AMPLITUDE_THRESHOLDS = (SATURATION_LEVEL / EXP_PGA_GAIN_VEC).astype(np.int64)[::-1]

# * what is the largest waiting time length covered by the Xylo-A3 -> how many bits suffice?
# NOTE: with a clock rate of 50K, and unsigned format for waiting times, this allows delaying PGA gain adjustment by 2^24/50K = 320 second
# this would be more than enough for all AGC applications in conventional audio and perhaps other closely-related applications
MAX_WAITING_BITWIDTH = 24


# ================================================
# *                Gain Smoother
# ================================================
# * in gain smoother we need to adjust the gain slowly to avoid fast gain jump.
# * since the gain changes at worst every `waiting time` samples, we make sure that gain variation settles down
# * how many time-constants is considered `INFINITY` in the low-pass filter used for gain adjustemt?
INIFINITY_OF_TRANSIENT_PHASE = 6

# * number of bits used for quantizing the gain ratio
NUM_BITS_GAIN_QUANTIZATION = 10
