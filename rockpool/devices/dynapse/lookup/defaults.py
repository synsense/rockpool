"""
Dynap-SE2 default simulation parameters

* Non User Facing *
"""

__all__ = [
    "NUM_CHIPS",
    "NUM_CORES",
    "NUM_NEURONS",
    "NUM_SYNAPSES",
    "NUM_DEST",
    "NUM_TAGS",
    "CORE_MAP",
    "CHIP_MAP",
    "CHIP_POS",
    "default_layout",
    "default_weights",
    "default_time_constants",
    "default_gain_ratios",
    "default_currents",
]

## -- Constants -- ##

NUM_CHIPS = 4
NUM_CORES = 4
NUM_NEURONS = 256
NUM_SYNAPSES = 64
NUM_DEST = 4
NUM_TAGS = 2048
CORE_MAP = [i // NUM_NEURONS for i in range(NUM_NEURONS * NUM_CORES * NUM_CHIPS)]
CHIP_MAP = {i: i // NUM_CORES for i in range(-NUM_CORES, NUM_CORES * NUM_CHIPS)}
CHIP_POS = {-1: (0, 0), 0: (1, 0), 1: (2, 0), 2: (3, 0), 3: (4, 0)}

## -- Maps -- ##

# Neuron ID -> Core ID
CORE_MAP = [i // NUM_NEURONS for i in range(NUM_NEURONS * NUM_CORES * NUM_CHIPS)]

# Core ID -> Chip ID
CHIP_MAP = {i: i // NUM_CORES for i in range(-NUM_CORES, NUM_CORES * NUM_CHIPS)}

# Chip ID -> Chip position in x-y coordinates
CHIP_POS = {-1: (0, 0), 0: (1, 0), 1: (2, 0), 3: (3, 0), 4: (4, 0)}

## -- Some Utilities -- ##

Itau_lambda = (
    lambda name: (
        (
            default_layout["Ut"]
            / ((default_layout["kappa_p"] + default_layout["kappa_n"]) / 2)
        )
        * default_layout[f"C_{name}"]
    )
    / default_time_constants[f"tau_{name}"]
)
Ipw_lambda = (
    lambda name: (default_layout["Vth"] * default_layout[f"C_{name}"])
    / default_time_constants[f"t_{name}"]
)
Igain_lambda = lambda name: Itau_lambda(name) * default_gain_ratios[f"r_gain_{name}"]

## -- Default Parameter Dictionaries -- ##

default_layout = {
    "C_ahp": 40e-12,
    "C_ampa": 24.5e-12,
    "C_gaba": 25e-12,
    "C_nmda": 25e-12,
    "C_pulse_ahp": 0.5e-12,
    "C_pulse": 0.5e-12,
    "C_ref": 1.5e-12,
    "C_shunt": 24.5e-12,
    "C_mem": 3e-12,
    "C_syn": 25e-12,
    "Io": 5e-13,
    "kappa_n": 0.75,
    "kappa_p": 0.66,
    "Ut": 25e-3,
    "Vth": 7e-1,
}

default_weights = {
    "Iw_0": 1e-9,
    "Iw_1": 2e-9,
    "Iw_2": 4e-9,
    "Iw_3": 8e-9,
    "Iscale": 1e-8,
}

default_time_constants = {
    "t_pulse_ahp": 1e-6,
    "t_pulse": 10e-6,
    "t_ref": 1e-3,
    "tau_ahp": 50e-3,
    "tau_ampa": 10e-3,
    "tau_gaba": 10e-3,
    "tau_nmda": 10e-3,
    "tau_shunt": 10e-3,
    "tau_mem": 20e-3,
}

default_gain_ratios = {
    "r_gain_ahp": 1,
    "r_gain_ampa": 100,
    "r_gain_gaba": 100,
    "r_gain_nmda": 100,
    "r_gain_shunt": 100,
    "r_gain_mem": 4,
}

default_currents = {
    "Idc": default_layout["Io"],
    "If_nmda": default_layout["Io"],
    "Igain_ahp": Igain_lambda("ahp"),
    "Igain_ampa": Igain_lambda("ampa"),
    "Igain_gaba": Igain_lambda("gaba"),
    "Igain_nmda": Igain_lambda("nmda"),
    "Igain_shunt": Igain_lambda("shunt"),
    "Igain_mem": Igain_lambda("mem"),
    "Ipulse_ahp": Ipw_lambda("pulse_ahp"),
    "Ipulse": Ipw_lambda("pulse"),
    "Iref": Ipw_lambda("ref"),
    "Ispkthr": 1e-7,
    "Itau_ahp": Itau_lambda("ahp"),
    "Itau_ampa": Itau_lambda("ampa"),
    "Itau_gaba": Itau_lambda("gaba"),
    "Itau_nmda": Itau_lambda("nmda"),
    "Itau_shunt": Itau_lambda("shunt"),
    "Itau_mem": Itau_lambda("mem"),
    "Iw_ahp": 5e-13,
}
