"""
Dynap-SE1/SE2 default simulation parameters

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
06/07/2022

[] TODO : merge with definitions
"""

from rockpool.devices.dynapse.definitions import NUM_NEURONS, NUM_CORES, NUM_CHIPS

Itau_lambda = (
    lambda name: (
        (dlayout["Ut"] / ((dlayout["kappa_p"] + dlayout["kappa_n"]) / 2))
        * dlayout[f"C_{name}"]
    )
    / dtime[f"tau_{name}"]
)
Ipw_lambda = lambda name: (dlayout["Vth"] * dlayout[f"C_{name}"]) / dtime[f"t_{name}"]
Igain_lambda = lambda name: Itau_lambda(name) * dgain[f"r_gain_{name}"]

dlayout = {
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

dweight = {
    "Iw_0": 1e-9,
    "Iw_1": 2e-9,
    "Iw_2": 4e-9,
    "Iw_3": 8e-9,
    "Iscale": 1e-8,
}

dtime = {
    "t_pulse_ahp": 1e-12,
    "t_pulse": 10e-6,
    "t_ref": 1e-3,
    "tau_ahp": 50e-3,
    "tau_ampa": 10e-3,
    "tau_gaba": 10e-3,
    "tau_nmda": 10e-3,
    "tau_shunt": 10e-3,
    "tau_mem": 20e-3,
}

dgain = {
    "r_gain_ahp": 1,
    "r_gain_ampa": 100,
    "r_gain_gaba": 100,
    "r_gain_nmda": 100,
    "r_gain_shunt": 100,
    "r_gain_mem": 4,
}

dcurrents = {
    "Idc": dlayout["Io"],
    "If_nmda": dlayout["Io"],
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
    "Iw_ahp": 0.0,
}

CORE_MAP = {i: i // NUM_NEURONS for i in range(NUM_NEURONS * NUM_CORES * NUM_CHIPS)}
CHIP_MAP = {i: i // NUM_CORES for i in range(-NUM_CORES, NUM_CORES * NUM_CHIPS)}
CHIP_POS = {-1: (0, 0), 0: (1, 0), 1: (2, 0), 3: (3, 0), 4: (4, 0)}
