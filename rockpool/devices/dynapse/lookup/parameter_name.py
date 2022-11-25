"""
DynapSE simulator parameter names and their reciprocals in SE1 and SE2 device configurations
0th column is for simulator, 1st column for SE1 2nd column is for SE2
The 0th column is actually a dummy column and makes it easier to reach the bias parameter name with the version number
table[Itau_ahp][1] -> SE1 bias parameter name
table[Itau_ahp][2] -> SE2 bias parameter name

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
18/01/2022
"""

__all__ = ["sim2device_se1", "sim2device_se2", "device2sim_se1", "device2sim_se2"]


table = {
    "Idc": ["Idc", "IF_DC_P", "SOIF_DC_P"],
    "If_nmda": ["If_nmda", "IF_NMDA_N", "DENM_NMREV_N"],
    "Igain_ahp": ["Igain_ahp", "IF_AHTHR_N", "SOAD_GAIN_P"],
    "Igain_ampa": ["Igain_ampa", "NPDPIE_THR_F_P", "DEAM_EGAIN_P"],
    "Igain_gaba": ["Igain_gaba", "NPDPII_THR_S_P", "DEGA_IGAIN_P"],
    "Igain_nmda": ["Igain_nmda", "NPDPIE_THR_S_P", "DENM_EGAIN_P"],
    "Igain_shunt": ["Igain_shunt", "NPDPII_THR_F_P", "DESC_IGAIN_P"],
    "Igain_mem": ["Igain_mem", "IF_THR_N", "SOIF_GAIN_N"],
    "Ipulse_ahp": ["Ipulse_ahp", None, "SOAD_PWTAU_N"],
    "Ipulse": ["Ipulse", "PULSE_PWLK_P", "SYPD_EXT_N"],
    "Iref": ["Iref", "IF_RFR_N", "SOIF_REFR_N"],
    "Ispkthr": ["Ispkthr", None, "SOIF_SPKTHR_P"],
    "Itau_ahp": ["Itau_ahp", "IF_AHTAU_N", "SOAD_TAU_P"],
    "Itau_ampa": ["Itau_ampa", "NPDPIE_TAU_F_P", "DEAM_ETAU_P"],
    "Itau_gaba": ["Itau_gaba", "NPDPII_TAU_S_P", "DEGA_ITAU_P"],
    "Itau_nmda": ["Itau_nmda", "NPDPIE_TAU_S_P", "DENM_ETAU_P"],
    "Itau_shunt": ["Itau_shunt", "NPDPII_TAU_F_P", "DESC_ITAU_P"],
    "Itau_mem": ["Itau_mem", "IF_TAU1_N", "SOIF_LEAK_N"],
    "Iw_0": ["Iw_0", "PS_WEIGHT_INH_S_N", "SYAM_W0_P"],  # GABA_B SE1
    "Iw_1": ["Iw_1", "PS_WEIGHT_INH_F_N", "SYAM_W1_P"],  # GABA_A SE1
    "Iw_2": ["Iw_2", "PS_WEIGHT_EXC_S_N", "SYAM_W2_P"],  # NMDA SE1
    "Iw_3": ["Iw_3", "PS_WEIGHT_EXC_F_N", "SYAM_W3_P"],  # AMPA SE1
    "Iw_ahp": ["Iw_ahp", "IF_AHW_P", "SOAD_W_N"],
}

sim2device_se1 = {k: v[1] for k, v in table.items() if v[1] is not None}
sim2device_se2 = {k: v[2] for k, v in table.items() if v[2] is not None}

device2sim_se1 = {v: k for k, v in sim2device_se1.items()}
device2sim_se2 = {v: k for k, v in sim2device_se2.items()}
